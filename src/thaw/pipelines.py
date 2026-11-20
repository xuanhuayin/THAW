"""Use the native Diffusers samplers and replace only the CFG prediction."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import torch

from .guidance import add_temporal_guidance
from .presets import WAN_NEGATIVE_PROMPT, Backend, Preset


@contextmanager
def use_thaw(
    pipeline,
    *,
    backend: Backend,
    strength: float,
    active_steps: int,
    causal: bool,
    latent_shape: tuple[int, int, int],
):
    """Apply THAW during a single pipeline call with standard CFG enabled."""
    if strength == 0 or active_steps == 0:
        yield
        return

    predictions: list[torch.Tensor] = []
    step_index = 0
    original_step = pipeline.scheduler.step
    batched_cfg = backend in ("cogvideox", "ltx")

    def capture(_module, _inputs, output):
        if step_index < active_steps:
            predictions.append(output[0])

    def update(stock: torch.Tensor) -> torch.Tensor:
        if batched_cfg:
            if len(predictions) != 1 or predictions[0].shape[0] != 2 * stock.shape[0]:
                raise RuntimeError("Expected one batched unconditional/conditional CFG prediction.")
            baseline, conditional = predictions.pop().chunk(2)
        else:
            if len(predictions) != 2:
                raise RuntimeError("Expected two CFG predictions in conditional/unconditional order.")
            conditional, baseline = predictions
            predictions.clear()

        if backend == "cogvideox":
            # CogVideoX uses [B, F, C, H, W].
            result = add_temporal_guidance(
                stock.transpose(1, 2), conditional.transpose(1, 2), baseline.transpose(1, 2),
                strength=strength, causal=causal,
            )
            return result.transpose(1, 2).contiguous()
        if backend == "ltx":
            # LTX packs its video latents into a token sequence.
            frames, height, width = latent_shape
            spatial_patch = pipeline.transformer_spatial_patch_size
            temporal_patch = pipeline.transformer_temporal_patch_size
            shape = (frames // temporal_patch, height // spatial_patch, width // spatial_patch)
            cfg_5d, cond_5d, base_5d = (
                pipeline._unpack_latents(t, *shape, spatial_patch, temporal_patch)
                for t in (stock, conditional, baseline)
            )
            result = add_temporal_guidance(
                cfg_5d, cond_5d, base_5d, strength=strength, causal=causal
            )
            return pipeline._pack_latents(result, spatial_patch, temporal_patch)

        return add_temporal_guidance(stock, conditional, baseline, strength=strength, causal=causal)

    @wraps(original_step)
    def step(model_output, *args, **kwargs):
        nonlocal step_index
        if step_index < active_steps:
            model_output = update(model_output)
        step_index += 1
        return original_step(model_output, *args, **kwargs)

    handle = pipeline.transformer.register_forward_hook(capture)
    try:
        with patch.object(pipeline.scheduler, "step", step):
            yield
    finally:
        handle.remove()


def load_pipeline(
    preset: Preset,
    *,
    checkpoint: Path | None,
    offload: str,
    local_files_only: bool,
    device: torch.device,
):
    from diffusers import (
        AutoencoderKLWan,
        ClassifierFreeGuidance,
        CogVideoXPipeline,
        HunyuanVideo15Pipeline,
        LTXPipeline,
        UniPCMultistepScheduler,
        WanImageToVideoPipeline,
        WanPipeline,
    )

    classes = {
        "wan": WanPipeline,
        "wan_i2v": WanImageToVideoPipeline,
        "cogvideox": CogVideoXPipeline,
        "ltx": LTXPipeline,
        "hunyuan": HunyuanVideo15Pipeline,
    }
    model = str(checkpoint) if checkpoint is not None else preset.checkpoint
    options = dict(local_files_only=local_files_only, use_safetensors=True)
    if checkpoint is None:
        options["revision"] = preset.revision

    components = {}
    if preset.backend in ("wan", "wan_i2v"):
        components["vae"] = AutoencoderKLWan.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32, **options
        )
    pipe = classes[preset.backend].from_pretrained(
        model, torch_dtype=torch.bfloat16, **components, **options
    )

    if preset.backend in ("wan", "wan_i2v"):
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=preset.flow_shift
        )
    if preset.backend == "hunyuan":
        pipe.guider = ClassifierFreeGuidance(guidance_scale=preset.cfg)
        pipe.vae.enable_tiling()

    if offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    elif offload == "model":
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    return pipe


def generate_frames(
    pipe,
    preset: Preset,
    *,
    prompt: str,
    negative_prompt: str | None,
    image: Path | None,
    height: int,
    width: int,
    frames: int,
    steps: int,
    strength: float,
    window: float,
    seed: int,
):
    from PIL import Image

    if negative_prompt is None:
        negative_prompt = WAN_NEGATIVE_PROMPT if preset.backend == "wan" else ""
    inputs = dict(
        prompt=prompt, negative_prompt=negative_prompt, height=height, width=width,
        num_frames=frames, num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="np",
    )
    if preset.backend != "hunyuan":
        inputs["guidance_scale"] = preset.cfg
    if preset.backend == "cogvideox":
        inputs.update(use_dynamic_cfg=False, eta=0.0)
    if preset.backend == "ltx":
        inputs.update(frame_rate=preset.fps, decode_timestep=0.05, decode_noise_scale=0.025, guidance_rescale=0.0)
    if image is not None:
        with Image.open(image) as source:
            inputs["image"] = source.convert("RGB")

    latent_shape = (
        (frames - 1) // preset.temporal_stride + 1,
        height // preset.spatial_stride,
        width // preset.spatial_stride,
    )
    with use_thaw(
        pipe, backend=preset.backend, strength=strength, active_steps=round(window * steps),
        causal=preset.causal, latent_shape=latent_shape,
    ):
        return pipe(**inputs).frames[0]
