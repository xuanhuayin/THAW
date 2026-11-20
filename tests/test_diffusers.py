"""Exercise upstream sampling loops without downloading checkpoint weights."""
import numpy as np
import pytest
import torch
from PIL import Image

from thaw.pipelines import use_thaw

diffusers = pytest.importorskip("diffusers")

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


def run_pair(pipe, inputs, backend, shape, forwards, scale=2):
    pipe.set_progress_bar_config(disable=True)
    calls = []
    observer = pipe.transformer.register_forward_hook(lambda *_: calls.append(1))
    outputs = []
    try:
        for strength in (0, scale):
            calls.clear()
            with use_thaw(
                pipe, backend=backend, strength=strength, active_steps=2, causal=True,
                latent_shape=shape,
            ):
                result = pipe(**inputs, generator=torch.Generator().manual_seed(101)).frames
            assert len(calls) == forwards
            assert torch.isfinite(result).all()
            outputs.append(result)
    finally:
        observer.remove()
    assert not pipe.transformer._forward_hooks
    assert not torch.equal(*outputs)


def test_native_wan_i2v():
    from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

    torch.manual_seed(71)
    pipe = diffusers.WanImageToVideoPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=diffusers.AutoencoderKLWan(
            base_dim=8, z_dim=4, dim_mult=[1, 2, 2, 4], num_res_blocks=1,
            latents_mean=[0.0] * 4, latents_std=[1.0] * 4,
        ),
        scheduler=diffusers.UniPCMultistepScheduler(
            use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=5.0
        ),
        image_processor=CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}),
        image_encoder=CLIPVisionModel(CLIPVisionConfig(
            image_size=32, patch_size=16, hidden_size=16, intermediate_size=32,
            num_attention_heads=2, num_hidden_layers=2,
        )),
        transformer=diffusers.WanTransformer3DModel(
            in_channels=12, out_channels=4, text_dim=16, freq_dim=8, ffn_dim=32, num_layers=1,
            num_attention_heads=2, attention_head_dim=8, image_dim=16, added_kv_proj_dim=16,
            rope_max_seq_len=32,
        ),
    )
    inputs = dict(
        image=Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)),
        height=32, width=32, num_frames=9, num_inference_steps=4, guidance_scale=5.0,
        prompt_embeds=torch.randn(1, 512, 16), negative_prompt_embeds=torch.randn(1, 512, 16),
        output_type="latent",
    )
    run_pair(pipe, inputs, "wan_i2v", (3, 4, 4), forwards=8)


def test_native_ltx_pipeline():
    torch.manual_seed(73)
    pipe = diffusers.LTXPipeline(
        tokenizer=None, text_encoder=None, vae=None,
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler(shift=1.0),
        transformer=diffusers.LTXVideoTransformer3DModel(
            in_channels=4, out_channels=4, caption_channels=16, cross_attention_dim=24,
            num_attention_heads=2, attention_head_dim=12, num_layers=1,
        ),
    )
    inputs = dict(
        height=64, width=64, num_frames=17, num_inference_steps=4, guidance_scale=3.0,
        prompt_embeds=torch.randn(1, 8, 16), negative_prompt_embeds=torch.randn(1, 8, 16),
        prompt_attention_mask=torch.ones(1, 8), negative_prompt_attention_mask=torch.ones(1, 8),
        output_type="latent",
    )
    run_pair(pipe, inputs, "ltx", (3, 2, 2), forwards=4, scale=4)


def test_native_hunyuan_pipeline():
    torch.manual_seed(75)
    pipe = diffusers.HunyuanVideo15Pipeline(
        text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None,
        vae=diffusers.AutoencoderKLHunyuanVideo15(
            latent_channels=4, block_out_channels=(32, 32, 32, 32, 32), layers_per_block=1,
        ),
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler(shift=5.0),
        guider=diffusers.ClassifierFreeGuidance(guidance_scale=6.0),
        transformer=diffusers.HunyuanVideo15Transformer3DModel(
            in_channels=9, out_channels=4, text_embed_dim=16, text_embed_2_dim=8,
            image_embed_dim=16, rope_axes_dim=(8, 8, 8), num_attention_heads=2,
            attention_head_dim=24, num_layers=1, num_refiner_layers=1, task_type="t2v",
        ),
    )
    mask = torch.ones(1, 8, dtype=torch.bool)
    inputs = dict(
        height=32, width=32, num_frames=9, num_inference_steps=4, output_type="latent",
        prompt_embeds=torch.randn(1, 8, 16), negative_prompt_embeds=torch.randn(1, 8, 16),
        prompt_embeds_mask=mask, negative_prompt_embeds_mask=mask,
        prompt_embeds_2=torch.randn(1, 8, 8), negative_prompt_embeds_2=torch.randn(1, 8, 8),
        prompt_embeds_mask_2=mask, negative_prompt_embeds_mask_2=mask,
    )
    run_pair(pipe, inputs, "hunyuan", (3, 2, 2), forwards=8, scale=4)
