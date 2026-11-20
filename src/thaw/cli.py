"""Run one prompt or a JSONL batch with fixed backbone settings."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import tempfile

import torch

from .pipelines import generate_frames, load_pipeline
from .presets import PRESETS, WAN_NEGATIVE_PROMPT

DEFAULT_MODEL = "wan-1.3b"


def read_jobs(args):
    if args.prompt is not None:
        if not args.prompt.strip():
            raise ValueError("The prompt must contain text.")
        return [dict(id=args.output.stem, prompt=args.prompt, image=args.image, seed=args.seed)]

    jobs = []
    seen = set()
    for line, entry in enumerate(args.prompts.read_text(encoding="utf-8").splitlines(), 1):
        if not entry.strip():
            continue
        row = json.loads(entry)
        if not isinstance(row, dict):
            raise ValueError(f"Batch line {line} must be a JSON object.")
        identifier, prompt = row.get("id"), row.get("prompt")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier):
            raise ValueError(f"Batch line {line} needs an id containing letters, numbers, underscores or hyphens.")
        if identifier in seen:
            raise ValueError(f"Duplicate batch id: {identifier}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Batch line {line} needs a nonempty prompt.")
        image = row.get("image")
        seed = row.get("seed", args.seed)
        if image is not None and (not isinstance(image, str) or not image):
            raise ValueError(f"Batch line {line} needs a valid image path.")
        if type(seed) is not int:
            raise ValueError(f"Batch line {line} needs an integer seed.")
        jobs.append(dict(
            id=identifier, prompt=prompt, seed=seed,
            image=args.prompts.parent / image if image is not None else None,
        ))
        seen.add(identifier)
    if not jobs:
        raise ValueError("The JSONL file contains no prompts.")
    return jobs


def save_video(frames, output: Path, fps: int):
    from diffusers.utils import export_to_video

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        candidate = Path(temporary) / "video.mp4"
        export_to_video(frames, str(candidate), fps=fps)
        candidate.replace(output)


def video_path(args, job, tag):
    target = args.output if args.prompt is not None else args.output_dir / f"{job['id']}.mp4"
    if args.paired:
        return target.with_name(f"{target.stem}_{tag}{target.suffix}")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(prog="thaw", description=__doc__)
    parser.add_argument("--model", choices=PRESETS, default=DEFAULT_MODEL)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prompt")
    modes.add_argument("--prompts", type=Path, help="JSONL with id, prompt and optional image and seed.")
    parser.add_argument("--negative-prompt", help="Defaults to the negative condition of the chosen preset.")
    parser.add_argument("--image", type=Path, help="Conditioning image for one wan-i2v-14b prompt.")
    parser.add_argument("--output", type=Path, default=Path("outputs/thaw.mp4"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory in batch mode.")
    parser.add_argument("--paired", action="store_true", help="Generate matched stock CFG and THAW clips.")
    parser.add_argument("--seed", type=int, default=42)
    for flag in ("steps", "frames", "height", "width"):
        parser.add_argument(f"--{flag}", type=int, help="Defaults to the selected model's paper setting.")
    parser.add_argument("--strength", type=float, help="Residual strength d.")
    parser.add_argument("--window", type=float, help="Fraction of steps in the initial window.")
    parser.add_argument("--device", default="cuda:0", help="CUDA device, such as cuda:0.")
    parser.add_argument("--offload", choices=("model", "sequential", "none"), default="model")
    parser.add_argument("--checkpoint", type=Path, help="Compatible local directory for the chosen model.")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    preset = PRESETS[args.model]
    values = {
        key: getattr(args, key) if getattr(args, key) is not None else getattr(preset, key)
        for key in ("height", "width", "frames", "steps", "strength", "window")
    }
    if values["steps"] <= 0:
        parser.error("--steps must be positive.")
    if any(values[k] <= 0 or values[k] % preset.spatial_stride for k in ("height", "width")):
        parser.error(f"Height and width must be positive multiples of {preset.spatial_stride}.")
    frames = values["frames"]
    if frames < 1 or (frames - 1) % preset.temporal_stride:
        parser.error(f"Frame count must be of the form {preset.temporal_stride}k+1.")
    if not math.isfinite(values["strength"]) or values["strength"] < 0:
        parser.error("--strength must be finite and nonnegative.")
    if not 0 <= values["window"] <= 1:
        parser.error("--window must be between 0 and 1.")
    active = round(values["window"] * values["steps"])
    if args.paired and (values["strength"] == 0 or active == 0):
        parser.error("--paired needs a nonzero strength and an active guidance window.")
    if values["strength"] > 0 and active and (frames - 1) // preset.temporal_stride < (2 if preset.causal else 1):
        parser.error("Choose enough latent frames for the selected temporal projection.")
    if args.prompt is not None and args.output.suffix.lower() != ".mp4":
        parser.error("--output must end in .mp4.")
    if args.prompts is not None and args.image is not None:
        parser.error("Specify batch images on individual JSONL rows.")

    try:
        jobs = read_jobs(args)
        device = torch.device(args.device)
    except (OSError, ValueError, RuntimeError) as e:
        parser.error(str(e))
    for job in jobs:
        image = job["image"]
        if (image is not None) != (preset.backend == "wan_i2v"):
            parser.error("Every wan-i2v-14b prompt needs an image. The other presets use text only.")
        if image is not None and not image.is_file():
            parser.error(f"Could not open image: {image}")
    if args.checkpoint is not None and not args.checkpoint.is_dir():
        parser.error("--checkpoint must name an existing Diffusers checkpoint directory.")
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("A CUDA device with bfloat16 support is required.")
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            parser.error("The chosen CUDA GPU does not support bfloat16.")

    pipe = load_pipeline(
        preset, checkpoint=args.checkpoint, offload=args.offload,
        local_files_only=args.local_files_only, device=device,
    )
    metadata = (args.output.parent / f"{args.output.stem}.jsonl") if args.prompt is not None else (
        args.output_dir / "generations.jsonl"
    )
    metadata.parent.mkdir(parents=True, exist_ok=True)
    effective_negative = args.negative_prompt if args.negative_prompt is not None else (
        WAN_NEGATIVE_PROMPT if preset.backend == "wan" else ""
    )
    with metadata.open("w", encoding="utf-8") as records:
        for job in jobs:
            strength = values["strength"]
            variants = (("stock", 0.0), ("thaw", strength)) if args.paired else (
                ("thaw" if strength and active else "stock", strength),
            )
            for tag, dose in variants:
                output = video_path(args, job, tag)
                config = {**values, "strength": dose}
                frames = generate_frames(
                    pipe, preset, prompt=job["prompt"], image=job["image"], seed=job["seed"],
                    negative_prompt=effective_negative, **config,
                )
                save_video(frames, output, preset.fps)
                record = {
                    **asdict(preset), **config, "model": args.model, "method": tag,
                    "active_steps": active if dose else 0, "prompt": job["prompt"],
                    "negative_prompt": effective_negative, "seed": job["seed"],
                    "image": job["image"].name if job["image"] else None,
                    "video": output.name, "id": job["id"],
                }
                if args.checkpoint is not None:
                    record.update(checkpoint=args.checkpoint.name, revision=None)
                records.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.flush()
                print(f"Saved {output}")
