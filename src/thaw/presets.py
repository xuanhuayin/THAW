"""Public checkpoints and settings for the final THAW method."""
from dataclasses import dataclass
from typing import Literal

Backend = Literal["wan", "wan_i2v", "cogvideox", "ltx", "hunyuan"]


@dataclass(frozen=True)
class Preset:
    backend: Backend
    checkpoint: str
    revision: str
    width: int
    height: int
    frames: int
    steps: int
    cfg: float
    strength: float
    window: float
    causal: bool
    fps: int
    spatial_stride: int
    temporal_stride: int
    flow_shift: float | None = None


PRESETS = {
    "wan-1.3b": Preset(
        "wan", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0fad780a534b6463e45facd96134c9f345acfa5b",
        832, 480, 49, 30, 5, 2, 0.4, True, 16, 16, 4, 3.0,
    ),
    "wan-14b": Preset(
        "wan", "Wan-AI/Wan2.1-T2V-14B-Diffusers", "38ec498cb3208fb688890f8cc7e94ede2cbd7f68",
        832, 480, 81, 50, 5, 2, 0.4, True, 16, 16, 4, 3.0,
    ),
    "cogvideox-5b": Preset(
        "cogvideox", "zai-org/CogVideoX-5b", "8fc5b281006c82b82d34fd2543d2f0ebb4e7e321",
        720, 480, 49, 50, 6, 3, 0.4, False, 8, 16, 4,
    ),
    "ltx-video-2b": Preset(
        "ltx", "Lightricks/LTX-Video-0.9.5", "e58e28c39631af4d1468ee57a853764e11c1d37e",
        704, 512, 161, 50, 3, 4, 0.2, True, 25, 32, 8,
    ),
    "hunyuanvideo-1.5": Preset(
        "hunyuan", "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "286be7ce72277246578a3e3cc2487e95ddae5bcf",
        848, 480, 81, 50, 6, 4, 0.6, True, 24, 16, 4,
    ),
    "wan-i2v-14b": Preset(
        "wan_i2v", "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "b184e23a8a16b20f108f727c902e769e873ffc73",
        832, 480, 81, 50, 5, 2, 0.4, True, 16, 16, 4, 5.0,
    ),
}

# Default negative prompt from the Wan text-to-video configuration.
WAN_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
