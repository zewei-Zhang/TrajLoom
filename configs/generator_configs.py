"""
Typed config models for TrajLoom generator training and inference.

These dataclasses mirror the JSON layout used by the current TrajLoom
generator config files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional

AxisOrder = Literal["xy", "yx"]


@dataclass
class TrajLoomVAECheckpointConfig:
    """Checkpoint reference plus embedded TrajLoom VAE architecture config."""

    ckpt_path: str
    config: Dict[str, Any]


# Backward-compatible alias used by older scripts and config payloads.
TrackVAECheckpointConfig = TrajLoomVAECheckpointConfig


@dataclass
class WanVideoVAEConfig:
    """Config for optional WAN video-VAE conditioning."""

    enable: bool = True
    wan2_repo_dir: str = ""  # Wan 2.1 repository root.
    ckpt_dir: str = ""  # Path containing or pointing to the WAN VAE checkpoint.
    ckpt_name: str = "Wan2.1_VAE.pth"
    dtype: str = "bf16"  # One of: bf16, fp16, fp32.
    latents_dir: str = ""  # Optional precomputed latents directory; empty means encode on the fly.
    resize_factor: int = 4  # Spatial downsample before WAN VAE encode.


@dataclass
class WanT5Config:
    """Config for optional WAN T5 text conditioning."""

    enable: bool = True
    wan2_repo_dir: str = ""  # Wan 2.1 repository root.
    ckpt_dir: str = ""  # Directory containing the T5 encoder checkpoint.
    ckpt_name: str = "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_name: str = "google/umt5-xxl"
    max_length: int = 128
    dropout_prob: float = 0.0
    dtype: str = "bf16"  # Kept for config compatibility; some wrappers ignore it.


@dataclass
class TrainConfig:
    """Training, evaluation, and logging settings for the generator."""

    resume: str = None
    out_dir: str = "/path/to/output/track_generator/"
    batch_size: int = 1
    num_workers: int = 4

    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"  # One of: bf16, fp16, no.

    log_every: int = 10
    save_every: int = 1000
    val_every: int = 500
    preview_every: int = 500
    max_val_batches: int = 2

    preview_steps: int = 16

    lambda_vis: float = 0.1
    vis_pos_weight: float = 4.0
    clip_grad_norm: float = 1.0

    latent_stats_json: str = ""
    latent_stats_key: str = "z_fut"

    fm_sigma: float = 0.0
    t_small_prob: float = 0.0
    t_small_max: float = 0.1

    x0_mode: str = "noise"
    x0_noise_std: float = 0.5
    x0_fix_first_latent: bool = False

    fm_use_vis_mask: bool = True
    fm_invis_weight: float = 0.1

    lambda_invis_v: float = 0.05

    seed: int = 42

    wandb_project: str = "track_generator"
    entity: str = "video-traj"
    wandb_name: str = "v001"


@dataclass
class TrajLoomGeneratorConfig:
    """Core TrajLoom generator architecture settings."""

    latent_channels: int = 16
    num_frames_latent: int = 21

    input_height: int = 480
    input_width: int = 832
    patch_size: int = 8

    hidden_size: int = 384
    num_heads: int = 6
    depth: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    use_video_cond: bool = True
    video_latent_channels: int = 16
    use_text_cond: bool = True
    text_embed_dim: int = 4096

    num_frames_in: int = 81
    latent_time_stride: int = 4
    use_overlap: bool = False

    temporal_first: bool = True
    spatial_ctx_mode: str = "pooled_last"
    spatial_ctx_pool: int = 8
    spatial_ctx_last_k: int = 1
    spatial_ctx_include_video: bool = True

    video_ctx_mode: str = "sum"
    track_fusion: str = "add_last_plus_vel"
    track_fusion_scale_init: float = 0.2
    track_fusion_use_time_gate: bool = True
    video_fusion: str = "none"
    video_fusion_scale_init: float = 0.05

    use_hist_vis_cond: bool = False
    hist_vis_scale_init: float = 1.0


@dataclass
class GeneratorDatasetConfig:
    """Dataset, temporal slicing, and latent-cache settings."""

    video_dir: str
    tracks_dir: str

    hist_len: int = 81
    future_len: int = 81
    use_overlap: bool = False

    raw_width: int = 832
    raw_height: int = 480

    track_stride: int = 8

    axis_order: AxisOrder = "xy"

    video_reader: str = "cv2"
    video_exts: str = ".mp4"
    max_frames: Optional[int] = 162

    train_fraction: float = 0.9
    seed: int = 42

    caption_csv: str = ""
    caption_text_col: str = "text"
    caption_video_col: str = "video_path"
    caption_vid_col: str = "videoid"

    use_cached_latents: bool = False
    cache_dir: str = ""
    cache_manifest_train: str = ""
    cache_manifest_eval: str = ""
    cache_manifest_val: str = ""


@dataclass(init=False)
class TrajLoomGeneratorFullConfig:
    """Top-level generator config with support for legacy `track_vae` payloads."""

    data: GeneratorDatasetConfig
    trajloom_vae: TrajLoomVAECheckpointConfig
    video_vae: WanVideoVAEConfig
    text_encoder: WanT5Config
    generator: TrajLoomGeneratorConfig
    train: TrainConfig

    def __init__(
        self,
        *,
        data: GeneratorDatasetConfig,
        trajloom_vae: Optional[TrajLoomVAECheckpointConfig] = None,
        track_vae: Optional[TrajLoomVAECheckpointConfig] = None,
        video_vae: Optional[WanVideoVAEConfig] = None,
        text_encoder: Optional[WanT5Config] = None,
        generator: TrajLoomGeneratorConfig,
        train: Optional[TrainConfig] = None,
    ) -> None:
        if trajloom_vae is None:
            trajloom_vae = track_vae
        if trajloom_vae is None:
            raise ValueError("Expected trajloom_vae (preferred) or track_vae.")
        self.data = data
        self.trajloom_vae = trajloom_vae
        self.video_vae = video_vae if video_vae is not None else WanVideoVAEConfig()
        self.text_encoder = text_encoder if text_encoder is not None else WanT5Config()
        self.generator = generator
        self.train = train if train is not None else TrainConfig()

    @property
    def track_vae(self) -> TrajLoomVAECheckpointConfig:
        return self.trajloom_vae

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajLoomGeneratorFullConfig":
        trajloom_vae = d.get("trajloom_vae", d.get("track_vae"))
        if trajloom_vae is None:
            raise KeyError("Expected 'trajloom_vae' (preferred) or 'track_vae' in generator config.")
        return cls(
            data=GeneratorDatasetConfig(**d["data"]),
            trajloom_vae=TrajLoomVAECheckpointConfig(**trajloom_vae),
            video_vae=WanVideoVAEConfig(**d.get("video_vae", {})),
            text_encoder=WanT5Config(**d.get("text_encoder", {})),
            generator=TrajLoomGeneratorConfig(**d["generator"]),
            train=TrainConfig(**d.get("train", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


FullConfig = TrajLoomGeneratorFullConfig


__all__ = [
    "AxisOrder",
    "TrajLoomVAECheckpointConfig",
    "TrackVAECheckpointConfig",
    "WanVideoVAEConfig",
    "WanT5Config",
    "TrainConfig",
    "TrajLoomGeneratorConfig",
    "GeneratorDatasetConfig",
    "TrajLoomGeneratorFullConfig",
    "FullConfig",
]
