"""
Config helpers for the standalone TrajLoom VAE settings.

This module mirrors the `data` and `trajloom_vae` sections used by
`configs/trajloom_generator_config.json` and consumed by
`run_trajloom_vae_recon.py`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Literal, Optional, Tuple


AxisOrder = Literal["xy", "yx"]


@dataclass
class TrajLoomVAEConfig:
    encoder_depth: int = 16
    decoder_depth: int = 16
    hidden_size: int = 512
    num_heads: int = 8
    patch_size: int = 32
    in_channels: int = 2
    latent_channels: int = 16
    input_height: int = 480
    input_width: int = 832
    num_frames_in: int = 81
    num_frames_latent: int = 21
    use_temp_compress: bool = True
    temp_stride: int = 4
    use_offsets: bool = True
    num_frame_tokens: int = 256
    learn_sigma: bool = False
    beta: float = 5e-5
    pos_embed_dim: int = 64


# Backward-compatible alias used by older code paths.
WanVAEConfig = TrajLoomVAEConfig


@dataclass
class TrajLoomVAECheckpointConfig:
    ckpt_path: str = "/path/to/checkpoints/trajloom_vae.pt"
    config: Dict[str, Any] = field(default_factory=lambda: asdict(TrajLoomVAEConfig()))


# Backward-compatible alias used by generator config helpers.
TrackVAECheckpointConfig = TrajLoomVAECheckpointConfig


@dataclass(init=False)
class TrajLoomVAEDataConfig:
    video_dir: str = "/path/to/data/videos/"
    trajectory_dir: str = "/path/to/data/trajectory/"
    caption_csv: str = "/path/to/data/captions.csv"
    caption_video_col: str = "video_path"
    caption_text_col: str = "text"
    caption_vid_col: str = "videoid"
    raw_height: int = 480
    raw_width: int = 832
    hist_len: int = 81
    future_len: int = 81
    use_overlap: bool = False
    track_stride: int = 32
    axis_order: AxisOrder = "xy"
    video_reader: str = "cv2"
    video_exts: str = ".mp4"
    max_frames: Optional[int] = 162
    train_fraction: float = 0.9
    seed: int = 42
    use_cached_latents: bool = True
    cache_dir: str = "/path/to/data/cache/"
    cache_manifest_train: str = "/path/to/data/cache/manifest_train.jsonl"
    cache_manifest_eval: str = "/path/to/data/cache/manifest_eval.jsonl"
    cache_manifest_val: str = ""

    def __init__(
        self,
        *,
        video_dir: str = "/path/to/data/videos/",
        trajectory_dir: Optional[str] = None,
        tracks_dir: Optional[str] = None,
        caption_csv: str = "/path/to/data/captions.csv",
        caption_video_col: str = "video_path",
        caption_text_col: str = "text",
        caption_vid_col: str = "videoid",
        raw_height: int = 480,
        raw_width: int = 832,
        hist_len: int = 81,
        future_len: int = 81,
        use_overlap: bool = False,
        track_stride: int = 32,
        axis_order: AxisOrder = "xy",
        video_reader: str = "cv2",
        video_exts: str = ".mp4",
        max_frames: Optional[int] = 162,
        train_fraction: float = 0.9,
        seed: int = 42,
        use_cached_latents: bool = True,
        cache_dir: str = "/path/to/data/cache/",
        cache_manifest_train: str = "/path/to/data/cache/manifest_train.jsonl",
        cache_manifest_eval: str = "/path/to/data/cache/manifest_eval.jsonl",
        cache_manifest_val: str = "",
    ) -> None:
        self.video_dir = str(video_dir)
        self.trajectory_dir = str(
            trajectory_dir
            if trajectory_dir not in (None, "")
            else (tracks_dir if tracks_dir not in (None, "") else "/path/to/data/trajectory/")
        )
        self.caption_csv = str(caption_csv)
        self.caption_video_col = str(caption_video_col)
        self.caption_text_col = str(caption_text_col)
        self.caption_vid_col = str(caption_vid_col)
        self.raw_height = int(raw_height)
        self.raw_width = int(raw_width)
        self.hist_len = int(hist_len)
        self.future_len = int(future_len)
        self.use_overlap = bool(use_overlap)
        self.track_stride = int(track_stride)
        self.axis_order = axis_order
        self.video_reader = str(video_reader)
        self.video_exts = str(video_exts)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.train_fraction = float(train_fraction)
        self.seed = int(seed)
        self.use_cached_latents = bool(use_cached_latents)
        self.cache_dir = str(cache_dir)
        self.cache_manifest_train = str(cache_manifest_train)
        self.cache_manifest_eval = str(cache_manifest_eval)
        self.cache_manifest_val = str(cache_manifest_val)

    @property
    def tracks_dir(self) -> str:
        return self.trajectory_dir


@dataclass
class VAETrainConfig:
    out_dir: str = "/path/to/output/trajloom_vae/"

    seed: int = 42
    num_epochs: int = 10
    batch_size: int = 2
    val_split: float = 0.10
    num_workers: int = 4

    mixed_precision: Literal["no", "fp16", "bf16", "fp32"] = "bf16"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = 1.0

    log_every: int = 50
    val_every: int = 200
    val_max_batches: int = 2
    save_every: int = 1000
    resume: Optional[str] = None

    vel_lambda: float = 0.1
    neighbor_lambda: float = 0.05

    entity: str = "video-traj"
    wandb_project: str = "trajloom-vae"
    wandb_name: str = "trajloom-vae"
    wandb_enabled: bool = True

    enable_activation_checkpointing: bool = True


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value):
        return asdict(value)
    return dict(value)


def _coerce_checkpoint_config(value: Any) -> TrajLoomVAECheckpointConfig:
    if isinstance(value, TrajLoomVAECheckpointConfig):
        return TrajLoomVAECheckpointConfig(
            ckpt_path=value.ckpt_path,
            config=_to_plain_dict(value.config),
        )

    payload = _to_plain_dict(value)
    if "config" not in payload:
        raise KeyError("Expected trajloom_vae.config in VAE config payload.")

    return TrajLoomVAECheckpointConfig(
        ckpt_path=str(payload.get("ckpt_path", TrajLoomVAECheckpointConfig().ckpt_path)),
        config=_to_plain_dict(payload["config"]),
    )


@dataclass(init=False)
class TrajLoomVAEFullConfig:
    data: TrajLoomVAEDataConfig
    trajloom_vae: TrajLoomVAECheckpointConfig
    train: Optional[VAETrainConfig]

    def __init__(
        self,
        *,
        data: Optional[TrajLoomVAEDataConfig | Dict[str, Any]] = None,
        trajloom_vae: Optional[TrajLoomVAECheckpointConfig | Dict[str, Any]] = None,
        track_vae: Optional[TrajLoomVAECheckpointConfig | Dict[str, Any]] = None,
        vae: Optional[TrajLoomVAEConfig | Dict[str, Any]] = None,
        train: Optional[VAETrainConfig | Dict[str, Any]] = None,
    ) -> None:
        if data is None:
            data = TrajLoomVAEDataConfig()
        self.data = data if isinstance(data, TrajLoomVAEDataConfig) else TrajLoomVAEDataConfig(**data)

        section = trajloom_vae if trajloom_vae is not None else track_vae
        if section is None:
            section = TrajLoomVAECheckpointConfig(config=_to_plain_dict(vae or TrajLoomVAEConfig()))
        self.trajloom_vae = _coerce_checkpoint_config(section)
        if train is None:
            self.train = None
        else:
            self.train = train if isinstance(train, VAETrainConfig) else VAETrainConfig(**train)

    @property
    def track_vae(self) -> TrajLoomVAECheckpointConfig:
        return self.trajloom_vae

    @property
    def vae(self) -> Dict[str, Any]:
        return dict(self.trajloom_vae.config)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrajLoomVAEFullConfig":
        return cls(
            data=payload.get("data"),
            trajloom_vae=payload.get("trajloom_vae"),
            track_vae=payload.get("track_vae"),
            vae=payload.get("vae"),
            train=payload.get("train"),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "data": asdict(self.data),
            "trajloom_vae": {
                "ckpt_path": self.trajloom_vae.ckpt_path,
                "config": dict(self.trajloom_vae.config),
            },
        }
        if self.train is not None:
            payload["train"] = asdict(self.train)
        return payload


TopConfig = TrajLoomVAEFullConfig
FullConfig = TrajLoomVAEFullConfig


__all__ = [
    "AxisOrder",
    "TrajLoomVAEConfig",
    "WanVAEConfig",
    "TrajLoomVAECheckpointConfig",
    "TrackVAECheckpointConfig",
    "TrajLoomVAEDataConfig",
    "VAETrainConfig",
    "TrajLoomVAEFullConfig",
    "TopConfig",
    "FullConfig",
]
