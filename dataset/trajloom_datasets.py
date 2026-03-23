"""
TrajLoom generator datasets.

This module provides the two dataset paths used by TrajLoom generator training:

  - `GeneratorVideoPairDataset` loads history video frames plus dense history
    and future track fields directly from videos and `.npz` track files.
  - `CachedLatentsDataset` loads cached TrajLoom-VAE latents from `.pt` files
    for the fast training path.

The video-and-track dataset intentionally follows the same decoding and
rasterization conventions as the TrajLoom-VAE data pipeline:

  - Video decoding: `read_video_frames(..., is_rgb=True)`
  - Frame conversion: `_to_bcthw(frames)`
  - Track loading: `load_tracks(npz_path)`
  - Sparse-to-dense rasterization: `flat_to_dense_raw(...)`

Important
---------
`_to_bcthw(...)` already returns `float32` values in `[-1, 1]`. Do not apply an
additional image normalization step such as `/127.5 - 1.0`.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch.utils.data import Dataset

from torch.utils.data import Dataset, DataLoader
# Keep the same utilities as the VAE dataset
from utils.dataset_utils import (
    load_trajectory,
    read_video_frames,
    _to_bcthw,
    _natural_key,
)

# Keep the same rasterization util as the VAE dataset
from utils.trajloom_vae_utils import flat_to_dense_raw
from configs.generator_configs import GeneratorDatasetConfig, AxisOrder


def _parse_video_exts(video_exts: str) -> List[str]:
    """Return normalized video filename patterns from the dataset config."""
    if not video_exts:
        return [".mp4"]
    parts = [part.strip() for part in str(video_exts).split(",") if part.strip()]
    return parts if parts else [".mp4"]


def _ensure_xy(tracks_nt2: torch.Tensor, axis_order: AxisOrder) -> torch.Tensor:
    """Return track coordinates in `(x, y)` order regardless of source layout."""
    return tracks_nt2 if axis_order == "xy" else tracks_nt2[..., [1, 0]]


class GeneratorVideoPairDataset(Dataset):
    """Load history video frames with aligned history and future track fields."""

    def __init__(self, cfg: GeneratorDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.video_dir = Path(cfg.video_dir)
        trajectory_dir = getattr(cfg, "trajectory_dir", getattr(cfg, "tracks_dir"))
        self.trajectory_dir = Path(trajectory_dir)
        self.tracks_dir = self.trajectory_dir

        # Build paired video/track records using the same matching scheme as the VAE dataset.
        video_ext_patterns = _parse_video_exts(cfg.video_exts)
        video_paths: List[str] = []
        for video_ext in video_ext_patterns:
            # Accept entries like ".mp4" or "*.mp4".
            glob_pattern = video_ext if "*" in video_ext else f"*{video_ext}"
            video_paths.extend(glob.glob(os.path.join(str(self.video_dir), glob_pattern)))

        video_paths = sorted(video_paths, key=_natural_key)
        if not video_paths:
            raise RuntimeError(f"No videos found in {self.video_dir} (exts={video_ext_patterns})")

        paired_records: List[Dict[str, str]] = []
        for video_path in video_paths:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            trajectory_path = os.path.join(str(self.trajectory_dir), video_id + ".npz")
            if os.path.isfile(trajectory_path):
                paired_records.append(
                    {"vid": video_id, "video_path": video_path, "trajectory_path": trajectory_path}
                )

        if not paired_records:
            raise RuntimeError(
                "No matched video/trajectory pairs under "
                f"video_dir={self.video_dir} and trajectory_dir={self.trajectory_dir}"
            )
        self.index = paired_records

        # Build a video-id-to-caption lookup when caption metadata is configured.
        self.caption_map: Dict[str, str] = {}
        if cfg.caption_csv:
            csv_path = Path(cfg.caption_csv)
            if not csv_path.exists():
                raise FileNotFoundError(f"caption_csv not found: {csv_path}")

            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Prefer the explicit video id column when it exists.
                    caption_video_id = (row.get(cfg.caption_vid_col) or "").strip()

                    # Otherwise derive the id from the source video path.
                    if not caption_video_id:
                        caption_video_path = (row.get(cfg.caption_video_col) or "").strip()
                        if caption_video_path:
                            caption_video_id = Path(caption_video_path).stem

                    if not caption_video_id:
                        continue

                    self.caption_map[caption_video_id] = (row.get(cfg.caption_text_col) or "")

    def __len__(self) -> int:
        return len(self.index)

    def _segment_bounds(self) -> Tuple[int, int, int, int]:
        """Return history and future segment bounds in frame indices."""
        hist_start = 0
        hist_end = hist_start + int(self.cfg.hist_len)
        fut_start = (hist_end - 1) if self.cfg.use_overlap else hist_end
        fut_end = fut_start + int(self.cfg.future_len)
        return hist_start, hist_end, fut_start, fut_end

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_record = self.index[index]
        video_id = sample_record["vid"]

        hist_start, hist_end, fut_start, fut_end = self._segment_bounds()
        required_frames = fut_end

        # Decode video frames using the same path as the VAE dataset.
        video_frames = read_video_frames(
            sample_record["video_path"],
            use_type=self.cfg.video_reader,
            is_rgb=True,
        )
        # `_to_bcthw` already returns `float32` values in `[-1, 1]`.
        video_frames_bcthw = _to_bcthw(video_frames)  # [3, T, H, W]
        hist_video_cthw = video_frames_bcthw[:, hist_start:hist_end].contiguous()  # [3, Th, H, W]

        # Load sparse tracks and visibility using the shared TrajLoom-VAE helpers.
        flat_trajectory_nt2, flat_vis_nt = load_trajectory(sample_record["trajectory_path"])  # [N, T, 2], [N, T]
        flat_trajectory_nt2 = torch.as_tensor(flat_trajectory_nt2, dtype=torch.float32)
        flat_vis_nt = torch.as_tensor(flat_vis_nt, dtype=torch.bool)

        flat_trajectory_nt2 = _ensure_xy(flat_trajectory_nt2, self.cfg.axis_order)

        if flat_trajectory_nt2.shape[1] < required_frames:
            raise RuntimeError(
                f"{video_id}: trajectory T={flat_trajectory_nt2.shape[1]} need >= {required_frames} "
                f"(hist_end={hist_end}, fut_end={fut_end})"
            )

        flat_trajectory_nt2 = flat_trajectory_nt2[:, :required_frames]
        flat_vis_nt = flat_vis_nt[:, :required_frames]

        # Rasterize sparse N x T tracks into dense fields that match VAE training inputs.
        dense_trajectory_thwd, dense_vis_thw = flat_to_dense_raw(
            tracks_nt2=flat_trajectory_nt2,
            vis_nt=flat_vis_nt,
            raw_hw=(int(self.cfg.raw_height), int(self.cfg.raw_width)),
            grid_stride=int(self.cfg.track_stride),
        )

        hist_trajectory = dense_trajectory_thwd[hist_start:hist_end].contiguous()
        hist_vis = dense_vis_thw[hist_start:hist_end].contiguous()
        fut_trajectory = dense_trajectory_thwd[fut_start:fut_end].contiguous()
        fut_vis = dense_vis_thw[fut_start:fut_end].contiguous()

        caption = self.caption_map.get(video_id, "")

        sample = {
            "vid": video_id,
            "caption": caption,
            "hist_trajectory_thwd": hist_trajectory.to(torch.float32),
            "hist_trajectory_vis_thw": hist_vis.to(torch.bool),
            "fut_trajectory_thwd": fut_trajectory.to(torch.float32),
            "fut_trajectory_vis_thw": fut_vis.to(torch.bool),
            "hist_video_cthw": hist_video_cthw.to(torch.float32),
            "trajectory_path": sample_record["trajectory_path"],
        }
        sample["hist_tracks_thwd"] = sample["hist_trajectory_thwd"]
        sample["hist_vis_thw"] = sample["hist_trajectory_vis_thw"]
        sample["fut_tracks_thwd"] = sample["fut_trajectory_thwd"]
        sample["fut_vis_thw"] = sample["fut_trajectory_vis_thw"]
        sample["tracks_path"] = sample["trajectory_path"]
        return sample


def generator_pair_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack raw generator samples into tensor-first training batches."""
    hist_trajectory = torch.stack([sample["hist_trajectory_thwd"] for sample in batch], dim=0)
    hist_visibility = torch.stack([sample["hist_trajectory_vis_thw"] for sample in batch], dim=0)
    fut_trajectory = torch.stack([sample["fut_trajectory_thwd"] for sample in batch], dim=0)
    fut_visibility = torch.stack([sample["fut_trajectory_vis_thw"] for sample in batch], dim=0)
    hist_video = torch.stack([sample["hist_video_cthw"] for sample in batch], dim=0)

    collated = {
        "vid": [sample["vid"] for sample in batch],
        "caption": [sample.get("caption", "") for sample in batch],
        "trajectory_path": [sample.get("trajectory_path", sample.get("tracks_path", "")) for sample in batch],
        "hist_trajectory_thwd": hist_trajectory,
        "hist_trajectory_vis_thw": hist_visibility,
        "fut_trajectory_thwd": fut_trajectory,
        "fut_trajectory_vis_thw": fut_visibility,
        "hist_video_cthw": hist_video,
    }
    collated["tracks_path"] = collated["trajectory_path"]
    collated["hist_tracks_thwd"] = collated["hist_trajectory_thwd"]
    collated["hist_vis_thw"] = collated["hist_trajectory_vis_thw"]
    collated["fut_tracks_thwd"] = collated["fut_trajectory_thwd"]
    collated["fut_vis_thw"] = collated["fut_trajectory_vis_thw"]
    return collated


# -----------------------------------------------------------------------------
# Cached-latent dataset (fast path for generator training)
# -----------------------------------------------------------------------------


class CachedLatentsDataset(Dataset):
    """Load cached TrajLoom-VAE latents and optional conditioning tensors from `.pt` files.

    Each sample is expected to be a dict saved by `cache_generator_latents.py`.

    Returned keys
    -------------
    Subset depends on what was cached:
      - vid: str
      - caption: str
      - z_hist:        [21, N, C]
      - z_fut:         [21, N, C]
      - z_video_hist:  [21, N, C_v] or None
      - text_cond:     [D] or [1, D] (whatever you cached) or None
      - hist_vis_lat:  [21, N] bool/uint8 or None
      - fut_vis_lat:   [21, N] bool/uint8 or None

    Notes
    -----
    - Shuffling works normally via `DataLoader(shuffle=True)` because the
      manifest is loaded into memory as a list of records.
    - This dataset does *no* scaling. Keep cached `z_*` values **unscaled** and
      apply SD/Wan-style scaling online in the trainer.
    """

    def __init__(self, cfg: GeneratorDatasetConfig, split: str = "train"):
        super().__init__()
        self.cfg = cfg
        # Normalize split naming: manifests now use `eval`, but older callers may still pass `val`.
        self.split = str(split).lower().strip()
        if self.split == "val":
            self.split = "eval"
        self.cache_dir = Path(cfg.cache_dir)
        if not self.cache_dir:
            raise ValueError("cfg.cache_dir is empty, but CachedLatentsDataset was requested")

        if self.split == "train":
            manifest_path = (
                Path(cfg.cache_manifest_train)
                if cfg.cache_manifest_train
                else (self.cache_dir / "manifest_train.jsonl")
            )

        elif self.split == "eval":
            # Prefer explicit overrides; support legacy cache_manifest_val.
            if getattr(cfg, "cache_manifest_eval", ""):
                manifest_path = Path(cfg.cache_manifest_eval)
            elif getattr(cfg, "cache_manifest_val", ""):
                manifest_path = Path(cfg.cache_manifest_val)
            else:
                # Prefer the new manifest name, then fall back to the legacy one if needed.
                eval_manifest_path = self.cache_dir / "manifest_eval.jsonl"
                val_manifest_path = self.cache_dir / "manifest_val.jsonl"
                manifest_path = eval_manifest_path if eval_manifest_path.exists() else val_manifest_path

        else:
            manifest_path = self.cache_dir / f"manifest_{self.split}.jsonl"

        if not manifest_path.exists():
            raise FileNotFoundError(f"cache manifest not found: {manifest_path}")

        self.records: List[Dict[str, Any]] = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "path" not in record:
                    raise KeyError(f"manifest line missing 'path': {record}")
                self.records.append(record)

        if len(self.records) == 0:
            raise RuntimeError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        cache_path = self.cache_dir / record["path"]
        cached_sample = torch.load(cache_path, map_location="cpu")

        # Keep output keys stable even if some optional values are missing.
        sample: Dict[str, Any] = {
            "vid": cached_sample.get("vid", record.get("vid", "")),
            "caption": cached_sample.get("caption", ""),
            "video_path": cached_sample.get("video_path", record.get("video_path", "")),
            "tracks_path": cached_sample.get("tracks_path", record.get("tracks_path", "")),
            "z_hist": cached_sample.get("z_hist", None),
            "z_fut": cached_sample.get("z_fut", None),
            "z_video_hist": cached_sample.get("z_video_hist", None),
            "text_cond": cached_sample.get("text_cond", None),
            "hist_vis_lat": cached_sample.get("hist_vis_lat", None),
            "fut_vis_lat": cached_sample.get("fut_vis_lat", None),
        }

        if sample["z_hist"] is None or sample["z_fut"] is None:
            raise KeyError(f"{cache_path} missing z_hist or z_fut")

        # Normalize optional singleton-batch cache layouts: [1, T, N, C] -> [T, N, C].
        for key in ["z_hist", "z_fut", "z_video_hist"]:
            value = sample.get(key, None)
            if torch.is_tensor(value) and value.dim() == 4 and value.size(0) == 1:
                sample[key] = value[0]

        # Cached visibility masks may be uint8; normalize them to bool for downstream code.
        for key in ["hist_vis_lat", "fut_vis_lat"]:
            value = sample.get(key, None)
            if torch.is_tensor(value) and value.dtype != torch.bool:
                sample[key] = value.to(torch.bool)

        return sample


def cached_latents_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack cached latent samples while preserving optional conditioning fields."""

    def _stack_optional(key: str):
        values = [sample.get(key, None) for sample in batch]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Mixed None/non-None for key='{key}' in a batch. Re-cache consistently.")
        return torch.stack(values, dim=0)

    collated_batch = {
        "vid": [sample.get("vid", "") for sample in batch],
        "caption": [sample.get("caption", "") for sample in batch],
        "video_path": [sample.get("video_path", "") for sample in batch],
        "tracks_path": [sample.get("tracks_path", "") for sample in batch],
        "z_hist": torch.stack([sample["z_hist"] for sample in batch], dim=0),
        "z_fut": torch.stack([sample["z_fut"] for sample in batch], dim=0),
        "z_video_hist": _stack_optional("z_video_hist"),
        "text_cond": _stack_optional("text_cond"),
        "hist_vis_lat": _stack_optional("hist_vis_lat"),
        "fut_vis_lat": _stack_optional("fut_vis_lat"),
    }
    return collated_batch


# -----------------------------------------------------------------------------
# Simple test (debug / sanity check)
# -----------------------------------------------------------------------------


def _print_tensor_stats(name: str, x: torch.Tensor, *, max_lines: int = 1) -> None:
    """Print tensor shape, dtype, and a compact summary of value ranges."""
    assert torch.is_tensor(x)
    shape = tuple(x.shape)
    dtype = x.dtype

    line = f"{name}: shape={shape} dtype={dtype}"
    if x.numel() == 0:
        print(line + " (empty)")
        return

    if dtype == torch.bool:
        ratio = float(x.float().mean().item())
        print(line + f"  true_ratio={ratio:.6f}")
        return

    if dtype.is_floating_point:
        x_det = x.detach()
        finite = bool(torch.isfinite(x_det).all().item())
        mn = float(x_det.min().item())
        mx = float(x_det.max().item())
        mean = float(x_det.mean().item())
        print(line + f"  min={mn:.6f} max={mx:.6f} mean={mean:.6f} finite={finite}")
        return

    # int types
    x_det = x.detach()
    mn = int(x_det.min().item())
    mx = int(x_det.max().item())
    print(line + f"  min={mn} max={mx}")


def _print_xy_stats(name: str, coords: torch.Tensor, vis: Optional[torch.Tensor] = None) -> None:
    """Print coordinate ranges, with optional visible-only stats from a mask."""
    assert coords.shape[-1] == 2, f"expected last dim=2, got {coords.shape}"

    x_coords = coords[..., 0]
    y_coords = coords[..., 1]

    def _minmax(values: torch.Tensor) -> Tuple[float, float]:
        return float(values.min().item()), float(values.max().item())

    xmn, xmx = _minmax(x_coords)
    ymn, ymx = _minmax(y_coords)
    print(f"{name}: x[min,max]=[{xmn:.3f},{xmx:.3f}]  y[min,max]=[{ymn:.3f},{ymx:.3f}]")

    if vis is None:
        return

    vis_mask = vis.to(torch.bool)
    if not bool(vis_mask.any().item()):
        print(f"{name}: visible mask is all-false")
        return

    # Visible-only min/max without allocating a masked_select vector.
    # We set invisible entries to +/-inf and reduce.
    inf = torch.tensor(float("inf"), device=coords.device, dtype=coords.dtype)
    ninf = torch.tensor(float("-inf"), device=coords.device, dtype=coords.dtype)

    x_vis_min = torch.where(vis_mask, x_coords, inf).min()
    x_vis_max = torch.where(vis_mask, x_coords, ninf).max()
    y_vis_min = torch.where(vis_mask, y_coords, inf).min()
    y_vis_max = torch.where(vis_mask, y_coords, ninf).max()

    vis_ratio = float(vis_mask.float().mean().item())
    print(
        f"{name} (visible-only): x[min,max]=[{float(x_vis_min):.3f},{float(x_vis_max):.3f}]  "
        f"y[min,max]=[{float(y_vis_min):.3f},{float(y_vis_max):.3f}]  vis_ratio={vis_ratio:.6f}"
    )
