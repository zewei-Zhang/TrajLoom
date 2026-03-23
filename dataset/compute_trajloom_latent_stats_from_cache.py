"""
Compute TrajLoom-VAE latent stats (per-channel mean/std) directly from cached
latent `.pt` files used by TrajLoom generator training.

Why this exists
---------------
TrajLoom generator training uses SD/Wan-style latent normalization:

    z_scaled = (z - mean) / std

where mean/std are **per latent channel** and loaded from a JSON.

This script:
  - reads cached latent `.pt` files (typically one per video)
  - accumulates stats for:
      * z_hist
      * z_fut
      * z_all  (= hist+fut combined)
  - writes a JSON compatible with
    `utils.trajloom_vae_utils.load_latent_stats_json()`.

Usage
-----
Single process:
  python compute_trajloom_latent_stats_from_cache.py \
    --cache_dir /path/to/cache \
    --split train \
    --out /path/to/latent_stats_cache.json

Multi-GPU (faster IO for huge caches):
  accelerate launch --num_processes 4 compute_trajloom_latent_stats_from_cache.py \
    --cache_dir /path/to/cache --split train --out /path/to/latent_stats_cache.json

Then set in your TrajLoom generator config:
  train.latent_stats_json = "/path/to/latent_stats_cache.json"
  train.latent_stats_key  = "z_all"
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm


@dataclass
class StatsOut:
    count: int
    mean: float
    std: float
    min: float
    max: float
    per_channel: Dict[str, Any]


class LatentStats:
    """Streaming per-channel stats accumulator."""

    def __init__(self, latent_channels: int):
        self.latent_channels = int(latent_channels)

        # Aggregate statistics across every latent value.
        self.count = 0
        self.sum = torch.tensor(0.0, dtype=torch.float64)
        self.sumsq = torch.tensor(0.0, dtype=torch.float64)
        self.min = float("inf")
        self.max = float("-inf")

        # Aggregate statistics for each latent channel independently.
        self.count_c = 0  # Number of flattened latent vectors; shared across channels.
        self.sum_c = torch.zeros(self.latent_channels, dtype=torch.float64)
        self.sumsq_c = torch.zeros(self.latent_channels, dtype=torch.float64)
        self.min_c = torch.full((self.latent_channels,), float("inf"), dtype=torch.float64)
        self.max_c = torch.full((self.latent_channels,), float("-inf"), dtype=torch.float64)

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> None:
        """Update running stats from a latent tensor with shape [..., C]."""
        if z is None:
            return
        if not torch.is_tensor(z):
            raise TypeError(f"Expected tensor, got {type(z)}")
        if z.numel() == 0:
            return
        if z.shape[-1] != self.latent_channels:
            raise ValueError(f"Expected last dim C={self.latent_channels}, got {tuple(z.shape)}")

        z_values = z.detach().to(dtype=torch.float32)
        self.count += int(z_values.numel())
        self.sum += z_values.sum(dtype=torch.float64).cpu()
        self.sumsq += (z_values * z_values).sum(dtype=torch.float64).cpu()
        self.min = min(self.min, float(z_values.min().cpu().item()))
        self.max = max(self.max, float(z_values.max().cpu().item()))

        z_channels = z_values.reshape(-1, self.latent_channels)
        self.count_c += int(z_channels.shape[0])
        self.sum_c += z_channels.sum(dim=0, dtype=torch.float64).cpu()
        self.sumsq_c += (z_channels * z_channels).sum(dim=0, dtype=torch.float64).cpu()
        self.min_c = torch.minimum(self.min_c, z_channels.min(dim=0).values.to(torch.float64).cpu())
        self.max_c = torch.maximum(self.max_c, z_channels.max(dim=0).values.to(torch.float64).cpu())

    def finalize(self) -> Dict[str, Any]:
        if self.count == 0:
            return {}

        mean = float((self.sum / self.count).item())
        var = float((self.sumsq / self.count - mean * mean).clamp_min(0.0).item())
        std = math.sqrt(var)

        mean_c = (self.sum_c / max(self.count_c, 1)).cpu()
        var_c = (self.sumsq_c / max(self.count_c, 1) - mean_c ** 2).clamp_min(0.0)
        std_c = torch.sqrt(var_c).cpu()

        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": float(std),
            "min": float(self.min),
            "max": float(self.max),
            "per_channel": {
                "count": int(self.count_c),
                "mean": [float(x) for x in mean_c.tolist()],
                "std": [float(x) for x in std_c.tolist()],
                "min": [float(x) for x in self.min_c.cpu().tolist()],
                "max": [float(x) for x in self.max_c.cpu().tolist()],
            },
        }


def _reduce_stats(stats: LatentStats, accelerator: Accelerator, device: torch.device) -> Optional[LatentStats]:
    """Reduce per-rank latent statistics into a single global accumulator."""
    # Move scalar/vector accumulators onto the active accelerator device for reduction.
    count = torch.tensor([stats.count], device=device, dtype=torch.float64)
    sum_ = stats.sum.to(device=device, dtype=torch.float64).reshape(1)
    sumsq = stats.sumsq.to(device=device, dtype=torch.float64).reshape(1)

    count_c = torch.tensor([stats.count_c], device=device, dtype=torch.float64)
    sum_c = stats.sum_c.to(device=device, dtype=torch.float64)
    sumsq_c = stats.sumsq_c.to(device=device, dtype=torch.float64)

    count = accelerator.reduce(count, reduction="sum")
    sum_ = accelerator.reduce(sum_, reduction="sum")
    sumsq = accelerator.reduce(sumsq, reduction="sum")
    count_c = accelerator.reduce(count_c, reduction="sum")
    sum_c = accelerator.reduce(sum_c, reduction="sum")
    sumsq_c = accelerator.reduce(sumsq_c, reduction="sum")

    mins = accelerator.gather(torch.tensor([stats.min], device=device, dtype=torch.float64))
    maxs = accelerator.gather(torch.tensor([stats.max], device=device, dtype=torch.float64))

    min_c_all = accelerator.gather(stats.min_c.to(device=device, dtype=torch.float64).unsqueeze(0))
    max_c_all = accelerator.gather(stats.max_c.to(device=device, dtype=torch.float64).unsqueeze(0))

    if accelerator.is_main_process:
        global_stats = LatentStats(stats.latent_channels)
        global_stats.count = int(count.item())
        global_stats.sum = sum_.detach().cpu().squeeze(0)
        global_stats.sumsq = sumsq.detach().cpu().squeeze(0)
        global_stats.min = float(mins.min().item())
        global_stats.max = float(maxs.max().item())

        global_stats.count_c = int(count_c.item())
        global_stats.sum_c = sum_c.detach().cpu()
        global_stats.sumsq_c = sumsq_c.detach().cpu()
        global_stats.min_c = min_c_all.min(dim=0).values.detach().cpu()
        global_stats.max_c = max_c_all.max(dim=0).values.detach().cpu()
        return global_stats
    return None


def _load_manifest(cache_dir: Path, split: str, manifest_path: Optional[str]) -> List[Path]:
    if manifest_path:
        manifest_file = Path(manifest_path)
    else:
        manifest_file = cache_dir / f"manifest_{split}.jsonl"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    paths: List[Path] = []
    with open(manifest_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            relative_path = record.get("path", "")
            if not relative_path:
                # Backward compatibility: older manifests may store `pt_path` directly.
                relative_path = record.get("pt_path", "")
            if not relative_path:
                continue
            paths.append(cache_dir / relative_path)
    if not paths:
        raise RuntimeError(f"Manifest {manifest_file} had 0 entries")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Root cache directory (contains manifest_*.jsonl).")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"],
                        help="Which manifest to use.")
    parser.add_argument("--manifest", type=str, default="", help="Optional override manifest path.")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Limit the number of cached files to scan (0 = all).")
    parser.add_argument("--latent_channels", type=int, default=0,
                        help="Override the latent channel count. If 0, infer it from the first cache file.")
    parser.add_argument("--require_unscaled", action="store_true",
                        help="Fail if any cached file reports stored_scaled_latents=True.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    cache_dir = Path(args.cache_dir)
    split = str(args.split)
    manifest = args.manifest if args.manifest else None

    paths = _load_manifest(cache_dir, split, manifest)
    if args.max_files and args.max_files > 0:
        paths = paths[: int(args.max_files)]

    # Split work evenly across accelerator ranks.
    rank = accelerator.process_index
    world = accelerator.num_processes
    paths_rank = paths[rank::world]

    if accelerator.is_main_process:
        print(f"[trajloom-latent-stats] cache_dir={cache_dir} split={split} files={len(paths)} world={world}")

    # Infer the latent channel count from the first available cache record when needed.
    latent_channels = int(args.latent_channels)
    if latent_channels <= 0:
        sample0 = torch.load(paths_rank[0] if paths_rank else paths[0], map_location="cpu")
        z_fut0 = sample0.get("z_fut", None)
        if z_fut0 is None:
            raise KeyError(f"{paths[0]} missing key 'z_fut'")
        latent_channels = int(z_fut0.shape[-1])

    stats_hist = LatentStats(latent_channels)
    stats_fut = LatentStats(latent_channels)
    stats_all = LatentStats(latent_channels)

    scaled_flags: List[bool] = []

    pbar = tqdm(paths_rank, disable=(not accelerator.is_local_main_process))
    for cache_path in pbar:
        cache_record = torch.load(cache_path, map_location="cpu")

        stored_scaled = bool(cache_record.get("stored_scaled_latents", False))
        scaled_flags.append(stored_scaled)
        if args.require_unscaled and stored_scaled:
            raise RuntimeError(
                f"Found stored_scaled_latents=True in {cache_path}. Re-cache with --store_scaled_latents off.")

        z_hist = cache_record.get("z_hist", None)
        z_fut = cache_record.get("z_fut", None)
        if z_hist is None or z_fut is None:
            raise KeyError(f"{cache_path} missing z_hist or z_fut")

        # Accept cached latents in either [T, N, C] or [1, T, N, C] form.
        if z_hist.dim() == 3:
            z_hist_sample = z_hist
        elif z_hist.dim() == 4 and z_hist.size(0) == 1:
            z_hist_sample = z_hist[0]
        else:
            raise ValueError(f"Unexpected z_hist shape {tuple(z_hist.shape)} in {cache_path}")

        if z_fut.dim() == 3:
            z_fut_sample = z_fut
        elif z_fut.dim() == 4 and z_fut.size(0) == 1:
            z_fut_sample = z_fut[0]
        else:
            raise ValueError(f"Unexpected z_fut shape {tuple(z_fut.shape)} in {cache_path}")

        stats_hist.update(z_hist_sample)
        stats_fut.update(z_fut_sample)
        stats_all.update(z_hist_sample)
        stats_all.update(z_fut_sample)

    # Merge per-rank accumulators into a single set of stats on the main process.
    stats_hist_g = _reduce_stats(stats_hist, accelerator, device)
    stats_fut_g = _reduce_stats(stats_fut, accelerator, device)
    stats_all_g = _reduce_stats(stats_all, accelerator, device)

    # Track whether any rank observed pre-scaled cache entries.
    any_scaled = torch.tensor([1.0 if any(scaled_flags) else 0.0], device=device, dtype=torch.float32)
    any_scaled = accelerator.reduce(any_scaled, reduction="max")

    if accelerator.is_main_process:
        stats_hist_main = stats_hist_g or stats_hist
        stats_fut_main = stats_fut_g or stats_fut
        stats_all_main = stats_all_g or stats_all

        out = {
            "meta": {
                "cache_dir": str(cache_dir),
                "split": split,
                "manifest": str(manifest) if manifest else str(cache_dir / f"manifest_{split}.jsonl"),
                "num_files": int(len(paths)),
                "world_size": int(world),
                "latent_channels": int(latent_channels),
                "stored_scaled_latents_present": bool(any_scaled.item() > 0.5),
            },
            "z_hist": stats_hist_main.finalize(),
            "z_fut": stats_fut_main.finalize(),
            "z_all": stats_all_main.finalize(),
        }

        z_std = out["z_all"].get("std", 0.0) if out.get("z_all") else 0.0
        out["suggested_latent_scale"] = float(1.0 / z_std) if (z_std and z_std > 0) else None

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        print("\n================= RESULTS =================")
        print(f"[saved] {out_path}")
        print(f"z_all mean={out['z_all']['mean']:.6f} std={out['z_all']['std']:.6f}")
        print(f"z_hist mean={out['z_hist']['mean']:.6f} std={out['z_hist']['std']:.6f}")
        print(f"z_fut  mean={out['z_fut']['mean']:.6f} std={out['z_fut']['std']:.6f}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
