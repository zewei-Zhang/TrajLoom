"""
Compute VEPE for VAE reconstruction benchmark outputs saved in the
per-video folder format.

Expected files per run directory
--------------------------------
- gt_tracks.npy        [T,N,2] float32
- pred_tracks.npy      [T,N,2] float32
- gt_visibility.npy    [T,N]   uint8/bool (optional; if missing, all visible)

Example
-------
python -m benchmark.compute_vae_recon_metrics \
  --runs_root "/path/to/vae_recon_runs/" \
  --out_json "/path/to/vae_recon_metrics.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _discover_run_dirs(root: Path) -> List[Path]:
    """Find directories containing both gt_tracks.npy and pred_tracks.npy."""
    if (root / "gt_tracks.npy").exists() and (root / "pred_tracks.npy").exists():
        return [root]
    out: List[Path] = []
    for p in root.rglob("gt_tracks.npy"):
        d = p.parent
        if (d / "pred_tracks.npy").exists():
            out.append(d)
    return sorted(set(out))


def _load_optional(path: Path) -> Optional[np.ndarray]:
    return np.load(str(path)) if path.exists() else None


def _safe_epe_px(pred_tn2: np.ndarray, gt_tn2: np.ndarray) -> np.ndarray:
    """Euclidean position error in pixels: [T,N]."""
    d = pred_tn2.astype(np.float32) - gt_tn2.astype(np.float32)
    return np.sqrt(np.sum(d * d, axis=-1, dtype=np.float32)).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute VEPE from saved VAE reconstruction runs.",
    )

    p.add_argument("--runs_root", type=str, required=True,
                   help="Run directory or benchmark root containing per-video subdirs.")
    p.add_argument(
        "--ignore_gt_visibility",
        action="store_true",
        help="Do NOT mask VEPE with gt_visibility.npy (treat all points visible).",
    )
    p.add_argument(
        "--exclude_t0",
        action="store_true",
        help="Exclude t=0 from VEPE (sometimes used if t=0 is an anchored overlap frame).",
    )
    p.add_argument(
        "--eval_len",
        type=int,
        default=0,
        help="If >0, compute VEPE on only the first eval_len frames.",
    )
    p.add_argument("--out_json", type=str, default="", help="Optional path to write a JSON summary.")

    # Legacy args accepted for compatibility with older wrappers; ignored now.
    p.add_argument("--gen_config", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--device", type=str, default="cuda:0", help=argparse.SUPPRESS)
    p.add_argument("--dtype", type=str, default="fp16", help=argparse.SUPPRESS)
    p.add_argument("--video_hw", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--grid_stride", type=int, default=32, help=argparse.SUPPRESS)
    p.add_argument("--grid_border", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--grid_origin", type=str, default="topleft", help=argparse.SUPPRESS)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.runs_root)

    run_dirs = _discover_run_dirs(root)
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs found under {root} (looking for gt_tracks.npy + pred_tracks.npy).")

    sum_vepe = 0.0
    cnt_vepe = 0
    per_video: List[Dict[str, Any]] = []

    for d in run_dirs:
        gt_p = d / "gt_tracks.npy"
        pr_p = d / "pred_tracks.npy"
        gt = np.load(str(gt_p))
        pr = np.load(str(pr_p))

        if gt.ndim == 4 and gt.shape[0] == 1:
            gt = gt[0]
        if pr.ndim == 4 and pr.shape[0] == 1:
            pr = pr[0]

        if gt.ndim != 3 or gt.shape[-1] != 2:
            raise ValueError(f"{gt_p} must be [T,N,2], got {gt.shape}")
        if pr.ndim != 3 or pr.shape[-1] != 2:
            raise ValueError(f"{pr_p} must be [T,N,2], got {pr.shape}")

        # Align shapes defensively if a run saved extra frames or points.
        Tm = min(int(gt.shape[0]), int(pr.shape[0]))
        Nm = min(int(gt.shape[1]), int(pr.shape[1]))
        if Tm != gt.shape[0] or Tm != pr.shape[0] or Nm != gt.shape[1] or Nm != pr.shape[1]:
            gt = gt[:Tm, :Nm]
            pr = pr[:Tm, :Nm]

        gt_vis = _load_optional(d / "gt_visibility.npy")
        if (not bool(args.ignore_gt_visibility)) and (gt_vis is not None):
            if gt_vis.ndim == 3 and gt_vis.shape[0] == 1:
                gt_vis = gt_vis[0]
            gt_vis = gt_vis[:Tm, :Nm]
            vis_mask = (gt_vis > 0.5)
        else:
            vis_mask = np.ones((Tm, Nm), dtype=bool)

        T_eval = int(Tm)
        if int(args.eval_len) > 0:
            T_eval = min(T_eval, int(args.eval_len))

        gt_eval = gt[:T_eval, :Nm]
        pr_eval = pr[:T_eval, :Nm]
        vis_mask_eval = vis_mask[:T_eval, :Nm]

        t_start = 1 if bool(args.exclude_t0) and T_eval > 1 else 0

        epe = _safe_epe_px(pr_eval, gt_eval)
        finite = np.isfinite(epe)
        mask = vis_mask_eval & finite

        m_vepe = mask[t_start:]
        if m_vepe.any():
            sum_vepe += float(epe[t_start:][m_vepe].sum())
            cnt_vepe += int(m_vepe.sum())
            vepe_val = float(epe[t_start:][m_vepe].mean())
        else:
            vepe_val = float("nan")

        per_video.append(
            {
                "run_dir": str(d),
                "T": int(Tm),
                "T_eval": int(T_eval),
                "N": int(Nm),
                "vepe_px": vepe_val,
            }
        )

    result: Dict[str, Any] = {
        "runs_root": str(root),
        "num_runs": int(len(run_dirs)),
        "ignore_gt_visibility": bool(args.ignore_gt_visibility),
        "exclude_t0": bool(args.exclude_t0),
        "eval_len": int(args.eval_len),
        "vepe_px": float(sum_vepe / max(1, cnt_vepe)) if cnt_vepe > 0 else float("nan"),
        "per_video": per_video,
    }

    print(json.dumps(result, indent=2))

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
