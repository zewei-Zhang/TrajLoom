"""
TrajLoom visibility predictor for latent-space future visibility logits.

This model file keeps the lightweight temporal visibility head used by
the final TrajLoom generator pipeline.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class VisPredictorConfig:
    latent_channels: int = 16
    hidden: int = 64
    depth: int = 3  # number of Conv1d blocks
    kernel_size: int = 3
    use_hist_last: bool = False  # concat last hist latent per token
    use_delta_to_hist: bool = False  # also concat (z_fut - z_hist_last)


class TinyVisPredictor(nn.Module):
    """
    Predict visibility logits in latent token space.
    Input:
      z_fut : [B, T_lat, N, C]
      z_hist: [B, T_lat, N, C] (optional, only last step is used)
    Output:
      logits_lat: [B, T_lat, N]  (BCEWithLogitsLoss target is {0,1})
    """

    def __init__(self, cfg: VisPredictorConfig):
        super().__init__()
        self.cfg = cfg

        in_c = cfg.latent_channels
        feat_c = in_c
        if cfg.use_hist_last:
            feat_c += in_c
            if cfg.use_delta_to_hist:
                feat_c += in_c

        self.in_proj = nn.Linear(feat_c, cfg.hidden)

        blocks = []
        for _ in range(cfg.depth):
            blocks += [
                nn.Conv1d(cfg.hidden, cfg.hidden, kernel_size=cfg.kernel_size,
                          padding=cfg.kernel_size // 2),
                nn.SiLU(),
            ]
        self.temporal_net = nn.Sequential(*blocks)

        self.out = nn.Conv1d(cfg.hidden, 1, kernel_size=1)

    def forward(self, z_fut: torch.Tensor, z_hist: torch.Tensor | None = None) -> torch.Tensor:
        assert z_fut.dim() == 4, f"expected [B,T,N,C], got {tuple(z_fut.shape)}"
        B, T, N, C = z_fut.shape

        x = z_fut

        if self.cfg.use_hist_last:
            assert z_hist is not None, "use_hist_last=True but z_hist is None"
            hist_last = z_hist[:, -1]  # [B,N,C]
            hist_rep = hist_last[:, None].expand(B, T, N, C)
            feats = [x, hist_rep]
            if self.cfg.use_delta_to_hist:
                feats.append(x - hist_rep)
            x = torch.cat(feats, dim=-1)  # [B,T,N,feat_c]

        # per-token temporal conv: reshape to [B*N, T, feat_c]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, x.shape[-1])
        x = self.in_proj(x)  # [B*N, T, hidden]
        x = x.permute(0, 2, 1)  # [B*N, hidden, T]

        x = self.temporal_net(x)  # [B*N, hidden, T]
        logits = self.out(x).squeeze(1)  # [B*N, T]

        logits = logits.view(B, N, T).permute(0, 2, 1)  # [B, T, N]
        return logits


# -----------------------------------------------------------------------------
# Project-aligned aliases
# -----------------------------------------------------------------------------

TrajLoomVisibilityPredictorConfig = VisPredictorConfig
TrajLoomVisibilityPredictor = TinyVisPredictor

__all__ = [
    "VisPredictorConfig",
    "TinyVisPredictor",
    "TrajLoomVisibilityPredictorConfig",
    "TrajLoomVisibilityPredictor",
]
