"""
TrajLoomGenerator (rectified-flow / flow-matching) in TrajLoomVAE latent space.

This public model file keeps the generator backbone and conditioning path used by
the final config, and excludes the visibility prediction branch.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp
from configs.generator_configs import TrajLoomGeneratorConfig


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sin/cos embedding for continuous timesteps t in [0,1].

    Args:
        t: [B]
        dim: embedding dimension

    Returns:
        [B, dim]
    """
    assert t.dim() == 1, f"expected [B], got {t.shape}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


def pos_embedding(pos: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sin/cos embedding for *absolute* (integer/float) positions.

    Used to align history (frames 0..80) and future (frames 81..161) temporal
    positions in the transformer.

    Args:
        pos: [L] positions (e.g. raw frame indices)
        dim: embedding dimension

    Returns:
        [L, dim]
    """
    assert pos.dim() == 1, f"expected [L], got {pos.shape}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=pos.device, dtype=torch.float32)
        / half
    )
    args = pos.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


# -----------------------------------------------------------------------------
# Attention + DiT block
# -----------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Multi-head cross-attention using torch.scaled_dot_product_attention.

    Queries come from `x` and keys/values come from `context`.

    We keep the same QK-normalization trick as SelfAttention for stability.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            *,
            qk_norm: bool = True,
            qk_norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False, eps=qk_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False, eps=qk_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, Lq, D], context: [B, Lc, D]
        B, Lq, D = x.shape
        Bc, Lc, Dc = context.shape
        assert Bc == B and Dc == D, f"context must be [B, Lc, D], got {context.shape} vs {x.shape}"

        q = self.q(x)
        k, v = self.kv(context).chunk(2, dim=-1)

        q = q.view(B, Lq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, Lc, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, Lc, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, Lq, D)
        return self.proj(y)


class SelfAttention(nn.Module):
    """Multi-head self-attention using torch.scaled_dot_product_attention.

    Minimal-but-important stability tweak from WHN/MovieGen: **QK-normalization**.
    When enabled, we apply RMSNorm over the per-head `head_dim` to queries and
    keys before attention. This stabilizes attention logits (especially in
    bf16/fp16) and is reported as "crucial" for flow-matching transformers.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            *,
            qk_norm: bool = True,
            qk_norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            # Normalize each [head_dim] vector. No affine: we only want magnitude stability.
            self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False, eps=qk_norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False, eps=qk_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.proj(y)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: [B,L,D], shift/scale: [B,D]
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """WAN-like AdaLN-Zero block.

    x <- x + gate_msa   * Attn(AdaLN_msa(x))
    x <- x + gate_xattn * XAttn(AdaLN_xattn(x), ctx)   (optional)
    x <- x + gate_mlp   * MLP(AdaLN_mlp(x))

    cond is a single vector [B, D].
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        # Cross-attention (used when `context` is passed to forward()).
        self.norm_xattn = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.xattn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        # Debug: store the *last* gate_xattn mean (after AdaLN) so you can log it.
        self.register_buffer("last_gate_xattn_mean", torch.tensor(0.0), persistent=False)

        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=nn.GELU,
            drop=0.0,
        )

        # WAN-like AdaLN: separate (shift, scale, gate) for
        #   - self-attention
        #   - cross-attention
        #   - MLP
        # => 9 * hidden_size
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))

        # Zero-init so we start close to identity
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        with torch.no_grad():
            hs = hidden_size
            self.adaLN[-1].bias[2 * hs:3 * hs].fill_(1.0)  # gate_msa = 1
            self.adaLN[-1].bias[5 * hs:6 * hs].fill_(1.0)  # gate_xattn
            # gate_xattn (chunk 5) stays 0.0 at init (off)
            self.adaLN[-1].bias[8 * hs:9 * hs].fill_(1.0)  # gate_mlp = 1

    def forward(self, x: torch.Tensor, cond: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,L,D]  cond: [B,D]
        (
            shift_msa, scale_msa, gate_msa,
            shift_xattn, scale_xattn, gate_xattn,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaLN(cond).chunk(9, dim=-1)

        x = x + gate_msa.unsqueeze(1) * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))

        if context is not None:
            # Store a lightweight debug stat.
            self.last_gate_xattn_mean.copy_(gate_xattn.detach().abs().mean())
            x = x + gate_xattn.unsqueeze(1) * self.xattn(
                _modulate(self.norm_xattn(x), shift_xattn, scale_xattn),
                context,
            )
        else:
            self.last_gate_xattn_mean.zero_()

        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# -----------------------------------------------------------------------------
# Generator
# -----------------------------------------------------------------------------

class TrajLoomGenerator(nn.Module):
    def __init__(self, cfg: TrajLoomGeneratorConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.depth % 2 == 0, "depth must be even (spatial/temporal pairs)"
        assert cfg.input_height % cfg.patch_size == 0
        assert cfg.input_width % cfg.patch_size == 0
        self.grid_h = cfg.input_height // cfg.patch_size
        self.grid_w = cfg.input_width // cfg.patch_size
        self.num_patches = self.grid_h * self.grid_w

        self.in_proj = nn.Linear(cfg.latent_channels, cfg.hidden_size, bias=True)

        # Project history track latents to transformer hidden size for cross-attention.
        self.hist_track_proj = nn.Linear(cfg.latent_channels, cfg.hidden_size, bias=True)

        # Optional: project history visibility (per-token) to hidden size and add to ctx_track.
        if bool(getattr(cfg, "use_hist_vis_cond", False)):
            self.hist_vis_proj = nn.Linear(1, cfg.hidden_size, bias=True)
            nn.init.zeros_(self.hist_vis_proj.weight)
            nn.init.zeros_(self.hist_vis_proj.bias)
            self.alpha_hist_vis = nn.Parameter(torch.tensor(float(getattr(cfg, "hist_vis_scale_init", 1.0))))
        else:
            self.hist_vis_proj = None
            self.alpha_hist_vis = None

        # Optional: project history video latents to transformer hidden size for cross-attention.
        # We keep this separate from hist_track_proj because the video latent semantics differ.
        if cfg.use_video_cond:
            self.hist_video_proj = nn.Linear(cfg.video_latent_channels, cfg.hidden_size, bias=True)
            # Learnable scale (0.0 warm-start; change to 1.0 if training from scratch)
            self.alpha_video_ctx = nn.Parameter(torch.tensor(1.0))
        else:
            self.hist_video_proj = None
            self.alpha_video_ctx = None

        # spatial pos embed per frame
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- temporal embeddings ---
        # We encode *absolute* frame indices so the model knows that future comes AFTER history.
        # History latent indices map to raw frames: 0,4,8,...,80
        # Future  latent indices map to raw frames: 81,85,...,161  (or 80,84,...,160 if overlap)
        stride = int(getattr(cfg, "latent_time_stride", 4))
        frames_in = int(getattr(cfg, "num_frames_in", 81))
        use_overlap = bool(getattr(cfg, "use_overlap", False))

        hist_pos = torch.arange(cfg.num_frames_latent, dtype=torch.float32) * stride  # [T]
        fut_start = (frames_in - 1) if use_overlap else frames_in
        fut_pos = fut_start + torch.arange(cfg.num_frames_latent, dtype=torch.float32) * stride  # [T]

        self.register_buffer(
            "abs_time_embed_hist",
            pos_embedding(hist_pos, cfg.hidden_size).unsqueeze(0),  # [1,T,D]
            persistent=False,
        )
        self.register_buffer(
            "abs_time_embed_fut",
            pos_embedding(fut_pos, cfg.hidden_size).unsqueeze(0),  # [1,T,D]
            persistent=False,
        )

        # Segment embeddings: 0=history context, 1=future/query
        self.seg_embed = nn.Parameter(torch.zeros(2, cfg.hidden_size))
        nn.init.trunc_normal_(self.seg_embed, std=0.02)

        # t + text -> AdaLN cond vector
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size * 4, cfg.hidden_size),
        )
        self.text_proj = nn.Linear(cfg.text_embed_dim, cfg.hidden_size, bias=True) if cfg.use_text_cond else None

        if cfg.use_video_cond:
            self.video_global_proj = nn.Linear(cfg.video_latent_channels, cfg.hidden_size, bias=True)
            # Learnable scale; set to 0.0 if you want to “warm start” from a track-only checkpoint.
            self.beta_video_cond = nn.Parameter(torch.tensor(1.0))
        else:
            self.video_global_proj = None
            self.beta_video_cond = None

        # ---------------------------------------------------------
        # NEW: track-global conditioning injected into AdaLN cond
        # ---------------------------------------------------------
        self.track_global_proj = nn.Linear(2 * cfg.latent_channels, cfg.hidden_size, bias=True)

        # Zero-init so it starts with *no effect* but learns immediately (stable like AdaLN-Zero).
        # nn.init.zeros_(self.track_global_proj.weight)
        # nn.init.zeros_(self.track_global_proj.bias)

        # Scalar scale; keep at 1.0 because proj is zero-init anyway.
        self.beta_track_cond = nn.Parameter(torch.tensor(1.0))

        self.blocks = nn.ModuleList(
            [DiTBlock(cfg.hidden_size, cfg.num_heads, mlp_ratio=cfg.mlp_ratio) for _ in range(cfg.depth)]
        )

        self.final_norm = nn.RMSNorm(cfg.hidden_size, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.latent_channels, bias=True)

        # -------------------------
        # Fusion parameters
        # -------------------------
        self.track_fusion_scale = nn.Parameter(torch.tensor(float(cfg.track_fusion_scale_init)))

        if bool(getattr(cfg, "track_fusion_use_time_gate", True)):
            # Initialize gate weights high early, lower later (learnable).
            T = int(cfg.num_frames_latent)
            g0 = torch.linspace(0.95, 0.35, T)  # early~0.95, late~0.35
            g0 = g0.clamp(1e-4, 1.0 - 1e-4)
            # store in logit-space so sigmoid(gate_logit) is in (0,1)
            gate_logit = torch.log(g0) - torch.log1p(-g0)
            self.track_fusion_gate_logit = nn.Parameter(gate_logit)
        else:
            self.track_fusion_gate_logit = None

        # Optional video fusion scale (only used if cfg.video_fusion != "none")
        if bool(getattr(cfg, "use_video_cond", False)) and str(getattr(cfg, "video_fusion", "none")) != "none":
            self.video_fusion_scale = nn.Parameter(torch.tensor(float(cfg.video_fusion_scale_init)))
        else:
            self.video_fusion_scale = None

    @torch.no_grad()
    def get_xattn_gate_stats(self):
        """
        Returns:
          (gate_spatial_mean, gate_temporal_mean) as scalar tensors on the current device.
        """
        if not hasattr(self, "blocks") or len(self.blocks) == 0:
            z = torch.tensor(0.0, device=next(self.parameters()).device)
            return z, z

        # by construction: blocks are [spatial, temporal, spatial, temporal, ...]
        g_sp = []
        g_tm = []
        for i, blk in enumerate(self.blocks):
            if not hasattr(blk, "last_gate_xattn_mean"):
                continue
            # scalar tensor saved from the most recent forward()
            g = blk.last_gate_xattn_mean.detach()
            if i % 2 == 0:
                g_sp.append(g)
            else:
                g_tm.append(g)

        dev = next(self.parameters()).device
        if len(g_sp) == 0:
            g_sp_mean = torch.tensor(0.0, device=dev)
        else:
            g_sp_mean = torch.stack(g_sp).mean()

        if len(g_tm) == 0:
            g_tm_mean = torch.tensor(0.0, device=dev)
        else:
            g_tm_mean = torch.stack(g_tm).mean()

        return g_sp_mean, g_tm_mean

    # -------------------------
    # Internal
    # -------------------------
    def _build_cond(
            self,
            t: torch.Tensor,
            text_cond: Optional[torch.Tensor],
            *,
            z_hist_track: Optional[torch.Tensor] = None,
            z_hist_video: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = timestep_embedding(t, self.cfg.hidden_size)
        temb = temb.to(
            device=self.time_mlp[0].weight.device,
            dtype=self.time_mlp[0].weight.dtype,
        )
        cond = self.time_mlp(temb)
        if self.text_proj is not None and text_cond is not None:
            cond = cond + self.text_proj(text_cond)

        # Add global video summary if enabled.
        # z_hist_video is [B,T,N,C_vid] -> mean over (T,N) -> [B,C_vid]
        if self.video_global_proj is not None:
            B = int(t.shape[0])
            if z_hist_video is None:
                vid_g = torch.zeros(
                    (B, self.cfg.video_latent_channels),
                    device=cond.device,
                    dtype=cond.dtype,
                )
            else:
                vid_g = z_hist_video.mean(dim=(1, 2)).to(dtype=cond.dtype)
            cond = cond + self.beta_video_cond * self.video_global_proj(vid_g)

        if hasattr(self, "track_global_proj") and (self.track_global_proj is not None):
            B = int(t.shape[0])
            C_lat = int(self.cfg.latent_channels)

            if z_hist_track is None:
                g = torch.zeros((B, 2 * C_lat), device=cond.device, dtype=cond.dtype)
            else:
                # Make sure dtype matches cond (bf16/fp16 safety)
                zh = z_hist_track.to(dtype=cond.dtype)

                g_mean = zh.mean(dim=(1, 2))  # [B,C_lat]
                g_last = zh[:, -1].mean(dim=1)  # [B,C_lat]
                g = torch.cat([g_mean, g_last], dim=-1)  # [B,2*C_lat]

            cond = cond + self.beta_track_cond * self.track_global_proj(g)
        return cond

    def _det_block_mean_pool2d(self, x: torch.Tensor, pool: int) -> torch.Tensor:
        """
        Deterministic pooling that behaves like "ceil" pooling:
          - Pads H/W to next multiple of `pool` with zeros
          - Computes per-block mean using reshape+sum (no adaptive_avg_pool2d)
          - Uses a mask so padded zeros do NOT bias the mean

        x: [B, C, H, W]
        returns: [B, C, ceil(H/pool), ceil(W/pool)]
        """
        assert x.dim() == 4, f"expected [B,C,H,W], got {tuple(x.shape)}"
        pool = int(pool)
        if pool <= 0:
            raise ValueError(f"pool must be >0, got {pool}")

        B, C, H, W = x.shape

        # Work in fp32 for stable sums; cast back at end
        orig_dtype = x.dtype
        x = x.float()

        pad_h = (pool - (H % pool)) % pool
        pad_w = (pool - (W % pool)) % pool

        if pad_h or pad_w:
            # pad order for 2D: (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            m = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)
            m = F.pad(m, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        else:
            m = torch.ones((B, 1, H, W), device=x.device, dtype=x.dtype)

        H2 = H + pad_h
        W2 = W + pad_w
        Ho = H2 // pool
        Wo = W2 // pool

        # Reshape into blocks and sum
        # x: [B,C,Ho,pool,Wo,pool]
        x = x.reshape(B, C, Ho, pool, Wo, pool)
        m = m.reshape(B, 1, Ho, pool, Wo, pool)

        sum_x = x.sum(dim=(3, 5))  # [B,C,Ho,Wo]
        cnt = m.sum(dim=(3, 5)).clamp_min(1.0)  # [B,1,Ho,Wo]
        out = sum_x / cnt  # broadcast over C

        return out.to(dtype=orig_dtype)

    def _run_blocks(
            self,
            x_btnd: torch.Tensor,
            cond_bd: torch.Tensor,
            *,
            ctx_hist_btnd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Deterministic-safe version:
          - temporal-first (so per-patch history is injected early)
          - pooled spatial ctx without adaptive_avg_pool2d (determinism-friendly)

        x_btnd: [B,T,N,D]
        cond_bd: [B,D]
        ctx_hist_btnd: [B,Tctx,N,D] where Tctx can be T or 2T (track+video)
        """
        B, T, N, D = x_btnd.shape
        assert N == self.num_patches, f"N={N} != num_patches={self.num_patches}"

        device = x_btnd.device
        dtype = x_btnd.dtype

        # -----------------------
        # Query embeddings
        # -----------------------
        fut_te = self.abs_time_embed_fut[:, :T].to(device=device, dtype=dtype)  # [1,T,D]
        x_btnd = x_btnd + fut_te.unsqueeze(2)  # [B,T,1,D]
        x_btnd = x_btnd + self.seg_embed[1].to(device=device, dtype=dtype).view(1, 1, 1, D)

        # spatial pos embedding on queries
        x_btnd = x_btnd + self.pos_embed.to(device=device, dtype=dtype).unsqueeze(1)  # [1,1,N,D]

        ctx_temp = None
        ctx_spatial = None

        if ctx_hist_btnd is not None:
            ctx = ctx_hist_btnd.to(device=device, dtype=dtype)
            Tctx = int(ctx.shape[1])

            # -----------------------
            # History time embedding + segment embedding
            # -----------------------
            hist_te = self.abs_time_embed_hist[:, :T].to(device=device, dtype=dtype)  # [1,T,D]
            if Tctx == T:
                ctx = ctx + hist_te.unsqueeze(2)
            elif Tctx == 2 * T:
                ctx[:, :T, ...] = ctx[:, :T, ...] + hist_te.unsqueeze(2)
                ctx[:, T:2 * T, ...] = ctx[:, T:2 * T, ...] + hist_te.unsqueeze(2)
            else:
                rep = (Tctx + T - 1) // T
                te = hist_te.repeat(1, rep, 1)[:, :Tctx]  # [1,Tctx,D]
                ctx = ctx + te.unsqueeze(2)

            ctx = ctx + self.seg_embed[0].to(device=device, dtype=dtype).view(1, 1, 1, D)

            # -----------------------
            # Temporal ctx (full per-patch history)  [B*N, Tctx, D]
            # -----------------------
            ctx_temp = rearrange(ctx, "b t n d -> (b n) t d")

            # -----------------------
            # Spatial ctx (pooled tokens) [B*T, Lc, D]
            # -----------------------
            mode = getattr(self.cfg, "spatial_ctx_mode", "pooled_last")

            if mode == "none":
                ctx_spatial = None

            elif mode == "time_mean":
                # Old behavior: loses per-patch structure (kept for ablation)
                ctx_time = ctx.mean(dim=2)  # [B,Tctx,D]
                ctx_spatial = rearrange(
                    ctx_time[:, None, :, :].expand(-1, T, -1, -1),
                    "b tf th d -> (b tf) th d",
                )  # [B*T, Tctx, D]

            elif mode == "pooled_last":
                pool = int(getattr(self.cfg, "spatial_ctx_pool", 8))
                last_k = int(getattr(self.cfg, "spatial_ctx_last_k", 1))
                include_video = bool(getattr(self.cfg, "spatial_ctx_include_video", True))

                last_k = max(1, min(last_k, T))

                parts = []

                # Track half is first T slots if ctx is [track, video] concatenated along time
                ctx_track = ctx[:, :min(Tctx, T), :, :]  # [B,<=T,N,D]
                parts.append(ctx_track[:, -last_k:, :, :])  # [B,last_k,N,D]

                if include_video and (Tctx >= 2 * T):
                    ctx_video = ctx[:, T:T + T, :, :]  # [B,T,N,D]
                    parts.append(ctx_video[:, -last_k:, :, :])

                ctx_sel = torch.cat(parts, dim=1)  # [B,K,N,D]
                K = int(ctx_sel.shape[1])

                # [B*K, D, Hgrid, Wgrid]
                ctx_grid = (
                    ctx_sel.reshape(B * K, self.grid_h, self.grid_w, D)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )

                # Deterministic pooling (NO adaptive_avg_pool2d)
                ctx_grid = self._det_block_mean_pool2d(ctx_grid, pool=pool)  # [B*K,D,Ho,Wo]
                Ho, Wo = int(ctx_grid.shape[-2]), int(ctx_grid.shape[-1])

                # tokens: [B, K*Ho*Wo, D]
                ctx_tokens = (
                    ctx_grid.permute(0, 2, 3, 1)
                    .reshape(B, K * Ho * Wo, D)
                )

                # pooled positional tokens (same Ho/Wo)
                pos_grid = (
                    self.pos_embed.to(device=device, dtype=dtype)
                    .reshape(1, self.grid_h, self.grid_w, D)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                pos_grid = self._det_block_mean_pool2d(pos_grid, pool=pool)  # [1,D,Ho,Wo]
                pos_tokens = (
                    pos_grid.permute(0, 2, 3, 1)
                    .reshape(1, Ho * Wo, D)
                    .repeat(1, K, 1)  # [1, K*Ho*Wo, D]
                )

                ctx_tokens = ctx_tokens + pos_tokens  # broadcast over B

                # same pooled ctx for every future time
                ctx_spatial = (
                    ctx_tokens[:, None, :, :]
                    .expand(-1, T, -1, -1)
                    .reshape(B * T, -1, D)
                )

            else:
                raise ValueError(f"Unknown spatial_ctx_mode='{mode}'")

        # -----------------------
        # Run blocks (temporal-first by default)
        # -----------------------
        temporal_first = bool(getattr(self.cfg, "temporal_first", True))

        for i in range(0, len(self.blocks), 2):
            spatial_block = self.blocks[i]
            temporal_block = self.blocks[i + 1]

            if temporal_first:
                # ---- temporal FIRST ----
                x_tmp = rearrange(x_btnd, "b t n d -> (b n) t d")
                c_tmp = repeat(cond_bd, "b d -> (b n) d", n=N)
                x_tmp = temporal_block(x_tmp, c_tmp, context=ctx_temp)
                x_btnd = rearrange(x_tmp, "(b n) t d -> b t n d", b=B, n=N)

                # ---- spatial SECOND ----
                x_sp = rearrange(x_btnd, "b t n d -> (b t) n d")
                c_sp = repeat(cond_bd, "b d -> (b t) d", t=T)
                x_sp = spatial_block(x_sp, c_sp, context=ctx_spatial)
                x_btnd = rearrange(x_sp, "(b t) n d -> b t n d", b=B, t=T)

            else:
                # ---- spatial THEN temporal (ablation) ----
                x_sp = rearrange(x_btnd, "b t n d -> (b t) n d")
                c_sp = repeat(cond_bd, "b d -> (b t) d", t=T)
                x_sp = spatial_block(x_sp, c_sp, context=ctx_spatial)
                x_btnd = rearrange(x_sp, "(b t) n d -> b t n d", b=B, t=T)

                x_tmp = rearrange(x_btnd, "b t n d -> (b n) t d")
                c_tmp = repeat(cond_bd, "b d -> (b n) d", n=N)
                x_tmp = temporal_block(x_tmp, c_tmp, context=ctx_temp)
                x_btnd = rearrange(x_tmp, "(b n) t d -> b t n d", b=B, n=N)

        return x_btnd

    # -------------------------
    # Flow model forward
    # -------------------------
    def forward(
            self,
            z_t: torch.Tensor,
            t: torch.Tensor,
            z_hist_track: torch.Tensor,
            hist_vis: Optional[torch.Tensor] = None,
            z_hist_video: Optional[torch.Tensor] = None,
            text_cond: Optional[torch.Tensor] = None,
    ):
        """Predict velocity field in track-latent space.

        Args:
            z_t:          [B, T_lat, N, C_lat] noisy latent at time t
            t:            [B] continuous in [0,1]
            z_hist_track: [B, T_lat, N, C_lat] history track latents (conditioning)
            hist_vis:     [B, T_lat, N] or [B, T_lat, N, 1] optional history visibility
            z_hist_video: [B, T_lat, N, C_vid] history video latents (optional)
            text_cond:    [B, text_embed_dim] pooled text embedding (optional)

        Returns:
            v_hat: [B, T_lat, N, C_lat]
        """
        B, T, N, C = z_t.shape
        assert T == self.cfg.num_frames_latent
        assert N == self.num_patches
        assert C == self.cfg.latent_channels
        assert z_hist_track.shape == z_t.shape

        if self.cfg.use_video_cond:
            if z_hist_video is None:
                z_hist_video = torch.zeros(
                    (B, T, N, self.cfg.video_latent_channels),
                    device=z_t.device,
                    dtype=z_t.dtype,
                )
            else:
                assert z_hist_video.shape[:3] == (B, T, N)

        x = self.in_proj(z_t)  # [B,T,N,D] queries
        ctx_track = self.hist_track_proj(z_hist_track)  # [B,T,N,D] track ctx

        if getattr(self, "hist_vis_proj", None) is not None:
            if hist_vis is None:
                vis_in = torch.zeros((B, T, N, 1), device=z_t.device, dtype=z_t.dtype)
            else:
                if hist_vis.dim() == 3:
                    vis_in = hist_vis.unsqueeze(-1)
                elif hist_vis.dim() == 4 and hist_vis.size(-1) == 1:
                    vis_in = hist_vis
                else:
                    raise ValueError(f"hist_vis must be [B,T,N] or [B,T,N,1], got {tuple(hist_vis.shape)}")
                vis_in = vis_in.to(device=z_t.device, dtype=z_t.dtype)

            vis_emb = self.hist_vis_proj(vis_in.to(dtype=ctx_track.dtype))
            alpha = self.alpha_hist_vis.to(dtype=ctx_track.dtype) if self.alpha_hist_vis is not None else 1.0
            ctx_track = ctx_track + alpha * vis_emb

        ctx_video = None
        if self.hist_video_proj is not None:
            if z_hist_video is None:
                z_hist_video_in = torch.zeros(
                    (B, T, N, self.cfg.video_latent_channels),
                    device=z_t.device,
                    dtype=z_t.dtype,
                )
            else:
                assert z_hist_video.shape[:3] == (B, T, N), f"z_hist_video must be [B,T,N,*], got {z_hist_video.shape}"
                z_hist_video_in = z_hist_video.to(dtype=z_t.dtype)

            ctx_video = self.hist_video_proj(z_hist_video_in)  # [B,T,N,D]

        # -----------------------------------------
        # Track/video context mixing (fix ambiguity)
        # -----------------------------------------
        ctx_hist = ctx_track
        if ctx_video is not None:
            mode = str(getattr(self.cfg, "video_ctx_mode", "sum"))
            if mode == "sum":
                ctx_hist = ctx_track + self.alpha_video_ctx * ctx_video  # [B,T,N,D]
            elif mode == "concat":
                ctx_hist = torch.cat([ctx_track, self.alpha_video_ctx * ctx_video], dim=1)  # [B,2T,N,D]
            else:
                raise ValueError(f"Unknown cfg.video_ctx_mode='{mode}'")

        # -----------------------------------------
        # Track fusion (inject aligned history)
        # -----------------------------------------
        fuse_mode = str(getattr(self.cfg, "track_fusion", "none"))
        if fuse_mode != "none":
            # Per-future-time gate (optional)
            if getattr(self, "track_fusion_gate_logit", None) is not None:
                gate = torch.sigmoid(self.track_fusion_gate_logit).to(device=x.device, dtype=x.dtype)  # [T]
                gate = gate.view(1, T, 1, 1)  # broadcast to [B,T,N,D]
            else:
                gate = 1.0

            last = ctx_track[:, -1:, ...]  # [B,1,N,D] broadcast across T

            if fuse_mode == "add_last":
                fused = last
            elif fuse_mode == "add_last_plus_vel":
                # Simple velocity hint from last two history latents (per patch)
                # vel: [B,1,N,D]
                if T >= 2:
                    vel = (ctx_track[:, -1, ...] - ctx_track[:, -2, ...]).unsqueeze(1)
                else:
                    vel = torch.zeros_like(last)

                # time weights 0..1 across future latent steps
                w = torch.linspace(0.0, 1.0, T, device=x.device, dtype=x.dtype).view(1, T, 1, 1)
                fused = last + w * vel
            else:
                raise ValueError(f"Unknown cfg.track_fusion='{fuse_mode}'")

            x = x + (self.track_fusion_scale.to(dtype=x.dtype) * gate) * fused

        # -----------------------------------------
        # Optional video fusion (start disabled)
        # -----------------------------------------
        vid_fuse = str(getattr(self.cfg, "video_fusion", "none"))
        if (vid_fuse != "none") and (ctx_video is not None) and (self.video_fusion_scale is not None):
            # reuse same time gate if present
            if getattr(self, "track_fusion_gate_logit", None) is not None:
                gate = torch.sigmoid(self.track_fusion_gate_logit).to(device=x.device, dtype=x.dtype).view(1, T, 1, 1)
            else:
                gate = 1.0

            if vid_fuse == "add_last":
                x = x + (self.video_fusion_scale.to(dtype=x.dtype) * gate) * ctx_video[:, -1:, ...]
            else:
                raise ValueError(f"Unknown cfg.video_fusion='{vid_fuse}'")

        # cond = self._build_cond(t, text_cond)  # [B,D]
        cond = self._build_cond(t, text_cond, z_hist_track=z_hist_track, z_hist_video=z_hist_video)
        x = self._run_blocks(x, cond, ctx_hist_btnd=ctx_hist)
        # x = self._run_blocks(x, cond, ctx_hist_btnd=ctx_track)

        x = self.final_norm(x)
        v_hat = self.out_proj(x)  # [B,T,N,C_lat]

        return v_hat

__all__ = [
    "TrajLoomGeneratorConfig",
    "TrajLoomGenerator",
]
