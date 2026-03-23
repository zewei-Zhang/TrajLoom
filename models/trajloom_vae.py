"""
TrajLoomVAE for dense trajectory encoding and reconstruction.

This public model file keeps the trajectory VAE used by the final TrajLoom
configs, with the temporal compressor/expander and latent sampling path.
"""

import math
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Mlp
from torch.distributions import Normal, kl_divergence


class TemporalConvCompressor(nn.Module):
    """
    Learned temporal downsampling using Conv1d over *all* frames.

    CHANGE vs original:
      - use replicate padding (manual F.pad) instead of Conv1d zero-padding.
      - conv padding is set to 0; we pad explicitly.
    """

    def __init__(
            self,
            num_frames_in: int,
            num_frames_latent: int,
            hidden_size: int,
            factor: int,
    ):
        super().__init__()
        assert num_frames_latent > 1
        # Wan-style: T_lat = 1 + (T_in - 1) / factor
        assert (num_frames_in - 1) % factor == 0, \
            f"(num_frames_in - 1) must be divisible by factor, got {num_frames_in}, factor={factor}"
        expected_lat = 1 + (num_frames_in - 1) // factor
        assert num_frames_latent == expected_lat, \
            f"num_frames_latent={num_frames_latent}, expected {expected_lat}"

        self.num_frames_in = num_frames_in
        self.num_frames_latent = num_frames_latent
        self.hidden_size = hidden_size
        self.factor = factor

        self.pad = factor - 1
        k = 2 * factor - 1

        # IMPORTANT: padding=0 (we pad manually with replicate)
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=k,
            stride=factor,
            padding=0,
            groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_in, N, D = x.shape
        assert T_in == self.num_frames_in
        assert D == self.hidden_size

        x_bn_dt = rearrange(x, "b t n d -> (b n) d t")  # [B*N, D, T_in]

        if self.pad > 0:
            # replicate-pad time: [left, right]
            x_bn_dt = F.pad(x_bn_dt, (self.pad, self.pad), mode="replicate")

        y_bn_dt = self.conv(x_bn_dt)  # [B*N, D, T_lat']

        # Defensive crop/pad in case of any off-by-one
        if y_bn_dt.size(-1) > self.num_frames_latent:
            y_bn_dt = y_bn_dt[..., : self.num_frames_latent]
        elif y_bn_dt.size(-1) < self.num_frames_latent:
            pad = self.num_frames_latent - y_bn_dt.size(-1)
            y_bn_dt = F.pad(y_bn_dt, (0, pad))

        y = rearrange(y_bn_dt, "(b n) d t -> b t n d", b=B, n=N)
        return y  # [B, T_lat, N, D]


class TemporalConvExpander(nn.Module):
    """
    Learned temporal upsampling using ConvTranspose1d over *all* frames.

    CHANGE vs original:
      - replicate-pad in latent time BEFORE deconv
      - then crop back to exactly num_frames_in
    """

    def __init__(
            self,
            num_frames_in: int,
            num_frames_latent: int,
            hidden_size: int,
            factor: int,
    ):
        super().__init__()
        assert num_frames_latent > 1

        self.num_frames_in = num_frames_in
        self.num_frames_latent = num_frames_latent
        self.hidden_size = hidden_size
        self.factor = factor

        self.pad_lat = factor - 1  # replicate pad amount in latent-time

        k = 2 * factor - 1
        p = factor - 1
        self.deconv = nn.ConvTranspose1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=k,
            stride=factor,
            padding=p,
            output_padding=0,
            groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T_lat, N, D = x.shape
        assert T_lat == self.num_frames_latent
        assert D == self.hidden_size

        x_bn_dt = rearrange(x, "b t n d -> (b n) d t")  # [B*N, D, T_lat]

        if self.pad_lat > 0:
            x_bn_dt = F.pad(x_bn_dt, (self.pad_lat, self.pad_lat), mode="replicate")

        y_bn_dt = self.deconv(x_bn_dt)  # [B*N, D, T_in' (longer because we padded input)]

        if self.pad_lat > 0:
            # Crop center region back to exactly T_in
            start = self.pad_lat * self.factor
            end = start + self.num_frames_in
            y_bn_dt = y_bn_dt[..., start:end]

        # Defensive crop/pad
        if y_bn_dt.size(-1) > self.num_frames_in:
            y_bn_dt = y_bn_dt[..., : self.num_frames_in]
        elif y_bn_dt.size(-1) < self.num_frames_in:
            pad = self.num_frames_in - y_bn_dt.size(-1)
            y_bn_dt = F.pad(y_bn_dt, (0, pad))

        y = rearrange(y_bn_dt, "(b n) d t -> b t n d", b=B, n=N)
        return y  # [B, T_in, N, D]


def get_normalized_patch_centers(
        image_size: tuple[int, int],
        num_horizontal_patches: int,
        num_vertical_patches: int,
        device: torch.device,
) -> torch.Tensor:
    """
    Compute normalized (x, y) centers for a H×W image patchified into
    (num_vertical_patches × num_horizontal_patches) patches.

    Returns:
        centers: [N_patches, 2] in [-1, 1], with order (x, y)
    """
    h, w = image_size

    # Patch indices (i: vertical, j: horizontal)
    i, j = torch.meshgrid(
        torch.linspace(0.5, num_vertical_patches - 0.5, num_vertical_patches, device=device),
        torch.linspace(0.5, num_horizontal_patches - 0.5, num_horizontal_patches, device=device),
        indexing="ij",
    )
    # Convert patch indices ↦ pixel centers
    centers_x = j * (w / num_horizontal_patches)
    centers_y = i * (h / num_vertical_patches)

    centers = torch.stack([centers_x, centers_y], dim=-1)  # [Hv, Hw, 2]
    centers = centers.view(num_vertical_patches * num_horizontal_patches, 2)  # [N_patches, 2]

    # Normalize to [-1, 1] using raw image width/height
    scale = torch.tensor([w, h], device=device)
    centers = (centers / scale) * 2.0 - 1.0
    return centers  # [N_patches, 2]


class VAEOutput(NamedTuple):
    recons: torch.Tensor
    latents: torch.Tensor
    elbo: torch.Tensor
    ll: torch.Tensor
    kl: torch.Tensor
    latent_dist: Normal


class LearnableFourierFeaturesEncoding(nn.Module):
    def __init__(
            self,
            pos_dim: int,
            embed_dim: int,
            mlp_dim: int,
            gamma: float = 1.0,
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        half_dim = embed_dim // 2
        self.w = nn.Parameter(
            torch.randn(half_dim, pos_dim) * (gamma ** 2),
            requires_grad=False,
        )
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.scale = 1.0 / math.sqrt(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, pos_dim]
        xw = x @ self.w.t()  # [B, N, half_dim]
        feats = self.scale * torch.cat([torch.cos(xw), torch.sin(xw)], dim=-1)
        return self.mlp(feats)


class TrackPosEncoder(nn.Module):
    """Append learned Fourier coord features to (x,y) tracks."""

    def __init__(self, embed_dim: int, pos_embed_dim: int):
        super().__init__()
        self.lff = LearnableFourierFeaturesEncoding(
            pos_dim=2,
            embed_dim=pos_embed_dim,
            mlp_dim=4 * embed_dim,
        )

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:
        # tracks: [B, T, 2, H, W] in pixel coordinates
        x = rearrange(tracks, "b t c h w -> b t h w c")  # [B,T,H,W,2]
        lff = self.lff(rearrange(x, "b t h w c -> b (t h w) c"))
        lff = rearrange(
            lff,
            "b (t h w) d -> b t h w d",
            t=x.shape[1], h=x.shape[2], w=x.shape[3],
        )
        x = torch.cat([x, lff], dim=-1)  # [B,T,H,W,2+pos_embed_dim]
        x = rearrange(x, "b t h w d -> b t d h w")
        return x


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TrajTransformerBlock(nn.Module):
    """Latte-style transformer block."""

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True)

        # Kept for checkpoint compatibility with older no-DINO checkpoints.
        self.norm2 = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)

        self.norm3 = nn.RMSNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.gate_msa * self.attn(self.norm1(x))
        x = x + self.gate_mlp * self.mlp(self.norm3(x))
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_1d_sincos_temp_embed(embed_dim: int, length: int) -> np.ndarray:
    pos = np.arange(length, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed_hw(embed_dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    grid_y = np.arange(grid_h, dtype=np.float32)
    grid_x = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_x, grid_y)  # (x,y)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_h, grid_w)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed  # [grid_h*grid_w, embed_dim]


class TrajLoomVAE(nn.Module):
    """
    Track VAE aligned with Wan 2.1 VACE with *mid‑encoder* temporal compression.

      - Input: tracks [B, T_in, 2, H_raw, W_raw] (normalized in [-1, 1])
      - Latent: [B, T_lat, H_p*W_p, latent_channels]

    Temporal behaviour:

      - use_temp_compress=False:
          * No temporal compression, T_lat == T_in.
          * All encoder / decoder blocks run at full T_in.

      - use_temp_compress=True:
          * Encoder:
                * First half of blocks run at full T_in.
                * Then T_in → T_lat with Conv1d temporal compression.
                * Second half of blocks run at T_lat.
          * Decoder:
                * First half of blocks run at T_lat.
                * Then T_lat → T_in with matching ConvTranspose1d upsampling.
                * Second half of blocks run at T_in.

    This reduces memory/compute because deeper layers see a shorter sequence.

    If use_offsets=True:
      - Internally the model operates on offsets from a fixed canonical pixel grid
        (normalized to [-1, 1]).
      - Encoded input:  x_offset = x_abs - grid
      - Decoded output: x_abs_hat = x_offset_hat + grid

    From the outside, inputs/outputs are still absolute normalized coords.
    """

    def __init__(
            self,
            input_height: int = 480,
            input_width: int = 832,
            patch_size: int = 8,
            in_channels: int = 2,
            latent_channels: int = 8,
            hidden_size: int = 384,
            encoder_depth: int = 12,
            decoder_depth: int = 12,
            num_heads: int = 6,
            mlp_ratio: float = 4.0,
            num_frames_in: int = 81,
            num_frames_latent: int = 21,  # used only when use_temp_compress=True
            learn_sigma: bool = False,
            pos_embed_dim: int = 128,
            beta: float = 5e-5,
            use_temp_compress: bool = True,
            temp_stride: int = 4,  # Wan‑like: 1 + (T‑1)//4
            use_offsets: bool = False,  # <<< NEW: offset vs absolute mode
            **legacy_kwargs,
    ):
        super().__init__()

        ignored_legacy_keys = {
            "use_dino",
            "frame_features_dim",
            "num_frame_tokens",
            "temp_method",
            "use_refine3d",
        }
        unexpected_legacy_keys = set(legacy_kwargs) - ignored_legacy_keys
        if unexpected_legacy_keys:
            unexpected_str = ", ".join(sorted(unexpected_legacy_keys))
            raise TypeError(f"Unexpected keyword arguments: {unexpected_str}")

        # ---------------- Basic config ----------------
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.beta = beta
        self.hidden_size = hidden_size
        self.latent_channels = latent_channels

        self.input_height = input_height
        self.input_width = input_width
        self.patch_size = patch_size

        self.num_frames_in = num_frames_in
        self.use_temp_compress = use_temp_compress
        self.temp_stride = temp_stride  # factor used in Wan‑like formula

        # ---------------- Offset (anchor grid) config ----------------
        self.use_offsets = use_offsets
        if self.use_offsets and self.learn_sigma:
            raise ValueError(
                "use_offsets=True with learn_sigma=True is not implemented cleanly. "
                "Please set learn_sigma=False when using offsets."
            )

        if self.use_offsets:
            # Precompute canonical pixel centers in normalized [-1, 1] coords.
            # Shape: [1, 1, 2, H, W] where channel order is (x, y).
            with torch.no_grad():
                ys = torch.linspace(0.5, input_height - 0.5, steps=input_height)
                xs = torch.linspace(0.5, input_width - 0.5, steps=input_width)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                coords = torch.stack([xx, yy], dim=0)  # [2, H, W]
                scale = torch.tensor(
                    [input_width, input_height]
                ).view(2, 1, 1)  # (x, y)
                coords = (coords / scale) * 2.0 - 1.0  # → [-1, 1]
                anchor = coords.unsqueeze(0).unsqueeze(0)  # [1, 1, 2, H, W]
            self.register_buffer("offset_anchor_grid", anchor)
        else:
            self.offset_anchor_grid = None  # not used

        # ---------------- Temporal compression config ----------------
        if self.use_temp_compress:
            # Wan‑style: T_lat = 1 + (T_in - 1) // temp_stride
            auto_T_lat = 1 + (num_frames_in - 1) // temp_stride
            if num_frames_latent is None or num_frames_latent >= num_frames_in:
                self.num_frames_latent = auto_T_lat
            else:
                assert num_frames_latent == auto_T_lat, (
                    f"num_frames_latent={num_frames_latent} is inconsistent with "
                    f"num_frames_in={num_frames_in} and temp_stride={temp_stride}. "
                    f"Expected {auto_T_lat}."
                )
                self.num_frames_latent = num_frames_latent

            self.temp_compressor = TemporalConvCompressor(
                num_frames_in=self.num_frames_in,
                num_frames_latent=self.num_frames_latent,
                hidden_size=self.hidden_size,
                factor=self.temp_stride,
            )
            self.temp_expander = TemporalConvExpander(
                num_frames_in=self.num_frames_in,
                num_frames_latent=self.num_frames_latent,
                hidden_size=self.hidden_size,
                factor=self.temp_stride,
            )
        else:
            # No temporal compression: latent sequence uses full T_in.
            self.num_frames_latent = num_frames_in
            self.temp_compressor = None
            self.temp_expander = None

        # Enforce even depths so that we always have (spatial, temporal) block pairs.
        assert (
                encoder_depth % 2 == 0
        ), "encoder_depth must be even (spatial+temporal pairs)."
        assert (
                decoder_depth % 2 == 0
        ), "decoder_depth must be even (spatial+temporal pairs)."
        # Split point for "middle" where compression / expansion happens.
        self.encoder_mid = encoder_depth // 2
        self.decoder_mid = decoder_depth // 2

        # ---------------- Patch embedding ----------------
        self.x_embedder = PatchEmbed(
            img_size=(input_height, input_width),
            patch_size=(patch_size, patch_size),
            in_chans=in_channels,  # (x, y)
            embed_dim=hidden_size,
            bias=True,
        )
        grid_h, grid_w = self.x_embedder.grid_size
        self.grid_h = grid_h  # 60 for 480p with patch=8
        self.grid_w = grid_w  # 104 for 832 with patch=8
        num_patches = grid_h * grid_w

        # ---------------- Patch‑center Fourier features ----------------
        self.patch_pos_lff = LearnableFourierFeaturesEncoding(
            pos_dim=2,
            embed_dim=pos_embed_dim,
            mlp_dim=4 * pos_embed_dim,
        )
        self.patch_pos_proj = nn.Linear(pos_embed_dim, hidden_size)

        # ---------------- Latent projection ----------------
        self.to_latents = nn.Linear(hidden_size, 2 * latent_channels)
        self.from_latents = nn.Linear(latent_channels, hidden_size)

        # ---------------- Positional embeddings ----------------
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )
        # Only defined at full T_in; deeper encoder / decoder blocks at T_lat
        # reuse the already time‑aware features instead of a new embedding.
        self.temp_embed_in = nn.Parameter(
            torch.zeros(1, num_frames_in, hidden_size), requires_grad=False
        )

        # ---------------- Encoder / decoder transformer blocks ----------------
        self.encoder_blocks = nn.ModuleList(
            [
                TrajTransformerBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                TrajTransformerBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(decoder_depth)
            ]
        )

        # ---------------- Final projection back to track field ----------------
        self.final_layer = nn.Linear(
            hidden_size, patch_size * patch_size * self.out_channels
        )

        self.initialize_weights()

    # ------------------------------------------------------------------ #
    # Init helpers
    # ------------------------------------------------------------------ #
    def initialize_weights(self):
        # Linear layers
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.apply(_basic_init)

        # PatchEmbed proj
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj.bias, 0.0)

        # Fixed sin‑cos position embeddings
        pos_embed = get_2d_sincos_pos_embed_hw(
            self.pos_embed.shape[-1], self.grid_h, self.grid_w
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_in = get_1d_sincos_temp_embed(
            self.temp_embed_in.shape[-1], self.num_frames_in
        )
        self.temp_embed_in.data.copy_(
            torch.from_numpy(temp_in).float().unsqueeze(0)
        )

        # Zero init final layer
        nn.init.constant_(self.final_layer.weight, 0.0)
        nn.init.constant_(self.final_layer.bias, 0.0)

    # ------------------------------------------------------------------ #
    # Utility: unpatchify
    # ------------------------------------------------------------------ #
    def unpatchify(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        x: [N, num_patches, patch_size*patch_size*C]
        Returns: [N, C, H_raw, W_raw]
        """
        N, L, _ = x.shape
        c = self.out_channels
        gh, gw = self.grid_h, self.grid_w
        p = patch_size
        assert L == gh * gw
        x = x.view(N, gh, gw, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [N, C, gh, p, gw, p]
        x = x.view(N, c, gh * p, gw * p)
        return x

    # ------------------------------------------------------------------ #
    # Offset helpers (internal)
    # ------------------------------------------------------------------ #
    def _to_offset_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 2, H, W] absolute normalized coords → offsets from canonical grid.
        """
        if not self.use_offsets:
            return x
        assert (
                self.offset_anchor_grid is not None
        ), "offset_anchor_grid is not initialized"
        B, T, C, H, W = x.shape
        assert (
                C == self.in_channels
        ), f"use_offsets expects {self.in_channels} input channels, got {C}"
        anchor = self.offset_anchor_grid.to(device=x.device, dtype=x.dtype)
        _, _, C_a, H_a, W_a = anchor.shape
        assert C_a == C and H_a == H and W_a == W
        anchor = anchor.expand(B, T, C_a, H_a, W_a)
        return x - anchor

    def _from_offset_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W] offsets → absolute coords (on first in_channels).
        Other channels (e.g. sigma) are passed through unchanged.
        """
        if not self.use_offsets:
            return x
        assert (
                self.offset_anchor_grid is not None
        ), "offset_anchor_grid is not initialized"
        B, T, C, H, W = x.shape
        anchor = self.offset_anchor_grid.to(device=x.device, dtype=x.dtype)
        _, _, C_a, H_a, W_a = anchor.shape
        assert C_a == self.in_channels and H_a == H and W_a == W
        anchor = anchor.expand(B, T, C_a, H_a, W_a)
        coords = x[:, :, : self.in_channels] + anchor
        if C == self.in_channels:
            return coords
        rest = x[:, :, self.in_channels:]
        return torch.cat([coords, rest], dim=2)

    # ------------------------------------------------------------------ #
    # Temporal downsample / upsample
    # ------------------------------------------------------------------ #
    def _temp_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T_in, N, D] -> [B, T_lat, N, D] (if compression enabled).
        """
        if not self.use_temp_compress or self.temp_compressor is None:
            return x
        return self.temp_compressor(x)

    def _temp_upsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T_lat, N, D] -> [B, T_in, N, D] (if compression enabled).
        """
        if not self.use_temp_compress or self.temp_expander is None:
            return x
        return self.temp_expander(x)

    # ------------------------------------------------------------------ #
    # Shared S/T transformer runner
    # ------------------------------------------------------------------ #
    def _run_blocks(
            self,
            x_tokens: torch.Tensor,
            blocks,
            f_spatial: Optional[torch.Tensor],
            f_temp: Optional[torch.Tensor],
            batch_size: int,
            num_frames: int,
            add_temp_embed: bool = False,
    ) -> torch.Tensor:
        """
        Run a sequence of TrackTransformerBlocks, alternating spatial+temporal,
        on tokens arranged as x_tokens: [(B*num_frames), N_patches, D].
        """
        if len(blocks) == 0:
            return x_tokens

        x = x_tokens
        for local_idx in range(0, len(blocks), 2):
            spatial_block = blocks[local_idx]
            x = spatial_block(x, f_spatial)

            if local_idx + 1 < len(blocks):
                temp_block = blocks[local_idx + 1]
                x = rearrange(
                    x,
                    "(b t) n d -> (b n) t d",
                    b=batch_size,
                    t=num_frames,
                )
                if add_temp_embed and local_idx == 0:
                    # Add temporal embedding once at the first temporal block
                    # of a stage that runs at full T_in.
                    # When num_frames != num_frames_in, caller must pass add_temp_embed=False.
                    x = x + self.temp_embed_in[:, :num_frames]
                x = temp_block(x, f_temp)
                x = rearrange(
                    x,
                    "(b n) t d -> (b t) n d",
                    b=batch_size,
                    n=self.grid_h * self.grid_w,
                )
        return x

    # ------------------------------------------------------------------ #
    # Encode / decode / forward
    # ------------------------------------------------------------------ #
    def encode(
            self,
            x: torch.Tensor,
            f: Optional[torch.Tensor] = None,
            use_fp16: bool = False,
    ) -> Normal:
        """
        x: [B, T_in, 2, H_raw, W_raw]  (tracks normalized in [-1,1])
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)
            if f is not None:
                f = f.to(dtype=torch.float16)

        # Apply offset transform if enabled (absolute → offsets)
        if self.use_offsets:
            x = self._to_offset_space(x)

        B, T, C, H, W = x.shape
        assert T == self.num_frames_in, (
            f"Expected num_frames_in={self.num_frames_in}, got {T}"
        )
        assert H == self.input_height and W == self.input_width, (
            f"Expected input size {(self.input_height, self.input_width)}, got {(H, W)}"
        )

        # Patchify tracks: (B,T,2,H,W) → (B*T, 2, H, W) → (B*T, N_patches, hidden_size)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.x_embedder(x)  # [(B*T), N_patches, D]
        # Add fixed 2D sin‑cos patch positional embedding
        x = x + self.pos_embed  # [1, N_patches, D] broadcast

        # Learnable Fourier features on patch centers (token‑level)
        patch_centers = get_normalized_patch_centers(
            image_size=(self.input_height, self.input_width),
            num_horizontal_patches=self.grid_w,
            num_vertical_patches=self.grid_h,
            device=x.device,
        )  # [N_patches, 2] in [-1,1]
        patch_centers = patch_centers.to(dtype=x.dtype)
        patch_centers = patch_centers.unsqueeze(0).expand(
            x.shape[0], -1, -1
        )  # [(B*T), N_patches, 2]

        lff = self.patch_pos_lff(patch_centers)  # [(B*T), N_patches, pos_embed_dim]
        lff = self.patch_pos_proj(lff)  # [(B*T), N_patches, D]
        x = x + lff

        # ---------------- Encoder stage 1: full T_in ----------------
        if not self.use_temp_compress:
            # No compression at all: run all encoder blocks at full T_in.
            x = self._run_blocks(
                x_tokens=x,
                blocks=self.encoder_blocks,
                f_spatial=None,
                f_temp=None,
                batch_size=B,
                num_frames=self.num_frames_in,
                add_temp_embed=True,
            )
            x_full = rearrange(
                x,
                "(b t) n d -> b t n d",
                b=B,
                t=self.num_frames_in,
            )  # [B, T_in, N, D]
            x_lat = x_full  # T_lat == T_in
        else:
            # First half of encoder at full T_in
            enc_blocks_full = list(self.encoder_blocks[: self.encoder_mid])
            x = self._run_blocks(
                x_tokens=x,
                blocks=enc_blocks_full,
                f_spatial=None,
                f_temp=None,
                batch_size=B,
                num_frames=self.num_frames_in,
                add_temp_embed=True,
            )
            x_full = rearrange(
                x,
                "(b t) n d -> b t n d",
                b=B,
                t=self.num_frames_in,
            )  # [B, T_in, N, D]

            # ---------------- Temporal compression in the middle ----------------
            x_lat = self._temp_downsample(x_full)  # [B, T_lat, N, D]

            # Second half of encoder at T_lat (if any blocks left)
            if self.encoder_mid < len(self.encoder_blocks):
                enc_blocks_lat = list(self.encoder_blocks[self.encoder_mid:])
                x_lat_tokens = rearrange(
                    x_lat,
                    "b t n d -> (b t) n d",
                    b=B,
                    t=self.num_frames_latent,
                )
                x_lat_tokens = self._run_blocks(
                    x_tokens=x_lat_tokens,
                    blocks=enc_blocks_lat,
                    f_spatial=None,
                    f_temp=None,
                    batch_size=B,
                    num_frames=self.num_frames_latent,
                    add_temp_embed=False,
                )
                x_lat = rearrange(
                    x_lat_tokens,
                    "(b t) n d -> b t n d",
                    b=B,
                    t=self.num_frames_latent,
                )  # [B, T_lat, N, D]

        # Project to Gaussian latent
        mean, logvar = self.to_latents(x_lat).chunk(2, dim=-1)
        scale = F.softplus(0.5 * logvar) + 1e-8
        return Normal(loc=mean, scale=scale)

    def decode(
            self,
            z: torch.Tensor,
            f: Optional[torch.Tensor] = None,
            use_fp16: bool = False,
    ) -> torch.Tensor:
        """
        z: [B, T_lat, N_patches, latent_channels]
        """
        if use_fp16:
            z = z.to(dtype=torch.float16)
            if f is not None:
                f = f.to(dtype=torch.float16)

        B, T_lat, N, C_lat = z.shape
        assert T_lat == self.num_frames_latent, (
            f"Expected T_lat={self.num_frames_latent}, got {T_lat}"
        )
        assert N == self.grid_h * self.grid_w

        # Latent projection
        x_lat = self.from_latents(z)  # [B, T_lat, N, D]

        # ---------------- Decoder stage 1 (near latent): at T_lat ----------------
        if not self.use_temp_compress:
            # No compression: decoder always runs at full T_in == T_lat
            x_full = x_lat  # [B, T_in, N, D]
            x_tokens = rearrange(
                x_full,
                "b t n d -> (b t) n d",
                b=B,
                t=self.num_frames_in,
            )
            x_tokens = self._run_blocks(
                x_tokens=x_tokens,
                blocks=self.decoder_blocks,
                f_spatial=None,
                f_temp=None,
                batch_size=B,
                num_frames=self.num_frames_in,
                add_temp_embed=False,  # original decoder had no temporal pos‑embed
            )
        else:
            dec_blocks_lat = list(self.decoder_blocks[: self.decoder_mid])
            dec_blocks_full = list(self.decoder_blocks[self.decoder_mid:])

            # Stage at T_lat
            if len(dec_blocks_lat) > 0:
                x_lat_tokens = rearrange(
                    x_lat,
                    "b t n d -> (b t) n d",
                    b=B,
                    t=self.num_frames_latent,
                )
                x_lat_tokens = self._run_blocks(
                    x_tokens=x_lat_tokens,
                    blocks=dec_blocks_lat,
                    f_spatial=None,
                    f_temp=None,
                    batch_size=B,
                    num_frames=self.num_frames_latent,
                    add_temp_embed=False,
                )
                x_lat = rearrange(
                    x_lat_tokens,
                    "(b t) n d -> b t n d",
                    b=B,
                    t=self.num_frames_latent,
                )

            # ---------------- Temporal expansion in the middle ----------------
            x_full = self._temp_upsample(x_lat)  # [B, T_in, N, D]

            # Final stage at full T_in
            x_tokens = rearrange(
                x_full,
                "b t n d -> (b t) n d",
                b=B,
                t=self.num_frames_in,
            )
            if len(dec_blocks_full) > 0:
                x_tokens = self._run_blocks(
                    x_tokens=x_tokens,
                    blocks=dec_blocks_full,
                    f_spatial=None,
                    f_temp=None,
                    batch_size=B,
                    num_frames=self.num_frames_in,
                    add_temp_embed=False,
                )

        # Final patch‑wise projection + unpatchify
        x_tokens = self.final_layer(x_tokens)  # [(B*T_in), N, p*p*C_out]
        x_images = self.unpatchify(
            x_tokens, patch_size=self.x_embedder.patch_size[0]
        )  # [(B*T_in), C_out, H_raw, W_raw]
        x_images = rearrange(
            x_images,
            "(b t) c h w -> b t c h w",
            b=B,
            t=self.num_frames_in,
        )

        # Convert offsets → absolute coords if needed
        if self.use_offsets:
            x_images = self._from_offset_space(x_images)

        return x_images

    def forward(
            self,
            x: torch.Tensor,
            f: Optional[torch.Tensor] = None,
            use_fp16: bool = False,
            deterministic: bool = False,
            vis_mask_bthw: Optional[torch.Tensor] = None,
    ) -> VAEOutput:
        """
        x: [B, T_in, 2, H_raw, W_raw]  (normalized in [-1,1] outside this module)
        vis_mask_bthw: [B, T_in, H_raw, W_raw] or None (visibility mask)
        """
        q_z_given_x = self.encode(x, use_fp16=use_fp16)
        p_z = Normal(
            loc=torch.zeros_like(q_z_given_x.loc),
            scale=torch.ones_like(q_z_given_x.scale),
        )

        if deterministic:
            z = q_z_given_x.mode
        else:
            z = q_z_given_x.rsample()

        x_hat = self.decode(z, use_fp16=use_fp16)

        # KL per‑sample
        kl = kl_divergence(q_z_given_x, p_z).mean(dim=[1, 2, 3])

        # Reconstruction log‑likelihood (Huber)
        ll_map = -F.huber_loss(x_hat, x, reduction="none")  # [B, T, C, H, W]

        if vis_mask_bthw is not None:
            m = vis_mask_bthw.to(dtype=ll_map.dtype, device=ll_map.device).unsqueeze(
                2
            )  # [B,T,1,H,W]
            vis_sum = (ll_map * m).sum(dim=[1, 2, 3, 4], dtype=torch.float32)
            vis_cnt = (
                    m.sum(dim=[1, 2, 3, 4], dtype=torch.float32) * ll_map.shape[2]
            ).clamp_min(1.0)
            ll = (vis_sum / vis_cnt).to(ll_map.dtype)
        else:
            ll = ll_map.mean(dim=[1, 2, 3, 4])

        if deterministic:
            elbo = ll
        else:
            elbo = ll - self.beta * kl

        return VAEOutput(
            recons=x_hat,
            latents=z,
            elbo=elbo,
            ll=ll,
            kl=kl,
            latent_dist=q_z_given_x,
        )

__all__ = [
    "TemporalConvCompressor",
    "TemporalConvExpander",
    "VAEOutput",
    "TrajLoomVAE",
]
