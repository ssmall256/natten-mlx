"""MLX module wrapper for 2D neighborhood attention."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from natten_mlx.functional import na2d, na2d_av, na2d_qk
from natten_mlx.utils.params import normalize_tuple_param


class NeighborhoodAttention2D(nn.Module):
    """2D Neighborhood Attention for MLX.

    Input:  [B, H, W, C]
    Output: [B, ceil(H/stride_h), ceil(W/stride_w), C]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size,
        stride=1,
        dilation=1,
        is_causal=False,
        qkv_bias: bool = True,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.kernel_size = normalize_tuple_param(kernel_size, 2, "kernel_size")
        self.stride = normalize_tuple_param(stride, 2, "stride")
        self.dilation = normalize_tuple_param(dilation, 2, "dilation")
        self.is_causal = normalize_tuple_param(is_causal, 2, "is_causal")
        self.scale = float(qk_scale) if qk_scale is not None else self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop_rate = float(attn_drop)
        self.attn_drop = nn.Dropout(self.attn_drop_rate) if self.attn_drop_rate > 0.0 else None
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop_rate = float(proj_drop)
        self.proj_drop = nn.Dropout(self.proj_drop_rate) if self.proj_drop_rate > 0.0 else None

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, H, W, C], got {x.shape}")

        batch, height, width, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Input channel dim ({channels}) must match embed_dim ({self.embed_dim})"
            )

        qkv = self.qkv(x).reshape(batch, height, width, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=3)
        q = q.squeeze(3)
        k = k.squeeze(3)
        v = v.squeeze(3)

        if self.attn_drop_rate > 0.0:
            if self.stride != (1, 1) or bool(self.is_causal[0]) or bool(self.is_causal[1]):
                raise NotImplementedError(
                    "attn_drop with stride != (1, 1) or is_causal=True is not supported yet. "
                    "Use attn_drop=0.0 for these configurations."
                )
            logits = na2d_qk(q, k, kernel_size=self.kernel_size, dilation=self.dilation)
            default_scale = self.head_dim ** -0.5
            if self.scale != default_scale:
                logits = logits * (self.scale / default_scale)
            attn = mx.softmax(logits, axis=-1)
            if self.attn_drop is not None:
                attn = self.attn_drop(attn)
            out = na2d_av(attn, v, kernel_size=self.kernel_size, dilation=self.dilation)
        else:
            out = na2d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                is_causal=self.is_causal,
                scale=self.scale,
            )

        out = out.reshape(out.shape[0], out.shape[1], out.shape[2], self.embed_dim)
        out = self.proj(out)
        if self.proj_drop is not None:
            out = self.proj_drop(out)
        return out
