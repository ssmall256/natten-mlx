"""MLX module wrapper for 3D neighborhood attention."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from natten_mlx.functional import na3d, na3d_av, na3d_qk
from natten_mlx.utils.params import normalize_tuple_param


class NeighborhoodAttention3D(nn.Module):
    """3D Neighborhood Attention for MLX.

    Input:  [B, D, H, W, C]
    Output: [B, ceil(D/stride_d), ceil(H/stride_h), ceil(W/stride_w), C]

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of query attention heads.
        kernel_size: Neighborhood window size (scalar or 3-tuple).
        num_kv_heads: Number of key/value heads for GQA/MQA.  When ``None``
            (default) it equals ``num_heads`` (standard MHA).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        is_causal: Union[bool, Tuple[bool, ...]] = False,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
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
        self.num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else self.num_heads

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

        self.kernel_size = normalize_tuple_param(kernel_size, 3, "kernel_size")
        self.stride = normalize_tuple_param(stride, 3, "stride")
        self.dilation = normalize_tuple_param(dilation, 3, "dilation")
        self.is_causal = normalize_tuple_param(is_causal, 3, "is_causal")
        self.scale = float(qk_scale) if qk_scale is not None else self.head_dim ** -0.5

        self._use_gqa = self.num_kv_heads != self.num_heads
        if self._use_gqa:
            self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=qkv_bias)
            self.kv_proj = nn.Linear(embed_dim, 2 * self.num_kv_heads * self.head_dim, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)

        self.attn_drop_rate = float(attn_drop)
        self.attn_drop = nn.Dropout(self.attn_drop_rate) if self.attn_drop_rate > 0.0 else None
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop_rate = float(proj_drop)
        self.proj_drop = nn.Dropout(self.proj_drop_rate) if self.proj_drop_rate > 0.0 else None

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape [B, D, H, W, C], got {x.shape}")

        batch, depth, height, width, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Input channel dim ({channels}) must match embed_dim ({self.embed_dim})"
            )

        if self._use_gqa:
            q = self.q_proj(x).reshape(batch, depth, height, width, self.num_heads, self.head_dim)
            kv = self.kv_proj(x).reshape(
                batch, depth, height, width, 2, self.num_kv_heads, self.head_dim
            )
            k, v = mx.split(kv, 2, axis=4)
            k = k.squeeze(4)
            v = v.squeeze(4)
        else:
            qkv = self.qkv(x).reshape(
                batch, depth, height, width, 3, self.num_heads, self.head_dim
            )
            q, k, v = mx.split(qkv, 3, axis=4)
            q = q.squeeze(4)
            k = k.squeeze(4)
            v = v.squeeze(4)

        if self.attn_drop_rate > 0.0:
            logits = na3d_qk(
                q,
                k,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=self.stride,
                is_causal=self.is_causal,
                scale=self.scale,
            )
            attn = mx.softmax(logits, axis=-1)
            if self.attn_drop is not None:
                attn = self.attn_drop(attn)
            out = na3d_av(
                attn,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=self.stride,
                is_causal=self.is_causal,
            )
        else:
            out = na3d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                is_causal=self.is_causal,
                scale=self.scale,
            )

        out = out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3], self.embed_dim)
        out = self.proj(out)
        if self.proj_drop is not None:
            out = self.proj_drop(out)
        return out
