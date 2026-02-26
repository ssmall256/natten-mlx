"""MLX module wrapper for 1D neighborhood attention."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from natten_mlx.functional import na1d, na1d_av, na1d_qk, na1d_varlen
from natten_mlx.utils.params import normalize_tuple_param


class NeighborhoodAttention1D(nn.Module):
    """1D Neighborhood Attention for MLX.

    Input:  [B, L, C]
    Output: [B, ceil(L/stride), C]

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of query attention heads.
        kernel_size: Neighborhood window size.
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

        self.kernel_size = normalize_tuple_param(kernel_size, 1, "kernel_size")
        self.stride = normalize_tuple_param(stride, 1, "stride")
        self.dilation = normalize_tuple_param(dilation, 1, "dilation")
        self.is_causal = normalize_tuple_param(is_causal, 1, "is_causal")
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

    def __call__(self, x: mx.array, seq_lens: mx.array | None = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input array of shape ``[B, L, C]``.
            seq_lens: Optional ``[B]`` int array of actual sequence lengths
                per batch element for variable-length attention.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, L, C], got {x.shape}")

        batch, length, channels = x.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"Input channel dim ({channels}) must match embed_dim ({self.embed_dim})"
            )

        if self._use_gqa:
            q = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim)
            kv = self.kv_proj(x).reshape(batch, length, 2, self.num_kv_heads, self.head_dim)
            k, v = mx.split(kv, 2, axis=2)
            k = k.squeeze(2)
            v = v.squeeze(2)
        else:
            qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
            q, k, v = mx.split(qkv, 3, axis=2)
            q = q.squeeze(2)
            k = k.squeeze(2)
            v = v.squeeze(2)

        if seq_lens is not None:
            out = na1d_varlen(
                q, k, v, seq_lens,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                scale=self.scale,
            )
        elif self.attn_drop_rate > 0.0:
            logits = na1d_qk(
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
            out = na1d_av(
                attn,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=self.stride,
                is_causal=self.is_causal,
            )
        else:
            out = na1d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                is_causal=self.is_causal,
                scale=self.scale,
            )

        out = out.reshape(out.shape[0], out.shape[1], self.embed_dim)
        out = self.proj(out)
        if self.proj_drop is not None:
            out = self.proj_drop(out)
        return out
