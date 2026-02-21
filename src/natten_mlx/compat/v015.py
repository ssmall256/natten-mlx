"""NATTEN v0.15.x compatibility layer for MLX tensors."""

from __future__ import annotations

import mlx.nn as nn

from .v014 import *  # noqa: F401,F403
from .v014 import __all__ as _v014_all
from natten_mlx.functional import na1d as _modern_na1d
from natten_mlx.functional import na1d_av, na1d_qk, na2d as _modern_na2d, na2d_av, na2d_qk, na3d as _modern_na3d
from natten_mlx.nn import NeighborhoodAttention3D as _ModernNeighborhoodAttention3D


def na1d(query, key, value, kernel_size, dilation=1, is_causal=False, scale=None):
    return _modern_na1d(
        query,
        key,
        value,
        kernel_size=kernel_size,
        stride=1,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
    )


def na2d(query, key, value, kernel_size, dilation=1, is_causal=False, scale=None):
    return _modern_na2d(
        query,
        key,
        value,
        kernel_size=kernel_size,
        stride=1,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
    )


def na3d(query, key, value, kernel_size, dilation=1, is_causal=False, scale=None):
    return _modern_na3d(
        query,
        key,
        value,
        kernel_size=kernel_size,
        stride=1,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
    )


class NeighborhoodAttention3D(nn.Module):
    """v0.15-style 3D module: uses ``dim``, no stride."""

    def __init__(
        self,
        dim,
        kernel_size,
        dilation=1,
        is_causal=False,
        num_heads=1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self._impl = _ModernNeighborhoodAttention3D(
            embed_dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def __call__(self, x):
        return self._impl(x)


def has_cuda() -> bool:
    return False


def has_mps() -> bool:
    return False


def has_mlx() -> bool:
    return True


def has_gemm() -> bool:
    return False


def has_fna() -> bool:
    return False


__all__ = list(_v014_all) + [
    "na1d",
    "na2d",
    "na3d",
    "na1d_qk",
    "na1d_av",
    "na2d_qk",
    "na2d_av",
    "NeighborhoodAttention3D",
    "has_cuda",
    "has_mps",
    "has_mlx",
    "has_gemm",
    "has_fna",
]
