"""NATTEN v0.15.x compatibility layer for MLX tensors."""

from __future__ import annotations

from .v014 import *  # noqa: F401,F403
from .v014 import __all__ as _v014_all
from natten_mlx.functional import na1d as _modern_na1d
from natten_mlx.functional import na1d_av, na1d_qk, na2d as _modern_na2d, na2d_av, na2d_qk


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


class NeighborhoodAttention3D:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("3D neighborhood attention is not supported in natten-mlx")


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
