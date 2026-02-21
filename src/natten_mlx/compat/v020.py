"""NATTEN v0.20+ compatibility layer for MLX tensors."""

from natten_mlx.functional import na1d, na1d_av, na1d_qk, na2d, na2d_av, na2d_qk, na3d, na3d_av, na3d_qk
from natten_mlx.nn import NeighborhoodAttention1D, NeighborhoodAttention2D, NeighborhoodAttention3D


def has_cuda() -> bool:
    return False


def has_mps() -> bool:
    return False


def has_mlx() -> bool:
    return True


def has_fna() -> bool:
    return False


__all__ = [
    "na1d",
    "na2d",
    "na1d_qk",
    "na1d_av",
    "na2d_qk",
    "na2d_av",
    "na3d",
    "na3d_qk",
    "na3d_av",
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "NeighborhoodAttention3D",
    "has_cuda",
    "has_mps",
    "has_mlx",
    "has_fna",
]
