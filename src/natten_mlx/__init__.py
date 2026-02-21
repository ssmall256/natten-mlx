"""Top-level API for natten-mlx."""

from __future__ import annotations

from natten_mlx._core import fast_metal, nanobind, ops
from natten_mlx.functional import na1d, na1d_av, na1d_qk, na2d, na2d_av, na2d_qk, na3d, na3d_av, na3d_qk
from natten_mlx.nn import NeighborhoodAttention1D, NeighborhoodAttention2D, NeighborhoodAttention3D
from natten_mlx.support_matrix import get_support_matrix
from natten_mlx.version import __version__


def has_metal() -> bool:
    return fast_metal.is_available()


def has_nanobind() -> bool:
    return nanobind.is_available()


def get_backend() -> str:
    return ops.get_backend()


def set_backend(name: str) -> None:
    ops.set_backend(name)


__all__ = [
    "__version__",
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
    "has_metal",
    "has_nanobind",
    "get_backend",
    "set_backend",
    "get_support_matrix",
]
