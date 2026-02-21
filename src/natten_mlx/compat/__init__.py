"""Compatibility API selector for historical NATTEN versions."""

from __future__ import annotations

from importlib import import_module
from packaging.version import Version


def for_version(version: str):
    """Return compat module for a target NATTEN version string."""
    parsed = Version(version)
    target = (parsed.major, parsed.minor)

    if target <= (0, 14):
        mod = "natten_mlx.compat.v014"
    elif target <= (0, 15):
        mod = "natten_mlx.compat.v015"
    elif target <= (0, 17):
        mod = "natten_mlx.compat.v017"
    else:
        mod = "natten_mlx.compat.v020"
    return import_module(mod)


__all__ = ["for_version"]
