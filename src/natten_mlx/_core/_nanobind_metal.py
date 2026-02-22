"""Nanobind tier Metal kernel backend -- delegates to fast_metal.

This module exists so the nanobind fallback path (``_nanobind_impl``)
can resolve a Metal kernel backend without coupling its import chain
directly to ``fast_metal``.  All public symbols are re-exported from
``fast_metal``; there is no independent implementation.
"""

from natten_mlx._core.fast_metal import *  # noqa: F401,F403
from natten_mlx._core.fast_metal import is_available  # noqa: F401 -- explicit for _nanobind_impl._choose()
