"""Built-in nanobind backend implementation.

Provides a stable backend contract in-tree, while still allowing users to
override with an external compiled extension via `NATTEN_MLX_NANOBIND_MODULE`.
"""

from __future__ import annotations

from . import fast_metal, pure


def _choose():
    return fast_metal if fast_metal.is_available() else pure


def na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)


def na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    return _choose().na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)


def na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)


def na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    return _choose().na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)
