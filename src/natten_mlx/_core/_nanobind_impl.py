"""Built-in nanobind backend implementation.

Provides a stable backend contract in-tree for the nanobind tier. This module
never delegates to ``fast_metal``; it routes to the nanobind-owned Metal kernel
backend when available, otherwise pure fallback.
"""

from __future__ import annotations

from . import pure

try:
    from . import _nanobind_metal
except Exception:  # pragma: no cover - guarded import fallback
    _nanobind_metal = None


def _choose():
    if _nanobind_metal is not None and _nanobind_metal.is_available():
        return _nanobind_metal
    return pure


def na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)


def na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    return _choose().na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)


def na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)


def na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    return _choose().na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)


def na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)


def na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    return _choose().na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)


def na1d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na1d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale)


def na2d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na2d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale)


def na3d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na3d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale)


def na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)


def na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    return _choose().na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)


def na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)


def na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    return _choose().na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)


def na3d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    return _choose().na3d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)


def na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    return _choose().na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)
