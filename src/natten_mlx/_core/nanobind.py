"""Tier 2: nanobind-backed backend adapter.

By default this loads the in-tree `natten_mlx._core._nanobind_impl` module,
and may be overridden to an external compiled module via
`NATTEN_MLX_NANOBIND_MODULE`.
"""

from __future__ import annotations

import importlib
import os

from . import pure

_MODULE_NAME = os.environ.get("NATTEN_MLX_NANOBIND_MODULE", "natten_mlx._core._nanobind_impl")

try:
    _EXT = importlib.import_module(_MODULE_NAME)
except Exception:
    _EXT = None


def is_available() -> bool:
    return _EXT is not None


def _call_or_fallback(name: str, fallback, *args):
    if _EXT is None:
        return fallback(*args)
    fn = getattr(_EXT, name, None)
    if fn is None:
        return fallback(*args)
    return fn(*args)


def na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na1d_forward",
        pure.na1d_forward,
        q,
        k,
        v,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
    )


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na2d_forward",
        pure.na2d_forward,
        q,
        k,
        v,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
    )


def na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na1d_qk_forward",
        pure.na1d_qk_forward,
        q,
        k,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
    )


def na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    return _call_or_fallback(
        "na1d_av_forward",
        pure.na1d_av_forward,
        attn,
        v,
        kernel_size,
        stride,
        dilation,
        is_causal,
    )


def na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na2d_qk_forward",
        pure.na2d_qk_forward,
        q,
        k,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
    )


def na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    return _call_or_fallback(
        "na2d_av_forward",
        pure.na2d_av_forward,
        attn,
        v,
        kernel_size,
        stride,
        dilation,
        is_causal,
    )


def na1d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na1d_backward",
        pure.na1d_backward,
        q,
        k,
        v,
        grad_out,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
    )


def na2d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na2d_backward",
        pure.na2d_backward,
        q,
        k,
        v,
        grad_out,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
    )


def na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na1d_qk_backward",
        pure.na1d_qk_backward,
        q,
        k,
        grad_attn,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
    )


def na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    return _call_or_fallback(
        "na1d_av_backward",
        pure.na1d_av_backward,
        attn,
        v,
        grad_out,
        kernel_size,
        stride,
        dilation,
        is_causal,
    )


def na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na2d_qk_backward",
        pure.na2d_qk_backward,
        q,
        k,
        grad_attn,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
    )


def na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    return _call_or_fallback(
        "na2d_av_backward",
        pure.na2d_av_backward,
        attn,
        v,
        grad_out,
        kernel_size,
        stride,
        dilation,
        is_causal,
    )
