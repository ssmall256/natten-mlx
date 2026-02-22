"""Tier 2: nanobind-backed backend adapter.

Resolution order:
1. `NATTEN_MLX_NANOBIND_MODULE` override (if set).
2. In-tree compiled extension: `natten_mlx._core._nanobind_ext`.
3. In-tree Python fallback: `natten_mlx._core._nanobind_impl`.

Availability semantics:
- `is_available()` reports compiled-extension availability only.
- If no compiled extension is available, functions still execute via the
  in-tree fallback implementation for explicit `set_backend("nanobind")`.
"""

from __future__ import annotations

import importlib
import os

import mlx.core as _mx  # noqa: F401

from . import pure

_OVERRIDE = os.environ.get("NATTEN_MLX_NANOBIND_MODULE")
_EXT: object | None = None
_FALLBACK_IMPL: object | None = None
_LOADED_MODULE_NAME: str | None = None

_EXT_CANDIDATES = [_OVERRIDE] if _OVERRIDE else ["natten_mlx._core._nanobind_ext"]
for _name in _EXT_CANDIDATES:
    if not _name:
        continue
    try:
        _candidate = importlib.import_module(_name)
        if _name == "natten_mlx._core._nanobind_impl":
            _FALLBACK_IMPL = _candidate
            _EXT = None
            _LOADED_MODULE_NAME = "natten_mlx._core._nanobind_impl"
            break
        _EXT = _candidate
        _LOADED_MODULE_NAME = _name
        break
    except Exception:
        continue

try:
    _FALLBACK_IMPL = importlib.import_module("natten_mlx._core._nanobind_impl")
    if _LOADED_MODULE_NAME is None:
        _LOADED_MODULE_NAME = "natten_mlx._core._nanobind_impl"
except Exception:
    _FALLBACK_IMPL = None


def is_available() -> bool:
    return _EXT is not None


def loaded_module_name() -> str | None:
    return _LOADED_MODULE_NAME


def _call_or_fallback(name: str, fallback, *args):
    if _EXT is not None:
        fn = getattr(_EXT, name, None)
        if fn is not None:
            try:
                return fn(*args)
            except (NotImplementedError, RuntimeError):
                pass
    if _FALLBACK_IMPL is not None:
        fn = getattr(_FALLBACK_IMPL, name, None)
        if fn is not None:
            return fn(*args)
    return fallback(*args)


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


def na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na3d_forward",
        pure.na3d_forward,
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


def na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na3d_qk_forward",
        pure.na3d_qk_forward,
        q,
        k,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
    )


def na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    return _call_or_fallback(
        "na3d_av_forward",
        pure.na3d_av_forward,
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


def na3d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na3d_backward",
        pure.na3d_backward,
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


def na3d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    return _call_or_fallback(
        "na3d_qk_backward",
        pure.na3d_qk_backward,
        q,
        k,
        grad_attn,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
    )


def na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    return _call_or_fallback(
        "na3d_av_backward",
        pure.na3d_av_backward,
        attn,
        v,
        grad_out,
        kernel_size,
        stride,
        dilation,
        is_causal,
    )
