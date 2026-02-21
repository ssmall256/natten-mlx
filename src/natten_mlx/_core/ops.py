"""Backend dispatch for natten-mlx."""

from __future__ import annotations

import os
from typing import Any

from . import fast_metal, nanobind, pure

_BACKEND_REGISTRY: dict[str, Any] = {}
_ACTIVE_BACKEND = os.environ.get("NATTEN_BACKEND", "auto")


def register_backend(name: str, module: Any) -> None:
    _BACKEND_REGISTRY[name] = module


def _resolve_backend() -> str:
    if _ACTIVE_BACKEND != "auto":
        return _ACTIVE_BACKEND

    if nanobind.is_available():
        return "nanobind"
    if fast_metal.is_available():
        return "fast_metal"
    return "pure"


def get_backend() -> str:
    return _resolve_backend()


def set_backend(name: str) -> None:
    global _ACTIVE_BACKEND
    valid = {"auto", "pure", "fast_metal", "nanobind"}
    if name not in valid:
        raise ValueError(f"Unknown backend {name!r}. Expected one of {sorted(valid)}")
    _ACTIVE_BACKEND = name


def _backend_module() -> Any:
    name = _resolve_backend()
    try:
        return _BACKEND_REGISTRY[name]
    except KeyError as exc:
        raise RuntimeError(f"Backend {name!r} is not registered") from exc


def na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _backend_module().na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _backend_module().na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    return _backend_module().na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    return _backend_module().na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)


def na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    return _backend_module().na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)


def na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    return _backend_module().na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)


def na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    return _backend_module().na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)


def na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    return _backend_module().na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)


def na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    return _backend_module().na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)


def _backend_fn_or_pure(name: str):
    backend_module = _backend_module()
    fn = getattr(backend_module, name, None)
    if fn is not None:
        return fn
    return getattr(pure, name)


def na1d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    fn = _backend_fn_or_pure("na1d_backward")
    return fn(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale)


def na2d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    fn = _backend_fn_or_pure("na2d_backward")
    return fn(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale)


def na3d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    fn = _backend_fn_or_pure("na3d_backward")
    return fn(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale)


def na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    fn = _backend_fn_or_pure("na1d_qk_backward")
    return fn(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)


def na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    fn = _backend_fn_or_pure("na1d_av_backward")
    return fn(attn, v, grad_out, kernel_size, stride, dilation, is_causal)


def na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    fn = _backend_fn_or_pure("na2d_qk_backward")
    return fn(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)


def na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    fn = _backend_fn_or_pure("na2d_av_backward")
    return fn(attn, v, grad_out, kernel_size, stride, dilation, is_causal)


def na3d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    fn = _backend_fn_or_pure("na3d_qk_backward")
    return fn(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)


def na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    fn = _backend_fn_or_pure("na3d_av_backward")
    return fn(attn, v, grad_out, kernel_size, stride, dilation, is_causal)


register_backend("pure", pure)
register_backend("fast_metal", fast_metal)
register_backend("nanobind", nanobind)

if _ACTIVE_BACKEND not in {"auto", *list(_BACKEND_REGISTRY.keys())}:
    _ACTIVE_BACKEND = "auto"
