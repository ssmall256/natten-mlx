"""Tier 1: MLX fast Metal kernel backend."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from . import pure
from ._metal_sources import (
    source_1d_av,
    source_1d_fused,
    source_1d_qk,
    source_2d_av,
    source_2d_fused,
    source_2d_qk,
)

_SUPPORTED_KERNELS = {3, 5, 7}
_KERNEL_BUILD_FAILED = False
_QK_1D_KERNELS: dict[int, Callable] = {}
_AV_1D_KERNELS: dict[int, Callable] = {}
_QK_2D_KERNELS: dict[int, Callable] = {}
_AV_2D_KERNELS: dict[int, Callable] = {}
_FUSED_1D_KERNELS: dict[int, Callable] = {}
_FUSED_2D_KERNELS: dict[int, Callable] = {}
_RPB_1D_CACHE: dict[tuple[int, int], mx.array] = {}
_RPB_2D_CACHE: dict[tuple[int, int], mx.array] = {}


def is_available() -> bool:
    return (
        not _KERNEL_BUILD_FAILED
        and hasattr(mx, "fast")
        and hasattr(mx.fast, "metal_kernel")
    )


def _threadgroup_1d(length: int) -> tuple[int, int, int]:
    return (min(max(length, 1), 256), 1, 1)


def _threadgroup_2d(height: int, width: int) -> tuple[int, int, int]:
    # 16x8 is a stable default for these simple kernels on Apple GPUs.
    return (min(max(width, 1), 16), min(max(height, 1), 8), 1)


def _to_metal_1d(x: mx.array) -> mx.array:
    # Kernels expect [B, heads, L, D].
    return mx.transpose(x, axes=(0, 2, 1, 3))


def _from_metal_1d(x: mx.array) -> mx.array:
    # Back to [B, L, heads, D] or [B, L, heads, K].
    return mx.transpose(x, axes=(0, 2, 1, 3))


def _to_metal_2d(x: mx.array) -> mx.array:
    # Kernels expect [B, heads, H, W, D].
    return mx.transpose(x, axes=(0, 3, 1, 2, 4))


def _from_metal_2d(x: mx.array) -> mx.array:
    # Back to [B, H, W, heads, D] or [B, H, W, heads, K2].
    return mx.transpose(x, axes=(0, 2, 3, 1, 4))


def _zero_rpb_1d(heads: int, kernel_size: int, dtype: mx.Dtype) -> mx.array:
    key = (heads, kernel_size)
    if key not in _RPB_1D_CACHE:
        _RPB_1D_CACHE[key] = mx.zeros((heads, 2 * kernel_size - 1), dtype=dtype)
    return _RPB_1D_CACHE[key]


def _zero_rpb_2d(heads: int, kernel_size: int, dtype: mx.Dtype) -> mx.array:
    key = (heads, kernel_size)
    if key not in _RPB_2D_CACHE:
        size = 2 * kernel_size - 1
        _RPB_2D_CACHE[key] = mx.zeros((heads, size, size), dtype=dtype)
    return _RPB_2D_CACHE[key]


def _get_1d_qk_kernel(kernel_size: int):
    if kernel_size not in _QK_1D_KERNELS:
        _QK_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_qk_k{kernel_size}",
            input_names=["query", "key", "rpb", "dilation_param"],
            output_names=["out"],
            source=source_1d_qk(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_1D_KERNELS[kernel_size]


def _get_1d_av_kernel(kernel_size: int):
    if kernel_size not in _AV_1D_KERNELS:
        _AV_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_av_k{kernel_size}",
            input_names=["attention_probs", "value", "dilation_param"],
            output_names=["out"],
            source=source_1d_av(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_1D_KERNELS[kernel_size]


def _get_2d_qk_kernel(kernel_size: int):
    if kernel_size not in _QK_2D_KERNELS:
        _QK_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_qk_k{kernel_size}",
            input_names=["query", "key", "rpb", "dilation_param"],
            output_names=["out"],
            source=source_2d_qk(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_2D_KERNELS[kernel_size]


def _get_2d_av_kernel(kernel_size: int):
    if kernel_size not in _AV_2D_KERNELS:
        _AV_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_av_k{kernel_size}",
            input_names=["attention_probs", "value", "dilation_param"],
            output_names=["out"],
            source=source_2d_av(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_2D_KERNELS[kernel_size]


def _get_1d_fused_kernel(kernel_size: int):
    if kernel_size not in _FUSED_1D_KERNELS:
        _FUSED_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_fused_k{kernel_size}",
            input_names=["query", "key", "value", "dilation_param", "scale_param"],
            output_names=["out"],
            source=source_1d_fused(kernel_size),
            ensure_row_contiguous=True,
        )
    return _FUSED_1D_KERNELS[kernel_size]


def _get_2d_fused_kernel(kernel_size: int):
    if kernel_size not in _FUSED_2D_KERNELS:
        _FUSED_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_fused_k{kernel_size}",
            input_names=["query", "key", "value", "dilation_param", "scale_param"],
            output_names=["out"],
            source=source_2d_fused(kernel_size),
            ensure_row_contiguous=True,
        )
    return _FUSED_2D_KERNELS[kernel_size]


def _supports_1d(kernel_size, stride, is_causal) -> bool:
    return (
        int(kernel_size[0]) in _SUPPORTED_KERNELS
        and int(stride[0]) == 1
        and not bool(is_causal[0])
    )


def _supports_2d(kernel_size, stride, dilation, is_causal) -> bool:
    return (
        int(kernel_size[0]) == int(kernel_size[1])
        and int(kernel_size[0]) in _SUPPORTED_KERNELS
        and int(stride[0]) == 1
        and int(stride[1]) == 1
        and int(dilation[0]) == int(dilation[1])
        and not bool(is_causal[0])
        and not bool(is_causal[1])
    )


def _with_fallback(fn, fallback):
    global _KERNEL_BUILD_FAILED
    try:
        return fn()
    except Exception:
        _KERNEL_BUILD_FAILED = True
        return fallback()


def _with_grad_fallback(fn, fallback):
    try:
        return fn()
    except Exception:
        return fallback()


def na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_1d(kernel_size, stride, is_causal):
        return pure.na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)
    batch, length, heads, _ = q.shape
    ksize = int(kernel_size[0])
    dil = int(dilation[0])
    scale_value = (q.shape[-1] ** -0.5) if scale is None else float(scale)

    def _run():
        kernel = _get_1d_fused_kernel(ksize)
        q_m = _to_metal_1d(mx.astype(q, mx.float32))
        k_m = _to_metal_1d(mx.astype(k, mx.float32))
        v_m = _to_metal_1d(mx.astype(v, mx.float32))
        dilation_param = mx.array([dil], dtype=mx.int32)
        scale_param = mx.array([scale_value], dtype=mx.float32)
        out = kernel(
            inputs=[q_m, k_m, v_m, dilation_param, scale_param],
            grid=(length, 1, batch * heads),
            threadgroup=_threadgroup_1d(length),
            output_shapes=[(batch, heads, length, q.shape[-1])],
            output_dtypes=[mx.float32],
        )[0]
        return mx.astype(_from_metal_1d(out), q.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale),
    )


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_2d(kernel_size, stride, dilation, is_causal):
        return pure.na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)
    batch, height, width, heads, _ = q.shape
    ksize = int(kernel_size[0])
    dil = int(dilation[0])
    scale_value = (q.shape[-1] ** -0.5) if scale is None else float(scale)

    def _run():
        kernel = _get_2d_fused_kernel(ksize)
        q_m = _to_metal_2d(mx.astype(q, mx.float32))
        k_m = _to_metal_2d(mx.astype(k, mx.float32))
        v_m = _to_metal_2d(mx.astype(v, mx.float32))
        dilation_param = mx.array([dil], dtype=mx.int32)
        scale_param = mx.array([scale_value], dtype=mx.float32)
        out = kernel(
            inputs=[q_m, k_m, v_m, dilation_param, scale_param],
            grid=(width, height, batch * heads),
            threadgroup=_threadgroup_2d(height, width),
            output_shapes=[(batch, heads, height, width, q.shape[-1])],
            output_dtypes=[mx.float32],
        )[0]
        return mx.astype(_from_metal_2d(out), q.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale),
    )


def na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_1d(kernel_size, stride, is_causal):
        return pure.na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)

    batch, length, heads, _ = q.shape
    ksize = int(kernel_size[0])
    dil = int(dilation[0])
    scale_value = (q.shape[-1] ** -0.5) if scale is None else float(scale)

    def _run():
        kernel = _get_1d_qk_kernel(ksize)
        q_m = _to_metal_1d(mx.astype(q, mx.float32))
        k_m = _to_metal_1d(mx.astype(k, mx.float32))
        rpb = _zero_rpb_1d(heads, ksize, mx.float32)
        dilation_param = mx.array([dil], dtype=mx.int32)
        out = kernel(
            inputs=[q_m, k_m, rpb, dilation_param],
            grid=(length, 1, batch * heads),
            threadgroup=_threadgroup_1d(length),
            output_shapes=[(batch, heads, length, ksize)],
            output_dtypes=[mx.float32],
        )[0]
        out = _from_metal_1d(out) * scale_value
        return mx.astype(out, q.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale),
    )


def na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    if not is_available() or not _supports_1d(kernel_size, stride, is_causal):
        return pure.na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)

    batch, length, heads, ksize = attn.shape
    dil = int(dilation[0])

    if ksize != int(kernel_size[0]):
        return pure.na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)

    def _run():
        kernel = _get_1d_av_kernel(ksize)
        attn_m = _to_metal_1d(mx.astype(attn, mx.float32))
        v_m = _to_metal_1d(mx.astype(v, mx.float32))
        dilation_param = mx.array([dil], dtype=mx.int32)
        out = kernel(
            inputs=[attn_m, v_m, dilation_param],
            grid=(length, 1, batch * heads),
            threadgroup=_threadgroup_1d(length),
            output_shapes=[(batch, heads, length, v.shape[-1])],
            output_dtypes=[mx.float32],
        )[0]
        return mx.astype(_from_metal_1d(out), v.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal),
    )


def na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_2d(kernel_size, stride, dilation, is_causal):
        return pure.na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)

    batch, height, width, heads, _ = q.shape
    ksize = int(kernel_size[0])
    dil = int(dilation[0])
    area = ksize * ksize
    scale_value = (q.shape[-1] ** -0.5) if scale is None else float(scale)

    def _run():
        kernel = _get_2d_qk_kernel(ksize)
        q_m = _to_metal_2d(mx.astype(q, mx.float32))
        k_m = _to_metal_2d(mx.astype(k, mx.float32))
        rpb = _zero_rpb_2d(heads, ksize, mx.float32)
        dilation_param = mx.array([dil], dtype=mx.int32)
        out = kernel(
            inputs=[q_m, k_m, rpb, dilation_param],
            grid=(width, height, batch * heads),
            threadgroup=_threadgroup_2d(height, width),
            output_shapes=[(batch, heads, height, width, area)],
            output_dtypes=[mx.float32],
        )[0]
        out = _from_metal_2d(out) * scale_value
        return mx.astype(out, q.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale),
    )


def na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    if not is_available() or not _supports_2d(kernel_size, stride, dilation, is_causal):
        return pure.na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)

    batch, height, width, heads, area = attn.shape
    ksize = int(kernel_size[0])
    dil = int(dilation[0])

    if area != ksize * ksize:
        return pure.na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)

    def _run():
        kernel = _get_2d_av_kernel(ksize)
        attn_m = _to_metal_2d(mx.astype(attn, mx.float32))
        v_m = _to_metal_2d(mx.astype(v, mx.float32))
        dilation_param = mx.array([dil], dtype=mx.int32)
        out = kernel(
            inputs=[attn_m, v_m, dilation_param],
            grid=(width, height, batch * heads),
            threadgroup=_threadgroup_2d(height, width),
            output_shapes=[(batch, heads, height, width, v.shape[-1])],
            output_dtypes=[mx.float32],
        )[0]
        return mx.astype(_from_metal_2d(out), v.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal),
    )


def na1d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_1d(kernel_size, stride, is_causal):
        return pure.na1d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale)

    def _run():
        def _loss_q(q_in):
            out = na1d_forward(q_in, k, v, kernel_size, stride, dilation, is_causal, scale)
            return mx.sum(out * grad_out)

        def _loss_k(k_in):
            out = na1d_forward(q, k_in, v, kernel_size, stride, dilation, is_causal, scale)
            return mx.sum(out * grad_out)

        def _loss_v(v_in):
            out = na1d_forward(q, k, v_in, kernel_size, stride, dilation, is_causal, scale)
            return mx.sum(out * grad_out)

        return mx.grad(_loss_q)(q), mx.grad(_loss_k)(k), mx.grad(_loss_v)(v)

    return _with_grad_fallback(
        _run,
        lambda: pure.na1d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale),
    )


def na2d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_2d(kernel_size, stride, dilation, is_causal):
        return pure.na2d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale)

    def _run():
        def _loss_q(q_in):
            out = na2d_forward(q_in, k, v, kernel_size, stride, dilation, is_causal, scale)
            return mx.sum(out * grad_out)

        def _loss_k(k_in):
            out = na2d_forward(q, k_in, v, kernel_size, stride, dilation, is_causal, scale)
            return mx.sum(out * grad_out)

        def _loss_v(v_in):
            out = na2d_forward(q, k, v_in, kernel_size, stride, dilation, is_causal, scale)
            return mx.sum(out * grad_out)

        return mx.grad(_loss_q)(q), mx.grad(_loss_k)(k), mx.grad(_loss_v)(v)

    return _with_grad_fallback(
        _run,
        lambda: pure.na2d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale),
    )


def na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_1d(kernel_size, stride, is_causal):
        return pure.na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)

    def _run():
        def _loss_q(q_in):
            out = na1d_qk_forward(q_in, k, kernel_size, stride, dilation, is_causal, scale)
            return mx.sum(out * grad_attn)

        def _loss_k(k_in):
            out = na1d_qk_forward(q, k_in, kernel_size, stride, dilation, is_causal, scale)
            return mx.sum(out * grad_attn)

        return mx.grad(_loss_q)(q), mx.grad(_loss_k)(k)

    return _with_grad_fallback(
        _run,
        lambda: pure.na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale),
    )


def na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    if not is_available() or not _supports_1d(kernel_size, stride, is_causal):
        return pure.na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)

    def _run():
        def _loss_attn(attn_in):
            out = na1d_av_forward(attn_in, v, kernel_size, stride, dilation, is_causal)
            return mx.sum(out * grad_out)

        def _loss_v(v_in):
            out = na1d_av_forward(attn, v_in, kernel_size, stride, dilation, is_causal)
            return mx.sum(out * grad_out)

        return mx.grad(_loss_attn)(attn), mx.grad(_loss_v)(v)

    return _with_grad_fallback(
        _run,
        lambda: pure.na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal),
    )


def na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_2d(kernel_size, stride, dilation, is_causal):
        return pure.na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)

    def _run():
        def _loss_q(q_in):
            out = na2d_qk_forward(q_in, k, kernel_size, stride, dilation, is_causal, scale)
            return mx.sum(out * grad_attn)

        def _loss_k(k_in):
            out = na2d_qk_forward(q, k_in, kernel_size, stride, dilation, is_causal, scale)
            return mx.sum(out * grad_attn)

        return mx.grad(_loss_q)(q), mx.grad(_loss_k)(k)

    return _with_grad_fallback(
        _run,
        lambda: pure.na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale),
    )


def na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    if not is_available() or not _supports_2d(kernel_size, stride, dilation, is_causal):
        return pure.na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)

    def _run():
        def _loss_attn(attn_in):
            out = na2d_av_forward(attn_in, v, kernel_size, stride, dilation, is_causal)
            return mx.sum(out * grad_out)

        def _loss_v(v_in):
            out = na2d_av_forward(attn, v_in, kernel_size, stride, dilation, is_causal)
            return mx.sum(out * grad_out)

        return mx.grad(_loss_attn)(attn), mx.grad(_loss_v)(v)

    return _with_grad_fallback(
        _run,
        lambda: pure.na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal),
    )
