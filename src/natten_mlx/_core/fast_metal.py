"""Tier 1: MLX fast Metal kernel backend."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from . import pure
from ._metal_sources import (
    source_1d_av,
    source_1d_av_backward_v,
    source_1d_fused,
    source_1d_qk,
    source_1d_qk_backward_k,
    source_2d_av,
    source_2d_av_backward_v,
    source_2d_fused,
    source_2d_qk,
    source_2d_qk_backward_k,
    source_3d_av,
    source_3d_av_backward_v,
    source_3d_fused,
    source_3d_qk,
    source_3d_qk_backward_k,
)

_KERNEL_BUILD_FAILED = False
_QK_1D_KERNELS: dict[int, Callable] = {}
_AV_1D_KERNELS: dict[int, Callable] = {}
_QK_2D_KERNELS: dict[int, Callable] = {}
_AV_2D_KERNELS: dict[int, Callable] = {}
_QK_3D_KERNELS: dict[int, Callable] = {}
_AV_3D_KERNELS: dict[int, Callable] = {}
_FUSED_1D_KERNELS: dict[int, Callable] = {}
_FUSED_2D_KERNELS: dict[int, Callable] = {}
_FUSED_3D_KERNELS: dict[int, Callable] = {}
_QK_BWD_K_1D_KERNELS: dict[int, Callable] = {}
_AV_BWD_V_1D_KERNELS: dict[int, Callable] = {}
_QK_BWD_K_2D_KERNELS: dict[int, Callable] = {}
_AV_BWD_V_2D_KERNELS: dict[int, Callable] = {}
_QK_BWD_K_3D_KERNELS: dict[int, Callable] = {}
_AV_BWD_V_3D_KERNELS: dict[int, Callable] = {}
_RPB_1D_CACHE: dict[tuple[int, int], mx.array] = {}
_RPB_2D_CACHE: dict[tuple[int, int], mx.array] = {}
_RPB_3D_CACHE: dict[tuple[int, int], mx.array] = {}


def is_available() -> bool:
    return (
        not _KERNEL_BUILD_FAILED
        and hasattr(mx, "fast")
        and hasattr(mx.fast, "metal_kernel")
    )


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _query_positions(length: int, stride: int) -> tuple[mx.array, int]:
    out_length = _ceil_div(length, stride)
    return mx.arange(out_length, dtype=mx.int32) * int(stride), out_length


def _cast(x: mx.array, dtype: mx.Dtype) -> mx.array:
    if hasattr(mx, "astype"):
        return mx.astype(x, dtype)
    return x.astype(dtype)


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


def _to_metal_3d(x: mx.array) -> mx.array:
    # Kernels expect [B, heads, D, H, W, C].
    return mx.transpose(x, axes=(0, 4, 1, 2, 3, 5))


def _from_metal_3d(x: mx.array) -> mx.array:
    # Back to [B, D, H, W, heads, C] or [B, D, H, W, heads, K3].
    return mx.transpose(x, axes=(0, 2, 3, 4, 1, 5))


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


def _zero_rpb_3d(heads: int, kernel_size: int, dtype: mx.Dtype) -> mx.array:
    key = (heads, kernel_size)
    if key not in _RPB_3D_CACHE:
        size = 2 * kernel_size - 1
        _RPB_3D_CACHE[key] = mx.zeros((heads, size, size, size), dtype=dtype)
    return _RPB_3D_CACHE[key]


def _get_1d_qk_kernel(kernel_size: int):
    if kernel_size not in _QK_1D_KERNELS:
        _QK_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_qk_k{kernel_size}",
            input_names=["query", "key", "rpb", "dilation_param", "causal_param"],
            output_names=["out"],
            source=source_1d_qk(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_1D_KERNELS[kernel_size]


def _get_1d_av_kernel(kernel_size: int):
    if kernel_size not in _AV_1D_KERNELS:
        _AV_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_av_k{kernel_size}",
            input_names=["attention_probs", "value", "dilation_param", "causal_param"],
            output_names=["out"],
            source=source_1d_av(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_1D_KERNELS[kernel_size]


def _get_2d_qk_kernel(kernel_size: int):
    if kernel_size not in _QK_2D_KERNELS:
        _QK_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_qk_k{kernel_size}",
            input_names=["query", "key", "rpb", "dilation_param", "causal_param"],
            output_names=["out"],
            source=source_2d_qk(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_2D_KERNELS[kernel_size]


def _get_2d_av_kernel(kernel_size: int):
    if kernel_size not in _AV_2D_KERNELS:
        _AV_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_av_k{kernel_size}",
            input_names=["attention_probs", "value", "dilation_param", "causal_param"],
            output_names=["out"],
            source=source_2d_av(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_2D_KERNELS[kernel_size]


def _get_3d_qk_kernel(kernel_size: int):
    if kernel_size not in _QK_3D_KERNELS:
        _QK_3D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na3d_qk_k{kernel_size}",
            input_names=["query", "key", "rpb", "dilation_param", "causal_param"],
            output_names=["out"],
            source=source_3d_qk(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_3D_KERNELS[kernel_size]


def _get_3d_av_kernel(kernel_size: int):
    if kernel_size not in _AV_3D_KERNELS:
        _AV_3D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na3d_av_k{kernel_size}",
            input_names=["attention_probs", "value", "dilation_param", "causal_param"],
            output_names=["out"],
            source=source_3d_av(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_3D_KERNELS[kernel_size]


def _get_3d_fused_kernel(kernel_size: int):
    if kernel_size not in _FUSED_3D_KERNELS:
        _FUSED_3D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na3d_fused_k{kernel_size}",
            input_names=[
                "query",
                "key",
                "value",
                "stride_param",
                "dilation_param",
                "causal_param",
                "scale_param",
            ],
            output_names=["out"],
            source=source_3d_fused(kernel_size),
            ensure_row_contiguous=True,
        )
    return _FUSED_3D_KERNELS[kernel_size]


def _get_1d_fused_kernel(kernel_size: int):
    if kernel_size not in _FUSED_1D_KERNELS:
        _FUSED_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_fused_k{kernel_size}",
            input_names=[
                "query",
                "key",
                "value",
                "stride_param",
                "dilation_param",
                "causal_param",
                "scale_param",
            ],
            output_names=["out"],
            source=source_1d_fused(kernel_size),
            ensure_row_contiguous=True,
        )
    return _FUSED_1D_KERNELS[kernel_size]


def _get_2d_fused_kernel(kernel_size: int):
    if kernel_size not in _FUSED_2D_KERNELS:
        _FUSED_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_fused_k{kernel_size}",
            input_names=[
                "query",
                "key",
                "value",
                "stride_param",
                "dilation_param",
                "causal_param",
                "scale_param",
            ],
            output_names=["out"],
            source=source_2d_fused(kernel_size),
            ensure_row_contiguous=True,
        )
    return _FUSED_2D_KERNELS[kernel_size]


def _get_1d_qk_backward_k_kernel(kernel_size: int):
    if kernel_size not in _QK_BWD_K_1D_KERNELS:
        _QK_BWD_K_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_qk_bwd_k_k{kernel_size}",
            input_names=[
                "grad_attn",
                "query",
                "stride_param",
                "dilation_param",
                "causal_param",
                "scale_param",
            ],
            output_names=["out"],
            source=source_1d_qk_backward_k(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_BWD_K_1D_KERNELS[kernel_size]


def _get_1d_av_backward_v_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_V_1D_KERNELS:
        _AV_BWD_V_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_av_bwd_v_k{kernel_size}",
            input_names=[
                "attention_probs",
                "grad_out",
                "target_shape_param",
                "stride_param",
                "dilation_param",
                "causal_param",
            ],
            output_names=["out"],
            source=source_1d_av_backward_v(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_V_1D_KERNELS[kernel_size]


def _get_2d_qk_backward_k_kernel(kernel_size: int):
    if kernel_size not in _QK_BWD_K_2D_KERNELS:
        _QK_BWD_K_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_qk_bwd_k_k{kernel_size}",
            input_names=[
                "grad_attn",
                "query",
                "stride_param",
                "dilation_param",
                "causal_param",
                "scale_param",
            ],
            output_names=["out"],
            source=source_2d_qk_backward_k(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_BWD_K_2D_KERNELS[kernel_size]


def _get_2d_av_backward_v_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_V_2D_KERNELS:
        _AV_BWD_V_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_av_bwd_v_k{kernel_size}",
            input_names=[
                "attention_probs",
                "grad_out",
                "target_shape_param",
                "stride_param",
                "dilation_param",
                "causal_param",
            ],
            output_names=["out"],
            source=source_2d_av_backward_v(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_V_2D_KERNELS[kernel_size]


def _get_3d_qk_backward_k_kernel(kernel_size: int):
    if kernel_size not in _QK_BWD_K_3D_KERNELS:
        _QK_BWD_K_3D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na3d_qk_bwd_k_k{kernel_size}",
            input_names=[
                "grad_attn",
                "query",
                "stride_param",
                "dilation_param",
                "causal_param",
                "scale_param",
            ],
            output_names=["out"],
            source=source_3d_qk_backward_k(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_BWD_K_3D_KERNELS[kernel_size]


def _get_3d_av_backward_v_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_V_3D_KERNELS:
        _AV_BWD_V_3D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na3d_av_bwd_v_k{kernel_size}",
            input_names=[
                "attention_probs",
                "grad_out",
                "target_shape_param",
                "stride_param",
                "dilation_param",
                "causal_param",
            ],
            output_names=["out"],
            source=source_3d_av_backward_v(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_V_3D_KERNELS[kernel_size]


def _is_valid_kernel_size(kernel_size: int) -> bool:
    return kernel_size > 0 and (kernel_size % 2 == 1)


def _supports_1d_split(kernel_size, stride, dilation, is_causal) -> bool:
    return (
        _is_valid_kernel_size(int(kernel_size[0]))
        and int(stride[0]) >= 1
        and int(dilation[0]) >= 1
    )


def _supports_2d_split(kernel_size, stride, dilation, is_causal) -> bool:
    return (
        int(kernel_size[0]) == int(kernel_size[1])
        and _is_valid_kernel_size(int(kernel_size[0]))
        and int(stride[0]) >= 1
        and int(stride[1]) >= 1
        and int(dilation[0]) >= 1
        and int(dilation[1]) >= 1
    )


def _supports_3d_split(kernel_size, stride, dilation, is_causal) -> bool:
    return (
        int(kernel_size[0]) == int(kernel_size[1]) == int(kernel_size[2])
        and _is_valid_kernel_size(int(kernel_size[0]))
        and int(stride[0]) >= 1
        and int(stride[1]) >= 1
        and int(stride[2]) >= 1
        and int(dilation[0]) >= 1
        and int(dilation[1]) >= 1
        and int(dilation[2]) >= 1
    )


def _supports_1d_fused(kernel_size, stride, dilation) -> bool:
    return (
        _is_valid_kernel_size(int(kernel_size[0]))
        and int(stride[0]) >= 1
        and int(dilation[0]) >= 1
    )


def _supports_2d_fused(kernel_size, stride, dilation) -> bool:
    return (
        int(kernel_size[0]) == int(kernel_size[1])
        and _is_valid_kernel_size(int(kernel_size[0]))
        and int(stride[0]) >= 1
        and int(stride[1]) >= 1
        and int(dilation[0]) >= 1
        and int(dilation[1]) >= 1
    )


def _supports_3d_fused(kernel_size, stride, dilation) -> bool:
    return (
        int(kernel_size[0]) == int(kernel_size[1]) == int(kernel_size[2])
        and _is_valid_kernel_size(int(kernel_size[0]))
        and int(stride[0]) >= 1
        and int(stride[1]) >= 1
        and int(stride[2]) >= 1
        and int(dilation[0]) >= 1
        and int(dilation[1]) >= 1
        and int(dilation[2]) >= 1
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
    if not is_available() or not _supports_1d_fused(kernel_size, stride, dilation):
        return pure.na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)
    batch, length, heads, _ = q.shape
    ksize = int(kernel_size[0])
    step = int(stride[0])
    dil = int(dilation[0])
    causal = 1 if bool(is_causal[0]) else 0
    out_length = _ceil_div(length, step)
    scale_value = (q.shape[-1] ** -0.5) if scale is None else float(scale)

    def _run():
        kernel = _get_1d_fused_kernel(ksize)
        q_m = _to_metal_1d(_cast(q, mx.float32))
        k_m = _to_metal_1d(_cast(k, mx.float32))
        v_m = _to_metal_1d(_cast(v, mx.float32))
        stride_param = mx.array([step], dtype=mx.int32)
        dilation_param = mx.array([dil], dtype=mx.int32)
        causal_param = mx.array([causal], dtype=mx.int32)
        scale_param = mx.array([scale_value], dtype=mx.float32)
        out = kernel(
            inputs=[q_m, k_m, v_m, stride_param, dilation_param, causal_param, scale_param],
            grid=(out_length, 1, batch * heads),
            threadgroup=_threadgroup_1d(out_length),
            output_shapes=[(batch, heads, out_length, q.shape[-1])],
            output_dtypes=[mx.float32],
        )[0]
        return _cast(_from_metal_1d(out), q.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale),
    )


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_2d_fused(kernel_size, stride, dilation):
        return pure.na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)
    batch, height, width, heads, _ = q.shape
    ksize = int(kernel_size[0])
    stride_h = int(stride[0])
    stride_w = int(stride[1])
    out_height = _ceil_div(height, stride_h)
    out_width = _ceil_div(width, stride_w)
    dil_h = int(dilation[0])
    dil_w = int(dilation[1])
    causal_h = 1 if bool(is_causal[0]) else 0
    causal_w = 1 if bool(is_causal[1]) else 0
    scale_value = (q.shape[-1] ** -0.5) if scale is None else float(scale)

    def _run():
        kernel = _get_2d_fused_kernel(ksize)
        q_m = _to_metal_2d(_cast(q, mx.float32))
        k_m = _to_metal_2d(_cast(k, mx.float32))
        v_m = _to_metal_2d(_cast(v, mx.float32))
        stride_param = mx.array([stride_h, stride_w], dtype=mx.int32)
        dilation_param = mx.array([dil_h, dil_w], dtype=mx.int32)
        causal_param = mx.array([causal_h, causal_w], dtype=mx.int32)
        scale_param = mx.array([scale_value], dtype=mx.float32)
        out = kernel(
            inputs=[q_m, k_m, v_m, stride_param, dilation_param, causal_param, scale_param],
            grid=(out_width, out_height, batch * heads),
            threadgroup=_threadgroup_2d(out_height, out_width),
            output_shapes=[(batch, heads, out_height, out_width, q.shape[-1])],
            output_dtypes=[mx.float32],
        )[0]
        return _cast(_from_metal_2d(out), q.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale),
    )


def na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    if not is_available():
        return pure.na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)
    if _supports_3d_fused(kernel_size, stride, dilation):
        batch, depth, height, width, heads, _ = q.shape
        ksize = int(kernel_size[0])
        stride_d = int(stride[0])
        stride_h = int(stride[1])
        stride_w = int(stride[2])
        out_depth = _ceil_div(depth, stride_d)
        out_height = _ceil_div(height, stride_h)
        out_width = _ceil_div(width, stride_w)
        dil_d = int(dilation[0])
        dil_h = int(dilation[1])
        dil_w = int(dilation[2])
        causal_d = 1 if bool(is_causal[0]) else 0
        causal_h = 1 if bool(is_causal[1]) else 0
        causal_w = 1 if bool(is_causal[2]) else 0
        scale_value = (q.shape[-1] ** -0.5) if scale is None else float(scale)

        def _run_fused():
            kernel = _get_3d_fused_kernel(ksize)
            q_m = _to_metal_3d(_cast(q, mx.float32))
            k_m = _to_metal_3d(_cast(k, mx.float32))
            v_m = _to_metal_3d(_cast(v, mx.float32))
            stride_param = mx.array([stride_d, stride_h, stride_w], dtype=mx.int32)
            dilation_param = mx.array([dil_d, dil_h, dil_w], dtype=mx.int32)
            causal_param = mx.array([causal_d, causal_h, causal_w], dtype=mx.int32)
            scale_param = mx.array([scale_value], dtype=mx.float32)
            out = kernel(
                inputs=[
                    q_m,
                    k_m,
                    v_m,
                    stride_param,
                    dilation_param,
                    causal_param,
                    scale_param,
                ],
                grid=(out_width, out_height, batch * heads * out_depth),
                threadgroup=_threadgroup_2d(out_height, out_width),
                output_shapes=[(batch, heads, out_depth, out_height, out_width, q.shape[-1])],
                output_dtypes=[mx.float32],
            )[0]
            return _cast(_from_metal_3d(out), q.dtype)

        return _with_fallback(
            _run_fused,
            lambda: pure.na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale),
        )

    if _supports_3d_split(kernel_size, stride, dilation, is_causal):
        def _run_split():
            logits = na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)
            attn = mx.softmax(logits, axis=-1)
            return na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)

        return _with_fallback(
            _run_split,
            lambda: pure.na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale),
        )

    return pure.na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale)


def na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_1d_split(kernel_size, stride, dilation, is_causal):
        return pure.na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)

    batch, length, heads, _ = q.shape
    ksize = int(kernel_size[0])
    step = int(stride[0])
    dil = int(dilation[0])
    causal = 1 if bool(is_causal[0]) else 0
    qpos, _ = _query_positions(length, step)
    scale_value = (q.shape[-1] ** -0.5) if scale is None else float(scale)

    def _run():
        kernel = _get_1d_qk_kernel(ksize)
        q_m = _to_metal_1d(_cast(q, mx.float32))
        k_m = _to_metal_1d(_cast(k, mx.float32))
        rpb = _zero_rpb_1d(heads, ksize, mx.float32)
        dilation_param = mx.array([dil], dtype=mx.int32)
        causal_param = mx.array([causal], dtype=mx.int32)
        out = kernel(
            inputs=[q_m, k_m, rpb, dilation_param, causal_param],
            grid=(length, 1, batch * heads),
            threadgroup=_threadgroup_1d(length),
            output_shapes=[(batch, heads, length, ksize)],
            output_dtypes=[mx.float32],
        )[0]
        out = _from_metal_1d(out) * scale_value
        if step != 1:
            out = mx.take(out, qpos, axis=1)
        return _cast(out, q.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale),
    )


def na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    if not is_available() or not _supports_1d_split(kernel_size, stride, dilation, is_causal):
        return pure.na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)

    batch, out_length, heads, ksize = attn.shape
    length = int(v.shape[1])
    step = int(stride[0])
    dil = int(dilation[0])
    causal = 1 if bool(is_causal[0]) else 0
    qpos, expected_out = _query_positions(length, step)

    if ksize != int(kernel_size[0]):
        return pure.na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)
    if out_length != expected_out:
        return pure.na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)

    def _run():
        kernel = _get_1d_av_kernel(ksize)
        attn_full = attn
        if step != 1:
            idx = mx.reshape(qpos, (1, expected_out, 1, 1))
            attn_full = mx.put_along_axis(
                mx.zeros((batch, length, heads, ksize), dtype=attn.dtype),
                idx,
                attn,
                axis=1,
            )
        attn_m = _to_metal_1d(_cast(attn_full, mx.float32))
        v_m = _to_metal_1d(_cast(v, mx.float32))
        dilation_param = mx.array([dil], dtype=mx.int32)
        causal_param = mx.array([causal], dtype=mx.int32)
        out = kernel(
            inputs=[attn_m, v_m, dilation_param, causal_param],
            grid=(length, 1, batch * heads),
            threadgroup=_threadgroup_1d(length),
            output_shapes=[(batch, heads, length, v.shape[-1])],
            output_dtypes=[mx.float32],
        )[0]
        out = _from_metal_1d(out)
        if step != 1:
            out = mx.take(out, qpos, axis=1)
        return _cast(out, v.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal),
    )


def na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_2d_split(kernel_size, stride, dilation, is_causal):
        return pure.na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)

    batch, height, width, heads, _ = q.shape
    ksize = int(kernel_size[0])
    step_h = int(stride[0])
    step_w = int(stride[1])
    dil_h = int(dilation[0])
    dil_w = int(dilation[1])
    causal_h = 1 if bool(is_causal[0]) else 0
    causal_w = 1 if bool(is_causal[1]) else 0
    qh, _ = _query_positions(height, step_h)
    qw, _ = _query_positions(width, step_w)
    area = ksize * ksize
    scale_value = (q.shape[-1] ** -0.5) if scale is None else float(scale)

    def _run():
        kernel = _get_2d_qk_kernel(ksize)
        q_m = _to_metal_2d(_cast(q, mx.float32))
        k_m = _to_metal_2d(_cast(k, mx.float32))
        rpb = _zero_rpb_2d(heads, ksize, mx.float32)
        dilation_param = mx.array([dil_h, dil_w], dtype=mx.int32)
        causal_param = mx.array([causal_h, causal_w], dtype=mx.int32)
        out = kernel(
            inputs=[q_m, k_m, rpb, dilation_param, causal_param],
            grid=(width, height, batch * heads),
            threadgroup=_threadgroup_2d(height, width),
            output_shapes=[(batch, heads, height, width, area)],
            output_dtypes=[mx.float32],
        )[0]
        out = _from_metal_2d(out) * scale_value
        if step_h != 1:
            out = mx.take(out, qh, axis=1)
        if step_w != 1:
            out = mx.take(out, qw, axis=2)
        return _cast(out, q.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale),
    )


def na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    if not is_available() or not _supports_2d_split(kernel_size, stride, dilation, is_causal):
        return pure.na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)

    batch, out_h, out_w, heads, area = attn.shape
    height = int(v.shape[1])
    width = int(v.shape[2])
    ksize = int(kernel_size[0])
    step_h = int(stride[0])
    step_w = int(stride[1])
    dil_h = int(dilation[0])
    dil_w = int(dilation[1])
    causal_h = 1 if bool(is_causal[0]) else 0
    causal_w = 1 if bool(is_causal[1]) else 0
    qh, expected_h = _query_positions(height, step_h)
    qw, expected_w = _query_positions(width, step_w)

    if area != ksize * ksize:
        return pure.na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)
    if out_h != expected_h or out_w != expected_w:
        return pure.na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)

    def _run():
        kernel = _get_2d_av_kernel(ksize)
        attn_full = attn
        if step_h != 1 or step_w != 1:
            idx_h = mx.reshape(qh, (1, expected_h, 1, 1, 1))
            attn_h = mx.put_along_axis(
                mx.zeros((batch, height, expected_w, heads, area), dtype=attn.dtype),
                idx_h,
                attn,
                axis=1,
            )
            idx_w = mx.reshape(qw, (1, 1, expected_w, 1, 1))
            attn_full = mx.put_along_axis(
                mx.zeros((batch, height, width, heads, area), dtype=attn.dtype),
                idx_w,
                attn_h,
                axis=2,
            )
        attn_m = _to_metal_2d(_cast(attn_full, mx.float32))
        v_m = _to_metal_2d(_cast(v, mx.float32))
        dilation_param = mx.array([dil_h, dil_w], dtype=mx.int32)
        causal_param = mx.array([causal_h, causal_w], dtype=mx.int32)
        out = kernel(
            inputs=[attn_m, v_m, dilation_param, causal_param],
            grid=(width, height, batch * heads),
            threadgroup=_threadgroup_2d(height, width),
            output_shapes=[(batch, heads, height, width, v.shape[-1])],
            output_dtypes=[mx.float32],
        )[0]
        out = _from_metal_2d(out)
        if step_h != 1:
            out = mx.take(out, qh, axis=1)
        if step_w != 1:
            out = mx.take(out, qw, axis=2)
        return _cast(out, v.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal),
    )


def na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_3d_split(kernel_size, stride, dilation, is_causal):
        return pure.na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)

    batch, depth, height, width, heads, _ = q.shape
    ksize = int(kernel_size[0])
    step_d = int(stride[0])
    step_h = int(stride[1])
    step_w = int(stride[2])
    dil_d = int(dilation[0])
    dil_h = int(dilation[1])
    dil_w = int(dilation[2])
    causal_d = 1 if bool(is_causal[0]) else 0
    causal_h = 1 if bool(is_causal[1]) else 0
    causal_w = 1 if bool(is_causal[2]) else 0
    qd, _ = _query_positions(depth, step_d)
    qh, _ = _query_positions(height, step_h)
    qw, _ = _query_positions(width, step_w)
    volume = ksize * ksize * ksize
    scale_value = (q.shape[-1] ** -0.5) if scale is None else float(scale)

    def _run():
        kernel = _get_3d_qk_kernel(ksize)
        q_m = _to_metal_3d(_cast(q, mx.float32))
        k_m = _to_metal_3d(_cast(k, mx.float32))
        rpb = _zero_rpb_3d(heads, ksize, mx.float32)
        dilation_param = mx.array([dil_d, dil_h, dil_w], dtype=mx.int32)
        causal_param = mx.array([causal_d, causal_h, causal_w], dtype=mx.int32)
        out = kernel(
            inputs=[q_m, k_m, rpb, dilation_param, causal_param],
            grid=(width, height, batch * heads * depth),
            threadgroup=_threadgroup_2d(height, width),
            output_shapes=[(batch, heads, depth, height, width, volume)],
            output_dtypes=[mx.float32],
        )[0]
        out = _from_metal_3d(out) * scale_value
        if step_d != 1:
            out = mx.take(out, qd, axis=1)
        if step_h != 1:
            out = mx.take(out, qh, axis=2)
        if step_w != 1:
            out = mx.take(out, qw, axis=3)
        return _cast(out, q.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale),
    )


def na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    if not is_available() or not _supports_3d_split(kernel_size, stride, dilation, is_causal):
        return pure.na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)

    batch, out_d, out_h, out_w, heads, volume = attn.shape
    depth = int(v.shape[1])
    height = int(v.shape[2])
    width = int(v.shape[3])
    ksize = int(kernel_size[0])
    step_d = int(stride[0])
    step_h = int(stride[1])
    step_w = int(stride[2])
    dil_d = int(dilation[0])
    dil_h = int(dilation[1])
    dil_w = int(dilation[2])
    causal_d = 1 if bool(is_causal[0]) else 0
    causal_h = 1 if bool(is_causal[1]) else 0
    causal_w = 1 if bool(is_causal[2]) else 0
    qd, expected_d = _query_positions(depth, step_d)
    qh, expected_h = _query_positions(height, step_h)
    qw, expected_w = _query_positions(width, step_w)

    if volume != ksize * ksize * ksize:
        return pure.na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)
    if out_d != expected_d or out_h != expected_h or out_w != expected_w:
        return pure.na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal)

    def _run():
        kernel = _get_3d_av_kernel(ksize)
        attn_full = attn
        if step_d != 1 or step_h != 1 or step_w != 1:
            idx_d = mx.reshape(qd, (1, expected_d, 1, 1, 1, 1))
            attn_d = mx.put_along_axis(
                mx.zeros((batch, depth, expected_h, expected_w, heads, volume), dtype=attn.dtype),
                idx_d,
                attn,
                axis=1,
            )
            idx_h = mx.reshape(qh, (1, 1, expected_h, 1, 1, 1))
            attn_h = mx.put_along_axis(
                mx.zeros((batch, depth, height, expected_w, heads, volume), dtype=attn.dtype),
                idx_h,
                attn_d,
                axis=2,
            )
            idx_w = mx.reshape(qw, (1, 1, 1, expected_w, 1, 1))
            attn_full = mx.put_along_axis(
                mx.zeros((batch, depth, height, width, heads, volume), dtype=attn.dtype),
                idx_w,
                attn_h,
                axis=3,
            )
        attn_m = _to_metal_3d(_cast(attn_full, mx.float32))
        v_m = _to_metal_3d(_cast(v, mx.float32))
        dilation_param = mx.array([dil_d, dil_h, dil_w], dtype=mx.int32)
        causal_param = mx.array([causal_d, causal_h, causal_w], dtype=mx.int32)
        out = kernel(
            inputs=[attn_m, v_m, dilation_param, causal_param],
            grid=(width, height, batch * heads * depth),
            threadgroup=_threadgroup_2d(height, width),
            output_shapes=[(batch, heads, depth, height, width, v.shape[-1])],
            output_dtypes=[mx.float32],
        )[0]
        out = _from_metal_3d(out)
        if step_d != 1:
            out = mx.take(out, qd, axis=1)
        if step_h != 1:
            out = mx.take(out, qh, axis=2)
        if step_w != 1:
            out = mx.take(out, qw, axis=3)
        return _cast(out, v.dtype)

    return _with_fallback(
        _run,
        lambda: pure.na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal),
    )


def na1d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_1d_fused(kernel_size, stride, dilation):
        return pure.na1d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale)

    def _run():
        logits = na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)
        attn = mx.softmax(logits, axis=-1)
        grad_attn, grad_v = na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)
        softmax_inner = mx.sum(grad_attn * attn, axis=-1, keepdims=True)
        grad_logits = attn * (grad_attn - softmax_inner)
        grad_q, grad_k = na1d_qk_backward(
            q, k, grad_logits, kernel_size, stride, dilation, is_causal, scale
        )
        return grad_q, grad_k, grad_v

    return _with_grad_fallback(
        _run,
        lambda: pure.na1d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale),
    )


def na2d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_2d_fused(kernel_size, stride, dilation):
        return pure.na2d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale)

    def _run():
        logits = na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)
        attn = mx.softmax(logits, axis=-1)
        grad_attn, grad_v = na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)
        softmax_inner = mx.sum(grad_attn * attn, axis=-1, keepdims=True)
        grad_logits = attn * (grad_attn - softmax_inner)
        grad_q, grad_k = na2d_qk_backward(
            q, k, grad_logits, kernel_size, stride, dilation, is_causal, scale
        )
        return grad_q, grad_k, grad_v

    return _with_grad_fallback(
        _run,
        lambda: pure.na2d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale),
    )


def na3d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not (
        _supports_3d_fused(kernel_size, stride, dilation)
        or _supports_3d_split(kernel_size, stride, dilation, is_causal)
    ):
        return pure.na3d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale)

    def _run():
        logits = na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)
        attn = mx.softmax(logits, axis=-1)
        grad_attn, grad_v = na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)
        softmax_inner = mx.sum(grad_attn * attn, axis=-1, keepdims=True)
        grad_logits = attn * (grad_attn - softmax_inner)
        grad_q, grad_k = na3d_qk_backward(
            q, k, grad_logits, kernel_size, stride, dilation, is_causal, scale
        )
        return grad_q, grad_k, grad_v

    return _with_grad_fallback(
        _run,
        lambda: pure.na3d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale),
    )


def na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_1d_split(kernel_size, stride, dilation, is_causal):
        return pure.na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)

    batch, length, heads, head_dim = q.shape
    ksize = int(kernel_size[0])
    step = int(stride[0])
    dil = int(dilation[0])
    causal = bool(is_causal[0])
    qpos_np = pure._query_positions(length, step)
    out_len = len(qpos_np)
    qpos = mx.array(qpos_np, dtype=mx.int32)
    indices, valid = pure._compute_neighbor_indices_1d(
        length, ksize, dil, causal, query_positions=qpos_np
    )
    valid_mask = valid[None, :, None, :]
    scale_value = (head_dim ** -0.5) if scale is None else float(scale)

    def _run():
        q_sel = mx.take(q, qpos, axis=1)
        flat_idx = mx.reshape(indices, (-1,))
        k_neighbors = mx.take(k, flat_idx, axis=1)
        k_neighbors = mx.reshape(k_neighbors, (batch, out_len, ksize, heads, head_dim))
        k_neighbors = mx.transpose(k_neighbors, axes=(0, 1, 3, 2, 4))

        grad_attn_masked = mx.where(valid_mask, grad_attn, mx.zeros_like(grad_attn))
        grad_q_sel = scale_value * mx.sum(grad_attn_masked[..., None] * k_neighbors, axis=-2)

        bwd_kernel = _get_1d_qk_backward_k_kernel(ksize)
        grad_attn_m = _to_metal_1d(_cast(grad_attn_masked, mx.float32))
        q_m = _to_metal_1d(_cast(q, mx.float32))
        stride_param = mx.array([step], dtype=mx.int32)
        dilation_param = mx.array([dil], dtype=mx.int32)
        causal_param = mx.array([1 if causal else 0], dtype=mx.int32)
        scale_param = mx.array([scale_value], dtype=mx.float32)
        grad_k_m = bwd_kernel(
            inputs=[grad_attn_m, q_m, stride_param, dilation_param, causal_param, scale_param],
            grid=(length, 1, batch * heads),
            threadgroup=_threadgroup_1d(length),
            output_shapes=[(batch, heads, length, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        grad_k = _cast(_from_metal_1d(grad_k_m), k.dtype)

        idx = mx.reshape(qpos, (1, out_len, 1, 1))
        grad_q = mx.put_along_axis(
            mx.zeros(q.shape, dtype=q.dtype),
            idx,
            _cast(grad_q_sel, q.dtype),
            axis=1,
        )
        return grad_q, grad_k

    return _with_grad_fallback(
        _run,
        lambda: pure.na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale),
    )


def na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    if not is_available() or not _supports_1d_split(kernel_size, stride, dilation, is_causal):
        return pure.na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)

    batch, out_len, heads, ksize = attn.shape
    _, length, _, head_dim = v.shape
    step = int(stride[0])
    dil = int(dilation[0])
    causal = bool(is_causal[0])
    qpos_np = pure._query_positions(length, step)
    if out_len != len(qpos_np):
        return pure.na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)
    qpos = mx.array(qpos_np, dtype=mx.int32)
    indices, valid = pure._compute_neighbor_indices_1d(
        length, ksize, dil, causal, query_positions=qpos_np
    )
    valid_mask = valid[None, :, None, :]

    def _run():
        flat_idx = mx.reshape(indices, (-1,))
        v_neighbors = mx.take(v, flat_idx, axis=1)
        v_neighbors = mx.reshape(v_neighbors, (batch, out_len, ksize, heads, head_dim))
        v_neighbors = mx.transpose(v_neighbors, axes=(0, 1, 3, 2, 4))

        grad_attn = mx.sum(grad_out[..., None, :] * v_neighbors, axis=-1)
        grad_attn = mx.where(valid_mask, grad_attn, mx.zeros_like(grad_attn))

        attn_masked = mx.where(valid_mask, attn, mx.zeros_like(attn))
        bwd_kernel = _get_1d_av_backward_v_kernel(ksize)
        attn_m = _to_metal_1d(_cast(attn_masked, mx.float32))
        grad_out_m = _to_metal_1d(_cast(grad_out, mx.float32))
        target_shape_param = mx.array([length], dtype=mx.int32)
        stride_param = mx.array([step], dtype=mx.int32)
        dilation_param = mx.array([dil], dtype=mx.int32)
        causal_param = mx.array([1 if causal else 0], dtype=mx.int32)
        grad_v_m = bwd_kernel(
            inputs=[
                attn_m,
                grad_out_m,
                target_shape_param,
                stride_param,
                dilation_param,
                causal_param,
            ],
            grid=(length, 1, batch * heads),
            threadgroup=_threadgroup_1d(length),
            output_shapes=[(batch, heads, length, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        grad_v = _cast(_from_metal_1d(grad_v_m), v.dtype)
        return _cast(grad_attn, attn.dtype), grad_v

    return _with_grad_fallback(
        _run,
        lambda: pure.na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal),
    )


def na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_2d_split(kernel_size, stride, dilation, is_causal):
        return pure.na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)

    batch, height, width, heads, head_dim = q.shape
    kh, kw = int(kernel_size[0]), int(kernel_size[1])
    area = kh * kw
    sh, sw = int(stride[0]), int(stride[1])
    qh_np = pure._query_positions(height, sh)
    qw_np = pure._query_positions(width, sw)
    out_h = len(qh_np)
    out_w = len(qw_np)
    qh = mx.array(qh_np, dtype=mx.int32)
    qw = mx.array(qw_np, dtype=mx.int32)
    indices, valid = pure._compute_neighbor_indices_2d(
        height,
        width,
        (kh, kw),
        (int(dilation[0]), int(dilation[1])),
        (bool(is_causal[0]), bool(is_causal[1])),
        query_h=qh_np,
        query_w=qw_np,
    )
    valid_mask = valid[None, :, :, None, :]
    scale_value = (head_dim ** -0.5) if scale is None else float(scale)
    flat_spatial = height * width

    def _run():
        q_rows = mx.take(q, qh, axis=1)
        q_sel = mx.take(q_rows, qw, axis=2)

        k_flat = mx.reshape(k, (batch, flat_spatial, heads, head_dim))
        flat_idx = mx.reshape(indices, (-1,))
        k_neighbors = mx.take(k_flat, flat_idx, axis=1)
        k_neighbors = mx.reshape(k_neighbors, (batch, out_h, out_w, area, heads, head_dim))
        k_neighbors = mx.transpose(k_neighbors, axes=(0, 1, 2, 4, 3, 5))

        grad_attn_masked = mx.where(valid_mask, grad_attn, mx.zeros_like(grad_attn))
        grad_q_sel = scale_value * mx.sum(grad_attn_masked[..., None] * k_neighbors, axis=-2)

        bwd_kernel = _get_2d_qk_backward_k_kernel(kh)
        grad_attn_m = _to_metal_2d(_cast(grad_attn_masked, mx.float32))
        q_m = _to_metal_2d(_cast(q, mx.float32))
        stride_param = mx.array([sh, sw], dtype=mx.int32)
        dilation_param = mx.array([int(dilation[0]), int(dilation[1])], dtype=mx.int32)
        causal_param = mx.array(
            [1 if bool(is_causal[0]) else 0, 1 if bool(is_causal[1]) else 0],
            dtype=mx.int32,
        )
        scale_param = mx.array([scale_value], dtype=mx.float32)
        grad_k_m = bwd_kernel(
            inputs=[grad_attn_m, q_m, stride_param, dilation_param, causal_param, scale_param],
            grid=(width, height, batch * heads),
            threadgroup=_threadgroup_2d(height, width),
            output_shapes=[(batch, heads, height, width, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        grad_k = _cast(_from_metal_2d(grad_k_m), k.dtype)

        idx_h = mx.reshape(qh, (1, out_h, 1, 1, 1))
        grad_q_h = mx.put_along_axis(
            mx.zeros((batch, height, out_w, heads, head_dim), dtype=q.dtype),
            idx_h,
            _cast(grad_q_sel, q.dtype),
            axis=1,
        )
        idx_w = mx.reshape(qw, (1, 1, out_w, 1, 1))
        grad_q = mx.put_along_axis(
            mx.zeros((batch, height, width, heads, head_dim), dtype=q.dtype),
            idx_w,
            grad_q_h,
            axis=2,
        )
        return grad_q, grad_k

    return _with_grad_fallback(
        _run,
        lambda: pure.na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale),
    )


def na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    if not is_available() or not _supports_2d_split(kernel_size, stride, dilation, is_causal):
        return pure.na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)

    batch, out_h, out_w, heads, area = attn.shape
    _, height, width, _, head_dim = v.shape
    flat_spatial = height * width
    qh_np = pure._query_positions(height, int(stride[0]))
    qw_np = pure._query_positions(width, int(stride[1]))
    if out_h != len(qh_np) or out_w != len(qw_np):
        return pure.na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)
    indices, valid = pure._compute_neighbor_indices_2d(
        height,
        width,
        (int(kernel_size[0]), int(kernel_size[1])),
        (int(dilation[0]), int(dilation[1])),
        (bool(is_causal[0]), bool(is_causal[1])),
        query_h=qh_np,
        query_w=qw_np,
    )
    valid_mask = valid[None, :, :, None, :]

    def _run():
        v_flat = mx.reshape(v, (batch, flat_spatial, heads, head_dim))
        flat_idx = mx.reshape(indices, (-1,))
        v_neighbors = mx.take(v_flat, flat_idx, axis=1)
        v_neighbors = mx.reshape(v_neighbors, (batch, out_h, out_w, area, heads, head_dim))
        v_neighbors = mx.transpose(v_neighbors, axes=(0, 1, 2, 4, 3, 5))

        grad_attn = mx.sum(grad_out[..., None, :] * v_neighbors, axis=-1)
        grad_attn = mx.where(valid_mask, grad_attn, mx.zeros_like(grad_attn))

        attn_masked = mx.where(valid_mask, attn, mx.zeros_like(attn))
        bwd_kernel = _get_2d_av_backward_v_kernel(int(kernel_size[0]))
        attn_m = _to_metal_2d(_cast(attn_masked, mx.float32))
        grad_out_m = _to_metal_2d(_cast(grad_out, mx.float32))
        target_shape_param = mx.array([height, width], dtype=mx.int32)
        stride_param = mx.array([int(stride[0]), int(stride[1])], dtype=mx.int32)
        dilation_param = mx.array([int(dilation[0]), int(dilation[1])], dtype=mx.int32)
        causal_param = mx.array(
            [1 if bool(is_causal[0]) else 0, 1 if bool(is_causal[1]) else 0],
            dtype=mx.int32,
        )
        grad_v_m = bwd_kernel(
            inputs=[
                attn_m,
                grad_out_m,
                target_shape_param,
                stride_param,
                dilation_param,
                causal_param,
            ],
            grid=(width, height, batch * heads),
            threadgroup=_threadgroup_2d(height, width),
            output_shapes=[(batch, heads, height, width, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        grad_v = _cast(_from_metal_2d(grad_v_m), v.dtype)
        return _cast(grad_attn, attn.dtype), grad_v

    return _with_grad_fallback(
        _run,
        lambda: pure.na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal),
    )


def na3d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    if not is_available() or not _supports_3d_split(kernel_size, stride, dilation, is_causal):
        return pure.na3d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)

    batch, depth, height, width, heads, head_dim = q.shape
    kd, kh, kw = int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2])
    volume = kd * kh * kw
    sd, sh, sw = int(stride[0]), int(stride[1]), int(stride[2])
    qd_np = pure._query_positions(depth, sd)
    qh_np = pure._query_positions(height, sh)
    qw_np = pure._query_positions(width, sw)
    out_d = len(qd_np)
    out_h = len(qh_np)
    out_w = len(qw_np)
    qd = mx.array(qd_np, dtype=mx.int32)
    qh = mx.array(qh_np, dtype=mx.int32)
    qw = mx.array(qw_np, dtype=mx.int32)
    indices, valid = pure._compute_neighbor_indices_3d(
        depth,
        height,
        width,
        (kd, kh, kw),
        (int(dilation[0]), int(dilation[1]), int(dilation[2])),
        (bool(is_causal[0]), bool(is_causal[1]), bool(is_causal[2])),
        query_d=qd_np,
        query_h=qh_np,
        query_w=qw_np,
    )
    valid_mask = valid[None, :, :, :, None, :]
    scale_value = (head_dim ** -0.5) if scale is None else float(scale)
    flat_spatial = depth * height * width

    def _run():
        q_depth = mx.take(q, qd, axis=1)
        q_rows = mx.take(q_depth, qh, axis=2)
        q_sel = mx.take(q_rows, qw, axis=3)

        k_flat = mx.reshape(k, (batch, flat_spatial, heads, head_dim))
        flat_idx = mx.reshape(indices, (-1,))
        k_neighbors = mx.take(k_flat, flat_idx, axis=1)
        k_neighbors = mx.reshape(
            k_neighbors, (batch, out_d, out_h, out_w, volume, heads, head_dim)
        )
        k_neighbors = mx.transpose(k_neighbors, axes=(0, 1, 2, 3, 5, 4, 6))

        grad_attn_masked = mx.where(valid_mask, grad_attn, mx.zeros_like(grad_attn))
        grad_q_sel = scale_value * mx.sum(grad_attn_masked[..., None] * k_neighbors, axis=-2)

        bwd_kernel = _get_3d_qk_backward_k_kernel(kd)
        grad_attn_m = _to_metal_3d(_cast(grad_attn_masked, mx.float32))
        q_m = _to_metal_3d(_cast(q, mx.float32))
        stride_param = mx.array([sd, sh, sw], dtype=mx.int32)
        dilation_param = mx.array(
            [int(dilation[0]), int(dilation[1]), int(dilation[2])],
            dtype=mx.int32,
        )
        causal_param = mx.array(
            [
                1 if bool(is_causal[0]) else 0,
                1 if bool(is_causal[1]) else 0,
                1 if bool(is_causal[2]) else 0,
            ],
            dtype=mx.int32,
        )
        scale_param = mx.array([scale_value], dtype=mx.float32)
        grad_k_m = bwd_kernel(
            inputs=[grad_attn_m, q_m, stride_param, dilation_param, causal_param, scale_param],
            grid=(width, height, batch * heads * depth),
            threadgroup=_threadgroup_2d(height, width),
            output_shapes=[(batch, heads, depth, height, width, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        grad_k = _cast(_from_metal_3d(grad_k_m), k.dtype)

        idx_d = mx.reshape(qd, (1, out_d, 1, 1, 1, 1))
        grad_q_d = mx.put_along_axis(
            mx.zeros((batch, depth, out_h, out_w, heads, head_dim), dtype=q.dtype),
            idx_d,
            _cast(grad_q_sel, q.dtype),
            axis=1,
        )
        idx_h = mx.reshape(qh, (1, 1, out_h, 1, 1, 1))
        grad_q_h = mx.put_along_axis(
            mx.zeros((batch, depth, height, out_w, heads, head_dim), dtype=q.dtype),
            idx_h,
            grad_q_d,
            axis=2,
        )
        idx_w = mx.reshape(qw, (1, 1, 1, out_w, 1, 1))
        grad_q = mx.put_along_axis(
            mx.zeros((batch, depth, height, width, heads, head_dim), dtype=q.dtype),
            idx_w,
            grad_q_h,
            axis=3,
        )
        return grad_q, grad_k

    return _with_grad_fallback(
        _run,
        lambda: pure.na3d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale),
    )


def na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    if not is_available() or not _supports_3d_split(kernel_size, stride, dilation, is_causal):
        return pure.na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)

    batch, out_d, out_h, out_w, heads, volume = attn.shape
    _, depth, height, width, _, head_dim = v.shape
    flat_spatial = depth * height * width
    qd_np = pure._query_positions(depth, int(stride[0]))
    qh_np = pure._query_positions(height, int(stride[1]))
    qw_np = pure._query_positions(width, int(stride[2]))
    if out_d != len(qd_np) or out_h != len(qh_np) or out_w != len(qw_np):
        return pure.na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)
    indices, valid = pure._compute_neighbor_indices_3d(
        depth,
        height,
        width,
        (int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2])),
        (int(dilation[0]), int(dilation[1]), int(dilation[2])),
        (bool(is_causal[0]), bool(is_causal[1]), bool(is_causal[2])),
        query_d=qd_np,
        query_h=qh_np,
        query_w=qw_np,
    )
    valid_mask = valid[None, :, :, :, None, :]

    def _run():
        v_flat = mx.reshape(v, (batch, flat_spatial, heads, head_dim))
        flat_idx = mx.reshape(indices, (-1,))
        v_neighbors = mx.take(v_flat, flat_idx, axis=1)
        v_neighbors = mx.reshape(
            v_neighbors, (batch, out_d, out_h, out_w, volume, heads, head_dim)
        )
        v_neighbors = mx.transpose(v_neighbors, axes=(0, 1, 2, 3, 5, 4, 6))

        grad_attn = mx.sum(grad_out[..., None, :] * v_neighbors, axis=-1)
        grad_attn = mx.where(valid_mask, grad_attn, mx.zeros_like(grad_attn))

        attn_masked = mx.where(valid_mask, attn, mx.zeros_like(attn))
        bwd_kernel = _get_3d_av_backward_v_kernel(int(kernel_size[0]))
        attn_m = _to_metal_3d(_cast(attn_masked, mx.float32))
        grad_out_m = _to_metal_3d(_cast(grad_out, mx.float32))
        target_shape_param = mx.array([depth, height, width], dtype=mx.int32)
        stride_param = mx.array([int(stride[0]), int(stride[1]), int(stride[2])], dtype=mx.int32)
        dilation_param = mx.array(
            [int(dilation[0]), int(dilation[1]), int(dilation[2])],
            dtype=mx.int32,
        )
        causal_param = mx.array(
            [
                1 if bool(is_causal[0]) else 0,
                1 if bool(is_causal[1]) else 0,
                1 if bool(is_causal[2]) else 0,
            ],
            dtype=mx.int32,
        )
        grad_v_m = bwd_kernel(
            inputs=[
                attn_m,
                grad_out_m,
                target_shape_param,
                stride_param,
                dilation_param,
                causal_param,
            ],
            grid=(width, height, batch * heads * depth),
            threadgroup=_threadgroup_2d(height, width),
            output_shapes=[(batch, heads, depth, height, width, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        grad_v = _cast(_from_metal_3d(grad_v_m), v.dtype)
        return _cast(grad_attn, attn.dtype), grad_v

    return _with_grad_fallback(
        _run,
        lambda: pure.na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal),
    )
