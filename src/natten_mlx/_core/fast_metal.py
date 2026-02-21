"""Tier 1: MLX fast Metal kernel backend."""

from __future__ import annotations

from collections.abc import Callable
import os

import mlx.core as mx
import numpy as np

from . import pure
from ._metal_sources import (
    source_1d_av,
    source_1d_av_backward_attn,
    source_1d_av_backward_fused,
    source_1d_av_backward_v,
    source_1d_av_backward_v_vec4,
    source_1d_fused,
    source_1d_qk,
    source_1d_qk_backward_q,
    source_1d_qk_backward_q_vec4,
    source_1d_qk_backward_k,
    source_1d_qk_backward_k_inverse,
    source_2d_av,
    source_2d_av_backward_attn,
    source_2d_av_backward_fused,
    source_2d_av_backward_v,
    source_2d_av_backward_v_vec4,
    source_2d_fused,
    source_2d_qk,
    source_2d_qk_backward_q,
    source_2d_qk_backward_k,
    source_3d_av,
    source_3d_av_backward_attn,
    source_3d_av_backward_fused,
    source_3d_av_backward_v,
    source_3d_av_backward_v_vec4,
    source_3d_fused,
    source_3d_qk,
    source_3d_qk_backward_q,
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
_QK_BWD_K_INV_1D_KERNELS: dict[int, Callable] = {}
_QK_BWD_Q_1D_KERNELS: dict[int, Callable] = {}
_QK_BWD_Q4_1D_KERNELS: dict[int, Callable] = {}
_AV_BWD_V_1D_KERNELS: dict[int, Callable] = {}
_AV_BWD_V4_1D_KERNELS: dict[int, Callable] = {}
_AV_BWD_ATTN_1D_KERNELS: dict[int, Callable] = {}
_AV_BWD_FUSED_1D_KERNELS: dict[int, Callable] = {}
_QK_BWD_K_2D_KERNELS: dict[int, Callable] = {}
_QK_BWD_Q_2D_KERNELS: dict[int, Callable] = {}
_AV_BWD_V_2D_KERNELS: dict[int, Callable] = {}
_AV_BWD_V4_2D_KERNELS: dict[int, Callable] = {}
_AV_BWD_ATTN_2D_KERNELS: dict[int, Callable] = {}
_AV_BWD_FUSED_2D_KERNELS: dict[int, Callable] = {}
_QK_BWD_K_3D_KERNELS: dict[int, Callable] = {}
_QK_BWD_Q_3D_KERNELS: dict[int, Callable] = {}
_AV_BWD_V_3D_KERNELS: dict[int, Callable] = {}
_AV_BWD_V4_3D_KERNELS: dict[int, Callable] = {}
_AV_BWD_ATTN_3D_KERNELS: dict[int, Callable] = {}
_AV_BWD_FUSED_3D_KERNELS: dict[int, Callable] = {}
_RPB_1D_CACHE: dict[tuple[int, int], mx.array] = {}
_RPB_2D_CACHE: dict[tuple[int, int], mx.array] = {}
_RPB_3D_CACHE: dict[tuple[int, int], mx.array] = {}
_INV_MAP_1D_CACHE: dict[tuple, tuple[mx.array, mx.array, mx.array]] = {}
_INV_MAP_1D_QK_CACHE: dict[tuple, tuple[mx.array, mx.array, mx.array]] = {}
_INV_MAP_2D_CACHE: dict[tuple, tuple[mx.array, mx.array, mx.array]] = {}
_INV_MAP_3D_CACHE: dict[tuple, tuple[mx.array, mx.array, mx.array]] = {}
_USE_AV_BWD_FUSION = os.getenv("NATTEN_MLX_AV_BWD_FUSION", "").strip() == "1"


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


def _threadgroup_1d_heavy(length: int) -> tuple[int, int, int]:
    # Backward kernels are register-heavy; smaller groups generally reduce
    # register pressure and improve occupancy.
    return (min(max(length, 1), 64), 1, 1)


def _threadgroup_2d_heavy(height: int, width: int) -> tuple[int, int, int]:
    # Keep backward kernels on 8x8 groups for better occupancy on heavy loops.
    return (min(max(width, 1), 8), min(max(height, 1), 8), 1)


def _threadgroup_grad_v(dim: int, values: int) -> tuple[int, int, int]:
    # grad_v kernels map x->channel and y->value index.
    d = max(dim, 1)
    v = max(values, 1)
    if d >= 32:
        return (min(d, 32), min(v, 4), 1)
    if d >= 16:
        return (min(d, 16), min(v, 8), 1)
    return (min(d, 8), min(v, 8), 1)


def _threadgroup_grad_v_1d_vec4(dim4: int, values: int) -> tuple[int, int, int]:
    # 1D vec4 grad_v path is more stable with slightly wider x and reduced y.
    return (min(max(dim4 * 2, 1), 16), min(max(values, 1), 8), 1)


def _threadgroup_grad_v_2d_vec4(dim4: int, values: int) -> tuple[int, int, int]:
    # 2D vec4 grad_v tends to run best with narrower x and wider y.
    x = 4 if dim4 >= 4 else max(dim4, 1)
    return (x, min(max(values, 1), 32), 1)


def _threadgroup_grad_v_3d_vec4(dim4: int, values: int) -> tuple[int, int, int]:
    # 3D vec4 grad_v benefits from a wider x lane utilization.
    x = min(max(dim4 * 2, 1), 8)
    return (x, min(max(values, 1), 32), 1)


def _threadgroup_1d_qk_bwd_k_inverse(dim: int, length: int) -> tuple[int, int, int]:
    # Inverse-map grad_k uses x->channel and y->token. Wider y helps at long
    # lengths once per-thread edge loops become dominant.
    d = max(dim, 1)
    l = max(length, 1)
    if d >= 64 and l >= 1024:
        return (8, 16, 1)
    if d >= 64 and l >= 512:
        return (16, 8, 1)
    return _threadgroup_grad_v(d, l)


def _threadgroup_1d_qk_bwd_q_vec4(dim4: int, length: int) -> tuple[int, int, int]:
    # Long-sequence 1D grad_q vec4 tends to benefit from wider y-groups.
    if dim4 >= 16 and length >= 1024:
        return (16, 16, 1)
    return _threadgroup_grad_v_1d_vec4(dim4, length)


def _use_vec4_1d_qk_grad_q(length: int, head_dim: int) -> bool:
    if head_dim % 4 != 0 or head_dim < 16:
        return False
    # Vec4 grad_q is a net win across short and long decode-like lengths.
    return length > 0


def _use_vec4_1d_grad_v(length: int, head_dim: int) -> bool:
    if head_dim % 4 != 0 or head_dim < 16:
        return False
    # Empirically, (L<=128, D=32) can be slower on vec4 in end-to-end backward.
    if head_dim == 32 and length <= 128:
        return False
    return True


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


def _build_inverse_csr(
    *,
    value_indices: np.ndarray,
    out_indices: np.ndarray,
    neighbor_indices: np.ndarray,
    num_values: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if value_indices.size == 0:
        offsets = np.zeros((num_values + 1,), dtype=np.int32)
        return offsets, np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    order = np.argsort(value_indices, kind="stable")
    vals = value_indices[order].astype(np.int32, copy=False)
    out_ids = out_indices[order].astype(np.int32, copy=False)
    nbr_ids = neighbor_indices[order].astype(np.int32, copy=False)

    counts = np.bincount(vals, minlength=num_values).astype(np.int32, copy=False)
    offsets = np.zeros((num_values + 1,), dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int64).astype(np.int32, copy=False)
    return offsets, out_ids, nbr_ids


def _to_compact_index_array(indices: np.ndarray) -> mx.array:
    if indices.size == 0:
        return mx.array(indices.astype(np.uint16, copy=False), dtype=mx.uint16)
    max_index = int(indices.max())
    if max_index <= np.iinfo(np.uint16).max:
        return mx.array(indices.astype(np.uint16, copy=False), dtype=mx.uint16)
    return mx.array(indices.astype(np.int32, copy=False), dtype=mx.int32)


def _inverse_map_1d(
    length: int,
    out_len: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    causal: bool,
    dim: int,
):
    key = (length, out_len, kernel_size, stride, dilation, causal, dim)
    cached = _INV_MAP_1D_CACHE.get(key)
    if cached is not None:
        return cached

    qpos = (np.arange(out_len, dtype=np.int32) * stride).astype(np.int32)
    idx, valid = pure._compute_axis_indices(qpos, length, kernel_size, dilation, causal)
    flat_valid = valid.reshape(-1)
    value_indices = idx.reshape(-1)[flat_valid].astype(np.int32, copy=False)
    out_indices = np.repeat(np.arange(out_len, dtype=np.int32), kernel_size)[flat_valid]
    neighbor_indices = np.tile(np.arange(kernel_size, dtype=np.int32), out_len)[flat_valid]
    offsets, out_ids, nbr_ids = _build_inverse_csr(
        value_indices=value_indices,
        out_indices=out_indices,
        neighbor_indices=neighbor_indices,
        num_values=length,
    )
    attn_base = (out_ids * kernel_size + nbr_ids).astype(np.int32, copy=False)
    grad_base = (out_ids * dim).astype(np.int32, copy=False)
    result = (
        mx.array(offsets, dtype=mx.int32),
        _to_compact_index_array(attn_base),
        _to_compact_index_array(grad_base),
    )
    _INV_MAP_1D_CACHE[key] = result
    return result


def _inverse_map_1d_qk(
    length: int,
    out_len: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    causal: bool,
    dim: int,
):
    key = (length, out_len, kernel_size, stride, dilation, causal, dim)
    cached = _INV_MAP_1D_QK_CACHE.get(key)
    if cached is not None:
        return cached

    qpos = (np.arange(out_len, dtype=np.int32) * stride).astype(np.int32)
    idx, valid = pure._compute_axis_indices(qpos, length, kernel_size, dilation, causal)
    flat_valid = valid.reshape(-1)
    value_indices = idx.reshape(-1)[flat_valid].astype(np.int32, copy=False)
    out_indices = np.repeat(np.arange(out_len, dtype=np.int32), kernel_size)[flat_valid]
    neighbor_indices = np.tile(np.arange(kernel_size, dtype=np.int32), out_len)[flat_valid]
    offsets, out_ids, nbr_ids = _build_inverse_csr(
        value_indices=value_indices,
        out_indices=out_indices,
        neighbor_indices=neighbor_indices,
        num_values=length,
    )
    attn_base = (out_ids * kernel_size + nbr_ids).astype(np.int32, copy=False)
    query_base = (out_ids * stride * dim).astype(np.int32, copy=False)
    result = (
        mx.array(offsets, dtype=mx.int32),
        _to_compact_index_array(attn_base),
        _to_compact_index_array(query_base),
    )
    _INV_MAP_1D_QK_CACHE[key] = result
    return result


def _inverse_map_2d(
    height: int,
    width: int,
    out_h: int,
    out_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int,
    dilation_w: int,
    causal_h: bool,
    causal_w: bool,
    dim: int,
):
    key = (
        height,
        width,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        causal_h,
        causal_w,
        dim,
    )
    cached = _INV_MAP_2D_CACHE.get(key)
    if cached is not None:
        return cached

    qh = (np.arange(out_h, dtype=np.int32) * stride_h).astype(np.int32)
    qw = (np.arange(out_w, dtype=np.int32) * stride_w).astype(np.int32)
    h_idx, h_valid = pure._compute_axis_indices(qh, height, kernel_h, dilation_h, causal_h)
    w_idx, w_valid = pure._compute_axis_indices(qw, width, kernel_w, dilation_w, causal_w)

    k_area = kernel_h * kernel_w
    lin = (
        h_idx[:, None, :, None].astype(np.int32) * width
        + w_idx[None, :, None, :].astype(np.int32)
    ).reshape(out_h, out_w, k_area)
    valid = (h_valid[:, None, :, None] & w_valid[None, :, None, :]).reshape(out_h, out_w, k_area)

    out_flat = np.arange(out_h * out_w, dtype=np.int32).reshape(out_h, out_w, 1)
    out_indices = np.broadcast_to(out_flat, (out_h, out_w, k_area)).reshape(-1)
    neighbor_indices = np.broadcast_to(np.arange(k_area, dtype=np.int32), (out_h, out_w, k_area)).reshape(-1)
    mask = valid.reshape(-1)
    offsets, out_ids, nbr_ids = _build_inverse_csr(
        value_indices=lin.reshape(-1)[mask].astype(np.int32, copy=False),
        out_indices=out_indices[mask],
        neighbor_indices=neighbor_indices[mask],
        num_values=height * width,
    )
    attn_base = (out_ids * k_area + nbr_ids).astype(np.int32, copy=False)
    grad_base = (out_ids * dim).astype(np.int32, copy=False)
    result = (
        mx.array(offsets, dtype=mx.int32),
        _to_compact_index_array(attn_base),
        _to_compact_index_array(grad_base),
    )
    _INV_MAP_2D_CACHE[key] = result
    return result


def _inverse_map_3d(
    depth: int,
    height: int,
    width: int,
    out_d: int,
    out_h: int,
    out_w: int,
    kernel_d: int,
    kernel_h: int,
    kernel_w: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    dilation_d: int,
    dilation_h: int,
    dilation_w: int,
    causal_d: bool,
    causal_h: bool,
    causal_w: bool,
    dim: int,
):
    key = (
        depth,
        height,
        width,
        out_d,
        out_h,
        out_w,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        dilation_d,
        dilation_h,
        dilation_w,
        causal_d,
        causal_h,
        causal_w,
        dim,
    )
    cached = _INV_MAP_3D_CACHE.get(key)
    if cached is not None:
        return cached

    qd = (np.arange(out_d, dtype=np.int32) * stride_d).astype(np.int32)
    qh = (np.arange(out_h, dtype=np.int32) * stride_h).astype(np.int32)
    qw = (np.arange(out_w, dtype=np.int32) * stride_w).astype(np.int32)

    d_idx, d_valid = pure._compute_axis_indices(qd, depth, kernel_d, dilation_d, causal_d)
    h_idx, h_valid = pure._compute_axis_indices(qh, height, kernel_h, dilation_h, causal_h)
    w_idx, w_valid = pure._compute_axis_indices(qw, width, kernel_w, dilation_w, causal_w)

    volume = kernel_d * kernel_h * kernel_w
    lin = (
        (
            d_idx[:, None, None, :, None, None].astype(np.int32) * height
            + h_idx[None, :, None, None, :, None].astype(np.int32)
        )
        * width
        + w_idx[None, None, :, None, None, :].astype(np.int32)
    ).reshape(out_d, out_h, out_w, volume)
    valid = (
        d_valid[:, None, None, :, None, None]
        & h_valid[None, :, None, None, :, None]
        & w_valid[None, None, :, None, None, :]
    ).reshape(out_d, out_h, out_w, volume)

    out_flat = np.arange(out_d * out_h * out_w, dtype=np.int32).reshape(out_d, out_h, out_w, 1)
    out_indices = np.broadcast_to(out_flat, (out_d, out_h, out_w, volume)).reshape(-1)
    neighbor_indices = np.broadcast_to(
        np.arange(volume, dtype=np.int32), (out_d, out_h, out_w, volume)
    ).reshape(-1)
    mask = valid.reshape(-1)
    offsets, out_ids, nbr_ids = _build_inverse_csr(
        value_indices=lin.reshape(-1)[mask].astype(np.int32, copy=False),
        out_indices=out_indices[mask],
        neighbor_indices=neighbor_indices[mask],
        num_values=depth * height * width,
    )
    attn_base = (out_ids * volume + nbr_ids).astype(np.int32, copy=False)
    grad_base = (out_ids * dim).astype(np.int32, copy=False)
    result = (
        mx.array(offsets, dtype=mx.int32),
        _to_compact_index_array(attn_base),
        _to_compact_index_array(grad_base),
    )
    _INV_MAP_3D_CACHE[key] = result
    return result


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


def _get_1d_qk_backward_k_inverse_kernel(kernel_size: int):
    if kernel_size not in _QK_BWD_K_INV_1D_KERNELS:
        _QK_BWD_K_INV_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_qk_bwd_k_inv_k{kernel_size}",
            input_names=[
                "grad_attn",
                "query",
                "inv_offsets",
                "inv_attn_base",
                "inv_query_base",
                "scale_param",
            ],
            output_names=["out"],
            source=source_1d_qk_backward_k_inverse(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_BWD_K_INV_1D_KERNELS[kernel_size]


def _get_1d_qk_backward_q_kernel(kernel_size: int):
    if kernel_size not in _QK_BWD_Q_1D_KERNELS:
        _QK_BWD_Q_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_qk_bwd_q_k{kernel_size}",
            input_names=[
                "grad_attn",
                "key",
                "stride_param",
                "dilation_param",
                "causal_param",
                "scale_param",
            ],
            output_names=["out"],
            source=source_1d_qk_backward_q(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_BWD_Q_1D_KERNELS[kernel_size]


def _get_1d_qk_backward_q_vec4_kernel(kernel_size: int):
    if kernel_size not in _QK_BWD_Q4_1D_KERNELS:
        _QK_BWD_Q4_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_qk_bwd_q4_k{kernel_size}",
            input_names=[
                "grad_attn",
                "key",
                "stride_param",
                "dilation_param",
                "causal_param",
                "scale_param",
            ],
            output_names=["out"],
            source=source_1d_qk_backward_q_vec4(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_BWD_Q4_1D_KERNELS[kernel_size]


def _get_1d_av_backward_v_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_V_1D_KERNELS:
        _AV_BWD_V_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_av_bwd_v_k{kernel_size}",
            input_names=[
                "attention_probs",
                "grad_out",
                "target_shape_param",
                "inv_offsets",
                "inv_attn_base",
                "inv_grad_base",
            ],
            output_names=["out"],
            source=source_1d_av_backward_v(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_V_1D_KERNELS[kernel_size]


def _get_1d_av_backward_v_vec4_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_V4_1D_KERNELS:
        _AV_BWD_V4_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_av_bwd_v4_k{kernel_size}",
            input_names=[
                "attention_probs",
                "grad_out",
                "target_shape_param",
                "inv_offsets",
                "inv_attn_base",
                "inv_grad_base",
            ],
            output_names=["out"],
            source=source_1d_av_backward_v_vec4(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_V4_1D_KERNELS[kernel_size]


def _get_1d_av_backward_attn_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_ATTN_1D_KERNELS:
        _AV_BWD_ATTN_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_av_bwd_attn_k{kernel_size}",
            input_names=[
                "grad_out",
                "value",
                "target_shape_param",
                "stride_param",
                "dilation_param",
                "causal_param",
            ],
            output_names=["out"],
            source=source_1d_av_backward_attn(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_ATTN_1D_KERNELS[kernel_size]


def _get_1d_av_backward_fused_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_FUSED_1D_KERNELS:
        _AV_BWD_FUSED_1D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na1d_av_bwd_fused_k{kernel_size}",
            input_names=[
                "attention_probs",
                "value",
                "grad_out",
                "target_shape_param",
                "stride_param",
                "dilation_param",
                "causal_param",
            ],
            output_names=["grad_attn", "grad_v"],
            source=source_1d_av_backward_fused(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_FUSED_1D_KERNELS[kernel_size]


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


def _get_2d_qk_backward_q_kernel(kernel_size: int):
    if kernel_size not in _QK_BWD_Q_2D_KERNELS:
        _QK_BWD_Q_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_qk_bwd_q_k{kernel_size}",
            input_names=[
                "grad_attn",
                "key",
                "stride_param",
                "dilation_param",
                "causal_param",
                "scale_param",
            ],
            output_names=["out"],
            source=source_2d_qk_backward_q(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_BWD_Q_2D_KERNELS[kernel_size]


def _get_2d_av_backward_v_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_V_2D_KERNELS:
        _AV_BWD_V_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_av_bwd_v_k{kernel_size}",
            input_names=[
                "attention_probs",
                "grad_out",
                "target_shape_param",
                "inv_offsets",
                "inv_attn_base",
                "inv_grad_base",
            ],
            output_names=["out"],
            source=source_2d_av_backward_v(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_V_2D_KERNELS[kernel_size]


def _get_2d_av_backward_v_vec4_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_V4_2D_KERNELS:
        _AV_BWD_V4_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_av_bwd_v4_k{kernel_size}",
            input_names=[
                "attention_probs",
                "grad_out",
                "target_shape_param",
                "inv_offsets",
                "inv_attn_base",
                "inv_grad_base",
            ],
            output_names=["out"],
            source=source_2d_av_backward_v_vec4(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_V4_2D_KERNELS[kernel_size]


def _get_2d_av_backward_attn_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_ATTN_2D_KERNELS:
        _AV_BWD_ATTN_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_av_bwd_attn_k{kernel_size}",
            input_names=[
                "grad_out",
                "value",
                "target_shape_param",
                "stride_param",
                "dilation_param",
                "causal_param",
            ],
            output_names=["out"],
            source=source_2d_av_backward_attn(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_ATTN_2D_KERNELS[kernel_size]


def _get_2d_av_backward_fused_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_FUSED_2D_KERNELS:
        _AV_BWD_FUSED_2D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na2d_av_bwd_fused_k{kernel_size}",
            input_names=[
                "attention_probs",
                "value",
                "grad_out",
                "target_shape_param",
                "stride_param",
                "dilation_param",
                "causal_param",
            ],
            output_names=["grad_attn", "grad_v"],
            source=source_2d_av_backward_fused(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_FUSED_2D_KERNELS[kernel_size]


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


def _get_3d_qk_backward_q_kernel(kernel_size: int):
    if kernel_size not in _QK_BWD_Q_3D_KERNELS:
        _QK_BWD_Q_3D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na3d_qk_bwd_q_k{kernel_size}",
            input_names=[
                "grad_attn",
                "key",
                "stride_param",
                "dilation_param",
                "causal_param",
                "scale_param",
            ],
            output_names=["out"],
            source=source_3d_qk_backward_q(kernel_size),
            ensure_row_contiguous=True,
        )
    return _QK_BWD_Q_3D_KERNELS[kernel_size]


def _get_3d_av_backward_v_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_V_3D_KERNELS:
        _AV_BWD_V_3D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na3d_av_bwd_v_k{kernel_size}",
            input_names=[
                "attention_probs",
                "grad_out",
                "target_shape_param",
                "inv_offsets",
                "inv_attn_base",
                "inv_grad_base",
            ],
            output_names=["out"],
            source=source_3d_av_backward_v(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_V_3D_KERNELS[kernel_size]


def _get_3d_av_backward_v_vec4_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_V4_3D_KERNELS:
        _AV_BWD_V4_3D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na3d_av_bwd_v4_k{kernel_size}",
            input_names=[
                "attention_probs",
                "grad_out",
                "target_shape_param",
                "inv_offsets",
                "inv_attn_base",
                "inv_grad_base",
            ],
            output_names=["out"],
            source=source_3d_av_backward_v_vec4(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_V4_3D_KERNELS[kernel_size]


def _get_3d_av_backward_attn_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_ATTN_3D_KERNELS:
        _AV_BWD_ATTN_3D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na3d_av_bwd_attn_k{kernel_size}",
            input_names=[
                "grad_out",
                "value",
                "target_shape_param",
                "stride_param",
                "dilation_param",
                "causal_param",
            ],
            output_names=["out"],
            source=source_3d_av_backward_attn(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_ATTN_3D_KERNELS[kernel_size]


def _get_3d_av_backward_fused_kernel(kernel_size: int):
    if kernel_size not in _AV_BWD_FUSED_3D_KERNELS:
        _AV_BWD_FUSED_3D_KERNELS[kernel_size] = mx.fast.metal_kernel(
            name=f"natten_mlx_na3d_av_bwd_fused_k{kernel_size}",
            input_names=[
                "attention_probs",
                "value",
                "grad_out",
                "target_shape_param",
                "stride_param",
                "dilation_param",
                "causal_param",
            ],
            output_names=["grad_attn", "grad_v"],
            source=source_3d_av_backward_fused(kernel_size),
            ensure_row_contiguous=True,
        )
    return _AV_BWD_FUSED_3D_KERNELS[kernel_size]


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
    out_len = _ceil_div(length, step)
    if int(grad_attn.shape[1]) != out_len:
        return pure.na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)
    scale_value = (head_dim ** -0.5) if scale is None else float(scale)

    def _run():
        grad_attn_m = _to_metal_1d(_cast(grad_attn, mx.float32))
        k_m = _to_metal_1d(_cast(k, mx.float32))
        q_m = _to_metal_1d(_cast(q, mx.float32))
        stride_param = mx.array([step], dtype=mx.int32)
        dilation_param = mx.array([dil], dtype=mx.int32)
        causal_param = mx.array([1 if causal else 0], dtype=mx.int32)
        scale_param = mx.array([scale_value], dtype=mx.float32)
        use_vec4_q = _use_vec4_1d_qk_grad_q(length, head_dim)
        grad_q_kernel = (
            _get_1d_qk_backward_q_vec4_kernel(ksize)
            if use_vec4_q
            else _get_1d_qk_backward_q_kernel(ksize)
        )
        grad_q_x = head_dim // 4 if use_vec4_q else length
        grad_q_y = length if use_vec4_q else 1
        grad_q_tg = (
            _threadgroup_1d_qk_bwd_q_vec4(grad_q_x, length)
            if use_vec4_q
            else _threadgroup_1d_heavy(length)
        )
        grad_q_m = grad_q_kernel(
            inputs=[grad_attn_m, k_m, stride_param, dilation_param, causal_param, scale_param],
            grid=(grad_q_x, grad_q_y, batch * heads),
            threadgroup=grad_q_tg,
            output_shapes=[(batch, heads, length, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        inv_offsets, inv_attn_base, inv_query_base = _inverse_map_1d_qk(
            length=length,
            out_len=out_len,
            kernel_size=ksize,
            stride=step,
            dilation=dil,
            causal=causal,
            dim=head_dim,
        )
        grad_k_kernel = _get_1d_qk_backward_k_inverse_kernel(ksize)
        grad_k_m = grad_k_kernel(
            inputs=[
                grad_attn_m,
                q_m,
                inv_offsets,
                inv_attn_base,
                inv_query_base,
                scale_param,
            ],
            grid=(head_dim, length, batch * heads),
            threadgroup=_threadgroup_1d_qk_bwd_k_inverse(head_dim, length),
            output_shapes=[(batch, heads, length, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        grad_q = _cast(_from_metal_1d(grad_q_m), q.dtype)
        grad_k = _cast(_from_metal_1d(grad_k_m), k.dtype)
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
    if out_len != _ceil_div(length, step):
        return pure.na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)

    def _run():
        v_m = _to_metal_1d(_cast(v, mx.float32))
        grad_out_m = _to_metal_1d(_cast(grad_out, mx.float32))
        target_shape_param = mx.array([length], dtype=mx.int32)
        stride_param = mx.array([step], dtype=mx.int32)
        dilation_param = mx.array([dil], dtype=mx.int32)
        causal_param = mx.array([1 if causal else 0], dtype=mx.int32)
        if _USE_AV_BWD_FUSION:
            attn_m = _to_metal_1d(_cast(attn, mx.float32))
            fused_kernel = _get_1d_av_backward_fused_kernel(ksize)
            max_len = max(length, out_len)
            grad_attn_m, grad_v_m = fused_kernel(
                inputs=[
                    attn_m,
                    v_m,
                    grad_out_m,
                    target_shape_param,
                    stride_param,
                    dilation_param,
                    causal_param,
                ],
                grid=(max_len, 1, batch * heads),
                threadgroup=_threadgroup_1d_heavy(max_len),
                output_shapes=[(batch, heads, out_len, ksize), (batch, heads, length, head_dim)],
                output_dtypes=[mx.float32, mx.float32],
            )
        else:
            grad_attn_kernel = _get_1d_av_backward_attn_kernel(ksize)
            grad_attn_m = grad_attn_kernel(
                inputs=[
                    grad_out_m,
                    v_m,
                    target_shape_param,
                    stride_param,
                    dilation_param,
                    causal_param,
                ],
                grid=(out_len, 1, batch * heads),
                threadgroup=_threadgroup_1d_heavy(out_len),
                output_shapes=[(batch, heads, out_len, ksize)],
                output_dtypes=[mx.float32],
            )[0]
            attn_m = _to_metal_1d(_cast(attn, mx.float32))
            inv_offsets, inv_attn_base, inv_grad_base = _inverse_map_1d(
                length=length,
                out_len=out_len,
                kernel_size=ksize,
                stride=step,
                dilation=dil,
                causal=causal,
                dim=head_dim,
            )
            use_vec4 = _use_vec4_1d_grad_v(length, head_dim)
            grad_v_kernel = (
                _get_1d_av_backward_v_vec4_kernel(ksize)
                if use_vec4
                else _get_1d_av_backward_v_kernel(ksize)
            )
            grad_v_x = head_dim // 4 if use_vec4 else head_dim
            grad_v_tg = (
                _threadgroup_grad_v_1d_vec4(grad_v_x, length)
                if use_vec4
                else _threadgroup_grad_v(grad_v_x, length)
            )
            grad_v_m = grad_v_kernel(
                inputs=[
                    attn_m,
                    grad_out_m,
                    target_shape_param,
                    inv_offsets,
                    inv_attn_base,
                    inv_grad_base,
                ],
                grid=(grad_v_x, length, batch * heads),
                threadgroup=grad_v_tg,
                output_shapes=[(batch, heads, length, head_dim)],
                output_dtypes=[mx.float32],
            )[0]
        grad_attn = _cast(_from_metal_1d(grad_attn_m), attn.dtype)
        grad_v = _cast(_from_metal_1d(grad_v_m), v.dtype)
        return grad_attn, grad_v

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
    out_h = _ceil_div(height, sh)
    out_w = _ceil_div(width, sw)
    if int(grad_attn.shape[1]) != out_h or int(grad_attn.shape[2]) != out_w:
        return pure.na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)
    if int(grad_attn.shape[4]) != area:
        return pure.na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)
    scale_value = (head_dim ** -0.5) if scale is None else float(scale)

    def _run():
        grad_attn_m = _to_metal_2d(_cast(grad_attn, mx.float32))
        q_m = _to_metal_2d(_cast(q, mx.float32))
        k_m = _to_metal_2d(_cast(k, mx.float32))
        stride_param = mx.array([sh, sw], dtype=mx.int32)
        dilation_param = mx.array([int(dilation[0]), int(dilation[1])], dtype=mx.int32)
        causal_param = mx.array(
            [1 if bool(is_causal[0]) else 0, 1 if bool(is_causal[1]) else 0],
            dtype=mx.int32,
        )
        scale_param = mx.array([scale_value], dtype=mx.float32)
        grad_q_kernel = _get_2d_qk_backward_q_kernel(kh)
        grad_q_m = grad_q_kernel(
            inputs=[grad_attn_m, k_m, stride_param, dilation_param, causal_param, scale_param],
            grid=(width, height, batch * heads),
            threadgroup=_threadgroup_2d_heavy(height, width),
            output_shapes=[(batch, heads, height, width, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        grad_k_kernel = _get_2d_qk_backward_k_kernel(kh)
        grad_k_m = grad_k_kernel(
            inputs=[grad_attn_m, q_m, stride_param, dilation_param, causal_param, scale_param],
            grid=(width, height, batch * heads),
            threadgroup=_threadgroup_2d_heavy(height, width),
            output_shapes=[(batch, heads, height, width, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        grad_q = _cast(_from_metal_2d(grad_q_m), q.dtype)
        grad_k = _cast(_from_metal_2d(grad_k_m), k.dtype)
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
    if out_h != _ceil_div(height, int(stride[0])) or out_w != _ceil_div(width, int(stride[1])):
        return pure.na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)
    if int(grad_out.shape[1]) != out_h or int(grad_out.shape[2]) != out_w:
        return pure.na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)
    if int(attn.shape[4]) != area:
        return pure.na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)

    def _run():
        v_m = _to_metal_2d(_cast(v, mx.float32))
        grad_out_m = _to_metal_2d(_cast(grad_out, mx.float32))
        target_shape_param = mx.array([height, width], dtype=mx.int32)
        stride_param = mx.array([int(stride[0]), int(stride[1])], dtype=mx.int32)
        dilation_param = mx.array([int(dilation[0]), int(dilation[1])], dtype=mx.int32)
        causal_param = mx.array(
            [1 if bool(is_causal[0]) else 0, 1 if bool(is_causal[1]) else 0],
            dtype=mx.int32,
        )
        if _USE_AV_BWD_FUSION:
            attn_m = _to_metal_2d(_cast(attn, mx.float32))
            fused_kernel = _get_2d_av_backward_fused_kernel(int(kernel_size[0]))
            max_h = max(height, out_h)
            max_w = max(width, out_w)
            grad_attn_m, grad_v_m = fused_kernel(
                inputs=[
                    attn_m,
                    v_m,
                    grad_out_m,
                    target_shape_param,
                    stride_param,
                    dilation_param,
                    causal_param,
                ],
                grid=(max_w, max_h, batch * heads),
                threadgroup=_threadgroup_2d_heavy(max_h, max_w),
                output_shapes=[
                    (batch, heads, out_h, out_w, area),
                    (batch, heads, height, width, head_dim),
                ],
                output_dtypes=[mx.float32, mx.float32],
            )
        else:
            grad_attn_kernel = _get_2d_av_backward_attn_kernel(int(kernel_size[0]))
            grad_attn_m = grad_attn_kernel(
                inputs=[
                    grad_out_m,
                    v_m,
                    target_shape_param,
                    stride_param,
                    dilation_param,
                    causal_param,
                ],
                grid=(out_w, out_h, batch * heads),
                threadgroup=_threadgroup_2d_heavy(out_h, out_w),
                output_shapes=[(batch, heads, out_h, out_w, area)],
                output_dtypes=[mx.float32],
            )[0]
            attn_m = _to_metal_2d(_cast(attn, mx.float32))
            inv_offsets, inv_attn_base, inv_grad_base = _inverse_map_2d(
                height=height,
                width=width,
                out_h=out_h,
                out_w=out_w,
                kernel_h=int(kernel_size[0]),
                kernel_w=int(kernel_size[1]),
                stride_h=int(stride[0]),
                stride_w=int(stride[1]),
                dilation_h=int(dilation[0]),
                dilation_w=int(dilation[1]),
                causal_h=bool(is_causal[0]),
                causal_w=bool(is_causal[1]),
                dim=head_dim,
            )
            use_vec4 = (head_dim % 4 == 0) and (head_dim >= 16)
            grad_v_kernel = (
                _get_2d_av_backward_v_vec4_kernel(int(kernel_size[0]))
                if use_vec4
                else _get_2d_av_backward_v_kernel(int(kernel_size[0]))
            )
            grad_v_x = head_dim // 4 if use_vec4 else head_dim
            grad_v_tg = (
                _threadgroup_grad_v_2d_vec4(grad_v_x, height * width)
                if use_vec4
                else _threadgroup_grad_v(grad_v_x, height * width)
            )
            grad_v_m = grad_v_kernel(
                inputs=[
                    attn_m,
                    grad_out_m,
                    target_shape_param,
                    inv_offsets,
                    inv_attn_base,
                    inv_grad_base,
                ],
                grid=(grad_v_x, height * width, batch * heads),
                threadgroup=grad_v_tg,
                output_shapes=[(batch, heads, height, width, head_dim)],
                output_dtypes=[mx.float32],
            )[0]
        grad_attn = _cast(_from_metal_2d(grad_attn_m), attn.dtype)
        grad_v = _cast(_from_metal_2d(grad_v_m), v.dtype)
        return grad_attn, grad_v

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
    out_d = _ceil_div(depth, sd)
    out_h = _ceil_div(height, sh)
    out_w = _ceil_div(width, sw)
    if (
        int(grad_attn.shape[1]) != out_d
        or int(grad_attn.shape[2]) != out_h
        or int(grad_attn.shape[3]) != out_w
    ):
        return pure.na3d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)
    if int(grad_attn.shape[5]) != volume:
        return pure.na3d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale)
    scale_value = (head_dim ** -0.5) if scale is None else float(scale)

    def _run():
        grad_attn_m = _to_metal_3d(_cast(grad_attn, mx.float32))
        q_m = _to_metal_3d(_cast(q, mx.float32))
        k_m = _to_metal_3d(_cast(k, mx.float32))
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
        grad_q_kernel = _get_3d_qk_backward_q_kernel(kd)
        grad_q_m = grad_q_kernel(
            inputs=[grad_attn_m, k_m, stride_param, dilation_param, causal_param, scale_param],
            grid=(width, height, batch * heads * depth),
            threadgroup=_threadgroup_2d_heavy(height, width),
            output_shapes=[(batch, heads, depth, height, width, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        grad_k_kernel = _get_3d_qk_backward_k_kernel(kd)
        grad_k_m = grad_k_kernel(
            inputs=[grad_attn_m, q_m, stride_param, dilation_param, causal_param, scale_param],
            grid=(width, height, batch * heads * depth),
            threadgroup=_threadgroup_2d_heavy(height, width),
            output_shapes=[(batch, heads, depth, height, width, head_dim)],
            output_dtypes=[mx.float32],
        )[0]
        grad_q = _cast(_from_metal_3d(grad_q_m), q.dtype)
        grad_k = _cast(_from_metal_3d(grad_k_m), k.dtype)
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
    if (
        out_d != _ceil_div(depth, int(stride[0]))
        or out_h != _ceil_div(height, int(stride[1]))
        or out_w != _ceil_div(width, int(stride[2]))
    ):
        return pure.na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)
    if (
        int(grad_out.shape[1]) != out_d
        or int(grad_out.shape[2]) != out_h
        or int(grad_out.shape[3]) != out_w
    ):
        return pure.na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal)

    def _run():
        v_m = _to_metal_3d(_cast(v, mx.float32))
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
        if _USE_AV_BWD_FUSION:
            attn_m = _to_metal_3d(_cast(attn, mx.float32))
            fused_kernel = _get_3d_av_backward_fused_kernel(int(kernel_size[0]))
            max_d = max(depth, out_d)
            max_h = max(height, out_h)
            max_w = max(width, out_w)
            grad_attn_m, grad_v_m = fused_kernel(
                inputs=[
                    attn_m,
                    v_m,
                    grad_out_m,
                    target_shape_param,
                    stride_param,
                    dilation_param,
                    causal_param,
                ],
                grid=(max_w, max_h, batch * heads * max_d),
                threadgroup=_threadgroup_2d_heavy(max_h, max_w),
                output_shapes=[
                    (batch, heads, out_d, out_h, out_w, volume),
                    (batch, heads, depth, height, width, head_dim),
                ],
                output_dtypes=[mx.float32, mx.float32],
            )
        else:
            grad_attn_kernel = _get_3d_av_backward_attn_kernel(int(kernel_size[0]))
            grad_attn_m = grad_attn_kernel(
                inputs=[
                    grad_out_m,
                    v_m,
                    target_shape_param,
                    stride_param,
                    dilation_param,
                    causal_param,
                ],
                grid=(out_w, out_h, batch * heads * out_d),
                threadgroup=_threadgroup_2d_heavy(out_h, out_w),
                output_shapes=[(batch, heads, out_d, out_h, out_w, volume)],
                output_dtypes=[mx.float32],
            )[0]
            attn_m = _to_metal_3d(_cast(attn, mx.float32))
            inv_offsets, inv_attn_base, inv_grad_base = _inverse_map_3d(
                depth=depth,
                height=height,
                width=width,
                out_d=out_d,
                out_h=out_h,
                out_w=out_w,
                kernel_d=int(kernel_size[0]),
                kernel_h=int(kernel_size[1]),
                kernel_w=int(kernel_size[2]),
                stride_d=int(stride[0]),
                stride_h=int(stride[1]),
                stride_w=int(stride[2]),
                dilation_d=int(dilation[0]),
                dilation_h=int(dilation[1]),
                dilation_w=int(dilation[2]),
                causal_d=bool(is_causal[0]),
                causal_h=bool(is_causal[1]),
                causal_w=bool(is_causal[2]),
                dim=head_dim,
            )
            use_vec4 = (head_dim % 4 == 0) and (head_dim >= 16)
            grad_v_kernel = (
                _get_3d_av_backward_v_vec4_kernel(int(kernel_size[0]))
                if use_vec4
                else _get_3d_av_backward_v_kernel(int(kernel_size[0]))
            )
            grad_v_x = head_dim // 4 if use_vec4 else head_dim
            grad_v_tg = (
                _threadgroup_grad_v_3d_vec4(grad_v_x, depth * height * width)
                if use_vec4
                else _threadgroup_grad_v(grad_v_x, depth * height * width)
            )
            grad_v_m = grad_v_kernel(
                inputs=[
                    attn_m,
                    grad_out_m,
                    target_shape_param,
                    inv_offsets,
                    inv_attn_base,
                    inv_grad_base,
                ],
                grid=(grad_v_x, depth * height * width, batch * heads),
                threadgroup=grad_v_tg,
                output_shapes=[(batch, heads, depth, height, width, head_dim)],
                output_dtypes=[mx.float32],
            )[0]
        grad_attn = _cast(_from_metal_3d(grad_attn_m), attn.dtype)
        grad_v = _cast(_from_metal_3d(grad_v_m), v.dtype)
        return grad_attn, grad_v

    return _with_grad_fallback(
        _run,
        lambda: pure.na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal),
    )
