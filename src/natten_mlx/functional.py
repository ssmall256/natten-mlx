"""Public functional API for natten-mlx."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import mlx.core as mx

from natten_mlx.autograd import (
    na1d_av_with_grad,
    na1d_qk_with_grad,
    na1d_with_grad,
    na2d_av_with_grad,
    na2d_qk_with_grad,
    na2d_with_grad,
)
from natten_mlx.utils.params import (
    check_dilation_kernel_vs_input,
    check_kernel_size_vs_input,
    check_stride_vs_kernel,
    normalize_kernel_size,
    normalize_tuple_param,
)


def _validate_1d_qkv(query: mx.array, key: mx.array, value: mx.array) -> tuple[int, int, int, int]:
    if query.ndim != 4:
        raise ValueError(f"query must be 4D [B, L, H, D], got shape {query.shape}")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError(
            "query, key, and value must have the same shape for 1D attention; "
            f"got {query.shape}, {key.shape}, {value.shape}"
        )
    return tuple(query.shape)


def _validate_2d_qkv(query: mx.array, key: mx.array, value: mx.array) -> tuple[int, int, int, int, int]:
    if query.ndim != 5:
        raise ValueError(f"query must be 5D [B, H, W, heads, head_dim], got {query.shape}")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError(
            "query, key, and value must have the same shape for 2D attention; "
            f"got {query.shape}, {key.shape}, {value.shape}"
        )
    return tuple(query.shape)


def na1d(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    kernel_size: Union[int, Tuple[int]],
    stride: Union[int, Tuple[int]] = 1,
    dilation: Union[int, Tuple[int]] = 1,
    is_causal: Union[bool, Tuple[bool]] = False,
    scale: Optional[float] = None,
) -> mx.array:
    """1D neighborhood attention.

    Layout: [batch, seqlen, heads, head_dim].
    """
    _, seqlen, _, _ = _validate_1d_qkv(query, key, value)

    ks = normalize_kernel_size(kernel_size, 1)
    st = normalize_tuple_param(stride, 1, "stride")
    dil = normalize_tuple_param(dilation, 1, "dilation")
    caus = normalize_tuple_param(is_causal, 1, "is_causal")

    check_kernel_size_vs_input(ks, (seqlen,))
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, (seqlen,))

    return na1d_with_grad(query, key, value, ks, st, dil, caus, scale)


def na2d(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    is_causal: Union[bool, Tuple[bool, bool]] = False,
    scale: Optional[float] = None,
) -> mx.array:
    """2D neighborhood attention.

    Layout: [batch, height, width, heads, head_dim].
    """
    _, height, width, _, _ = _validate_2d_qkv(query, key, value)

    ks = normalize_kernel_size(kernel_size, 2)
    st = normalize_tuple_param(stride, 2, "stride")
    dil = normalize_tuple_param(dilation, 2, "dilation")
    caus = normalize_tuple_param(is_causal, 2, "is_causal")

    check_kernel_size_vs_input(ks, (height, width))
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, (height, width))

    return na2d_with_grad(query, key, value, ks, st, dil, caus, scale)


def na1d_qk(
    query: mx.array,
    key: mx.array,
    kernel_size,
    dilation=1,
    *,
    stride=1,
    is_causal=False,
    scale: Optional[float] = None,
) -> mx.array:
    """Separate 1D query-key logits. Returns [B, ceil(L/stride), H, K]."""
    _, seqlen, _, _ = _validate_1d_qkv(query, key, key)
    ks = normalize_kernel_size(kernel_size, 1)
    st = normalize_tuple_param(stride, 1, "stride")
    dil = normalize_tuple_param(dilation, 1, "dilation")
    caus = normalize_tuple_param(is_causal, 1, "is_causal")

    check_kernel_size_vs_input(ks, (seqlen,))
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, (seqlen,))

    return na1d_qk_with_grad(query, key, ks, st, dil, caus, scale)


def na1d_av(
    attn: mx.array,
    value: mx.array,
    kernel_size,
    dilation=1,
    *,
    stride=1,
    is_causal=False,
) -> mx.array:
    """Separate 1D attention-value op. attn: [B, ceil(L/stride), H, K]."""
    if attn.ndim != 4 or value.ndim != 4:
        raise ValueError("attn and value must be 4D for na1d_av")
    if attn.shape[0] != value.shape[0] or attn.shape[2] != value.shape[2]:
        raise ValueError(
            "attn and value must match batch and heads dimensions; "
            f"got {attn.shape} and {value.shape}"
        )

    ks = normalize_kernel_size(kernel_size, 1)
    st = normalize_tuple_param(stride, 1, "stride")
    dil = normalize_tuple_param(dilation, 1, "dilation")
    caus = normalize_tuple_param(is_causal, 1, "is_causal")

    check_kernel_size_vs_input(ks, (value.shape[1],))
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, (value.shape[1],))

    return na1d_av_with_grad(attn, value, ks, st, dil, caus)


def na2d_qk(
    query: mx.array,
    key: mx.array,
    kernel_size,
    dilation=1,
    *,
    stride=1,
    is_causal=False,
    scale: Optional[float] = None,
) -> mx.array:
    """Separate 2D query-key logits. Returns [B, Oh, Ow, heads, Kh*Kw]."""
    _, height, width, _, _ = _validate_2d_qkv(query, key, key)
    ks = normalize_kernel_size(kernel_size, 2)
    st = normalize_tuple_param(stride, 2, "stride")
    dil = normalize_tuple_param(dilation, 2, "dilation")
    caus = normalize_tuple_param(is_causal, 2, "is_causal")

    check_kernel_size_vs_input(ks, (height, width))
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, (height, width))

    return na2d_qk_with_grad(query, key, ks, st, dil, caus, scale)


def na2d_av(
    attn: mx.array,
    value: mx.array,
    kernel_size,
    dilation=1,
    *,
    stride=1,
    is_causal=False,
) -> mx.array:
    """Separate 2D attention-value op."""
    if attn.ndim != 5 or value.ndim != 5:
        raise ValueError("attn and value must be 5D for na2d_av")
    if attn.shape[0] != value.shape[0] or attn.shape[3] != value.shape[3]:
        raise ValueError(
            "attn and value must match batch and heads dimensions; "
            f"got {attn.shape} and {value.shape}"
        )

    ks = normalize_kernel_size(kernel_size, 2)
    st = normalize_tuple_param(stride, 2, "stride")
    dil = normalize_tuple_param(dilation, 2, "dilation")
    caus = normalize_tuple_param(is_causal, 2, "is_causal")

    check_kernel_size_vs_input(ks, (value.shape[1], value.shape[2]))
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, (value.shape[1], value.shape[2]))

    return na2d_av_with_grad(attn, value, ks, st, dil, caus)


__all__ = ["na1d", "na2d", "na1d_qk", "na1d_av", "na2d_qk", "na2d_av"]
