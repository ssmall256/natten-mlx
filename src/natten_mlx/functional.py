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
    na3d_av_with_grad,
    na3d_qk_with_grad,
    na3d_with_grad,
)
from natten_mlx.utils.params import (
    check_dilation_kernel_vs_input,
    check_kernel_size_vs_input,
    check_stride_vs_kernel,
    normalize_kernel_size,
    normalize_tuple_param,
)


def _repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """Expand KV heads to match Q heads for GQA/MQA."""
    if n_rep == 1:
        return x
    return mx.repeat(x, n_rep, axis=-2)


def _validate_qkv(query: mx.array, key: mx.array, value: mx.array, rank: int) -> int:
    """Validate Q/K/V shapes with GQA support. Returns kv_repeat factor."""
    expected_ndim = rank + 3
    layouts = {1: "[B, L, H, D]", 2: "[B, H, W, heads, dim]", 3: "[B, D, H, W, heads, dim]"}
    if query.ndim != expected_ndim or key.ndim != expected_ndim or value.ndim != expected_ndim:
        raise ValueError(f"na{rank}d expects query/key/value with shape {layouts[rank]}.")

    spatial_q = query.shape[1:-2]
    spatial_k = key.shape[1:-2]
    spatial_v = value.shape[1:-2]
    if spatial_q != spatial_k or spatial_q != spatial_v:
        raise ValueError(
            f"Spatial dimensions must match: Q={spatial_q}, K={spatial_k}, V={spatial_v}."
        )

    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Batch dimensions must match.")

    heads_q, dim_q = query.shape[-2], query.shape[-1]
    heads_kv, dim_k = key.shape[-2], key.shape[-1]
    if dim_q != dim_k:
        raise ValueError(f"Head dim must match: Q has {dim_q}, K has {dim_k}.")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("K and V must have same number of heads.")
    if key.shape[-1] != value.shape[-1]:
        raise ValueError(f"Head dim must match: K has {dim_k}, V has {value.shape[-1]}.")

    if heads_q % heads_kv != 0:
        raise ValueError(
            f"heads_q ({heads_q}) must be divisible by heads_kv ({heads_kv}) for GQA."
        )

    return heads_q // heads_kv


def _full_attn_with_lse(
    query: mx.array, key: mx.array, value: mx.array, scale: float
) -> tuple[mx.array, mx.array]:
    """Full (non-neighborhood) attention with LSE return.

    Q: [B, ..spatial.., H, D], K/V: [B, N_extra, H, D].
    Returns (output, lse) where output has Q's spatial shape and lse has shape
    [B, ..spatial.., H].
    """
    spatial_dims = query.shape[1:-2]
    B, H, D = query.shape[0], query.shape[-2], query.shape[-1]

    # Flatten spatial dims: [B, S, H, D]
    S = 1
    for s in spatial_dims:
        S *= s
    q_flat = query.reshape(B, S, H, D)

    # [B, H, S, D] and [B, H, N_extra, D]
    q_t = mx.transpose(q_flat, axes=(0, 2, 1, 3))
    k_t = mx.transpose(key, axes=(0, 2, 1, 3))
    logits = (q_t @ mx.transpose(k_t, axes=(0, 1, 3, 2))) * scale  # [B, H, S, N_extra]

    lse = mx.logsumexp(logits, axis=-1)    # [B, H, S]
    attn = mx.softmax(logits, axis=-1)     # [B, H, S, N_extra]

    v_t = mx.transpose(value, axes=(0, 2, 1, 3))  # [B, H, N_extra, D]
    out = attn @ v_t  # [B, H, S, D]

    # Back to [B, ..spatial.., H, D]
    out = mx.transpose(out, axes=(0, 2, 1, 3)).reshape(*([B] + list(spatial_dims) + [H, D]))
    lse = mx.transpose(lse, axes=(0, 2, 1)).reshape(*([B] + list(spatial_dims) + [H]))

    return out, lse


def _validate_additional_kv(
    query: mx.array,
    additional_keys: Optional[mx.array],
    additional_values: Optional[mx.array],
    kv_repeat: int,
) -> tuple[Optional[mx.array], Optional[mx.array]]:
    """Validate and optionally expand additional K/V for GQA."""
    if additional_keys is None and additional_values is None:
        return None, None
    if additional_keys is None or additional_values is None:
        raise ValueError("additional_keys and additional_values must both be provided or both None.")

    if additional_keys.ndim != 4 or additional_values.ndim != 4:
        raise ValueError(
            "additional_keys/additional_values must be 4D: [B, N_extra, heads_kv, dim]."
        )
    if additional_keys.shape[0] != query.shape[0]:
        raise ValueError("additional_keys batch must match query batch.")
    if additional_values.shape[0] != query.shape[0]:
        raise ValueError("additional_values batch must match query batch.")
    if additional_keys.shape[-1] != query.shape[-1]:
        raise ValueError("additional_keys head dim must match query head dim.")
    if additional_values.shape[-1] != query.shape[-1]:
        raise ValueError("additional_values head dim must match query head dim.")
    if additional_keys.shape[-2] != additional_values.shape[-2]:
        raise ValueError("additional_keys and additional_values must have same number of heads.")
    if additional_keys.shape[1] != additional_values.shape[1]:
        raise ValueError("additional_keys and additional_values must have same N_extra.")

    ak = _repeat_kv(additional_keys, kv_repeat)
    av = _repeat_kv(additional_values, kv_repeat)
    return ak, av


def _is_full_attention(kernel_size: tuple, spatial_shape: tuple) -> bool:
    """Check if kernel covers all spatial dims (NA degenerates to global attention)."""
    return all(ks >= s for ks, s in zip(kernel_size, spatial_shape))


def _sdpa_forward(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float,
    spatial_shape: tuple,
    return_lse: bool = False,
    additional_keys=None,
    additional_values=None,
) -> mx.array:
    """Full-attention fast path via SDPA when kernel covers entire spatial extent."""
    B, H, D = query.shape[0], query.shape[-2], query.shape[-1]

    S = 1
    for s in spatial_shape:
        S *= s

    k_flat = key.reshape(B, S, H, D)
    v_flat = value.reshape(B, S, H, D)

    if additional_keys is not None:
        k_flat = mx.concatenate([k_flat, additional_keys], axis=1)
        v_flat = mx.concatenate([v_flat, additional_values], axis=1)

    q_flat = query.reshape(B, -1, H, D)

    # MLX SDPA expects [B, H, S, D] (heads before sequence)
    q_t = mx.transpose(q_flat, axes=(0, 2, 1, 3))
    k_t = mx.transpose(k_flat, axes=(0, 2, 1, 3))
    v_t = mx.transpose(v_flat, axes=(0, 2, 1, 3))
    out = mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
    out = mx.transpose(out, axes=(0, 2, 1, 3))  # back to [B, S, H, D]
    out = out.reshape(*([B] + list(spatial_shape) + [H, D]))

    if return_lse:
        # Compute LSE
        q_t = mx.transpose(q_flat, axes=(0, 2, 1, 3))
        k_t = mx.transpose(k_flat, axes=(0, 2, 1, 3))
        logits = (q_t @ mx.transpose(k_t, axes=(0, 1, 3, 2))) * scale
        lse = mx.logsumexp(logits, axis=-1)
        lse = mx.transpose(lse, axes=(0, 2, 1)).reshape(*([B] + list(spatial_shape) + [H]))
        return out, lse

    return out


def _validate_1d_qkv(query: mx.array, key: mx.array, value: mx.array) -> tuple[int, int, int, int]:
    _validate_qkv(query, key, value, 1)
    return tuple(query.shape)


def _validate_2d_qkv(query: mx.array, key: mx.array, value: mx.array) -> tuple[int, int, int, int, int]:
    _validate_qkv(query, key, value, 2)
    return tuple(query.shape)


def _validate_3d_qkv(
    query: mx.array, key: mx.array, value: mx.array
) -> tuple[int, int, int, int, int, int]:
    _validate_qkv(query, key, value, 3)
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
    return_lse: bool = False,
    additional_keys: Optional[mx.array] = None,
    additional_values: Optional[mx.array] = None,
) -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """1D neighborhood attention.

    Each query attends to a local window of ``kernel_size`` neighbors along
    the sequence dimension.  Supports GQA/MQA when K/V have fewer heads
    than Q, and optional extra global tokens via ``additional_keys``/
    ``additional_values``.

    Args:
        query: ``[B, L, H_q, D]``.
        key: ``[B, L, H_kv, D]``.  ``H_q`` must be divisible by ``H_kv``.
        value: ``[B, L, H_kv, D]``.
        kernel_size: Neighborhood window size (scalar or 1-tuple).
        stride: Output stride for downsampling.  Default ``1``.
        dilation: Gap between attended positions.  Default ``1``.
        is_causal: Causal masking (attend only to past/current).
        scale: Logit scaling factor.  Default ``D ** -0.5``.
        return_lse: If ``True``, return ``(output, lse)`` where ``lse``
            has shape ``[B, L_out, H_q]``.
        additional_keys: ``[B, N_extra, H_kv, D]`` — extra tokens every
            query attends to (global attention).  Requires
            ``additional_values``.
        additional_values: ``[B, N_extra, H_kv, D]``.

    Returns:
        ``[B, L_out, H_q, D]``, or ``(output, lse)`` when
        ``return_lse=True``.

    Note:
        When ``kernel_size >= L`` the call is dispatched to
        ``mx.fast.scaled_dot_product_attention`` automatically.
    """
    kv_repeat = _validate_qkv(query, key, value, 1)
    add_k, add_v = _validate_additional_kv(query, additional_keys, additional_values, kv_repeat)
    _, seqlen, _, head_dim = query.shape

    key = _repeat_kv(key, kv_repeat)
    value = _repeat_kv(value, kv_repeat)

    ks = normalize_kernel_size(kernel_size, 1)
    st = normalize_tuple_param(stride, 1, "stride")
    dil = normalize_tuple_param(dilation, 1, "dilation")
    caus = normalize_tuple_param(is_causal, 1, "is_causal")

    spatial_shape = (seqlen,)
    check_kernel_size_vs_input(ks, spatial_shape)
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    scale_value = head_dim ** -0.5 if scale is None else float(scale)

    # FMHA fast path: kernel covers all spatial dims → use SDPA
    if _is_full_attention(ks, spatial_shape) and all(s == 1 for s in st) and not any(caus):
        return _sdpa_forward(
            query, key, value, scale_value, spatial_shape,
            return_lse=return_lse, additional_keys=add_k, additional_values=add_v,
        )

    has_additional = add_k is not None

    if has_additional or return_lse:
        logits = na1d_qk(query, key, kernel_size=ks, dilation=dil, stride=st, is_causal=caus)
        logits_scaled = logits * scale_value
        lse = mx.logsumexp(logits_scaled, axis=-1)
        attn = mx.softmax(logits_scaled, axis=-1)
        out = na1d_av(attn, value, kernel_size=ks, dilation=dil, stride=st, is_causal=caus)

        if has_additional:
            from natten_mlx.merge import merge_attentions
            out_extra, lse_extra = _full_attn_with_lse(query, add_k, add_v, scale_value)
            out, lse = merge_attentions([out, out_extra], [lse, lse_extra])

        if return_lse:
            return out, lse
        return out

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
    return_lse: bool = False,
    additional_keys: Optional[mx.array] = None,
    additional_values: Optional[mx.array] = None,
) -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """2D neighborhood attention.

    Each query attends to a local ``kernel_size × kernel_size`` window on the
    spatial grid.  Supports GQA/MQA when K/V have fewer heads than Q, and
    optional extra global tokens via ``additional_keys``/``additional_values``.

    Args:
        query: ``[B, H, W, H_q, D]``.
        key: ``[B, H, W, H_kv, D]``.  ``H_q`` must be divisible by ``H_kv``.
        value: ``[B, H, W, H_kv, D]``.
        kernel_size: Neighborhood window size (scalar or ``(kH, kW)``).
        stride: Output stride for downsampling.  Default ``1``.
        dilation: Gap between attended positions.  Default ``1``.
        is_causal: Causal masking per axis, e.g. ``(False, True)``.
        scale: Logit scaling factor.  Default ``D ** -0.5``.
        return_lse: If ``True``, return ``(output, lse)`` where ``lse``
            has shape ``[B, H_out, W_out, H_q]``.
        additional_keys: ``[B, N_extra, H_kv, D]`` — extra tokens every
            query attends to (global attention).  Requires
            ``additional_values``.
        additional_values: ``[B, N_extra, H_kv, D]``.

    Returns:
        ``[B, H_out, W_out, H_q, D]``, or ``(output, lse)`` when
        ``return_lse=True``.

    Note:
        When ``kernel_size >= (H, W)`` the call is dispatched to
        ``mx.fast.scaled_dot_product_attention`` automatically.
    """
    kv_repeat = _validate_qkv(query, key, value, 2)
    add_k, add_v = _validate_additional_kv(query, additional_keys, additional_values, kv_repeat)
    _, height, width, _, head_dim = query.shape

    key = _repeat_kv(key, kv_repeat)
    value = _repeat_kv(value, kv_repeat)

    ks = normalize_kernel_size(kernel_size, 2)
    st = normalize_tuple_param(stride, 2, "stride")
    dil = normalize_tuple_param(dilation, 2, "dilation")
    caus = normalize_tuple_param(is_causal, 2, "is_causal")

    spatial_shape = (height, width)
    check_kernel_size_vs_input(ks, spatial_shape)
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    scale_value = head_dim ** -0.5 if scale is None else float(scale)

    # FMHA fast path: kernel covers all spatial dims → use SDPA
    if _is_full_attention(ks, spatial_shape) and all(s == 1 for s in st) and not any(caus):
        return _sdpa_forward(
            query, key, value, scale_value, spatial_shape,
            return_lse=return_lse, additional_keys=add_k, additional_values=add_v,
        )

    has_additional = add_k is not None

    if has_additional or return_lse:
        logits = na2d_qk(query, key, kernel_size=ks, dilation=dil, stride=st, is_causal=caus)
        logits_scaled = logits * scale_value
        lse = mx.logsumexp(logits_scaled, axis=-1)
        attn = mx.softmax(logits_scaled, axis=-1)
        out = na2d_av(attn, value, kernel_size=ks, dilation=dil, stride=st, is_causal=caus)

        if has_additional:
            from natten_mlx.merge import merge_attentions
            out_extra, lse_extra = _full_attn_with_lse(query, add_k, add_v, scale_value)
            out, lse = merge_attentions([out, out_extra], [lse, lse_extra])

        if return_lse:
            return out, lse
        return out

    return na2d_with_grad(query, key, value, ks, st, dil, caus, scale)


def na3d(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = 1,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    is_causal: Union[bool, Tuple[bool, bool, bool]] = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    additional_keys: Optional[mx.array] = None,
    additional_values: Optional[mx.array] = None,
) -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """3D neighborhood attention.

    Each query attends to a local ``kernel_size³`` window in the volumetric
    spatial grid.  Supports GQA/MQA when K/V have fewer heads than Q, and
    optional extra global tokens via ``additional_keys``/``additional_values``.

    Args:
        query: ``[B, D1, D2, D3, H_q, D]``.
        key: ``[B, D1, D2, D3, H_kv, D]``.  ``H_q`` must be divisible by
            ``H_kv``.
        value: ``[B, D1, D2, D3, H_kv, D]``.
        kernel_size: Neighborhood window size (scalar or ``(k1, k2, k3)``).
        stride: Output stride for downsampling.  Default ``1``.
        dilation: Gap between attended positions.  Default ``1``.
        is_causal: Causal masking per axis.
        scale: Logit scaling factor.  Default ``D ** -0.5``.
        return_lse: If ``True``, return ``(output, lse)`` where ``lse``
            has shape ``[B, D1_out, D2_out, D3_out, H_q]``.
        additional_keys: ``[B, N_extra, H_kv, D]`` — extra tokens every
            query attends to (global attention).  Requires
            ``additional_values``.
        additional_values: ``[B, N_extra, H_kv, D]``.

    Returns:
        ``[B, D1_out, D2_out, D3_out, H_q, D]``, or ``(output, lse)``
        when ``return_lse=True``.

    Note:
        When ``kernel_size >= (D1, D2, D3)`` the call is dispatched to
        ``mx.fast.scaled_dot_product_attention`` automatically.
    """
    kv_repeat = _validate_qkv(query, key, value, 3)
    add_k, add_v = _validate_additional_kv(query, additional_keys, additional_values, kv_repeat)
    _, depth, height, width, _, head_dim = query.shape

    key = _repeat_kv(key, kv_repeat)
    value = _repeat_kv(value, kv_repeat)

    ks = normalize_kernel_size(kernel_size, 3)
    st = normalize_tuple_param(stride, 3, "stride")
    dil = normalize_tuple_param(dilation, 3, "dilation")
    caus = normalize_tuple_param(is_causal, 3, "is_causal")

    spatial_shape = (depth, height, width)
    check_kernel_size_vs_input(ks, spatial_shape)
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, spatial_shape)

    scale_value = head_dim ** -0.5 if scale is None else float(scale)

    # FMHA fast path: kernel covers all spatial dims → use SDPA
    if _is_full_attention(ks, spatial_shape) and all(s == 1 for s in st) and not any(caus):
        return _sdpa_forward(
            query, key, value, scale_value, spatial_shape,
            return_lse=return_lse, additional_keys=add_k, additional_values=add_v,
        )

    has_additional = add_k is not None

    if has_additional or return_lse:
        logits = na3d_qk(query, key, kernel_size=ks, dilation=dil, stride=st, is_causal=caus)
        logits_scaled = logits * scale_value
        lse = mx.logsumexp(logits_scaled, axis=-1)
        attn = mx.softmax(logits_scaled, axis=-1)
        out = na3d_av(attn, value, kernel_size=ks, dilation=dil, stride=st, is_causal=caus)

        if has_additional:
            from natten_mlx.merge import merge_attentions
            out_extra, lse_extra = _full_attn_with_lse(query, add_k, add_v, scale_value)
            out, lse = merge_attentions([out, out_extra], [lse, lse_extra])

        if return_lse:
            return out, lse
        return out

    return na3d_with_grad(query, key, value, ks, st, dil, caus, scale)


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

def na3d_qk(
    query: mx.array,
    key: mx.array,
    kernel_size,
    dilation=1,
    *,
    stride=1,
    is_causal=False,
    scale: Optional[float] = None,
) -> mx.array:
    """Separate 3D query-key logits. Returns [B, Od, Oh, Ow, heads, Kd*Kh*Kw]."""
    _, depth, height, width, _, _ = _validate_3d_qkv(query, key, key)
    ks = normalize_kernel_size(kernel_size, 3)
    st = normalize_tuple_param(stride, 3, "stride")
    dil = normalize_tuple_param(dilation, 3, "dilation")
    caus = normalize_tuple_param(is_causal, 3, "is_causal")

    check_kernel_size_vs_input(ks, (depth, height, width))
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, (depth, height, width))

    return na3d_qk_with_grad(query, key, ks, st, dil, caus, scale)


def na3d_av(
    attn: mx.array,
    value: mx.array,
    kernel_size,
    dilation=1,
    *,
    stride=1,
    is_causal=False,
) -> mx.array:
    """Separate 3D attention-value op."""
    if attn.ndim != 6 or value.ndim != 6:
        raise ValueError("attn and value must be 6D for na3d_av")
    if attn.shape[0] != value.shape[0] or attn.shape[4] != value.shape[4]:
        raise ValueError(
            "attn and value must match batch and heads dimensions; "
            f"got {attn.shape} and {value.shape}"
        )

    ks = normalize_kernel_size(kernel_size, 3)
    st = normalize_tuple_param(stride, 3, "stride")
    dil = normalize_tuple_param(dilation, 3, "dilation")
    caus = normalize_tuple_param(is_causal, 3, "is_causal")

    check_kernel_size_vs_input(ks, (value.shape[1], value.shape[2], value.shape[3]))
    check_stride_vs_kernel(st, ks)
    check_dilation_kernel_vs_input(dil, ks, (value.shape[1], value.shape[2], value.shape[3]))

    return na3d_av_with_grad(attn, value, ks, st, dil, caus)


def _validate_spatial_sizes(
    spatial_sizes: mx.array,
    batch_size: int,
    max_spatial: tuple,
    kernel_size: tuple,
    rank: int,
) -> None:
    """Validate spatial_sizes for variable-length 2D/3D attention."""
    if spatial_sizes.ndim != 2 or spatial_sizes.shape[0] != batch_size or spatial_sizes.shape[1] != rank:
        raise ValueError(
            f"spatial_sizes must be a [B, {rank}] array, "
            f"got shape {tuple(spatial_sizes.shape)}."
        )
    if spatial_sizes.dtype not in (mx.int32, mx.int64):
        raise ValueError(f"spatial_sizes must be int32 or int64, got {spatial_sizes.dtype}.")
    for d in range(rank):
        min_dim = int(spatial_sizes[:, d].min().item())
        max_dim = int(spatial_sizes[:, d].max().item())
        if min_dim < kernel_size[d]:
            raise ValueError(
                f"All spatial_sizes[:,{d}] must be >= kernel_size[{d}] ({kernel_size[d]}), "
                f"but min={min_dim}."
            )
        if max_dim > max_spatial[d]:
            raise ValueError(
                f"All spatial_sizes[:,{d}] must be <= max_spatial[{d}] ({max_spatial[d]}), "
                f"but max={max_dim}."
            )


def _validate_seq_lens(
    seq_lens: mx.array, batch_size: int, l_max: int, kernel_size: int,
) -> None:
    """Validate seq_lens for variable-length attention."""
    if seq_lens.ndim != 1 or seq_lens.shape[0] != batch_size:
        raise ValueError(
            f"seq_lens must be a 1-D array of length B={batch_size}, "
            f"got shape {tuple(seq_lens.shape)}."
        )
    if seq_lens.dtype not in (mx.int32, mx.int64):
        raise ValueError(f"seq_lens must be int32 or int64, got {seq_lens.dtype}.")
    min_len = int(seq_lens.min().item())
    max_len = int(seq_lens.max().item())
    if min_len < kernel_size:
        raise ValueError(
            f"All seq_lens must be >= kernel_size ({kernel_size}), "
            f"but min(seq_lens)={min_len}."
        )
    if max_len > l_max:
        raise ValueError(
            f"All seq_lens must be <= L_max ({l_max}), "
            f"but max(seq_lens)={max_len}."
        )


def na1d_varlen(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    seq_lens: mx.array,
    kernel_size: Union[int, Tuple[int]],
    dilation: Union[int, Tuple[int]] = 1,
    scale: Optional[float] = None,
) -> mx.array:
    """Variable-length 1D neighborhood attention.

    Tensors are padded to a common ``L_max`` but each batch element only
    attends within its actual length given by ``seq_lens``.  Positions
    beyond ``seq_lens[b]`` are zeroed in the output.

    Args:
        query: ``[B, L_max, H, D]``.
        key: ``[B, L_max, H, D]``.
        value: ``[B, L_max, H, D]``.
        seq_lens: ``[B]`` int array — actual length per batch element.
            Must satisfy ``kernel_size <= seq_lens[b] <= L_max``.
        kernel_size: Neighborhood window size (scalar or 1-tuple).
        dilation: Gap between attended positions.  Default ``1``.
        scale: Logit scaling factor.  Default ``D ** -0.5``.

    Returns:
        ``[B, L_max, H, D]`` — output with padding positions zeroed.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("na1d_varlen expects query/key/value with shape [B, L, H, D].")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes.")

    B, L_max = query.shape[0], query.shape[1]
    ks = normalize_kernel_size(kernel_size, 1)
    dil = normalize_tuple_param(dilation, 1, "dilation")

    _validate_seq_lens(seq_lens, B, L_max, ks[0])

    scale_value = query.shape[-1] ** -0.5 if scale is None else float(scale)

    from natten_mlx.autograd import na1d_varlen_with_grad
    return na1d_varlen_with_grad(query, key, value, seq_lens, ks, dil, scale_value)


def na2d_varlen(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    spatial_sizes: mx.array,
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]] = 1,
    scale: Optional[float] = None,
) -> mx.array:
    """Variable-length 2D neighborhood attention.

    Tensors are padded to a common ``(H_max, W_max)`` but each batch element
    only attends within its actual spatial extent given by ``spatial_sizes``.
    Positions beyond ``(H_b, W_b)`` are zeroed in the output.

    Args:
        query: ``[B, H_max, W_max, heads, D]``.
        key: ``[B, H_max, W_max, heads, D]``.
        value: ``[B, H_max, W_max, heads, D]``.
        spatial_sizes: ``[B, 2]`` int array — ``(H_b, W_b)`` per batch element.
        kernel_size: Neighborhood window size (scalar or ``(kH, kW)``).
        dilation: Gap between attended positions.  Default ``1``.
        scale: Logit scaling factor.  Default ``D ** -0.5``.

    Returns:
        ``[B, H_max, W_max, heads, D]`` — output with padding positions zeroed.
    """
    if query.ndim != 5 or key.ndim != 5 or value.ndim != 5:
        raise ValueError("na2d_varlen expects query/key/value with shape [B, H, W, heads, D].")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes.")

    B, H_max, W_max = query.shape[0], query.shape[1], query.shape[2]
    ks = normalize_kernel_size(kernel_size, 2)
    dil = normalize_tuple_param(dilation, 2, "dilation")

    _validate_spatial_sizes(spatial_sizes, B, (H_max, W_max), ks, 2)

    scale_value = query.shape[-1] ** -0.5 if scale is None else float(scale)

    from natten_mlx.autograd import na2d_varlen_with_grad
    return na2d_varlen_with_grad(query, key, value, spatial_sizes, ks, dil, scale_value)


def na3d_varlen(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    spatial_sizes: mx.array,
    kernel_size: Union[int, Tuple[int, int, int]],
    dilation: Union[int, Tuple[int, int, int]] = 1,
    scale: Optional[float] = None,
) -> mx.array:
    """Variable-length 3D neighborhood attention.

    Tensors are padded to a common ``(D_max, H_max, W_max)`` but each batch
    element only attends within its actual spatial extent given by
    ``spatial_sizes``.  Positions beyond ``(D_b, H_b, W_b)`` are zeroed.

    Args:
        query: ``[B, D_max, H_max, W_max, heads, D]``.
        key: ``[B, D_max, H_max, W_max, heads, D]``.
        value: ``[B, D_max, H_max, W_max, heads, D]``.
        spatial_sizes: ``[B, 3]`` int array — ``(D_b, H_b, W_b)`` per batch.
        kernel_size: Neighborhood window size (scalar or ``(kD, kH, kW)``).
        dilation: Gap between attended positions.  Default ``1``.
        scale: Logit scaling factor.  Default ``D ** -0.5``.

    Returns:
        ``[B, D_max, H_max, W_max, heads, D]`` — output with padding zeroed.
    """
    if query.ndim != 6 or key.ndim != 6 or value.ndim != 6:
        raise ValueError("na3d_varlen expects query/key/value with shape [B, D, H, W, heads, dim].")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes.")

    B = query.shape[0]
    D_max, H_max, W_max = query.shape[1], query.shape[2], query.shape[3]
    ks = normalize_kernel_size(kernel_size, 3)
    dil = normalize_tuple_param(dilation, 3, "dilation")

    _validate_spatial_sizes(spatial_sizes, B, (D_max, H_max, W_max), ks, 3)

    scale_value = query.shape[-1] ** -0.5 if scale is None else float(scale)

    from natten_mlx.autograd import na3d_varlen_with_grad
    return na3d_varlen_with_grad(query, key, value, spatial_sizes, ks, dil, scale_value)


__all__ = [
    "na1d",
    "na1d_varlen",
    "na2d",
    "na2d_varlen",
    "na3d",
    "na3d_varlen",
    "na1d_qk",
    "na1d_av",
    "na2d_qk",
    "na2d_av",
    "na3d_qk",
    "na3d_av",
]
