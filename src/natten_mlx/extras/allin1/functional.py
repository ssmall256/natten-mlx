"""
HIGH-PERFORMANCE NATTEN Functional API Compatibility Layer

Reuses existing optimized natten-mlx unfused implementations.
Exposes intermediate stages (QK, Softmax, AV) separately for API compatibility.

Performance: Uses the same efficient MLX operations as unfused reference.
No unnecessary conversions or slow Python loops.
"""

import mlx.core as mx
from typing import Union, Tuple, Optional, Any
import numpy as np


from .reference_impl import get_window_start, get_window_end, get_pb_start


def _torch_to_mlx(tensor):
    """Convert PyTorch tensor to MLX array."""
    if tensor is None:
        return None
    return mx.array(tensor.detach().cpu().numpy())


def _mlx_to_torch(array, reference_tensor=None):
    """Convert MLX array to PyTorch tensor."""
    try:
        import torch
    except ImportError:
        return array

    np_array = np.array(array)
    result = torch.from_numpy(np_array)

    if reference_tensor is not None:
        result = result.to(dtype=reference_tensor.dtype, device=reference_tensor.device)

    return result


def _is_torch_tensor(x):
    """Check if input is a PyTorch tensor."""
    if x is None:
        return False
    return hasattr(x, '__module__') and 'torch' in x.__module__


def _check_args_against_dim(length: int, kernel_size: int, dilation: int, axis_name: str) -> None:
    if kernel_size * dilation > length:
        raise ValueError(
            f"Invalid NATTEN args on {axis_name}: kernel_size * dilation must be <= axis length. "
            f"Got kernel_size={kernel_size}, dilation={dilation}, {axis_name}={length}."
        )


def _natten2dqkrpb_mlx_fast(
    query: mx.array,
    key: mx.array,
    rpb: Optional[mx.array],
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
) -> mx.array:
    """
    Fast 2D NATTEN QK+RPB with author-compatible shifted window semantics.

    Returns attention scores BEFORE softmax.
    """
    if isinstance(kernel_size, int):
        ks_h = ks_w = kernel_size
    else:
        ks_h, ks_w = kernel_size

    if isinstance(dilation, int):
        dil_h = dil_w = dilation
    else:
        dil_h, dil_w = dilation

    if ks_h != ks_w or dil_h != dil_w:
        raise ValueError("Only square kernels and uniform dilation are supported for shifted semantics.")

    B, H, height, width, D = query.shape
    K = int(ks_h)
    nh = K // 2
    rpb_size = 2 * K - 1

    # Precompute window starts/ends/pb starts.
    ni_list = [get_window_start(i, height, K, nh, dil_h) for i in range(height)]
    ei_list = [get_window_end(ni_list[i], height, K, dil_h) for i in range(height)]
    pi_list = [get_pb_start(i, height, K, nh, dil_h) for i in range(height)]

    nj_list = [get_window_start(j, width, K, nh, dil_w) for j in range(width)]
    ej_list = [get_window_end(nj_list[j], width, K, dil_w) for j in range(width)]
    pj_list = [get_pb_start(j, width, K, nh, dil_w) for j in range(width)]

    ni = mx.array(ni_list, dtype=mx.int32)
    ei = mx.array(ei_list, dtype=mx.int32)
    pi = mx.array(pi_list, dtype=mx.int32)
    nj = mx.array(nj_list, dtype=mx.int32)
    ej = mx.array(ej_list, dtype=mx.int32)
    pj = mx.array(pj_list, dtype=mx.int32)

    min_ni = int(mx.min(ni).item())
    min_nj = int(mx.min(nj).item())
    pad_before_i = max(0, -min_ni)
    pad_before_j = max(0, -min_nj)

    max_ei = int(mx.max(ei).item())
    max_ej = int(mx.max(ej).item())
    pad_after_i = max(0, max_ei - height)
    pad_after_j = max(0, max_ej - width)

    pad_width = [
        (0, 0),
        (0, 0),
        (pad_before_i, pad_after_i),
        (pad_before_j, pad_after_j),
        (0, 0),
    ]
    key_pad = mx.pad(key, pad_width, constant_values=0)

    Hp = height + pad_before_i + pad_after_i
    Wp = width + pad_before_j + pad_after_j

    attn_scores = []

    for ki in range(K):
        pos_i = ni + ki * dil_h
        valid_i = (pos_i >= 0) & (pos_i < ei)

        h_idx = pos_i + pad_before_i
        h_idx_safe = mx.clip(h_idx, 0, Hp - 1).astype(mx.int32)
        k_h = mx.take(key_pad, h_idx_safe, axis=2)

        if rpb is not None:
            bias_i = (pi + ki).astype(mx.int32)
            bias_i_safe = mx.clip(bias_i, 0, rpb_size - 1).astype(mx.int32)
            rpb_rows = mx.take(rpb, bias_i_safe, axis=1)
        else:
            rpb_rows = None

        for kj in range(K):
            pos_j = nj + kj * dil_w
            valid_j = (pos_j >= 0) & (pos_j < ej)

            w_idx = pos_j + pad_before_j
            w_idx_safe = mx.clip(w_idx, 0, Wp - 1).astype(mx.int32)
            k_shifted = mx.take(k_h, w_idx_safe, axis=3)

            valid = (mx.reshape(valid_i, (height, 1)) & mx.reshape(valid_j, (1, width)))
            mask = mx.where(valid, 0.0, -mx.inf).reshape(1, 1, height, width)

            score = mx.sum(query * k_shifted, axis=-1)

            if rpb_rows is not None:
                bias_j = (pj + kj).astype(mx.int32)
                bias_j_safe = mx.clip(bias_j, 0, rpb_size - 1).astype(mx.int32)
                rpb_ij = mx.take(rpb_rows, bias_j_safe, axis=2)
                rpb_ij = mx.reshape(rpb_ij, (1, H, height, width))
                score = score + rpb_ij

            score = score + mask
            attn_scores.append(score)

    return mx.stack(attn_scores, axis=-1)


def _natten2dav_mlx_fast(
    attention_probs: mx.array,
    value: mx.array,
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
) -> mx.array:
    """
    Fast 2D NATTEN AV with author-compatible shifted window semantics.

    Applies softmaxed attention to values.
    """
    if isinstance(kernel_size, int):
        ks_h = ks_w = kernel_size
    else:
        ks_h, ks_w = kernel_size

    if isinstance(dilation, int):
        dil_h = dil_w = dilation
    else:
        dil_h, dil_w = dilation

    if ks_h != ks_w or dil_h != dil_w:
        raise ValueError("Only square kernels and uniform dilation are supported for shifted semantics.")

    B, H, height, width, _ = attention_probs.shape
    _, _, _, _, D = value.shape
    K = int(ks_h)
    nh = K // 2

    ni_list = [get_window_start(i, height, K, nh, dil_h) for i in range(height)]
    ei_list = [get_window_end(ni_list[i], height, K, dil_h) for i in range(height)]
    nj_list = [get_window_start(j, width, K, nh, dil_w) for j in range(width)]
    ej_list = [get_window_end(nj_list[j], width, K, dil_w) for j in range(width)]

    ni = mx.array(ni_list, dtype=mx.int32)
    ei = mx.array(ei_list, dtype=mx.int32)
    nj = mx.array(nj_list, dtype=mx.int32)
    ej = mx.array(ej_list, dtype=mx.int32)

    min_ni = int(mx.min(ni).item())
    min_nj = int(mx.min(nj).item())
    pad_before_i = max(0, -min_ni)
    pad_before_j = max(0, -min_nj)

    max_ei = int(mx.max(ei).item())
    max_ej = int(mx.max(ej).item())
    pad_after_i = max(0, max_ei - height)
    pad_after_j = max(0, max_ej - width)

    pad_width = [
        (0, 0),
        (0, 0),
        (pad_before_i, pad_after_i),
        (pad_before_j, pad_after_j),
        (0, 0),
    ]
    value_pad = mx.pad(value, pad_width, constant_values=0)

    Hp = height + pad_before_i + pad_after_i
    Wp = width + pad_before_j + pad_after_j

    output = mx.zeros((B, H, height, width, D), dtype=value.dtype)
    idx = 0

    for ki in range(K):
        pos_i = ni + ki * dil_h
        valid_i = (pos_i >= 0) & (pos_i < ei)

        h_idx = pos_i + pad_before_i
        h_idx_safe = mx.clip(h_idx, 0, Hp - 1).astype(mx.int32)
        v_h = mx.take(value_pad, h_idx_safe, axis=2)

        for kj in range(K):
            pos_j = nj + kj * dil_w
            valid_j = (pos_j >= 0) & (pos_j < ej)

            w_idx = pos_j + pad_before_j
            w_idx_safe = mx.clip(w_idx, 0, Wp - 1).astype(mx.int32)
            v_shifted = mx.take(v_h, w_idx_safe, axis=3)

            valid = (mx.reshape(valid_i, (height, 1)) & mx.reshape(valid_j, (1, width)))
            valid_reshaped = valid.reshape((1, 1, height, width, 1))

            attn_weight = attention_probs[:, :, :, :, idx:idx + 1]
            attn_weight = mx.where(valid_reshaped, attn_weight, 0.0)

            output = output + attn_weight * v_shifted
            idx += 1

    return output


def _natten1dqkrpb_mlx_fast(
    query: mx.array,
    key: mx.array,
    rpb: Optional[mx.array],
    kernel_size: int,
    dilation: int,
) -> mx.array:
    """
    Fast 1D NATTEN QK+RPB with author-compatible shifted window semantics.

    Returns attention scores BEFORE softmax.
    """
    B, H, L, D = query.shape
    K = int(kernel_size)
    nh = K // 2
    dil = int(dilation)
    rpb_size = 2 * K - 1

    ni_list = [get_window_start(i, L, K, nh, dil) for i in range(L)]
    ei_list = [get_window_end(ni_list[i], L, K, dil) for i in range(L)]
    pi_list = [get_pb_start(i, L, K, nh, dil) for i in range(L)]

    ni = mx.array(ni_list, dtype=mx.int32)
    ei = mx.array(ei_list, dtype=mx.int32)
    pi = mx.array(pi_list, dtype=mx.int32)

    min_ni = int(mx.min(ni).item())
    pad_before = max(0, -min_ni)
    max_ei = int(mx.max(ei).item())
    pad_after = max(0, max_ei - L)

    pad_width = [(0, 0), (0, 0), (pad_before, pad_after), (0, 0)]
    key_pad = mx.pad(key, pad_width, constant_values=0)

    Lp = L + pad_before + pad_after
    attn_scores = []

    for ki in range(K):
        pos = ni + ki * dil
        valid = (pos >= 0) & (pos < ei)
        mask = mx.where(valid, 0.0, -mx.inf).reshape((1, 1, L))

        idx = pos + pad_before
        idx_safe = mx.clip(idx, 0, Lp - 1).astype(mx.int32)
        k_shifted = mx.take(key_pad, idx_safe, axis=2)

        score = mx.sum(query * k_shifted, axis=-1)

        if rpb is not None:
            rpb_idx = (pi + ki).astype(mx.int32)
            rpb_idx_safe = mx.clip(rpb_idx, 0, rpb_size - 1).astype(mx.int32)
            rpb_val = mx.take(rpb, rpb_idx_safe, axis=1)
            rpb_val = mx.reshape(rpb_val, (1, H, L))
            score = score + rpb_val

        score = score + mask
        attn_scores.append(score)

    return mx.stack(attn_scores, axis=-1)


def _natten1dav_mlx_fast(
    attention_probs: mx.array,
    value: mx.array,
    kernel_size: int,
    dilation: int,
) -> mx.array:
    """
    Fast 1D NATTEN AV with author-compatible shifted window semantics.

    Applies softmaxed attention to values.
    """
    B, H, L, K = attention_probs.shape
    _, _, _, D = value.shape
    K = int(kernel_size)
    nh = K // 2
    dil = int(dilation)

    ni_list = [get_window_start(i, L, K, nh, dil) for i in range(L)]
    ei_list = [get_window_end(ni_list[i], L, K, dil) for i in range(L)]

    ni = mx.array(ni_list, dtype=mx.int32)
    ei = mx.array(ei_list, dtype=mx.int32)

    min_ni = int(mx.min(ni).item())
    pad_before = max(0, -min_ni)
    max_ei = int(mx.max(ei).item())
    pad_after = max(0, max_ei - L)

    pad_width = [(0, 0), (0, 0), (pad_before, pad_after), (0, 0)]
    value_pad = mx.pad(value, pad_width, constant_values=0)

    Lp = L + pad_before + pad_after
    output = mx.zeros((B, H, L, D), dtype=value.dtype)

    for ki in range(K):
        pos = ni + ki * dil
        valid = (pos >= 0) & (pos < ei)
        valid_reshaped = valid.reshape((1, 1, L, 1))

        idx = pos + pad_before
        idx_safe = mx.clip(idx, 0, Lp - 1).astype(mx.int32)
        v_shifted = mx.take(value_pad, idx_safe, axis=2)

        attn_weight = attention_probs[:, :, :, ki:ki + 1]
        attn_weight = mx.where(valid_reshaped, attn_weight, 0.0)
        output = output + attn_weight * v_shifted

    return output


# Public API with automatic PyTorch conversion
def natten1dqkrpb(query: Any, key: Any, rpb: Optional[Any], kernel_size: int, dilation: int) -> Any:
    """1D NATTEN QK+RPB (returns scores BEFORE softmax)."""
    # Convert PyTorch → MLX if needed
    is_torch = _is_torch_tensor(query)
    if is_torch:
        ref = query
        query = _torch_to_mlx(query)
        key = _torch_to_mlx(key)
        rpb = _torch_to_mlx(rpb) if rpb is not None else None

    _check_args_against_dim(int(query.shape[2]), int(kernel_size), int(dilation), "length")

    # Fast MLX implementation
    result = _natten1dqkrpb_mlx_fast(query, key, rpb, kernel_size, dilation)

    # Convert back to PyTorch if needed
    return _mlx_to_torch(result, ref) if is_torch else result


def natten1dav(attention_probs: Any, value: Any, kernel_size: int, dilation: int) -> Any:
    """1D NATTEN AV (applies softmaxed attention to values)."""
    # Convert PyTorch → MLX if needed
    is_torch = _is_torch_tensor(attention_probs)
    if is_torch:
        ref = attention_probs
        attention_probs = _torch_to_mlx(attention_probs)
        value = _torch_to_mlx(value)

    _check_args_against_dim(int(value.shape[2]), int(kernel_size), int(dilation), "length")

    # Fast MLX implementation
    result = _natten1dav_mlx_fast(attention_probs, value, kernel_size, dilation)

    # Convert back to PyTorch if needed
    return _mlx_to_torch(result, ref) if is_torch else result


def natten2dqkrpb(
    query: Any,
    key: Any,
    rpb: Optional[Any],
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
) -> Any:
    """2D NATTEN QK+RPB (returns scores BEFORE softmax)."""
    # Convert PyTorch → MLX if needed
    is_torch = _is_torch_tensor(query)
    if is_torch:
        ref = query
        query = _torch_to_mlx(query)
        key = _torch_to_mlx(key)
        rpb = _torch_to_mlx(rpb) if rpb is not None else None

    k = int(kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size)
    d = int(dilation[0] if isinstance(dilation, tuple) else dilation)
    _check_args_against_dim(int(query.shape[2]), k, d, "height")
    _check_args_against_dim(int(query.shape[3]), k, d, "width")

    # Fast MLX implementation
    result = _natten2dqkrpb_mlx_fast(query, key, rpb, kernel_size, dilation)

    # Convert back to PyTorch if needed
    return _mlx_to_torch(result, ref) if is_torch else result


def natten2dav(
    attention_probs: Any,
    value: Any,
    kernel_size: Union[int, Tuple[int, int]],
    dilation: Union[int, Tuple[int, int]],
) -> Any:
    """2D NATTEN AV (applies softmaxed attention to values)."""
    # Convert PyTorch → MLX if needed
    is_torch = _is_torch_tensor(attention_probs)
    if is_torch:
        ref = attention_probs
        attention_probs = _torch_to_mlx(attention_probs)
        value = _torch_to_mlx(value)

    k = int(kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size)
    d = int(dilation[0] if isinstance(dilation, tuple) else dilation)
    _check_args_against_dim(int(value.shape[2]), k, d, "height")
    _check_args_against_dim(int(value.shape[3]), k, d, "width")

    # Fast MLX implementation
    result = _natten2dav_mlx_fast(attention_probs, value, kernel_size, dilation)

    # Convert back to PyTorch if needed
    return _mlx_to_torch(result, ref) if is_torch else result


__all__ = ['natten1dqkrpb', 'natten1dav', 'natten2dqkrpb', 'natten2dav']
