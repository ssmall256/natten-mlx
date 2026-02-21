"""Tier 0: pure MLX backend for neighborhood attention."""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from natten_mlx.utils.window import compute_window_start_end


def is_available() -> bool:
    return True


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _query_positions(length: int, stride: int) -> np.ndarray:
    out_length = _ceil_div(length, stride)
    return (np.arange(out_length, dtype=np.int32) * stride).astype(np.int32)


def _compute_neighbor_indices_1d(
    seqlen: int,
    kernel_size: int,
    dilation: int,
    is_causal: bool,
    query_positions: np.ndarray | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute 1D neighbor indices and validity mask for each query position."""
    if query_positions is None:
        query_positions = np.arange(seqlen, dtype=np.int32)

    qpos = np.asarray(query_positions, dtype=np.int32)
    k_steps = np.arange(kernel_size, dtype=np.int32)

    if is_causal:
        start = qpos - (kernel_size - 1) * dilation
        raw = start[:, None] + k_steps[None, :] * dilation
        valid = (raw >= 0) & (raw <= qpos[:, None]) & (raw < seqlen)
        clipped = np.clip(raw, 0, seqlen - 1).astype(np.int32)
        return mx.array(clipped, dtype=mx.int32), mx.array(valid, dtype=mx.bool_)

    starts, ends = compute_window_start_end(qpos, seqlen, kernel_size, dilation)
    raw = starts[:, None] + k_steps[None, :] * dilation
    valid = (raw >= 0) & (raw < ends[:, None])
    clipped = np.clip(raw, 0, seqlen - 1).astype(np.int32)
    return mx.array(clipped, dtype=mx.int32), mx.array(valid, dtype=mx.bool_)


def _compute_axis_indices(
    query_positions: np.ndarray,
    spatial_size: int,
    kernel_size: int,
    dilation: int,
    is_causal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(query_positions, dtype=np.int32)
    k_steps = np.arange(kernel_size, dtype=np.int32)

    if is_causal:
        start = qpos - (kernel_size - 1) * dilation
        raw = start[:, None] + k_steps[None, :] * dilation
        valid = (raw >= 0) & (raw <= qpos[:, None]) & (raw < spatial_size)
        clipped = np.clip(raw, 0, spatial_size - 1).astype(np.int32)
        return clipped, valid.astype(np.bool_)

    starts, ends = compute_window_start_end(qpos, spatial_size, kernel_size, dilation)
    raw = starts[:, None] + k_steps[None, :] * dilation
    valid = (raw >= 0) & (raw < ends[:, None])
    clipped = np.clip(raw, 0, spatial_size - 1).astype(np.int32)
    return clipped, valid.astype(np.bool_)


def _compute_neighbor_indices_2d(
    height: int,
    width: int,
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
    is_causal: tuple[bool, bool],
    query_h: np.ndarray | None = None,
    query_w: np.ndarray | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute 2D neighbor linear indices and validity mask."""
    if query_h is None:
        query_h = np.arange(height, dtype=np.int32)
    if query_w is None:
        query_w = np.arange(width, dtype=np.int32)

    kh, kw = kernel_size
    dh, dw = dilation
    causal_h, causal_w = is_causal

    h_idx, h_valid = _compute_axis_indices(query_h, height, kh, dh, causal_h)
    w_idx, w_valid = _compute_axis_indices(query_w, width, kw, dw, causal_w)

    lin = (
        h_idx[:, None, :, None].astype(np.int32) * width
        + w_idx[None, :, None, :].astype(np.int32)
    )
    valid = h_valid[:, None, :, None] & w_valid[None, :, None, :]

    lin = lin.reshape(len(query_h), len(query_w), kh * kw)
    valid = valid.reshape(len(query_h), len(query_w), kh * kw)
    return mx.array(lin, dtype=mx.int32), mx.array(valid, dtype=mx.bool_)


def na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    """Fused 1D neighborhood attention."""
    batch, seqlen, heads, head_dim = q.shape
    ksize = kernel_size[0]
    step = stride[0]
    dil = dilation[0]
    causal = is_causal[0]

    if scale is None:
        scale = head_dim ** -0.5

    qpos_np = _query_positions(seqlen, step)
    qpos = mx.array(qpos_np, dtype=mx.int32)
    out_len = len(qpos_np)

    q_sel = mx.take(q, qpos, axis=1)
    indices, valid = _compute_neighbor_indices_1d(
        seqlen, ksize, dil, causal, query_positions=qpos_np
    )

    flat_idx = mx.reshape(indices, (-1,))
    k_neighbors = mx.take(k, flat_idx, axis=1)
    v_neighbors = mx.take(v, flat_idx, axis=1)

    k_neighbors = mx.reshape(k_neighbors, (batch, out_len, ksize, heads, head_dim))
    v_neighbors = mx.reshape(v_neighbors, (batch, out_len, ksize, heads, head_dim))

    k_neighbors = mx.transpose(k_neighbors, axes=(0, 1, 3, 2, 4))
    v_neighbors = mx.transpose(v_neighbors, axes=(0, 1, 3, 2, 4))

    logits = mx.sum(q_sel[..., None, :] * k_neighbors, axis=-1) * float(scale)

    valid_mask = valid[None, :, None, :]
    neg_inf = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
    logits = mx.where(valid_mask, logits, neg_inf)

    weights = mx.softmax(logits, axis=-1)
    return mx.sum(weights[..., None] * v_neighbors, axis=-2)


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    """Fused 2D neighborhood attention."""
    batch, height, width, heads, head_dim = q.shape
    kh, kw = kernel_size
    sh, sw = stride

    if scale is None:
        scale = head_dim ** -0.5

    qh_np = _query_positions(height, sh)
    qw_np = _query_positions(width, sw)
    qh = mx.array(qh_np, dtype=mx.int32)
    qw = mx.array(qw_np, dtype=mx.int32)

    out_h = len(qh_np)
    out_w = len(qw_np)
    kernel_area = kh * kw

    q_rows = mx.take(q, qh, axis=1)
    q_sel = mx.take(q_rows, qw, axis=2)

    indices, valid = _compute_neighbor_indices_2d(
        height,
        width,
        kernel_size,
        dilation,
        is_causal,
        query_h=qh_np,
        query_w=qw_np,
    )

    k_flat = mx.reshape(k, (batch, height * width, heads, head_dim))
    v_flat = mx.reshape(v, (batch, height * width, heads, head_dim))

    flat_idx = mx.reshape(indices, (-1,))
    k_neighbors = mx.take(k_flat, flat_idx, axis=1)
    v_neighbors = mx.take(v_flat, flat_idx, axis=1)

    k_neighbors = mx.reshape(
        k_neighbors, (batch, out_h, out_w, kernel_area, heads, head_dim)
    )
    v_neighbors = mx.reshape(
        v_neighbors, (batch, out_h, out_w, kernel_area, heads, head_dim)
    )

    k_neighbors = mx.transpose(k_neighbors, axes=(0, 1, 2, 4, 3, 5))
    v_neighbors = mx.transpose(v_neighbors, axes=(0, 1, 2, 4, 3, 5))

    logits = mx.sum(q_sel[..., None, :] * k_neighbors, axis=-1) * float(scale)

    valid_mask = valid[None, :, :, None, :]
    neg_inf = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
    logits = mx.where(valid_mask, logits, neg_inf)

    weights = mx.softmax(logits, axis=-1)
    return mx.sum(weights[..., None] * v_neighbors, axis=-2)


def na1d_qk_forward(q, k, kernel_size, dilation):
    """Separate 1D QK operation returning attention logits."""
    batch, seqlen, heads, head_dim = q.shape
    ksize = kernel_size[0]
    dil = dilation[0]

    qpos_np = np.arange(seqlen, dtype=np.int32)
    indices, _ = _compute_neighbor_indices_1d(
        seqlen, ksize, dil, is_causal=False, query_positions=qpos_np
    )

    flat_idx = mx.reshape(indices, (-1,))
    k_neighbors = mx.take(k, flat_idx, axis=1)
    k_neighbors = mx.reshape(k_neighbors, (batch, seqlen, ksize, heads, head_dim))
    k_neighbors = mx.transpose(k_neighbors, axes=(0, 1, 3, 2, 4))

    logits = mx.sum(q[..., None, :] * k_neighbors, axis=-1)
    return logits


def na1d_av_forward(attn, v, kernel_size, dilation):
    """Separate 1D AV operation."""
    batch, out_len, heads, ksize = attn.shape
    _, seqlen, _, head_dim = v.shape

    if ksize != kernel_size[0]:
        raise ValueError(
            f"attn kernel dimension ({ksize}) must match kernel_size ({kernel_size[0]})"
        )

    qpos_np = np.arange(out_len, dtype=np.int32)
    indices, _ = _compute_neighbor_indices_1d(
        seqlen, kernel_size[0], dilation[0], is_causal=False, query_positions=qpos_np
    )

    flat_idx = mx.reshape(indices, (-1,))
    v_neighbors = mx.take(v, flat_idx, axis=1)
    v_neighbors = mx.reshape(v_neighbors, (batch, out_len, ksize, heads, head_dim))
    v_neighbors = mx.transpose(v_neighbors, axes=(0, 1, 3, 2, 4))

    return mx.sum(attn[..., None] * v_neighbors, axis=-2)


def na2d_qk_forward(q, k, kernel_size, dilation):
    """Separate 2D QK operation returning attention logits."""
    batch, height, width, heads, head_dim = q.shape
    kh, kw = kernel_size
    kernel_area = kh * kw

    qh_np = np.arange(height, dtype=np.int32)
    qw_np = np.arange(width, dtype=np.int32)

    indices, _ = _compute_neighbor_indices_2d(
        height,
        width,
        kernel_size,
        dilation,
        is_causal=(False, False),
        query_h=qh_np,
        query_w=qw_np,
    )

    k_flat = mx.reshape(k, (batch, height * width, heads, head_dim))
    flat_idx = mx.reshape(indices, (-1,))
    k_neighbors = mx.take(k_flat, flat_idx, axis=1)
    k_neighbors = mx.reshape(
        k_neighbors, (batch, height, width, kernel_area, heads, head_dim)
    )
    k_neighbors = mx.transpose(k_neighbors, axes=(0, 1, 2, 4, 3, 5))

    logits = mx.sum(q[..., None, :] * k_neighbors, axis=-1)
    return logits


def na2d_av_forward(attn, v, kernel_size, dilation):
    """Separate 2D AV operation."""
    batch, out_h, out_w, heads, kernel_area = attn.shape
    _, height, width, _, head_dim = v.shape

    if kernel_area != kernel_size[0] * kernel_size[1]:
        raise ValueError(
            "attn neighborhood dimension must equal kernel area; "
            f"got {kernel_area} vs {kernel_size[0] * kernel_size[1]}"
        )

    qh_np = np.arange(out_h, dtype=np.int32)
    qw_np = np.arange(out_w, dtype=np.int32)

    indices, _ = _compute_neighbor_indices_2d(
        height,
        width,
        kernel_size,
        dilation,
        is_causal=(False, False),
        query_h=qh_np,
        query_w=qw_np,
    )

    v_flat = mx.reshape(v, (batch, height * width, heads, head_dim))
    flat_idx = mx.reshape(indices, (-1,))
    v_neighbors = mx.take(v_flat, flat_idx, axis=1)
    v_neighbors = mx.reshape(
        v_neighbors, (batch, out_h, out_w, kernel_area, heads, head_dim)
    )
    v_neighbors = mx.transpose(v_neighbors, axes=(0, 1, 2, 4, 3, 5))

    return mx.sum(attn[..., None] * v_neighbors, axis=-2)
