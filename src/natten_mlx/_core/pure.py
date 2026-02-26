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


def _compute_neighbor_indices_3d(
    depth: int,
    height: int,
    width: int,
    kernel_size: tuple[int, int, int],
    dilation: tuple[int, int, int],
    is_causal: tuple[bool, bool, bool],
    query_d: np.ndarray | None = None,
    query_h: np.ndarray | None = None,
    query_w: np.ndarray | None = None,
) -> tuple[mx.array, mx.array]:
    """Compute 3D neighbor linear indices and validity mask."""
    if query_d is None:
        query_d = np.arange(depth, dtype=np.int32)
    if query_h is None:
        query_h = np.arange(height, dtype=np.int32)
    if query_w is None:
        query_w = np.arange(width, dtype=np.int32)

    kd, kh, kw = kernel_size
    dd, dh, dw = dilation
    causal_d, causal_h, causal_w = is_causal

    d_idx, d_valid = _compute_axis_indices(query_d, depth, kd, dd, causal_d)
    h_idx, h_valid = _compute_axis_indices(query_h, height, kh, dh, causal_h)
    w_idx, w_valid = _compute_axis_indices(query_w, width, kw, dw, causal_w)

    lin = (
        (
            d_idx[:, None, None, :, None, None].astype(np.int32) * height
            + h_idx[None, :, None, None, :, None].astype(np.int32)
        )
        * width
        + w_idx[None, None, :, None, None, :].astype(np.int32)
    )
    valid = (
        d_valid[:, None, None, :, None, None]
        & h_valid[None, :, None, None, :, None]
        & w_valid[None, None, :, None, None, :]
    )

    lin = lin.reshape(len(query_d), len(query_h), len(query_w), kd * kh * kw)
    valid = valid.reshape(len(query_d), len(query_h), len(query_w), kd * kh * kw)
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


def na3d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    """Fused 3D neighborhood attention."""
    logits = na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale)
    weights = mx.softmax(logits, axis=-1)
    return na3d_av_forward(weights, v, kernel_size, stride, dilation, is_causal)


def na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    """Separate 1D QK operation returning attention logits."""
    batch, seqlen, heads, head_dim = q.shape
    ksize = kernel_size[0]
    step = stride[0]
    dil = dilation[0]
    causal = bool(is_causal[0])
    out_len = _ceil_div(seqlen, step)
    scale_value = head_dim ** -0.5 if scale is None else float(scale)

    qpos_np = _query_positions(seqlen, step)
    qpos = mx.array(qpos_np, dtype=mx.int32)
    q_sel = mx.take(q, qpos, axis=1)
    indices, valid = _compute_neighbor_indices_1d(
        seqlen, ksize, dil, is_causal=causal, query_positions=qpos_np
    )

    flat_idx = mx.reshape(indices, (-1,))
    k_neighbors = mx.take(k, flat_idx, axis=1)
    k_neighbors = mx.reshape(k_neighbors, (batch, out_len, ksize, heads, head_dim))
    k_neighbors = mx.transpose(k_neighbors, axes=(0, 1, 3, 2, 4))

    logits = mx.sum(q_sel[..., None, :] * k_neighbors, axis=-1) * scale_value
    valid_mask = valid[None, :, None, :]
    neg_inf = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
    return mx.where(valid_mask, logits, neg_inf)


def na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    """Separate 1D AV operation."""
    batch, out_len, heads, ksize = attn.shape
    _, seqlen, _, head_dim = v.shape

    if ksize != kernel_size[0]:
        raise ValueError(
            f"attn kernel dimension ({ksize}) must match kernel_size ({kernel_size[0]})"
        )

    qpos_np = _query_positions(seqlen, stride[0])
    if out_len != len(qpos_np):
        raise ValueError(
            f"attn length ({out_len}) must match ceil(input_len/stride) ({len(qpos_np)})"
        )

    indices, valid = _compute_neighbor_indices_1d(
        seqlen,
        kernel_size[0],
        dilation[0],
        is_causal=bool(is_causal[0]),
        query_positions=qpos_np,
    )

    flat_idx = mx.reshape(indices, (-1,))
    v_neighbors = mx.take(v, flat_idx, axis=1)
    v_neighbors = mx.reshape(v_neighbors, (batch, out_len, ksize, heads, head_dim))
    v_neighbors = mx.transpose(v_neighbors, axes=(0, 1, 3, 2, 4))

    valid_mask = valid[None, :, None, :]
    masked_attn = mx.where(valid_mask, attn, mx.zeros(attn.shape, dtype=attn.dtype))
    return mx.sum(masked_attn[..., None] * v_neighbors, axis=-2)


def na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    """Separate 2D QK operation returning attention logits."""
    batch, height, width, heads, head_dim = q.shape
    kh, kw = kernel_size
    sh, sw = stride
    kernel_area = kh * kw
    scale_value = head_dim ** -0.5 if scale is None else float(scale)

    qh_np = _query_positions(height, sh)
    qw_np = _query_positions(width, sw)
    qh = mx.array(qh_np, dtype=mx.int32)
    qw = mx.array(qw_np, dtype=mx.int32)
    out_h = len(qh_np)
    out_w = len(qw_np)

    q_rows = mx.take(q, qh, axis=1)
    q_sel = mx.take(q_rows, qw, axis=2)

    indices, valid = _compute_neighbor_indices_2d(
        height,
        width,
        kernel_size,
        dilation,
        is_causal=(bool(is_causal[0]), bool(is_causal[1])),
        query_h=qh_np,
        query_w=qw_np,
    )

    k_flat = mx.reshape(k, (batch, height * width, heads, head_dim))
    flat_idx = mx.reshape(indices, (-1,))
    k_neighbors = mx.take(k_flat, flat_idx, axis=1)
    k_neighbors = mx.reshape(
        k_neighbors, (batch, out_h, out_w, kernel_area, heads, head_dim)
    )
    k_neighbors = mx.transpose(k_neighbors, axes=(0, 1, 2, 4, 3, 5))

    logits = mx.sum(q_sel[..., None, :] * k_neighbors, axis=-1) * scale_value
    valid_mask = valid[None, :, :, None, :]
    neg_inf = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
    return mx.where(valid_mask, logits, neg_inf)


def na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    """Separate 2D AV operation."""
    batch, out_h, out_w, heads, kernel_area = attn.shape
    _, height, width, _, head_dim = v.shape

    if kernel_area != kernel_size[0] * kernel_size[1]:
        raise ValueError(
            "attn neighborhood dimension must equal kernel area; "
            f"got {kernel_area} vs {kernel_size[0] * kernel_size[1]}"
        )

    qh_np = _query_positions(height, stride[0])
    qw_np = _query_positions(width, stride[1])
    if out_h != len(qh_np) or out_w != len(qw_np):
        raise ValueError(
            "attn spatial dimensions must match ceil(input/stride); "
            f"got ({out_h}, {out_w}) vs ({len(qh_np)}, {len(qw_np)})"
        )

    indices, valid = _compute_neighbor_indices_2d(
        height,
        width,
        kernel_size,
        dilation,
        is_causal=(bool(is_causal[0]), bool(is_causal[1])),
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

    valid_mask = valid[None, :, :, None, :]
    masked_attn = mx.where(valid_mask, attn, mx.zeros(attn.shape, dtype=attn.dtype))
    return mx.sum(masked_attn[..., None] * v_neighbors, axis=-2)


def na3d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    """Separate 3D QK operation returning attention logits."""
    batch, depth, height, width, heads, head_dim = q.shape
    kd, kh, kw = kernel_size
    sd, sh, sw = stride
    kernel_volume = kd * kh * kw
    scale_value = head_dim ** -0.5 if scale is None else float(scale)

    qd_np = _query_positions(depth, sd)
    qh_np = _query_positions(height, sh)
    qw_np = _query_positions(width, sw)
    qd = mx.array(qd_np, dtype=mx.int32)
    qh = mx.array(qh_np, dtype=mx.int32)
    qw = mx.array(qw_np, dtype=mx.int32)
    out_d = len(qd_np)
    out_h = len(qh_np)
    out_w = len(qw_np)

    q_depth = mx.take(q, qd, axis=1)
    q_rows = mx.take(q_depth, qh, axis=2)
    q_sel = mx.take(q_rows, qw, axis=3)

    indices, valid = _compute_neighbor_indices_3d(
        depth,
        height,
        width,
        kernel_size,
        dilation,
        is_causal=(bool(is_causal[0]), bool(is_causal[1]), bool(is_causal[2])),
        query_d=qd_np,
        query_h=qh_np,
        query_w=qw_np,
    )

    k_flat = mx.reshape(k, (batch, depth * height * width, heads, head_dim))
    flat_idx = mx.reshape(indices, (-1,))
    k_neighbors = mx.take(k_flat, flat_idx, axis=1)
    k_neighbors = mx.reshape(
        k_neighbors, (batch, out_d, out_h, out_w, kernel_volume, heads, head_dim)
    )
    k_neighbors = mx.transpose(k_neighbors, axes=(0, 1, 2, 3, 5, 4, 6))

    logits = mx.sum(q_sel[..., None, :] * k_neighbors, axis=-1) * scale_value
    valid_mask = valid[None, :, :, :, None, :]
    neg_inf = mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
    return mx.where(valid_mask, logits, neg_inf)


def na3d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    """Separate 3D AV operation."""
    batch, out_d, out_h, out_w, heads, kernel_volume = attn.shape
    _, depth, height, width, _, head_dim = v.shape

    if kernel_volume != kernel_size[0] * kernel_size[1] * kernel_size[2]:
        raise ValueError(
            "attn neighborhood dimension must equal kernel volume; "
            f"got {kernel_volume} vs {kernel_size[0] * kernel_size[1] * kernel_size[2]}"
        )

    qd_np = _query_positions(depth, stride[0])
    qh_np = _query_positions(height, stride[1])
    qw_np = _query_positions(width, stride[2])
    if out_d != len(qd_np) or out_h != len(qh_np) or out_w != len(qw_np):
        raise ValueError(
            "attn spatial dimensions must match ceil(input/stride); "
            f"got ({out_d}, {out_h}, {out_w}) vs ({len(qd_np)}, {len(qh_np)}, {len(qw_np)})"
        )

    indices, valid = _compute_neighbor_indices_3d(
        depth,
        height,
        width,
        kernel_size,
        dilation,
        is_causal=(bool(is_causal[0]), bool(is_causal[1]), bool(is_causal[2])),
        query_d=qd_np,
        query_h=qh_np,
        query_w=qw_np,
    )

    v_flat = mx.reshape(v, (batch, depth * height * width, heads, head_dim))
    flat_idx = mx.reshape(indices, (-1,))
    v_neighbors = mx.take(v_flat, flat_idx, axis=1)
    v_neighbors = mx.reshape(
        v_neighbors, (batch, out_d, out_h, out_w, kernel_volume, heads, head_dim)
    )
    v_neighbors = mx.transpose(v_neighbors, axes=(0, 1, 2, 3, 5, 4, 6))

    valid_mask = valid[None, :, :, :, None, :]
    masked_attn = mx.where(valid_mask, attn, mx.zeros(attn.shape, dtype=attn.dtype))
    return mx.sum(masked_attn[..., None] * v_neighbors, axis=-2)


def na1d_varlen_forward(q, k, v, seq_lens, kernel_size, dilation, scale):
    """Variable-length 1D NA: per-sample loop over existing na1d_forward."""
    batch, l_max, heads, head_dim = q.shape
    stride = (1,)
    is_causal = (False,)

    if scale is None:
        scale = head_dim ** -0.5

    parts = []
    for b in range(batch):
        L_b = int(seq_lens[b].item())
        out_b = na1d_forward(
            q[b:b+1, :L_b], k[b:b+1, :L_b], v[b:b+1, :L_b],
            kernel_size, stride, dilation, is_causal, scale,
        )
        if L_b < l_max:
            pad = mx.zeros((1, l_max - L_b, heads, head_dim), dtype=q.dtype)
            out_b = mx.concatenate([out_b, pad], axis=1)
        parts.append(out_b)
    return mx.concatenate(parts, axis=0)


def na1d_varlen_backward(q, k, v, grad_out, seq_lens, kernel_size, dilation, scale):
    """Variable-length 1D NA backward: per-sample autodiff."""
    batch, l_max, heads, head_dim = q.shape
    stride = (1,)
    is_causal = (False,)

    if scale is None:
        scale = head_dim ** -0.5

    dq_parts, dk_parts, dv_parts = [], [], []
    for b in range(batch):
        L_b = int(seq_lens[b].item())
        dq_b, dk_b, dv_b = na1d_backward(
            q[b:b+1, :L_b], k[b:b+1, :L_b], v[b:b+1, :L_b],
            grad_out[b:b+1, :L_b],
            kernel_size, stride, dilation, is_causal, scale,
        )
        if L_b < l_max:
            pad = mx.zeros((1, l_max - L_b, heads, head_dim), dtype=q.dtype)
            dq_b = mx.concatenate([dq_b, pad], axis=1)
            dk_b = mx.concatenate([dk_b, pad], axis=1)
            dv_b = mx.concatenate([dv_b, pad], axis=1)
        dq_parts.append(dq_b)
        dk_parts.append(dk_b)
        dv_parts.append(dv_b)
    return mx.concatenate(dq_parts, axis=0), mx.concatenate(dk_parts, axis=0), mx.concatenate(dv_parts, axis=0)


def na2d_varlen_forward(q, k, v, spatial_sizes, kernel_size, dilation, scale):
    """Variable-length 2D NA: per-sample loop over existing na2d_forward."""
    batch, h_max, w_max, heads, head_dim = q.shape
    stride = (1, 1)
    is_causal = (False, False)

    if scale is None:
        scale = head_dim ** -0.5

    parts = []
    for b in range(batch):
        H_b = int(spatial_sizes[b, 0].item())
        W_b = int(spatial_sizes[b, 1].item())
        out_b = na2d_forward(
            q[b:b+1, :H_b, :W_b], k[b:b+1, :H_b, :W_b], v[b:b+1, :H_b, :W_b],
            kernel_size, stride, dilation, is_causal, scale,
        )
        if H_b < h_max or W_b < w_max:
            padded = mx.zeros((1, h_max, w_max, heads, head_dim), dtype=q.dtype)
            # Can't do slice assignment in MLX, so build via concatenation
            if W_b < w_max:
                w_pad = mx.zeros((1, H_b, w_max - W_b, heads, head_dim), dtype=q.dtype)
                out_b = mx.concatenate([out_b, w_pad], axis=2)
            if H_b < h_max:
                h_pad = mx.zeros((1, h_max - H_b, w_max, heads, head_dim), dtype=q.dtype)
                out_b = mx.concatenate([out_b, h_pad], axis=1)
        parts.append(out_b)
    return mx.concatenate(parts, axis=0)


def na2d_varlen_backward(q, k, v, grad_out, spatial_sizes, kernel_size, dilation, scale):
    """Variable-length 2D NA backward: per-sample autodiff."""
    batch, h_max, w_max, heads, head_dim = q.shape
    stride = (1, 1)
    is_causal = (False, False)

    if scale is None:
        scale = head_dim ** -0.5

    dq_parts, dk_parts, dv_parts = [], [], []
    for b in range(batch):
        H_b = int(spatial_sizes[b, 0].item())
        W_b = int(spatial_sizes[b, 1].item())
        dq_b, dk_b, dv_b = na2d_backward(
            q[b:b+1, :H_b, :W_b], k[b:b+1, :H_b, :W_b], v[b:b+1, :H_b, :W_b],
            grad_out[b:b+1, :H_b, :W_b],
            kernel_size, stride, dilation, is_causal, scale,
        )
        if H_b < h_max or W_b < w_max:
            if W_b < w_max:
                w_pad = mx.zeros((1, H_b, w_max - W_b, heads, head_dim), dtype=q.dtype)
                dq_b = mx.concatenate([dq_b, w_pad], axis=2)
                dk_b = mx.concatenate([dk_b, w_pad], axis=2)
                dv_b = mx.concatenate([dv_b, w_pad], axis=2)
            if H_b < h_max:
                h_pad = mx.zeros((1, h_max - H_b, w_max, heads, head_dim), dtype=q.dtype)
                dq_b = mx.concatenate([dq_b, h_pad], axis=1)
                dk_b = mx.concatenate([dk_b, h_pad], axis=1)
                dv_b = mx.concatenate([dv_b, h_pad], axis=1)
        dq_parts.append(dq_b)
        dk_parts.append(dk_b)
        dv_parts.append(dv_b)
    return mx.concatenate(dq_parts, axis=0), mx.concatenate(dk_parts, axis=0), mx.concatenate(dv_parts, axis=0)


def na3d_varlen_forward(q, k, v, spatial_sizes, kernel_size, dilation, scale):
    """Variable-length 3D NA: per-sample loop over existing na3d_forward."""
    batch, d_max, h_max, w_max, heads, head_dim = q.shape
    stride = (1, 1, 1)
    is_causal = (False, False, False)

    if scale is None:
        scale = head_dim ** -0.5

    parts = []
    for b in range(batch):
        D_b = int(spatial_sizes[b, 0].item())
        H_b = int(spatial_sizes[b, 1].item())
        W_b = int(spatial_sizes[b, 2].item())
        out_b = na3d_forward(
            q[b:b+1, :D_b, :H_b, :W_b], k[b:b+1, :D_b, :H_b, :W_b], v[b:b+1, :D_b, :H_b, :W_b],
            kernel_size, stride, dilation, is_causal, scale,
        )
        # Pad back to full size
        if W_b < w_max:
            w_pad = mx.zeros((1, D_b, H_b, w_max - W_b, heads, head_dim), dtype=q.dtype)
            out_b = mx.concatenate([out_b, w_pad], axis=3)
        if H_b < h_max:
            h_pad = mx.zeros((1, D_b, h_max - H_b, w_max, heads, head_dim), dtype=q.dtype)
            out_b = mx.concatenate([out_b, h_pad], axis=2)
        if D_b < d_max:
            d_pad = mx.zeros((1, d_max - D_b, h_max, w_max, heads, head_dim), dtype=q.dtype)
            out_b = mx.concatenate([out_b, d_pad], axis=1)
        parts.append(out_b)
    return mx.concatenate(parts, axis=0)


def na3d_varlen_backward(q, k, v, grad_out, spatial_sizes, kernel_size, dilation, scale):
    """Variable-length 3D NA backward: per-sample autodiff."""
    batch, d_max, h_max, w_max, heads, head_dim = q.shape
    stride = (1, 1, 1)
    is_causal = (False, False, False)

    if scale is None:
        scale = head_dim ** -0.5

    dq_parts, dk_parts, dv_parts = [], [], []
    for b in range(batch):
        D_b = int(spatial_sizes[b, 0].item())
        H_b = int(spatial_sizes[b, 1].item())
        W_b = int(spatial_sizes[b, 2].item())
        dq_b, dk_b, dv_b = na3d_backward(
            q[b:b+1, :D_b, :H_b, :W_b], k[b:b+1, :D_b, :H_b, :W_b], v[b:b+1, :D_b, :H_b, :W_b],
            grad_out[b:b+1, :D_b, :H_b, :W_b],
            kernel_size, stride, dilation, is_causal, scale,
        )
        if W_b < w_max:
            w_pad = mx.zeros((1, D_b, H_b, w_max - W_b, heads, head_dim), dtype=q.dtype)
            dq_b = mx.concatenate([dq_b, w_pad], axis=3)
            dk_b = mx.concatenate([dk_b, w_pad], axis=3)
            dv_b = mx.concatenate([dv_b, w_pad], axis=3)
        if H_b < h_max:
            h_pad = mx.zeros((1, D_b, h_max - H_b, w_max, heads, head_dim), dtype=q.dtype)
            dq_b = mx.concatenate([dq_b, h_pad], axis=2)
            dk_b = mx.concatenate([dk_b, h_pad], axis=2)
            dv_b = mx.concatenate([dv_b, h_pad], axis=2)
        if D_b < d_max:
            d_pad = mx.zeros((1, d_max - D_b, h_max, w_max, heads, head_dim), dtype=q.dtype)
            dq_b = mx.concatenate([dq_b, d_pad], axis=1)
            dk_b = mx.concatenate([dk_b, d_pad], axis=1)
            dv_b = mx.concatenate([dv_b, d_pad], axis=1)
        dq_parts.append(dq_b)
        dk_parts.append(dk_b)
        dv_parts.append(dv_b)
    return mx.concatenate(dq_parts, axis=0), mx.concatenate(dk_parts, axis=0), mx.concatenate(dv_parts, axis=0)


def na1d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
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


def na2d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
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


def na3d_backward(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale):
    def _loss_q(q_in):
        out = na3d_forward(q_in, k, v, kernel_size, stride, dilation, is_causal, scale)
        return mx.sum(out * grad_out)

    def _loss_k(k_in):
        out = na3d_forward(q, k_in, v, kernel_size, stride, dilation, is_causal, scale)
        return mx.sum(out * grad_out)

    def _loss_v(v_in):
        out = na3d_forward(q, k, v_in, kernel_size, stride, dilation, is_causal, scale)
        return mx.sum(out * grad_out)

    return mx.grad(_loss_q)(q), mx.grad(_loss_k)(k), mx.grad(_loss_v)(v)


def na1d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    def _loss_q(q_in):
        out = na1d_qk_forward(q_in, k, kernel_size, stride, dilation, is_causal, scale)
        return mx.sum(out * grad_attn)

    def _loss_k(k_in):
        out = na1d_qk_forward(q, k_in, kernel_size, stride, dilation, is_causal, scale)
        return mx.sum(out * grad_attn)

    return mx.grad(_loss_q)(q), mx.grad(_loss_k)(k)


def na1d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    def _loss_attn(attn_in):
        out = na1d_av_forward(attn_in, v, kernel_size, stride, dilation, is_causal)
        return mx.sum(out * grad_out)

    def _loss_v(v_in):
        out = na1d_av_forward(attn, v_in, kernel_size, stride, dilation, is_causal)
        return mx.sum(out * grad_out)

    return mx.grad(_loss_attn)(attn), mx.grad(_loss_v)(v)


def na2d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    def _loss_q(q_in):
        out = na2d_qk_forward(q_in, k, kernel_size, stride, dilation, is_causal, scale)
        return mx.sum(out * grad_attn)

    def _loss_k(k_in):
        out = na2d_qk_forward(q, k_in, kernel_size, stride, dilation, is_causal, scale)
        return mx.sum(out * grad_attn)

    return mx.grad(_loss_q)(q), mx.grad(_loss_k)(k)


def na2d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    def _loss_attn(attn_in):
        out = na2d_av_forward(attn_in, v, kernel_size, stride, dilation, is_causal)
        return mx.sum(out * grad_out)

    def _loss_v(v_in):
        out = na2d_av_forward(attn, v_in, kernel_size, stride, dilation, is_causal)
        return mx.sum(out * grad_out)

    return mx.grad(_loss_attn)(attn), mx.grad(_loss_v)(v)


def na3d_qk_backward(q, k, grad_attn, kernel_size, stride, dilation, is_causal, scale):
    def _loss_q(q_in):
        out = na3d_qk_forward(q_in, k, kernel_size, stride, dilation, is_causal, scale)
        return mx.sum(out * grad_attn)

    def _loss_k(k_in):
        out = na3d_qk_forward(q, k_in, kernel_size, stride, dilation, is_causal, scale)
        return mx.sum(out * grad_attn)

    return mx.grad(_loss_q)(q), mx.grad(_loss_k)(k)


def na3d_av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal):
    def _loss_attn(attn_in):
        out = na3d_av_forward(attn_in, v, kernel_size, stride, dilation, is_causal)
        return mx.sum(out * grad_out)

    def _loss_v(v_in):
        out = na3d_av_forward(attn, v_in, kernel_size, stride, dilation, is_causal)
        return mx.sum(out * grad_out)

    return mx.grad(_loss_attn)(attn), mx.grad(_loss_v)(v)
