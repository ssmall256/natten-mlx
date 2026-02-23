"""Fused QK+RPB and AV Metal kernels for all-in-one (DiNAT) models.

These provide fused neighborhood attention ops with relative position bias,
optimized for the specific shapes used by DiNAT (k∈{3,5,7}, D=12).

Layout: spatial-first [B, ..., H, D] — transposition to heads-first
is handled internally.

Example usage::

    from natten_mlx.extras.allin1 import na1d_qk_rpb, na1d_av_fused

    logits = na1d_qk_rpb(q, k, rpb, kernel_size=5, dilation=2, scale=0.288)
    attn = mx.softmax(logits, axis=-1)
    out = na1d_av_fused(attn, v, kernel_size=5, dilation=2)
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


def _try_import_metal():
    try:
        from natten_mlx.extras.allin1.functional_metal import (
            _natten1dqkrpb_metal,
            _natten1dav_metal,
            _natten2dqkrpb_metal,
            _natten2dav_metal,
        )
        return _natten1dqkrpb_metal, _natten1dav_metal, _natten2dqkrpb_metal, _natten2dav_metal
    except Exception:
        return None, None, None, None

_metal_1d_qkrpb, _metal_1d_av, _metal_2d_qkrpb, _metal_2d_av = _try_import_metal()


def na1d_qk_rpb(
    query: mx.array,
    key: mx.array,
    rpb: Optional[mx.array],
    kernel_size: int,
    dilation: int = 1,
    *,
    scale: Optional[float] = None,
) -> mx.array:
    """Fused 1D QK + RPB via Metal kernel.

    Layout: spatial-first [B, L, H, D].
    RPB: [H, 2*kernel_size - 1] or None.
    Returns: [B, L, H, K] (attention logits before softmax).
    """
    if query.ndim != 4:
        raise ValueError(f"query must be 4D [B, L, H, D], got shape {query.shape}")

    K = int(kernel_size)
    dil = int(dilation)

    if scale is not None:
        query = query * scale

    H = query.shape[2]
    if rpb is None:
        rpb = mx.zeros((H, 2 * K - 1))

    if _metal_1d_qkrpb is not None:
        q_hf = mx.transpose(query, axes=(0, 2, 1, 3))
        k_hf = mx.transpose(key, axes=(0, 2, 1, 3))
        out_hf = _metal_1d_qkrpb(q_hf, k_hf, rpb, K, dil)
        return mx.transpose(out_hf, axes=(0, 2, 1, 3))

    # Fallback: pure MLX
    from natten_mlx.extras.allin1.functional import _natten1dqkrpb_mlx_fast
    q_hf = mx.transpose(query, axes=(0, 2, 1, 3))
    k_hf = mx.transpose(key, axes=(0, 2, 1, 3))
    out_hf = _natten1dqkrpb_mlx_fast(q_hf, k_hf, rpb, K, dil)
    return mx.transpose(out_hf, axes=(0, 2, 1, 3))


def na1d_av_fused(
    attn: mx.array,
    value: mx.array,
    kernel_size: int,
    dilation: int = 1,
) -> mx.array:
    """Fused 1D AV via Metal kernel.

    Layout: spatial-first [B, L, H, K] for attn, [B, L, H, D] for value.
    Returns: [B, L, H, D].
    """
    if attn.ndim != 4 or value.ndim != 4:
        raise ValueError("attn and value must be 4D for na1d_av_fused")

    K = int(kernel_size)
    dil = int(dilation)

    if _metal_1d_av is not None:
        attn_hf = mx.transpose(attn, axes=(0, 2, 1, 3))
        v_hf = mx.transpose(value, axes=(0, 2, 1, 3))
        out_hf = _metal_1d_av(attn_hf, v_hf, K, dil)
        return mx.transpose(out_hf, axes=(0, 2, 1, 3))

    from natten_mlx.extras.allin1.functional import _natten1dav_mlx_fast
    attn_hf = mx.transpose(attn, axes=(0, 2, 1, 3))
    v_hf = mx.transpose(value, axes=(0, 2, 1, 3))
    out_hf = _natten1dav_mlx_fast(attn_hf, v_hf, K, dil)
    return mx.transpose(out_hf, axes=(0, 2, 1, 3))


def na2d_qk_rpb(
    query: mx.array,
    key: mx.array,
    rpb: Optional[mx.array],
    kernel_size: int,
    dilation: int = 1,
    *,
    scale: Optional[float] = None,
) -> mx.array:
    """Fused 2D QK + RPB via Metal kernel.

    Layout: spatial-first [B, Hh, Hw, H, D].
    RPB: [H, 2*K-1, 2*K-1] or None.
    Returns: [B, Hh, Hw, H, K*K].
    """
    if query.ndim != 5:
        raise ValueError(f"query must be 5D [B, Hh, Hw, H, D], got shape {query.shape}")

    K = int(kernel_size)
    dil = int(dilation)

    if scale is not None:
        query = query * scale

    H = query.shape[3]
    if rpb is None:
        rpb = mx.zeros((H, 2 * K - 1, 2 * K - 1))

    if _metal_2d_qkrpb is not None:
        q_hf = mx.transpose(query, axes=(0, 3, 1, 2, 4))
        k_hf = mx.transpose(key, axes=(0, 3, 1, 2, 4))
        out_hf = _metal_2d_qkrpb(q_hf, k_hf, rpb, K, dil)
        return mx.transpose(out_hf, axes=(0, 2, 3, 1, 4))

    from natten_mlx.extras.allin1.functional import _natten2dqkrpb_mlx_fast
    q_hf = mx.transpose(query, axes=(0, 3, 1, 2, 4))
    k_hf = mx.transpose(key, axes=(0, 3, 1, 2, 4))
    out_hf = _natten2dqkrpb_mlx_fast(q_hf, k_hf, rpb, K, dil)
    return mx.transpose(out_hf, axes=(0, 2, 3, 1, 4))


def na2d_av_fused(
    attn: mx.array,
    value: mx.array,
    kernel_size: int,
    dilation: int = 1,
) -> mx.array:
    """Fused 2D AV via Metal kernel.

    Layout: spatial-first [B, Hh, Hw, H, K*K] for attn, [B, Hh, Hw, H, D] for value.
    Returns: [B, Hh, Hw, H, D].
    """
    if attn.ndim != 5 or value.ndim != 5:
        raise ValueError("attn and value must be 5D for na2d_av_fused")

    K = int(kernel_size)
    dil = int(dilation)

    if _metal_2d_av is not None:
        attn_hf = mx.transpose(attn, axes=(0, 3, 1, 2, 4))
        v_hf = mx.transpose(value, axes=(0, 3, 1, 2, 4))
        out_hf = _metal_2d_av(attn_hf, v_hf, K, dil)
        return mx.transpose(out_hf, axes=(0, 2, 3, 1, 4))

    from natten_mlx.extras.allin1.functional import _natten2dav_mlx_fast
    attn_hf = mx.transpose(attn, axes=(0, 3, 1, 2, 4))
    v_hf = mx.transpose(value, axes=(0, 3, 1, 2, 4))
    out_hf = _natten2dav_mlx_fast(attn_hf, v_hf, K, dil)
    return mx.transpose(out_hf, axes=(0, 2, 3, 1, 4))


__all__ = [
    "na1d_qk_rpb",
    "na1d_av_fused",
    "na2d_qk_rpb",
    "na2d_av_fused",
]
