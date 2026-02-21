"""NATTEN v0.14.x API surface for MLX tensors."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from natten_mlx.functional import na1d, na1d_av, na1d_qk, na2d, na2d_av, na2d_qk
from natten_mlx.utils.params import (
    check_dilation_kernel_vs_input,
    check_kernel_size_vs_input,
    normalize_tuple_param,
)
from natten_mlx.utils.window import compute_pb_start


def _rpb_indices_1d(length: int, kernel_size: int, dilation: int) -> np.ndarray:
    """RPB gather indices [L, K] using NATTEN pb_start coupling."""
    qpos = np.arange(length, dtype=np.int32)
    pb = compute_pb_start(qpos, length, kernel_size, dilation)
    idx = pb[:, None] + np.arange(kernel_size, dtype=np.int32)[None, :]
    return np.clip(idx, 0, 2 * kernel_size - 2).astype(np.int32)


def _rpb_indices_2d(
    height: int,
    width: int,
    kernel_size: tuple[int, int],
    dilation: tuple[int, int],
) -> np.ndarray:
    """RPB flattened gather indices [H, W, Kh*Kw] using pb_start coupling."""
    kh, kw = kernel_size
    dh, dw = dilation

    i_pb = compute_pb_start(np.arange(height, dtype=np.int32), height, kh, dh)
    j_pb = compute_pb_start(np.arange(width, dtype=np.int32), width, kw, dw)

    i_idx = i_pb[:, None] + np.arange(kh, dtype=np.int32)[None, :]
    j_idx = j_pb[:, None] + np.arange(kw, dtype=np.int32)[None, :]

    i_idx = np.clip(i_idx, 0, 2 * kh - 2)
    j_idx = np.clip(j_idx, 0, 2 * kw - 2)

    idx_h = np.broadcast_to(i_idx[:, None, :, None], (height, width, kh, kw))
    idx_w = np.broadcast_to(j_idx[None, :, None, :], (height, width, kh, kw))

    pair = idx_h * (2 * kw - 1) + idx_w
    return pair.reshape(height, width, kh * kw).astype(np.int32)


class NeighborhoodAttention1D(nn.Module):
    """v0.14-style 1D module: uses ``dim``, no stride, no causal."""

    def __init__(
        self,
        dim,
        kernel_size,
        dilation=1,
        num_heads=1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = normalize_tuple_param(kernel_size, 1, "kernel_size")
        self.dilation = normalize_tuple_param(dilation, 1, "dilation")
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop_rate = float(attn_drop)
        self.attn_drop = nn.Dropout(self.attn_drop_rate) if self.attn_drop_rate > 0.0 else None
        self.proj_drop_rate = float(proj_drop)
        self.proj_drop = nn.Dropout(self.proj_drop_rate) if self.proj_drop_rate > 0.0 else None

    def __call__(self, x):
        batch, length, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got channels={channels}")

        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=2)
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)

        if self.attn_drop_rate > 0.0:
            logits = na1d_qk(q, k, kernel_size=self.kernel_size, dilation=self.dilation)
            default_scale = self.head_dim ** -0.5
            if self.scale != default_scale:
                logits = logits * (self.scale / default_scale)
            attn = mx.softmax(logits, axis=-1)
            if self.attn_drop is not None:
                attn = self.attn_drop(attn)
            out = na1d_av(attn, v, kernel_size=self.kernel_size, dilation=self.dilation)
        else:
            out = na1d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                stride=(1,),
                dilation=self.dilation,
                is_causal=(False,),
                scale=self.scale,
            )
        out = out.reshape(batch, out.shape[1], channels)
        out = self.proj(out)
        if self.proj_drop is not None:
            out = self.proj_drop(out)
        return out


class NeighborhoodAttention2D(nn.Module):
    """v0.14-style 2D module: uses ``dim``, no stride, no causal."""

    def __init__(
        self,
        dim,
        kernel_size,
        dilation=1,
        num_heads=1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = normalize_tuple_param(kernel_size, 2, "kernel_size")
        self.dilation = normalize_tuple_param(dilation, 2, "dilation")
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop_rate = float(attn_drop)
        self.attn_drop = nn.Dropout(self.attn_drop_rate) if self.attn_drop_rate > 0.0 else None
        self.proj_drop_rate = float(proj_drop)
        self.proj_drop = nn.Dropout(self.proj_drop_rate) if self.proj_drop_rate > 0.0 else None

    def __call__(self, x):
        batch, height, width, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got channels={channels}")

        qkv = self.qkv(x).reshape(batch, height, width, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=3)
        q = q.squeeze(3)
        k = k.squeeze(3)
        v = v.squeeze(3)

        if self.attn_drop_rate > 0.0:
            logits = na2d_qk(q, k, kernel_size=self.kernel_size, dilation=self.dilation)
            default_scale = self.head_dim ** -0.5
            if self.scale != default_scale:
                logits = logits * (self.scale / default_scale)
            attn = mx.softmax(logits, axis=-1)
            if self.attn_drop is not None:
                attn = self.attn_drop(attn)
            out = na2d_av(attn, v, kernel_size=self.kernel_size, dilation=self.dilation)
        else:
            out = na2d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                stride=(1, 1),
                dilation=self.dilation,
                is_causal=(False, False),
                scale=self.scale,
            )
        out = out.reshape(batch, out.shape[1], out.shape[2], channels)
        out = self.proj(out)
        if self.proj_drop is not None:
            out = self.proj_drop(out)
        return out


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    """v0.14-style 1D QK with RPB.

    query/key: [B, H, L, D]
    rpb: [H, 2*K-1]
    returns: [B, H, L, K]
    """
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must be [B, H, L, D]")

    ks = normalize_tuple_param(kernel_size, 1, "kernel_size")
    dil = normalize_tuple_param(dilation, 1, "dilation")
    ksize = ks[0]
    length = int(query.shape[2])

    check_kernel_size_vs_input(ks, (length,))
    check_dilation_kernel_vs_input(dil, ks, (length,))

    if rpb.ndim != 2 or int(rpb.shape[1]) != 2 * ksize - 1:
        raise ValueError(
            f"rpb must be [heads, {2 * ksize - 1}] for kernel_size={ksize}, got {rpb.shape}"
        )

    q = mx.transpose(query, axes=(0, 2, 1, 3))
    k = mx.transpose(key, axes=(0, 2, 1, 3))

    logits = na1d_qk(q, k, ks, dil)

    rpb_index = _rpb_indices_1d(length, ksize, dil[0])
    flat_idx = mx.array(rpb_index.reshape(-1), dtype=mx.int32)

    bias = mx.take(rpb, flat_idx, axis=1)
    bias = mx.reshape(bias, (rpb.shape[0], length, ksize))
    bias = mx.transpose(bias, axes=(1, 0, 2))

    logits = logits + bias[None, :, :, :]
    return mx.transpose(logits, axes=(0, 2, 1, 3))


def natten1dav(attn, value, kernel_size, dilation):
    """v0.14-style 1D AV.

    attn: [B, H, L, K]
    value: [B, H, L, D]
    returns: [B, H, L, D]
    """
    a = mx.transpose(attn, axes=(0, 2, 1, 3))
    v = mx.transpose(value, axes=(0, 2, 1, 3))

    ks = normalize_tuple_param(kernel_size, 1, "kernel_size")
    dil = normalize_tuple_param(dilation, 1, "dilation")

    check_kernel_size_vs_input(ks, (int(v.shape[1]),))
    check_dilation_kernel_vs_input(dil, ks, (int(v.shape[1]),))

    out = na1d_av(a, v, ks, dil)
    return mx.transpose(out, axes=(0, 2, 1, 3))


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    """v0.14-style 2D QK+RPB.

    query/key: [B, H, Hh, Hw, D]
    rpb: [H, 2*Kh-1, 2*Kw-1]
    returns: [B, H, Hh, Hw, Kh*Kw]
    """
    if query.ndim != 5 or key.ndim != 5:
        raise ValueError("query and key must be [B, H, Hh, Hw, D]")

    ks = normalize_tuple_param(kernel_size, 2, "kernel_size")
    dil = normalize_tuple_param(dilation, 2, "dilation")
    kh, kw = ks

    height = int(query.shape[2])
    width = int(query.shape[3])

    check_kernel_size_vs_input(ks, (height, width))
    check_dilation_kernel_vs_input(dil, ks, (height, width))

    if rpb.ndim != 3 or int(rpb.shape[1]) != 2 * kh - 1 or int(rpb.shape[2]) != 2 * kw - 1:
        raise ValueError(
            "rpb must be [heads, 2*Kh-1, 2*Kw-1] for the provided kernel_size; "
            f"got {rpb.shape}"
        )

    q = mx.transpose(query, axes=(0, 2, 3, 1, 4))
    k = mx.transpose(key, axes=(0, 2, 3, 1, 4))

    logits = na2d_qk(q, k, ks, dil)

    pair_idx = _rpb_indices_2d(height, width, ks, dil)
    rpb_flat = mx.reshape(rpb, (rpb.shape[0], (2 * kh - 1) * (2 * kw - 1)))

    flat_idx = mx.array(pair_idx.reshape(-1), dtype=mx.int32)
    bias = mx.take(rpb_flat, flat_idx, axis=1)
    bias = mx.reshape(bias, (rpb.shape[0], height, width, kh * kw))
    bias = mx.transpose(bias, axes=(1, 2, 0, 3))

    logits = logits + bias[None, :, :, :, :]
    return mx.transpose(logits, axes=(0, 3, 1, 2, 4))


def natten2dav(attn, value, kernel_size, dilation):
    """v0.14-style 2D AV.

    attn: [B, H, Hh, Hw, Kh*Kw]
    value: [B, H, Hh, Hw, D]
    returns: [B, H, Hh, Hw, D]
    """
    a = mx.transpose(attn, axes=(0, 2, 3, 1, 4))
    v = mx.transpose(value, axes=(0, 2, 3, 1, 4))

    ks = normalize_tuple_param(kernel_size, 2, "kernel_size")
    dil = normalize_tuple_param(dilation, 2, "dilation")

    check_kernel_size_vs_input(ks, (int(v.shape[1]), int(v.shape[2])))
    check_dilation_kernel_vs_input(dil, ks, (int(v.shape[1]), int(v.shape[2])))

    out = na2d_av(a, v, ks, dil)
    return mx.transpose(out, axes=(0, 3, 1, 2, 4))


def add_natten_handle(flop_counter):
    """Compatibility stub for fvcore flop counting."""
    return flop_counter


__all__ = [
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "natten1dqkrpb",
    "natten1dav",
    "natten2dqkrpb",
    "natten2dav",
    "add_natten_handle",
]
