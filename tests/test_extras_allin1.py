"""Tests for natten_mlx.extras.allin1 fused Metal kernels.

Verifies that the fused QK+RPB and AV functions produce results matching
the decomposed na1d_qk/na1d_av + manual RPB gather reference path.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.extras.allin1 import (
    na1d_qk_rpb,
    na1d_av_fused,
    na2d_qk_rpb,
    na2d_av_fused,
)
from natten_mlx.functional import na1d_qk, na1d_av, na2d_qk, na2d_av
from natten_mlx.utils.window import compute_pb_start


# ---------------------------------------------------------------------------
# Reference RPB index builders (same logic as original DiNAT code)
# ---------------------------------------------------------------------------

def _rpb_indices_1d(length, kernel_size, dilation):
    qpos = np.arange(length, dtype=np.int32)
    pb = compute_pb_start(qpos, length, kernel_size, dilation)
    idx = pb[:, None] + np.arange(kernel_size, dtype=np.int32)[None, :]
    idx = np.clip(idx, 0, 2 * kernel_size - 2).astype(np.int32)
    return mx.array(idx)


def _rpb_indices_2d(height, width, kernel_size, dilation):
    kh = kw = kernel_size
    dh = dw = dilation
    i_pb = compute_pb_start(np.arange(height, dtype=np.int32), height, kh, dh)
    j_pb = compute_pb_start(np.arange(width, dtype=np.int32), width, kw, dw)
    i_idx = np.clip(i_pb[:, None] + np.arange(kh, dtype=np.int32)[None, :], 0, 2 * kh - 2)
    j_idx = np.clip(j_pb[:, None] + np.arange(kw, dtype=np.int32)[None, :], 0, 2 * kw - 2)
    idx_h = np.broadcast_to(i_idx[:, None, :, None], (height, width, kh, kw))
    idx_w = np.broadcast_to(j_idx[None, :, None, :], (height, width, kh, kw))
    pair = (idx_h * (2 * kw - 1) + idx_w).reshape(height, width, kh * kw).astype(np.int32)
    return mx.array(pair)


def _reference_1d_qk_rpb(q, k, rpb, kernel_size, dilation, scale):
    """Decomposed reference: na1d_qk + manual RPB gather."""
    logits = na1d_qk(q, k, kernel_size, dilation, scale=scale)
    if rpb is not None:
        L = q.shape[1]
        idx = _rpb_indices_1d(L, kernel_size, dilation)
        bias = rpb[:, idx]  # [H, L, K]
        logits = logits + mx.transpose(bias, axes=(1, 0, 2))[None, :, :, :]
    return logits


def _reference_2d_qk_rpb(q, k, rpb, kernel_size, dilation, scale):
    """Decomposed reference: na2d_qk + manual RPB gather."""
    logits = na2d_qk(q, k, kernel_size, dilation, scale=scale)
    if rpb is not None:
        Hh, Hw = q.shape[1], q.shape[2]
        kk = kernel_size
        rpb_flat = mx.reshape(rpb, (rpb.shape[0], (2 * kk - 1) * (2 * kk - 1)))
        idx = _rpb_indices_2d(Hh, Hw, kernel_size, dilation)
        bias = rpb_flat[:, idx]  # [H, Hh, Hw, K*K]
        logits = logits + mx.transpose(bias, axes=(1, 2, 0, 3))[None, :, :, :, :]
    return logits


# ---------------------------------------------------------------------------
# 1D tests
# ---------------------------------------------------------------------------

class TestExtrasAllin1_1D:
    """1D fused kernel correctness tests."""

    @pytest.mark.parametrize("dilation", [1, 2, 4, 8, 16, 32])
    def test_qk_rpb_matches_reference(self, dilation):
        B, L, H, D, K = 4, 216, 2, 12, 5
        if (K - 1) * dilation >= L:
            pytest.skip(f"dilation={dilation} too large for L={L}")

        mx.random.seed(42)
        q = mx.random.normal((B, L, H, D))
        k = mx.random.normal((B, L, H, D))
        rpb = mx.random.normal((H, 2 * K - 1))
        scale = D ** -0.5

        fused = na1d_qk_rpb(q, k, rpb, K, dilation, scale=scale)
        ref = _reference_1d_qk_rpb(q, k, rpb, K, dilation, scale)
        mx.eval(fused, ref)

        diff = mx.abs(fused - ref).max().item()
        assert diff < 1e-4, f"1D QK+RPB dilation={dilation}: max diff {diff}"

    @pytest.mark.parametrize("dilation", [1, 2, 4, 8, 16, 32])
    def test_av_matches_reference(self, dilation):
        B, L, H, D, K = 4, 216, 2, 12, 5
        if (K - 1) * dilation >= L:
            pytest.skip(f"dilation={dilation} too large for L={L}")

        mx.random.seed(42)
        # Generate valid attention weights
        logits = mx.random.normal((B, L, H, K))
        attn = mx.softmax(logits, axis=-1)
        v = mx.random.normal((B, L, H, D))

        fused = na1d_av_fused(attn, v, K, dilation)
        ref = na1d_av(attn, v, K, dilation)
        mx.eval(fused, ref)

        diff = mx.abs(fused - ref).max().item()
        assert diff < 1e-4, f"1D AV dilation={dilation}: max diff {diff}"

    def test_qk_rpb_none(self):
        """RPB=None should still work (QK only, no bias)."""
        B, L, H, D, K = 2, 64, 2, 12, 5
        mx.random.seed(42)
        q = mx.random.normal((B, L, H, D))
        k = mx.random.normal((B, L, H, D))

        fused = na1d_qk_rpb(q, k, None, K, 1, scale=D ** -0.5)
        ref = na1d_qk(q, k, K, 1, scale=D ** -0.5)
        mx.eval(fused, ref)

        diff = mx.abs(fused - ref).max().item()
        assert diff < 1e-4, f"1D QK rpb=None: max diff {diff}"

    def test_output_shapes(self):
        B, L, H, D, K = 2, 32, 2, 12, 5
        mx.random.seed(42)
        q = mx.random.normal((B, L, H, D))
        k = mx.random.normal((B, L, H, D))
        rpb = mx.random.normal((H, 2 * K - 1))

        logits = na1d_qk_rpb(q, k, rpb, K, 1, scale=0.288)
        assert logits.shape == (B, L, H, K)

        attn = mx.softmax(logits, axis=-1)
        v = mx.random.normal((B, L, H, D))
        out = na1d_av_fused(attn, v, K, 1)
        assert out.shape == (B, L, H, D)


# ---------------------------------------------------------------------------
# 2D tests
# ---------------------------------------------------------------------------

class TestExtrasAllin1_2D:
    """2D fused kernel correctness tests."""

    def test_qk_rpb_matches_reference(self):
        B, Hh, Hw, H, D, K = 4, 8, 108, 2, 12, 5
        mx.random.seed(42)
        q = mx.random.normal((B, Hh, Hw, H, D))
        k = mx.random.normal((B, Hh, Hw, H, D))
        rpb = mx.random.normal((H, 2 * K - 1, 2 * K - 1))
        scale = D ** -0.5

        fused = na2d_qk_rpb(q, k, rpb, K, 1, scale=scale)
        ref = _reference_2d_qk_rpb(q, k, rpb, K, 1, scale)
        mx.eval(fused, ref)

        diff = mx.abs(fused - ref).max().item()
        assert diff < 1e-4, f"2D QK+RPB: max diff {diff}"

    def test_av_matches_reference(self):
        B, Hh, Hw, H, D, K = 4, 8, 108, 2, 12, 5
        mx.random.seed(42)
        logits = mx.random.normal((B, Hh, Hw, H, K * K))
        attn = mx.softmax(logits, axis=-1)
        v = mx.random.normal((B, Hh, Hw, H, D))

        fused = na2d_av_fused(attn, v, K, 1)
        ref = na2d_av(attn, v, K, 1)
        mx.eval(fused, ref)

        diff = mx.abs(fused - ref).max().item()
        assert diff < 1e-4, f"2D AV: max diff {diff}"

    def test_qk_rpb_none(self):
        B, Hh, Hw, H, D, K = 2, 8, 32, 2, 12, 5
        mx.random.seed(42)
        q = mx.random.normal((B, Hh, Hw, H, D))
        k = mx.random.normal((B, Hh, Hw, H, D))

        fused = na2d_qk_rpb(q, k, None, K, 1, scale=D ** -0.5)
        ref = na2d_qk(q, k, K, 1, scale=D ** -0.5)
        mx.eval(fused, ref)

        diff = mx.abs(fused - ref).max().item()
        assert diff < 1e-4, f"2D QK rpb=None: max diff {diff}"

    def test_output_shapes(self):
        B, Hh, Hw, H, D, K = 2, 4, 16, 2, 12, 5
        mx.random.seed(42)
        q = mx.random.normal((B, Hh, Hw, H, D))
        k = mx.random.normal((B, Hh, Hw, H, D))
        rpb = mx.random.normal((H, 2 * K - 1, 2 * K - 1))

        logits = na2d_qk_rpb(q, k, rpb, K, 1, scale=0.288)
        assert logits.shape == (B, Hh, Hw, H, K * K)

        attn = mx.softmax(logits, axis=-1)
        v = mx.random.normal((B, Hh, Hw, H, D))
        out = na2d_av_fused(attn, v, K, 1)
        assert out.shape == (B, Hh, Hw, H, D)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestExtrasAllin1_Errors:
    """Input validation tests."""

    def test_1d_wrong_ndim(self):
        with pytest.raises(ValueError, match="4D"):
            na1d_qk_rpb(mx.zeros((2, 3)), mx.zeros((2, 3)), None, 3)

    def test_2d_wrong_ndim(self):
        with pytest.raises(ValueError, match="5D"):
            na2d_qk_rpb(mx.zeros((2, 3, 4)), mx.zeros((2, 3, 4)), None, 3)

    def test_1d_av_wrong_ndim(self):
        with pytest.raises(ValueError, match="4D"):
            na1d_av_fused(mx.zeros((2, 3)), mx.zeros((2, 3)), 3)

    def test_2d_av_wrong_ndim(self):
        with pytest.raises(ValueError, match="5D"):
            na2d_av_fused(mx.zeros((2, 3, 4)), mx.zeros((2, 3, 4)), 3)
