"""Tests for variable-length 1D neighborhood attention (MLX)."""

import pytest
import mlx.core as mx
import numpy as np

from natten_mlx.functional import na1d, na1d_varlen


def _varlen_reference(q, k, v, seq_lens, kernel_size, dilation, scale):
    """Per-sample reference: slice each batch element and run na1d independently."""
    B = q.shape[0]
    L_max, H, D = q.shape[1], q.shape[2], q.shape[3]
    parts = []
    for b in range(B):
        L = int(seq_lens[b].item())
        out_b = na1d(
            q[b:b+1, :L], k[b:b+1, :L], v[b:b+1, :L],
            kernel_size=kernel_size, dilation=dilation, scale=scale,
        )
        if L < L_max:
            pad = mx.zeros((1, L_max - L, H, D), dtype=q.dtype)
            out_b = mx.concatenate([out_b, pad], axis=1)
        parts.append(out_b)
    return mx.concatenate(parts, axis=0)


def assert_close(a, b, atol=1e-5, rtol=1e-5, msg=""):
    """Helper to compare MLX arrays."""
    a_np = np.array(a)
    b_np = np.array(b)
    np.testing.assert_allclose(a_np, b_np, atol=atol, rtol=rtol, err_msg=msg)


class TestVarlenForward:
    """Forward-pass correctness tests."""

    def test_uniform_lengths(self):
        """All seq_lens == L_max must match na1d exactly."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = mx.random.normal((B, L, H, D))
        k = mx.random.normal((B, L, H, D))
        v = mx.random.normal((B, L, H, D))
        seq_lens = mx.array([L, L], dtype=mx.int32)

        out_varlen = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        out_na1d = na1d(q, k, v, kernel_size=K)
        assert_close(out_varlen, out_na1d)

    def test_mixed_lengths(self):
        """B=4 with different lengths per sample."""
        B, L_max, H, D, K = 4, 32, 4, 16, 7
        q = mx.random.normal((B, L_max, H, D))
        k = mx.random.normal((B, L_max, H, D))
        v = mx.random.normal((B, L_max, H, D))
        seq_lens = mx.array([32, 24, 16, 8], dtype=mx.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        ref = _varlen_reference(q, k, v, seq_lens, K, 1, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_minimum_lengths(self):
        """All seq_lens == kernel_size (smallest valid)."""
        B, K, H, D = 3, 7, 2, 8
        L_max = 32
        q = mx.random.normal((B, L_max, H, D))
        k = mx.random.normal((B, L_max, H, D))
        v = mx.random.normal((B, L_max, H, D))
        seq_lens = mx.array([K, K, K], dtype=mx.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        ref = _varlen_reference(q, k, v, seq_lens, K, 1, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_single_batch(self):
        """B=1 should work."""
        B, L, H, D, K = 1, 64, 4, 16, 7
        q = mx.random.normal((B, L, H, D))
        k = mx.random.normal((B, L, H, D))
        v = mx.random.normal((B, L, H, D))
        seq_lens = mx.array([48], dtype=mx.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        ref = _varlen_reference(q, k, v, seq_lens, K, 1, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_per_sample_parity(self):
        """Each slice must match an independent na1d call."""
        B, L_max, H, D, K = 3, 64, 4, 16, 7
        q = mx.random.normal((B, L_max, H, D))
        k = mx.random.normal((B, L_max, H, D))
        v = mx.random.normal((B, L_max, H, D))
        seq_lens = mx.array([64, 32, 16], dtype=mx.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        for b in range(B):
            L = int(seq_lens[b].item())
            expected = na1d(q[b:b+1, :L], k[b:b+1, :L], v[b:b+1, :L], kernel_size=K)
            assert_close(out[b, :L], expected[0], msg=f"Mismatch at batch {b}")

    def test_dilation(self):
        """dilation=2 must produce correct results."""
        B, L_max, H, D, K, dil = 2, 64, 4, 16, 7, 2
        q = mx.random.normal((B, L_max, H, D))
        k = mx.random.normal((B, L_max, H, D))
        v = mx.random.normal((B, L_max, H, D))
        seq_lens = mx.array([64, 32], dtype=mx.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K, dilation=dil)
        ref = _varlen_reference(q, k, v, seq_lens, K, dil, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_large_kernel(self):
        """K=15, L_max=16 — kernel nearly covers entire sequence."""
        B, L_max, H, D, K = 2, 16, 2, 8, 15
        q = mx.random.normal((B, L_max, H, D))
        k = mx.random.normal((B, L_max, H, D))
        v = mx.random.normal((B, L_max, H, D))
        seq_lens = mx.array([16, 15], dtype=mx.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        ref = _varlen_reference(q, k, v, seq_lens, K, 1, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_padding_positions_zero(self):
        """Output beyond seq_lens[b] must be zero."""
        B, L_max, H, D, K = 2, 32, 4, 16, 7
        q = mx.random.normal((B, L_max, H, D))
        k = mx.random.normal((B, L_max, H, D))
        v = mx.random.normal((B, L_max, H, D))
        seq_lens = mx.array([16, 8], dtype=mx.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K)
        out_np = np.array(out)
        assert (out_np[0, 16:] == 0).all(), "Batch 0 padding should be zero"
        assert (out_np[1, 8:] == 0).all(), "Batch 1 padding should be zero"

    def test_custom_scale(self):
        """Explicit scale parameter should be honored."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = mx.random.normal((B, L, H, D))
        k = mx.random.normal((B, L, H, D))
        v = mx.random.normal((B, L, H, D))
        seq_lens = mx.array([24, 16], dtype=mx.int32)

        out = na1d_varlen(q, k, v, seq_lens, kernel_size=K, scale=0.1)
        ref = _varlen_reference(q, k, v, seq_lens, K, 1, 0.1)
        assert_close(out, ref)


class TestVarlenValidation:
    """Input validation tests."""

    def test_rejects_short_seqlens(self):
        """seq_len < kernel_size should raise ValueError."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = mx.random.normal((B, L, H, D))
        k = mx.random.normal((B, L, H, D))
        v = mx.random.normal((B, L, H, D))
        seq_lens = mx.array([6, 32], dtype=mx.int32)  # 6 < K=7

        with pytest.raises(ValueError, match="kernel_size"):
            na1d_varlen(q, k, v, seq_lens, kernel_size=K)

    def test_rejects_exceeding_lmax(self):
        """seq_len > L_max should raise ValueError."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = mx.random.normal((B, L, H, D))
        k = mx.random.normal((B, L, H, D))
        v = mx.random.normal((B, L, H, D))
        seq_lens = mx.array([32, 33], dtype=mx.int32)  # 33 > L_max=32

        with pytest.raises(ValueError, match="L_max"):
            na1d_varlen(q, k, v, seq_lens, kernel_size=K)

    def test_rejects_wrong_seq_lens_shape(self):
        """seq_lens shape must be (B,)."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = mx.random.normal((B, L, H, D))
        k = mx.random.normal((B, L, H, D))
        v = mx.random.normal((B, L, H, D))
        seq_lens = mx.array([[32, 32]], dtype=mx.int32)

        with pytest.raises(ValueError):
            na1d_varlen(q, k, v, seq_lens, kernel_size=K)

    def test_rejects_float_seq_lens(self):
        """seq_lens must be integer dtype."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = mx.random.normal((B, L, H, D))
        k = mx.random.normal((B, L, H, D))
        v = mx.random.normal((B, L, H, D))
        seq_lens = mx.array([32.0, 16.0], dtype=mx.float32)

        with pytest.raises(ValueError, match="int32 or int64"):
            na1d_varlen(q, k, v, seq_lens, kernel_size=K)

    def test_rejects_mismatched_shapes(self):
        """Q/K/V must have identical shapes."""
        B, L, H, D, K = 2, 32, 4, 16, 7
        q = mx.random.normal((B, L, H, D))
        k = mx.random.normal((B, L, H, D + 1))
        v = mx.random.normal((B, L, H, D))
        seq_lens = mx.array([32, 16], dtype=mx.int32)

        with pytest.raises(ValueError):
            na1d_varlen(q, k, v, seq_lens, kernel_size=K)


class TestVarlenBackward:
    """Backward gradient tests."""

    def test_backward_gradients(self):
        """Backward gradients match per-sample reference."""
        B, L_max, H, D, K = 2, 32, 4, 16, 7
        q = mx.random.normal((B, L_max, H, D))
        k = mx.random.normal((B, L_max, H, D))
        v = mx.random.normal((B, L_max, H, D))
        seq_lens = mx.array([32, 16], dtype=mx.int32)

        def loss_fn(q, k, v):
            return mx.sum(na1d_varlen(q, k, v, seq_lens, kernel_size=K))

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
        dq, dk, dv = grad_fn(q, k, v)

        # Reference: per-sample grads
        def ref_loss_b(q_b, k_b, v_b):
            return mx.sum(na1d(q_b, k_b, v_b, kernel_size=K))

        ref_grad = mx.grad(ref_loss_b, argnums=(0, 1, 2))

        for b in range(B):
            L = int(seq_lens[b].item())
            ref_dq, ref_dk, ref_dv = ref_grad(
                q[b:b+1, :L], k[b:b+1, :L], v[b:b+1, :L],
            )
            assert_close(dq[b, :L], ref_dq[0], atol=1e-4, rtol=1e-4,
                         msg=f"dq mismatch at batch {b}")
            assert_close(dk[b, :L], ref_dk[0], atol=1e-4, rtol=1e-4,
                         msg=f"dk mismatch at batch {b}")
            assert_close(dv[b, :L], ref_dv[0], atol=1e-4, rtol=1e-4,
                         msg=f"dv mismatch at batch {b}")

    def test_backward_padding_zero(self):
        """Gradients at padding positions must be zero."""
        B, L_max, H, D, K = 2, 32, 4, 16, 7
        q = mx.random.normal((B, L_max, H, D))
        k = mx.random.normal((B, L_max, H, D))
        v = mx.random.normal((B, L_max, H, D))
        seq_lens = mx.array([16, 8], dtype=mx.int32)

        def loss_fn(q, k, v):
            return mx.sum(na1d_varlen(q, k, v, seq_lens, kernel_size=K))

        dq, dk, dv = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        dq_np, dk_np, dv_np = np.array(dq), np.array(dk), np.array(dv)

        assert (dq_np[0, 16:] == 0).all(), "dq padding should be zero"
        assert (dq_np[1, 8:] == 0).all(), "dq padding should be zero"
        assert (dv_np[0, 16:] == 0).all(), "dv padding should be zero"
        assert (dv_np[1, 8:] == 0).all(), "dv padding should be zero"
