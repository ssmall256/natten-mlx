"""Tests for variable-length 2D neighborhood attention (MLX)."""

import pytest
import mlx.core as mx
import numpy as np

from natten_mlx.functional import na2d, na2d_varlen


def _varlen_reference(q, k, v, spatial_sizes, kernel_size, dilation, scale):
    """Per-sample reference: slice each batch element and run na2d independently."""
    B = q.shape[0]
    H_max, W_max, heads, D = q.shape[1], q.shape[2], q.shape[3], q.shape[4]
    parts = []
    for b in range(B):
        H_b = int(spatial_sizes[b, 0].item())
        W_b = int(spatial_sizes[b, 1].item())
        out_b = na2d(
            q[b:b+1, :H_b, :W_b], k[b:b+1, :H_b, :W_b], v[b:b+1, :H_b, :W_b],
            kernel_size=kernel_size, dilation=dilation, scale=scale,
        )
        if H_b < H_max or W_b < W_max:
            if W_b < W_max:
                w_pad = mx.zeros((1, H_b, W_max - W_b, heads, D), dtype=q.dtype)
                out_b = mx.concatenate([out_b, w_pad], axis=2)
            if H_b < H_max:
                h_pad = mx.zeros((1, H_max - H_b, W_max, heads, D), dtype=q.dtype)
                out_b = mx.concatenate([out_b, h_pad], axis=1)
        parts.append(out_b)
    return mx.concatenate(parts, axis=0)


def assert_close(a, b, atol=1e-5, rtol=1e-5, msg=""):
    a_np = np.array(a)
    b_np = np.array(b)
    np.testing.assert_allclose(a_np, b_np, atol=atol, rtol=rtol, err_msg=msg)


class TestVarlen2DForward:
    """Forward-pass correctness tests."""

    def test_uniform_sizes(self):
        """All spatial_sizes == (H_max, W_max) must match na2d exactly."""
        B, H, W, heads, D, K = 2, 8, 8, 2, 16, 3
        q = mx.random.normal((B, H, W, heads, D))
        k = mx.random.normal((B, H, W, heads, D))
        v = mx.random.normal((B, H, W, heads, D))
        spatial_sizes = mx.array([[H, W], [H, W]], dtype=mx.int32)

        out_varlen = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        out_na2d = na2d(q, k, v, kernel_size=K)
        assert_close(out_varlen, out_na2d)

    def test_mixed_sizes(self):
        """B=3 with different spatial sizes per sample."""
        B, H_max, W_max, heads, D, K = 3, 12, 12, 2, 16, 3
        q = mx.random.normal((B, H_max, W_max, heads, D))
        k = mx.random.normal((B, H_max, W_max, heads, D))
        v = mx.random.normal((B, H_max, W_max, heads, D))
        spatial_sizes = mx.array([[12, 12], [8, 10], [6, 6]], dtype=mx.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_minimum_sizes(self):
        """All spatial dims == kernel_size (smallest valid)."""
        B, K, heads, D = 2, 3, 2, 8
        H_max, W_max = 12, 12
        q = mx.random.normal((B, H_max, W_max, heads, D))
        k = mx.random.normal((B, H_max, W_max, heads, D))
        v = mx.random.normal((B, H_max, W_max, heads, D))
        spatial_sizes = mx.array([[K, K], [K, K]], dtype=mx.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_single_batch(self):
        """B=1 should work."""
        B, H, W, heads, D, K = 1, 16, 16, 2, 16, 3
        q = mx.random.normal((B, H, W, heads, D))
        k = mx.random.normal((B, H, W, heads, D))
        v = mx.random.normal((B, H, W, heads, D))
        spatial_sizes = mx.array([[10, 12]], dtype=mx.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_per_sample_parity(self):
        """Each slice must match an independent na2d call."""
        B, H_max, W_max, heads, D, K = 3, 16, 16, 2, 16, 3
        q = mx.random.normal((B, H_max, W_max, heads, D))
        k = mx.random.normal((B, H_max, W_max, heads, D))
        v = mx.random.normal((B, H_max, W_max, heads, D))
        spatial_sizes = mx.array([[16, 16], [10, 12], [8, 8]], dtype=mx.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        for b in range(B):
            H_b = int(spatial_sizes[b, 0].item())
            W_b = int(spatial_sizes[b, 1].item())
            expected = na2d(q[b:b+1, :H_b, :W_b], k[b:b+1, :H_b, :W_b], v[b:b+1, :H_b, :W_b], kernel_size=K)
            assert_close(out[b, :H_b, :W_b], expected[0], msg=f"Mismatch at batch {b}")

    def test_dilation(self):
        """dilation=2 must produce correct results."""
        B, H_max, W_max, heads, D, K, dil = 2, 16, 16, 2, 16, 3, 2
        q = mx.random.normal((B, H_max, W_max, heads, D))
        k = mx.random.normal((B, H_max, W_max, heads, D))
        v = mx.random.normal((B, H_max, W_max, heads, D))
        spatial_sizes = mx.array([[16, 16], [10, 12]], dtype=mx.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K, dilation=dil)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, dil, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_padding_positions_zero(self):
        """Output beyond spatial_sizes must be zero."""
        B, H_max, W_max, heads, D, K = 2, 12, 12, 2, 16, 3
        q = mx.random.normal((B, H_max, W_max, heads, D))
        k = mx.random.normal((B, H_max, W_max, heads, D))
        v = mx.random.normal((B, H_max, W_max, heads, D))
        spatial_sizes = mx.array([[8, 6], [4, 4]], dtype=mx.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        out_np = np.array(out)
        assert (out_np[0, 8:] == 0).all(), "Batch 0 row padding should be zero"
        assert (out_np[0, :8, 6:] == 0).all(), "Batch 0 col padding should be zero"
        assert (out_np[1, 4:] == 0).all(), "Batch 1 row padding should be zero"
        assert (out_np[1, :4, 4:] == 0).all(), "Batch 1 col padding should be zero"

    def test_custom_scale(self):
        """Explicit scale parameter should be honored."""
        B, H, W, heads, D, K = 2, 8, 8, 2, 16, 3
        q = mx.random.normal((B, H, W, heads, D))
        k = mx.random.normal((B, H, W, heads, D))
        v = mx.random.normal((B, H, W, heads, D))
        spatial_sizes = mx.array([[8, 6], [6, 8]], dtype=mx.int32)

        out = na2d_varlen(q, k, v, spatial_sizes, kernel_size=K, scale=0.1)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, 0.1)
        assert_close(out, ref)


class TestVarlen2DValidation:
    """Input validation tests."""

    def test_rejects_small_spatial(self):
        B, H, W, heads, D, K = 2, 12, 12, 2, 16, 5
        q = mx.random.normal((B, H, W, heads, D))
        k = mx.random.normal((B, H, W, heads, D))
        v = mx.random.normal((B, H, W, heads, D))
        spatial_sizes = mx.array([[12, 12], [4, 12]], dtype=mx.int32)

        with pytest.raises(ValueError, match="kernel_size"):
            na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_exceeding_max(self):
        B, H, W, heads, D, K = 2, 12, 12, 2, 16, 3
        q = mx.random.normal((B, H, W, heads, D))
        k = mx.random.normal((B, H, W, heads, D))
        v = mx.random.normal((B, H, W, heads, D))
        spatial_sizes = mx.array([[12, 12], [12, 13]], dtype=mx.int32)

        with pytest.raises(ValueError, match="max_spatial"):
            na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_wrong_shape(self):
        B, H, W, heads, D, K = 2, 12, 12, 2, 16, 3
        q = mx.random.normal((B, H, W, heads, D))
        k = mx.random.normal((B, H, W, heads, D))
        v = mx.random.normal((B, H, W, heads, D))
        spatial_sizes = mx.array([8, 8], dtype=mx.int32)

        with pytest.raises(ValueError):
            na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_float_spatial_sizes(self):
        B, H, W, heads, D, K = 2, 12, 12, 2, 16, 3
        q = mx.random.normal((B, H, W, heads, D))
        k = mx.random.normal((B, H, W, heads, D))
        v = mx.random.normal((B, H, W, heads, D))
        spatial_sizes = mx.array([[8.0, 8.0], [6.0, 6.0]], dtype=mx.float32)

        with pytest.raises(ValueError, match="int32 or int64"):
            na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_mismatched_shapes(self):
        B, H, W, heads, D, K = 2, 12, 12, 2, 16, 3
        q = mx.random.normal((B, H, W, heads, D))
        k = mx.random.normal((B, H, W, heads, D + 1))
        v = mx.random.normal((B, H, W, heads, D))
        spatial_sizes = mx.array([[12, 12], [8, 8]], dtype=mx.int32)

        with pytest.raises(ValueError):
            na2d_varlen(q, k, v, spatial_sizes, kernel_size=K)


class TestVarlen2DBackward:
    """Backward gradient tests."""

    def test_backward_gradients(self):
        """Backward gradients match per-sample reference."""
        B, H_max, W_max, heads, D, K = 2, 8, 8, 2, 16, 3
        q = mx.random.normal((B, H_max, W_max, heads, D))
        k = mx.random.normal((B, H_max, W_max, heads, D))
        v = mx.random.normal((B, H_max, W_max, heads, D))
        spatial_sizes = mx.array([[8, 8], [6, 6]], dtype=mx.int32)

        def loss_fn(q, k, v):
            return mx.sum(na2d_varlen(q, k, v, spatial_sizes, kernel_size=K))

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
        dq, dk, dv = grad_fn(q, k, v)

        def ref_loss_b(q_b, k_b, v_b):
            return mx.sum(na2d(q_b, k_b, v_b, kernel_size=K))

        ref_grad = mx.grad(ref_loss_b, argnums=(0, 1, 2))

        for b in range(B):
            H_b = int(spatial_sizes[b, 0].item())
            W_b = int(spatial_sizes[b, 1].item())
            ref_dq, ref_dk, ref_dv = ref_grad(
                q[b:b+1, :H_b, :W_b], k[b:b+1, :H_b, :W_b], v[b:b+1, :H_b, :W_b],
            )
            assert_close(dq[b, :H_b, :W_b], ref_dq[0], atol=1e-4, rtol=1e-4,
                         msg=f"dq mismatch at batch {b}")
            assert_close(dk[b, :H_b, :W_b], ref_dk[0], atol=1e-4, rtol=1e-4,
                         msg=f"dk mismatch at batch {b}")
            assert_close(dv[b, :H_b, :W_b], ref_dv[0], atol=1e-4, rtol=1e-4,
                         msg=f"dv mismatch at batch {b}")

    def test_backward_padding_zero(self):
        """Gradients at padding positions must be zero."""
        B, H_max, W_max, heads, D, K = 2, 12, 12, 2, 16, 3
        q = mx.random.normal((B, H_max, W_max, heads, D))
        k = mx.random.normal((B, H_max, W_max, heads, D))
        v = mx.random.normal((B, H_max, W_max, heads, D))
        spatial_sizes = mx.array([[8, 6], [4, 4]], dtype=mx.int32)

        def loss_fn(q, k, v):
            return mx.sum(na2d_varlen(q, k, v, spatial_sizes, kernel_size=K))

        dq, dk, dv = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        dq_np, dv_np = np.array(dq), np.array(dv)

        assert (dq_np[0, 8:] == 0).all(), "dq row padding should be zero"
        assert (dq_np[0, :8, 6:] == 0).all(), "dq col padding should be zero"
        assert (dv_np[0, 8:] == 0).all(), "dv row padding should be zero"
        assert (dv_np[0, :8, 6:] == 0).all(), "dv col padding should be zero"
