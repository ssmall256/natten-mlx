"""Tests for variable-length 3D neighborhood attention (MLX)."""

import pytest
import mlx.core as mx
import numpy as np

from natten_mlx.functional import na3d, na3d_varlen


def _varlen_reference(q, k, v, spatial_sizes, kernel_size, dilation, scale):
    """Per-sample reference: slice each batch element and run na3d independently."""
    B = q.shape[0]
    d_max, h_max, w_max, heads, D = q.shape[1], q.shape[2], q.shape[3], q.shape[4], q.shape[5]
    parts = []
    for b in range(B):
        D_b = int(spatial_sizes[b, 0].item())
        H_b = int(spatial_sizes[b, 1].item())
        W_b = int(spatial_sizes[b, 2].item())
        out_b = na3d(
            q[b:b+1, :D_b, :H_b, :W_b], k[b:b+1, :D_b, :H_b, :W_b], v[b:b+1, :D_b, :H_b, :W_b],
            kernel_size=kernel_size, dilation=dilation, scale=scale,
        )
        if W_b < w_max:
            w_pad = mx.zeros((1, D_b, H_b, w_max - W_b, heads, D), dtype=q.dtype)
            out_b = mx.concatenate([out_b, w_pad], axis=3)
        if H_b < h_max:
            h_pad = mx.zeros((1, D_b, h_max - H_b, w_max, heads, D), dtype=q.dtype)
            out_b = mx.concatenate([out_b, h_pad], axis=2)
        if D_b < d_max:
            d_pad = mx.zeros((1, d_max - D_b, h_max, w_max, heads, D), dtype=q.dtype)
            out_b = mx.concatenate([out_b, d_pad], axis=1)
        parts.append(out_b)
    return mx.concatenate(parts, axis=0)


def assert_close(a, b, atol=1e-5, rtol=1e-5, msg=""):
    a_np = np.array(a)
    b_np = np.array(b)
    np.testing.assert_allclose(a_np, b_np, atol=atol, rtol=rtol, err_msg=msg)


class TestVarlen3DForward:
    """Forward-pass correctness tests."""

    def test_uniform_sizes(self):
        B, D, H, W, heads, dim, K = 2, 4, 4, 4, 2, 8, 3
        q = mx.random.normal((B, D, H, W, heads, dim))
        k = mx.random.normal((B, D, H, W, heads, dim))
        v = mx.random.normal((B, D, H, W, heads, dim))
        spatial_sizes = mx.array([[D, H, W], [D, H, W]], dtype=mx.int32)

        out_varlen = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        out_na3d = na3d(q, k, v, kernel_size=K)
        assert_close(out_varlen, out_na3d)

    def test_mixed_sizes(self):
        B, D_max, H_max, W_max, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        k = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        v = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        spatial_sizes = mx.array([[6, 6, 6], [4, 5, 3]], dtype=mx.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_minimum_sizes(self):
        B, K, heads, dim = 2, 3, 2, 8
        D_max, H_max, W_max = 6, 6, 6
        q = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        k = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        v = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        spatial_sizes = mx.array([[K, K, K], [K, K, K]], dtype=mx.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_single_batch(self):
        B, D, H, W, heads, dim, K = 1, 6, 6, 6, 2, 8, 3
        q = mx.random.normal((B, D, H, W, heads, dim))
        k = mx.random.normal((B, D, H, W, heads, dim))
        v = mx.random.normal((B, D, H, W, heads, dim))
        spatial_sizes = mx.array([[4, 5, 3]], dtype=mx.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, q.shape[-1] ** -0.5)
        assert_close(out, ref)

    def test_per_sample_parity(self):
        B, D_max, H_max, W_max, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        k = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        v = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        spatial_sizes = mx.array([[6, 6, 6], [3, 4, 5]], dtype=mx.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        for b in range(B):
            D_b, H_b, W_b = [int(spatial_sizes[b, d].item()) for d in range(3)]
            expected = na3d(q[b:b+1, :D_b, :H_b, :W_b], k[b:b+1, :D_b, :H_b, :W_b],
                           v[b:b+1, :D_b, :H_b, :W_b], kernel_size=K)
            assert_close(out[b, :D_b, :H_b, :W_b], expected[0], msg=f"Mismatch at batch {b}")

    def test_padding_positions_zero(self):
        B, D_max, H_max, W_max, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        k = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        v = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        spatial_sizes = mx.array([[4, 3, 5], [3, 3, 3]], dtype=mx.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)
        out_np = np.array(out)
        assert (out_np[0, 4:] == 0).all(), "Batch 0 depth padding should be zero"
        assert (out_np[0, :4, 3:] == 0).all(), "Batch 0 height padding should be zero"
        assert (out_np[0, :4, :3, 5:] == 0).all(), "Batch 0 width padding should be zero"

    def test_custom_scale(self):
        B, D, H, W, heads, dim, K = 2, 4, 4, 4, 2, 8, 3
        q = mx.random.normal((B, D, H, W, heads, dim))
        k = mx.random.normal((B, D, H, W, heads, dim))
        v = mx.random.normal((B, D, H, W, heads, dim))
        spatial_sizes = mx.array([[4, 4, 4], [3, 3, 3]], dtype=mx.int32)

        out = na3d_varlen(q, k, v, spatial_sizes, kernel_size=K, scale=0.1)
        ref = _varlen_reference(q, k, v, spatial_sizes, K, 1, 0.1)
        assert_close(out, ref)


class TestVarlen3DValidation:

    def test_rejects_small_spatial(self):
        B, D, H, W, heads, dim, K = 2, 6, 6, 6, 2, 8, 5
        q = mx.random.normal((B, D, H, W, heads, dim))
        k = mx.random.normal((B, D, H, W, heads, dim))
        v = mx.random.normal((B, D, H, W, heads, dim))
        spatial_sizes = mx.array([[6, 6, 6], [4, 6, 6]], dtype=mx.int32)

        with pytest.raises(ValueError, match="kernel_size"):
            na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_exceeding_max(self):
        B, D, H, W, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = mx.random.normal((B, D, H, W, heads, dim))
        k = mx.random.normal((B, D, H, W, heads, dim))
        v = mx.random.normal((B, D, H, W, heads, dim))
        spatial_sizes = mx.array([[6, 6, 6], [6, 7, 6]], dtype=mx.int32)

        with pytest.raises(ValueError, match="max_spatial"):
            na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)

    def test_rejects_wrong_shape(self):
        B, D, H, W, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = mx.random.normal((B, D, H, W, heads, dim))
        k = mx.random.normal((B, D, H, W, heads, dim))
        v = mx.random.normal((B, D, H, W, heads, dim))
        spatial_sizes = mx.array([[6, 6], [6, 6]], dtype=mx.int32)

        with pytest.raises(ValueError):
            na3d_varlen(q, k, v, spatial_sizes, kernel_size=K)


class TestVarlen3DBackward:

    def test_backward_gradients(self):
        B, D_max, H_max, W_max, heads, dim, K = 2, 4, 4, 4, 2, 8, 3
        q = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        k = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        v = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        spatial_sizes = mx.array([[4, 4, 4], [3, 3, 3]], dtype=mx.int32)

        def loss_fn(q, k, v):
            return mx.sum(na3d_varlen(q, k, v, spatial_sizes, kernel_size=K))

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
        dq, dk, dv = grad_fn(q, k, v)

        def ref_loss_b(q_b, k_b, v_b):
            return mx.sum(na3d(q_b, k_b, v_b, kernel_size=K))

        ref_grad = mx.grad(ref_loss_b, argnums=(0, 1, 2))

        for b in range(B):
            D_b, H_b, W_b = [int(spatial_sizes[b, d].item()) for d in range(3)]
            ref_dq, ref_dk, ref_dv = ref_grad(
                q[b:b+1, :D_b, :H_b, :W_b], k[b:b+1, :D_b, :H_b, :W_b], v[b:b+1, :D_b, :H_b, :W_b],
            )
            assert_close(dq[b, :D_b, :H_b, :W_b], ref_dq[0], atol=1e-4, rtol=1e-4,
                         msg=f"dq mismatch at batch {b}")
            assert_close(dk[b, :D_b, :H_b, :W_b], ref_dk[0], atol=1e-4, rtol=1e-4,
                         msg=f"dk mismatch at batch {b}")
            assert_close(dv[b, :D_b, :H_b, :W_b], ref_dv[0], atol=1e-4, rtol=1e-4,
                         msg=f"dv mismatch at batch {b}")

    def test_backward_padding_zero(self):
        B, D_max, H_max, W_max, heads, dim, K = 2, 6, 6, 6, 2, 8, 3
        q = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        k = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        v = mx.random.normal((B, D_max, H_max, W_max, heads, dim))
        spatial_sizes = mx.array([[4, 3, 5], [3, 3, 3]], dtype=mx.int32)

        def loss_fn(q, k, v):
            return mx.sum(na3d_varlen(q, k, v, spatial_sizes, kernel_size=K))

        dq, dk, dv = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        dq_np, dv_np = np.array(dq), np.array(dv)

        assert (dq_np[0, 4:] == 0).all(), "dq depth padding should be zero"
        assert (dq_np[0, :4, 3:] == 0).all(), "dq height padding should be zero"
        assert (dv_np[1, 3:] == 0).all(), "dv depth padding should be zero"
