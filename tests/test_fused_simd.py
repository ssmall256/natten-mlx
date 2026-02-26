"""Tests for fused SIMD cooperative kernels in natten-mlx."""

import mlx.core as mx
import numpy as np
import pytest

from natten_mlx._core import fast_metal, pure


def _skip_if_unavailable():
    if not fast_metal.is_available():
        pytest.skip("fast_metal not available")


@pytest.mark.parametrize("D", [32, 64, 128])
def test_1d_basic(D):
    _skip_if_unavailable()
    np.random.seed(42)
    B, L, H, K = 2, 32, 4, 7
    q = mx.array(np.random.randn(B, L, H, D).astype(np.float32))
    k = mx.array(np.random.randn(B, L, H, D).astype(np.float32))
    v = mx.array(np.random.randn(B, L, H, D).astype(np.float32))

    ks = (K,)
    dil = (1,)
    stride = (1,)
    causal = (False,)
    scale = float(D ** -0.5)

    out_ref = pure.na1d_forward(q, k, v, ks, stride, dil, causal, scale)
    out_fused, lse = fast_metal.na1d_fused_simd_forward(q, k, v, ks, stride, dil, causal, scale)

    mx.eval(out_ref, out_fused)
    np.testing.assert_allclose(
        np.array(out_ref), np.array(out_fused), atol=1e-4, rtol=1e-3
    )


@pytest.mark.parametrize("D", [32, 64])
def test_1d_causal(D):
    _skip_if_unavailable()
    np.random.seed(42)
    B, L, H, K = 1, 16, 2, 5
    q = mx.array(np.random.randn(B, L, H, D).astype(np.float32))
    k = mx.array(np.random.randn(B, L, H, D).astype(np.float32))
    v = mx.array(np.random.randn(B, L, H, D).astype(np.float32))

    ks = (K,)
    dil = (1,)
    stride = (1,)
    causal = (True,)
    scale = float(D ** -0.5)

    out_ref = pure.na1d_forward(q, k, v, ks, stride, dil, causal, scale)
    out_fused, lse = fast_metal.na1d_fused_simd_forward(q, k, v, ks, stride, dil, causal, scale)

    mx.eval(out_ref, out_fused)
    np.testing.assert_allclose(
        np.array(out_ref), np.array(out_fused), atol=1e-4, rtol=1e-3
    )


def test_1d_strided():
    _skip_if_unavailable()
    np.random.seed(42)
    B, L, H, D, K = 1, 32, 2, 64, 7
    q = mx.array(np.random.randn(B, L, H, D).astype(np.float32))
    k = mx.array(np.random.randn(B, L, H, D).astype(np.float32))
    v = mx.array(np.random.randn(B, L, H, D).astype(np.float32))

    ks = (K,)
    dil = (1,)
    stride = (2,)
    causal = (False,)
    scale = float(D ** -0.5)

    out_ref = pure.na1d_forward(q, k, v, ks, stride, dil, causal, scale)
    out_fused, lse = fast_metal.na1d_fused_simd_forward(q, k, v, ks, stride, dil, causal, scale)

    mx.eval(out_ref, out_fused)
    np.testing.assert_allclose(
        np.array(out_ref), np.array(out_fused), atol=1e-4, rtol=1e-3
    )


@pytest.mark.parametrize("D", [32, 64])
def test_2d_basic(D):
    _skip_if_unavailable()
    np.random.seed(42)
    B, Hi, Wi, H = 1, 8, 8, 2
    Kh = 3
    q = mx.array(np.random.randn(B, Hi, Wi, H, D).astype(np.float32))
    k = mx.array(np.random.randn(B, Hi, Wi, H, D).astype(np.float32))
    v = mx.array(np.random.randn(B, Hi, Wi, H, D).astype(np.float32))

    ks = (Kh, Kh)
    dil = (1, 1)
    stride = (1, 1)
    causal = (False, False)
    scale = float(D ** -0.5)

    out_ref = pure.na2d_forward(q, k, v, ks, stride, dil, causal, scale)
    out_fused, lse = fast_metal.na2d_fused_simd_forward(q, k, v, ks, stride, dil, causal, scale)

    mx.eval(out_ref, out_fused)
    np.testing.assert_allclose(
        np.array(out_ref), np.array(out_fused), atol=1e-4, rtol=1e-3
    )


@pytest.mark.parametrize("D", [32, 64])
def test_3d_basic(D):
    _skip_if_unavailable()
    np.random.seed(42)
    B, Dp, Hi, Wi, H = 1, 4, 4, 4, 2
    Kd = 3
    q = mx.array(np.random.randn(B, Dp, Hi, Wi, H, D).astype(np.float32))
    k = mx.array(np.random.randn(B, Dp, Hi, Wi, H, D).astype(np.float32))
    v = mx.array(np.random.randn(B, Dp, Hi, Wi, H, D).astype(np.float32))

    ks = (Kd, Kd, Kd)
    dil = (1, 1, 1)
    stride = (1, 1, 1)
    causal = (False, False, False)
    scale = float(D ** -0.5)

    out_ref = pure.na3d_forward(q, k, v, ks, stride, dil, causal, scale)
    out_fused, lse = fast_metal.na3d_fused_simd_forward(q, k, v, ks, stride, dil, causal, scale)

    mx.eval(out_ref, out_fused)
    np.testing.assert_allclose(
        np.array(out_ref), np.array(out_fused), atol=1e-4, rtol=1e-3
    )


def test_1d_integration():
    """Verify na1d_forward auto-selects fused SIMD path."""
    _skip_if_unavailable()
    np.random.seed(42)
    B, L, H, D, K = 2, 32, 4, 64, 7
    q = mx.array(np.random.randn(B, L, H, D).astype(np.float32))
    k = mx.array(np.random.randn(B, L, H, D).astype(np.float32))
    v = mx.array(np.random.randn(B, L, H, D).astype(np.float32))

    ks = (K,)
    dil = (1,)
    stride = (1,)
    causal = (False,)
    scale = float(D ** -0.5)

    out_integrated = fast_metal.na1d_forward(q, k, v, ks, stride, dil, causal, scale)
    out_ref = pure.na1d_forward(q, k, v, ks, stride, dil, causal, scale)

    mx.eval(out_integrated, out_ref)
    np.testing.assert_allclose(
        np.array(out_ref), np.array(out_integrated), atol=1e-4, rtol=1e-3
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
