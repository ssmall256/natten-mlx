import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx import na1d, na2d, set_backend
from natten_mlx.functional import na3d
from natten_mlx.nn import NeighborhoodAttention1D, NeighborhoodAttention2D, NeighborhoodAttention3D


def _run_backend(backend: str, fn):
    set_backend(backend)
    try:
        return fn()
    finally:
        set_backend("auto")


@pytest.mark.parametrize("backend", ["pure", "fast_metal", "nanobind"])
def test_backend_forced_matches_pure_supported_fused(backend: str):
    q = mx.random.normal((1, 9, 2, 4))
    k = mx.random.normal((1, 9, 2, 4))
    v = mx.random.normal((1, 9, 2, 4))

    out_pure = _run_backend(
        "pure",
        lambda: na1d(q, k, v, kernel_size=3, stride=1, dilation=1, is_causal=False, scale=0.37),
    )
    out_backend = _run_backend(
        backend,
        lambda: na1d(q, k, v, kernel_size=3, stride=1, dilation=1, is_causal=False, scale=0.37),
    )
    mx.eval(out_pure, out_backend)
    np.testing.assert_allclose(np.array(out_backend), np.array(out_pure), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["pure", "fast_metal", "nanobind"])
def test_backend_forced_matches_pure_stride_causal(backend: str):
    q = mx.random.normal((1, 8, 7, 2, 3))
    k = mx.random.normal((1, 8, 7, 2, 3))
    v = mx.random.normal((1, 8, 7, 2, 3))

    out_pure = _run_backend(
        "pure",
        lambda: na2d(
            q,
            k,
            v,
            kernel_size=(3, 3),
            stride=(2, 3),
            dilation=(2, 2),
            is_causal=(True, False),
            scale=0.41,
        ),
    )
    out_backend = _run_backend(
        backend,
        lambda: na2d(
            q,
            k,
            v,
            kernel_size=(3, 3),
            stride=(2, 3),
            dilation=(2, 2),
            is_causal=(True, False),
            scale=0.41,
        ),
    )
    mx.eval(out_pure, out_backend)
    np.testing.assert_allclose(np.array(out_backend), np.array(out_pure), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["pure", "fast_metal", "nanobind"])
def test_backend_forced_matches_pure_expanded_fused_na1d(backend: str):
    q = mx.random.normal((1, 25, 2, 4))
    k = mx.random.normal((1, 25, 2, 4))
    v = mx.random.normal((1, 25, 2, 4))

    out_pure = _run_backend(
        "pure",
        lambda: na1d(
            q,
            k,
            v,
            kernel_size=9,
            stride=2,
            dilation=2,
            is_causal=True,
            scale=0.29,
        ),
    )
    out_backend = _run_backend(
        backend,
        lambda: na1d(
            q,
            k,
            v,
            kernel_size=9,
            stride=2,
            dilation=2,
            is_causal=True,
            scale=0.29,
        ),
    )
    mx.eval(out_pure, out_backend)
    np.testing.assert_allclose(np.array(out_backend), np.array(out_pure), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["pure", "fast_metal", "nanobind"])
def test_backend_forced_matches_pure_expanded_fused_na2d(backend: str):
    q = mx.random.normal((1, 19, 17, 2, 3))
    k = mx.random.normal((1, 19, 17, 2, 3))
    v = mx.random.normal((1, 19, 17, 2, 3))

    out_pure = _run_backend(
        "pure",
        lambda: na2d(
            q,
            k,
            v,
            kernel_size=(9, 9),
            stride=(2, 3),
            dilation=(2, 1),
            is_causal=(True, False),
            scale=0.33,
        ),
    )
    out_backend = _run_backend(
        backend,
        lambda: na2d(
            q,
            k,
            v,
            kernel_size=(9, 9),
            stride=(2, 3),
            dilation=(2, 1),
            is_causal=(True, False),
            scale=0.33,
        ),
    )
    mx.eval(out_pure, out_backend)
    np.testing.assert_allclose(np.array(out_backend), np.array(out_pure), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["pure", "fast_metal", "nanobind"])
def test_backend_forced_matches_pure_na3d_stride_causal(backend: str):
    q = mx.random.normal((1, 9, 8, 7, 2, 3))
    k = mx.random.normal((1, 9, 8, 7, 2, 3))
    v = mx.random.normal((1, 9, 8, 7, 2, 3))

    out_pure = _run_backend(
        "pure",
        lambda: na3d(
            q,
            k,
            v,
            kernel_size=(3, 3, 3),
            stride=(2, 3, 2),
            dilation=(2, 1, 1),
            is_causal=(True, False, True),
            scale=0.29,
        ),
    )
    out_backend = _run_backend(
        backend,
        lambda: na3d(
            q,
            k,
            v,
            kernel_size=(3, 3, 3),
            stride=(2, 3, 2),
            dilation=(2, 1, 1),
            is_causal=(True, False, True),
            scale=0.29,
        ),
    )
    mx.eval(out_pure, out_backend)
    np.testing.assert_allclose(np.array(out_backend), np.array(out_pure), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["pure", "fast_metal", "nanobind"])
def test_backend_forced_attn_drop_causal_stride_smoke(backend: str):
    x1 = mx.random.normal((2, 11, 16))
    x2 = mx.random.normal((2, 7, 5, 12))

    def _run():
        layer1 = NeighborhoodAttention1D(
            embed_dim=16,
            num_heads=4,
            kernel_size=3,
            stride=2,
            is_causal=True,
            attn_drop=0.2,
        )
        layer2 = NeighborhoodAttention2D(
            embed_dim=12,
            num_heads=3,
            kernel_size=(3, 3),
            stride=(2, 1),
            is_causal=(True, False),
            attn_drop=0.2,
        )
        y1 = layer1(x1)
        y2 = layer2(x2)
        mx.eval(y1, y2)
        return y1, y2

    y1, y2 = _run_backend(backend, _run)
    assert y1.shape == (2, 6, 16)
    assert y2.shape == (2, 4, 5, 12)
    assert np.isfinite(np.array(y1)).all()
    assert np.isfinite(np.array(y2)).all()


@pytest.mark.parametrize("backend", ["pure", "fast_metal", "nanobind"])
def test_backend_forced_attn_drop_causal_stride_3d_smoke(backend: str):
    x = mx.random.normal((2, 7, 6, 5, 12))

    def _run():
        layer = NeighborhoodAttention3D(
            embed_dim=12,
            num_heads=3,
            kernel_size=(3, 3, 3),
            stride=(2, 1, 2),
            is_causal=(True, False, True),
            attn_drop=0.2,
        )
        y = layer(x)
        mx.eval(y)
        return y

    y = _run_backend(backend, _run)
    assert y.shape == (2, 4, 6, 3, 12)
    assert np.isfinite(np.array(y)).all()


@pytest.mark.parametrize("backend", ["pure", "fast_metal", "nanobind"])
def test_backend_forced_backward_smoke(backend: str):
    q = mx.random.normal((1, 7, 2, 3))
    k = mx.random.normal((1, 7, 2, 3))
    v = mx.random.normal((1, 7, 2, 3))

    def _run():
        def loss_fn(q_in):
            out = na1d(q_in, k, v, kernel_size=3, stride=1, dilation=1, is_causal=False)
            return mx.sum(out)

        grad_q = mx.grad(loss_fn)(q)
        mx.eval(grad_q)
        return grad_q

    grad_q = _run_backend(backend, _run)
    assert grad_q.shape == q.shape
    assert np.isfinite(np.array(grad_q)).all()


@pytest.mark.parametrize("backend", ["pure", "fast_metal", "nanobind"])
def test_backend_forced_dropout_backward_smoke(backend: str):
    x = mx.random.normal((1, 10, 12))

    def _run():
        layer = NeighborhoodAttention1D(
            embed_dim=12,
            num_heads=3,
            kernel_size=3,
            stride=1,
            is_causal=False,
            attn_drop=0.2,
        )

        def loss_fn(x_in):
            return mx.sum(layer(x_in))

        grad_x = mx.grad(loss_fn)(x)
        mx.eval(grad_x)
        return grad_x

    grad_x = _run_backend(backend, _run)
    assert grad_x.shape == x.shape
    assert np.isfinite(np.array(grad_x)).all()
