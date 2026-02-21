import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx import set_backend
from natten_mlx._core import ops
from natten_mlx.functional import na1d, na1d_av, na1d_qk, na2d, na2d_av, na2d_qk, na3d, na3d_av, na3d_qk


def _run_backend(backend: str, fn):
    set_backend(backend)
    try:
        return fn()
    finally:
        set_backend("auto")


def _to_np_tuple(values):
    mx.eval(*values)
    return tuple(np.array(v) for v in values)


@pytest.mark.parametrize("backend", ["fast_metal", "nanobind"])
def test_fused_na1d_backward_matches_pure(backend: str):
    rng = np.random.default_rng(9001)
    q = mx.array(rng.standard_normal((1, 12, 2, 4), dtype=np.float32))
    k = mx.array(rng.standard_normal((1, 12, 2, 4), dtype=np.float32))
    v = mx.array(rng.standard_normal((1, 12, 2, 4), dtype=np.float32))
    cotangent = mx.array(rng.standard_normal((1, 12, 2, 4), dtype=np.float32))

    def _grads():
        def _loss_q(q_in):
            out = na1d(
                q_in,
                k,
                v,
                kernel_size=3,
                stride=1,
                dilation=1,
                is_causal=False,
                scale=0.37,
            )
            return mx.sum(out * cotangent)

        def _loss_k(k_in):
            out = na1d(
                q,
                k_in,
                v,
                kernel_size=3,
                stride=1,
                dilation=1,
                is_causal=False,
                scale=0.37,
            )
            return mx.sum(out * cotangent)

        def _loss_v(v_in):
            out = na1d(
                q,
                k,
                v_in,
                kernel_size=3,
                stride=1,
                dilation=1,
                is_causal=False,
                scale=0.37,
            )
            return mx.sum(out * cotangent)

        return _to_np_tuple((mx.grad(_loss_q)(q), mx.grad(_loss_k)(k), mx.grad(_loss_v)(v)))

    pure_grads = _run_backend("pure", _grads)
    backend_grads = _run_backend(backend, _grads)
    for got, want in zip(backend_grads, pure_grads):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", ["fast_metal", "nanobind"])
def test_fused_na2d_backward_matches_pure(backend: str):
    rng = np.random.default_rng(9002)
    q = mx.array(rng.standard_normal((1, 7, 8, 2, 3), dtype=np.float32))
    k = mx.array(rng.standard_normal((1, 7, 8, 2, 3), dtype=np.float32))
    v = mx.array(rng.standard_normal((1, 7, 8, 2, 3), dtype=np.float32))
    cotangent = mx.array(rng.standard_normal((1, 7, 8, 2, 3), dtype=np.float32))

    def _grads():
        def _loss_q(q_in):
            out = na2d(
                q_in,
                k,
                v,
                kernel_size=(3, 3),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=(False, False),
                scale=0.41,
            )
            return mx.sum(out * cotangent)

        def _loss_k(k_in):
            out = na2d(
                q,
                k_in,
                v,
                kernel_size=(3, 3),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=(False, False),
                scale=0.41,
            )
            return mx.sum(out * cotangent)

        def _loss_v(v_in):
            out = na2d(
                q,
                k,
                v_in,
                kernel_size=(3, 3),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=(False, False),
                scale=0.41,
            )
            return mx.sum(out * cotangent)

        return _to_np_tuple((mx.grad(_loss_q)(q), mx.grad(_loss_k)(k), mx.grad(_loss_v)(v)))

    pure_grads = _run_backend("pure", _grads)
    backend_grads = _run_backend(backend, _grads)
    for got, want in zip(backend_grads, pure_grads):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", ["fast_metal", "nanobind"])
def test_fused_na3d_backward_matches_pure(backend: str):
    rng = np.random.default_rng(9005)
    q = mx.array(rng.standard_normal((1, 6, 7, 8, 2, 3), dtype=np.float32))
    k = mx.array(rng.standard_normal((1, 6, 7, 8, 2, 3), dtype=np.float32))
    v = mx.array(rng.standard_normal((1, 6, 7, 8, 2, 3), dtype=np.float32))
    cotangent = mx.array(rng.standard_normal((1, 3, 4, 4, 2, 3), dtype=np.float32))

    def _grads():
        def _loss_q(q_in):
            out = na3d(
                q_in,
                k,
                v,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                dilation=(1, 1, 2),
                is_causal=(True, False, False),
                scale=0.31,
            )
            return mx.sum(out * cotangent)

        def _loss_k(k_in):
            out = na3d(
                q,
                k_in,
                v,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                dilation=(1, 1, 2),
                is_causal=(True, False, False),
                scale=0.31,
            )
            return mx.sum(out * cotangent)

        def _loss_v(v_in):
            out = na3d(
                q,
                k,
                v_in,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                dilation=(1, 1, 2),
                is_causal=(True, False, False),
                scale=0.31,
            )
            return mx.sum(out * cotangent)

        return _to_np_tuple((mx.grad(_loss_q)(q), mx.grad(_loss_k)(k), mx.grad(_loss_v)(v)))

    pure_grads = _run_backend("pure", _grads)
    backend_grads = _run_backend(backend, _grads)
    for got, want in zip(backend_grads, pure_grads):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", ["fast_metal", "nanobind"])
def test_split_na1d_qk_av_backward_entrypoints_match_pure(backend: str):
    rng = np.random.default_rng(9003)
    q = mx.array(rng.standard_normal((1, 10, 2, 4), dtype=np.float32))
    k = mx.array(rng.standard_normal((1, 10, 2, 4), dtype=np.float32))
    v = mx.array(rng.standard_normal((1, 10, 2, 4), dtype=np.float32))
    ks = (3,)
    st = (1,)
    dil = (1,)
    caus = (False,)
    scale = 0.33

    logits = _run_backend("pure", lambda: na1d_qk(q, k, kernel_size=ks, stride=st, dilation=dil, is_causal=caus, scale=scale))
    attn = mx.softmax(logits, axis=-1)
    grad_logits = mx.array(rng.standard_normal(logits.shape, dtype=np.float32))
    grad_out = mx.array(rng.standard_normal((1, 10, 2, 4), dtype=np.float32))

    pure_qk = _run_backend(
        "pure",
        lambda: _to_np_tuple(ops.na1d_qk_backward(q, k, grad_logits, ks, st, dil, caus, scale)),
    )
    backend_qk = _run_backend(
        backend,
        lambda: _to_np_tuple(ops.na1d_qk_backward(q, k, grad_logits, ks, st, dil, caus, scale)),
    )
    for got, want in zip(backend_qk, pure_qk):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)

    pure_av = _run_backend(
        "pure",
        lambda: _to_np_tuple(ops.na1d_av_backward(attn, v, grad_out, ks, st, dil, caus)),
    )
    backend_av = _run_backend(
        backend,
        lambda: _to_np_tuple(ops.na1d_av_backward(attn, v, grad_out, ks, st, dil, caus)),
    )
    for got, want in zip(backend_av, pure_av):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", ["fast_metal", "nanobind"])
def test_split_na2d_qk_av_backward_entrypoints_match_pure(backend: str):
    rng = np.random.default_rng(9004)
    q = mx.array(rng.standard_normal((1, 6, 7, 2, 3), dtype=np.float32))
    k = mx.array(rng.standard_normal((1, 6, 7, 2, 3), dtype=np.float32))
    v = mx.array(rng.standard_normal((1, 6, 7, 2, 3), dtype=np.float32))
    ks = (3, 3)
    st = (1, 1)
    dil = (1, 1)
    caus = (False, False)
    scale = 0.27

    logits = _run_backend("pure", lambda: na2d_qk(q, k, kernel_size=ks, stride=st, dilation=dil, is_causal=caus, scale=scale))
    attn = mx.softmax(logits, axis=-1)
    grad_logits = mx.array(rng.standard_normal(logits.shape, dtype=np.float32))
    grad_out = mx.array(rng.standard_normal((1, 6, 7, 2, 3), dtype=np.float32))

    pure_qk = _run_backend(
        "pure",
        lambda: _to_np_tuple(ops.na2d_qk_backward(q, k, grad_logits, ks, st, dil, caus, scale)),
    )
    backend_qk = _run_backend(
        backend,
        lambda: _to_np_tuple(ops.na2d_qk_backward(q, k, grad_logits, ks, st, dil, caus, scale)),
    )
    for got, want in zip(backend_qk, pure_qk):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)

    pure_av = _run_backend(
        "pure",
        lambda: _to_np_tuple(ops.na2d_av_backward(attn, v, grad_out, ks, st, dil, caus)),
    )
    backend_av = _run_backend(
        backend,
        lambda: _to_np_tuple(ops.na2d_av_backward(attn, v, grad_out, ks, st, dil, caus)),
    )
    for got, want in zip(backend_av, pure_av):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", ["fast_metal", "nanobind"])
def test_split_na3d_qk_av_backward_entrypoints_match_pure(backend: str):
    rng = np.random.default_rng(9006)
    q = mx.array(rng.standard_normal((1, 6, 7, 8, 2, 3), dtype=np.float32))
    k = mx.array(rng.standard_normal((1, 6, 7, 8, 2, 3), dtype=np.float32))
    v = mx.array(rng.standard_normal((1, 6, 7, 8, 2, 3), dtype=np.float32))
    ks = (3, 3, 3)
    st = (1, 1, 1)
    dil = (1, 1, 1)
    caus = (False, False, False)
    scale = 0.23

    logits = _run_backend(
        "pure",
        lambda: na3d_qk(q, k, kernel_size=ks, stride=st, dilation=dil, is_causal=caus, scale=scale),
    )
    attn = mx.softmax(logits, axis=-1)
    grad_logits = mx.array(rng.standard_normal(logits.shape, dtype=np.float32))
    grad_out = mx.array(rng.standard_normal((1, 6, 7, 8, 2, 3), dtype=np.float32))

    pure_qk = _run_backend(
        "pure",
        lambda: _to_np_tuple(ops.na3d_qk_backward(q, k, grad_logits, ks, st, dil, caus, scale)),
    )
    backend_qk = _run_backend(
        backend,
        lambda: _to_np_tuple(ops.na3d_qk_backward(q, k, grad_logits, ks, st, dil, caus, scale)),
    )
    for got, want in zip(backend_qk, pure_qk):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)

    pure_av = _run_backend(
        "pure",
        lambda: _to_np_tuple(ops.na3d_av_backward(attn, v, grad_out, ks, st, dil, caus)),
    )
    backend_av = _run_backend(
        backend,
        lambda: _to_np_tuple(ops.na3d_av_backward(attn, v, grad_out, ks, st, dil, caus)),
    )
    for got, want in zip(backend_av, pure_av):
        np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)
