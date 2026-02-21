import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx._core import fast_metal, pure


def test_fast_metal_na1d_backward_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 25, 2, 4))
    k = mx.random.normal((1, 25, 2, 4))
    v = mx.random.normal((1, 25, 2, 4))
    grad_out = mx.random.normal((1, 13, 2, 4))
    ks = (9,)
    st = (2,)
    dil = (2,)
    caus = (True,)
    scale = 0.29

    gq_p, gk_p, gv_p = pure.na1d_backward(q, k, v, grad_out, ks, st, dil, caus, scale)
    mx.eval(gq_p, gk_p, gv_p)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_backward")

    monkeypatch.setattr(pure, "na1d_backward", _no_fallback)
    gq_f, gk_f, gv_f = fast_metal.na1d_backward(q, k, v, grad_out, ks, st, dil, caus, scale)
    mx.eval(gq_f, gk_f, gv_f)

    np.testing.assert_allclose(np.array(gq_f), np.array(gq_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gk_f), np.array(gk_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gv_f), np.array(gv_p), rtol=1e-5, atol=1e-5)


def test_fast_metal_na2d_backward_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 19, 17, 2, 3))
    k = mx.random.normal((1, 19, 17, 2, 3))
    v = mx.random.normal((1, 19, 17, 2, 3))
    grad_out = mx.random.normal((1, 10, 6, 2, 3))
    ks = (9, 9)
    st = (2, 3)
    dil = (2, 1)
    caus = (True, False)
    scale = 0.33

    gq_p, gk_p, gv_p = pure.na2d_backward(q, k, v, grad_out, ks, st, dil, caus, scale)
    mx.eval(gq_p, gk_p, gv_p)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na2d_backward")

    monkeypatch.setattr(pure, "na2d_backward", _no_fallback)
    gq_f, gk_f, gv_f = fast_metal.na2d_backward(q, k, v, grad_out, ks, st, dil, caus, scale)
    mx.eval(gq_f, gk_f, gv_f)

    np.testing.assert_allclose(np.array(gq_f), np.array(gq_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gk_f), np.array(gk_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gv_f), np.array(gv_p), rtol=1e-5, atol=1e-5)


def test_fast_metal_na3d_backward_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 13, 11, 9, 2, 3))
    k = mx.random.normal((1, 13, 11, 9, 2, 3))
    v = mx.random.normal((1, 13, 11, 9, 2, 3))
    grad_out = mx.random.normal((1, 7, 4, 5, 2, 3))
    ks = (5, 5, 5)
    st = (2, 3, 2)
    dil = (2, 1, 2)
    caus = (True, False, True)
    scale = 0.29

    gq_p, gk_p, gv_p = pure.na3d_backward(q, k, v, grad_out, ks, st, dil, caus, scale)
    mx.eval(gq_p, gk_p, gv_p)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_backward")

    monkeypatch.setattr(pure, "na3d_backward", _no_fallback)
    gq_f, gk_f, gv_f = fast_metal.na3d_backward(q, k, v, grad_out, ks, st, dil, caus, scale)
    mx.eval(gq_f, gk_f, gv_f)

    np.testing.assert_allclose(np.array(gq_f), np.array(gq_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gk_f), np.array(gk_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gv_f), np.array(gv_p), rtol=1e-5, atol=1e-5)


def test_fast_metal_na1d_av_backward_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    attn = mx.random.normal((1, 13, 2, 7))
    v = mx.random.normal((1, 25, 2, 4))
    grad_out = mx.random.normal((1, 13, 2, 4))
    ks = (7,)
    st = (2,)
    dil = (2,)
    caus = (True,)

    ga_p, gv_p = pure.na1d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_p, gv_p)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_av_backward")

    monkeypatch.setattr(pure, "na1d_av_backward", _no_fallback)
    ga_f, gv_f = fast_metal.na1d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_f, gv_f)

    np.testing.assert_allclose(np.array(ga_f), np.array(ga_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gv_f), np.array(gv_p), rtol=1e-5, atol=1e-5)


def test_fast_metal_na1d_qk_backward_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 25, 2, 4))
    k = mx.random.normal((1, 25, 2, 4))
    grad_attn = mx.random.normal((1, 13, 2, 9))
    ks = (9,)
    st = (2,)
    dil = (2,)
    caus = (True,)
    scale = 0.31

    gq_p, gk_p = pure.na1d_qk_backward(q, k, grad_attn, ks, st, dil, caus, scale)
    mx.eval(gq_p, gk_p)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_qk_backward")

    monkeypatch.setattr(pure, "na1d_qk_backward", _no_fallback)
    gq_f, gk_f = fast_metal.na1d_qk_backward(q, k, grad_attn, ks, st, dil, caus, scale)
    mx.eval(gq_f, gk_f)

    np.testing.assert_allclose(np.array(gq_f), np.array(gq_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gk_f), np.array(gk_p), rtol=1e-5, atol=1e-5)


def test_fast_metal_na1d_av_backward_vec4_path_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    attn = mx.random.normal((1, 13, 2, 7))
    v = mx.random.normal((1, 25, 2, 16))
    grad_out = mx.random.normal((1, 13, 2, 16))
    ks = (7,)
    st = (2,)
    dil = (2,)
    caus = (True,)

    ga_p, gv_p = pure.na1d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_p, gv_p)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_av_backward")

    monkeypatch.setattr(pure, "na1d_av_backward", _no_fallback)
    ga_f, gv_f = fast_metal.na1d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_f, gv_f)

    np.testing.assert_allclose(np.array(ga_f), np.array(ga_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gv_f), np.array(gv_p), rtol=1e-5, atol=1e-5)


def test_fast_metal_na2d_av_backward_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    attn = mx.random.normal((1, 10, 6, 2, 25))
    v = mx.random.normal((1, 19, 17, 2, 3))
    grad_out = mx.random.normal((1, 10, 6, 2, 3))
    ks = (5, 5)
    st = (2, 3)
    dil = (2, 1)
    caus = (True, False)

    ga_p, gv_p = pure.na2d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_p, gv_p)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na2d_av_backward")

    monkeypatch.setattr(pure, "na2d_av_backward", _no_fallback)
    ga_f, gv_f = fast_metal.na2d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_f, gv_f)

    np.testing.assert_allclose(np.array(ga_f), np.array(ga_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gv_f), np.array(gv_p), rtol=1e-5, atol=1e-5)


def test_fast_metal_na2d_av_backward_vec4_path_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    attn = mx.random.normal((1, 10, 6, 2, 25))
    v = mx.random.normal((1, 19, 17, 2, 16))
    grad_out = mx.random.normal((1, 10, 6, 2, 16))
    ks = (5, 5)
    st = (2, 3)
    dil = (2, 1)
    caus = (True, False)

    ga_p, gv_p = pure.na2d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_p, gv_p)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na2d_av_backward")

    monkeypatch.setattr(pure, "na2d_av_backward", _no_fallback)
    ga_f, gv_f = fast_metal.na2d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_f, gv_f)

    np.testing.assert_allclose(np.array(ga_f), np.array(ga_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gv_f), np.array(gv_p), rtol=1e-5, atol=1e-5)


def test_fast_metal_na3d_av_backward_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    attn = mx.random.normal((1, 7, 4, 5, 2, 27))
    v = mx.random.normal((1, 13, 11, 9, 2, 3))
    grad_out = mx.random.normal((1, 7, 4, 5, 2, 3))
    ks = (3, 3, 3)
    st = (2, 3, 2)
    dil = (2, 1, 2)
    caus = (True, False, True)

    ga_p, gv_p = pure.na3d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_p, gv_p)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_av_backward")

    monkeypatch.setattr(pure, "na3d_av_backward", _no_fallback)
    ga_f, gv_f = fast_metal.na3d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_f, gv_f)

    np.testing.assert_allclose(np.array(ga_f), np.array(ga_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gv_f), np.array(gv_p), rtol=1e-5, atol=1e-5)


def test_fast_metal_na3d_av_backward_vec4_path_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    attn = mx.random.normal((1, 7, 4, 5, 2, 27))
    v = mx.random.normal((1, 13, 11, 9, 2, 16))
    grad_out = mx.random.normal((1, 7, 4, 5, 2, 16))
    ks = (3, 3, 3)
    st = (2, 3, 2)
    dil = (2, 1, 2)
    caus = (True, False, True)

    ga_p, gv_p = pure.na3d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_p, gv_p)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_av_backward")

    monkeypatch.setattr(pure, "na3d_av_backward", _no_fallback)
    ga_f, gv_f = fast_metal.na3d_av_backward(attn, v, grad_out, ks, st, dil, caus)
    mx.eval(ga_f, gv_f)

    np.testing.assert_allclose(np.array(ga_f), np.array(ga_p), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(gv_f), np.array(gv_p), rtol=1e-5, atol=1e-5)


def test_inverse_map_uses_compact_index_dtype_when_safe():
    _, inv_attn_base, inv_grad_base = fast_metal._inverse_map_1d(
        length=64,
        out_len=64,
        kernel_size=7,
        stride=1,
        dilation=1,
        causal=False,
        dim=32,
    )
    assert inv_attn_base.dtype == mx.uint16
    assert inv_grad_base.dtype == mx.uint16


def test_inverse_map_upcasts_grad_index_dtype_on_overflow():
    _, inv_attn_base, inv_grad_base = fast_metal._inverse_map_1d(
        length=512,
        out_len=512,
        kernel_size=7,
        stride=1,
        dilation=1,
        causal=False,
        dim=256,
    )
    assert inv_attn_base.dtype == mx.uint16
    assert inv_grad_base.dtype == mx.int32
