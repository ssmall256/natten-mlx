import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx._core import fast_metal, pure


def test_fast_metal_na1d_split_stride_k9_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 29, 2, 4))
    k = mx.random.normal((1, 29, 2, 4))
    v = mx.random.normal((1, 29, 2, 4))
    ks = (9,)
    st = (2,)
    dil = (2,)
    caus = (False,)
    scale = 0.31

    logits_pure = pure.na1d_qk_forward(q, k, ks, st, dil, caus, scale)
    attn = mx.softmax(logits_pure, axis=-1)
    out_pure = pure.na1d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_pure, out_pure)

    def _no_qk_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_qk_forward")

    def _no_av_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_av_forward")

    monkeypatch.setattr(pure, "na1d_qk_forward", _no_qk_fallback)
    monkeypatch.setattr(pure, "na1d_av_forward", _no_av_fallback)

    logits_fast = fast_metal.na1d_qk_forward(q, k, ks, st, dil, caus, scale)
    out_fast = fast_metal.na1d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_fast, out_fast)

    np.testing.assert_allclose(np.array(logits_fast), np.array(logits_pure), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na2d_split_stride_k9_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 21, 19, 2, 3))
    k = mx.random.normal((1, 21, 19, 2, 3))
    v = mx.random.normal((1, 21, 19, 2, 3))
    ks = (9, 9)
    st = (2, 3)
    dil = (2, 1)
    caus = (False, False)
    scale = 0.27

    logits_pure = pure.na2d_qk_forward(q, k, ks, st, dil, caus, scale)
    attn = mx.softmax(logits_pure, axis=-1)
    out_pure = pure.na2d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_pure, out_pure)

    def _no_qk_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na2d_qk_forward")

    def _no_av_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na2d_av_forward")

    monkeypatch.setattr(pure, "na2d_qk_forward", _no_qk_fallback)
    monkeypatch.setattr(pure, "na2d_av_forward", _no_av_fallback)

    logits_fast = fast_metal.na2d_qk_forward(q, k, ks, st, dil, caus, scale)
    out_fast = fast_metal.na2d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_fast, out_fast)

    np.testing.assert_allclose(np.array(logits_fast), np.array(logits_pure), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na3d_split_stride_k5_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 13, 11, 10, 2, 3))
    k = mx.random.normal((1, 13, 11, 10, 2, 3))
    v = mx.random.normal((1, 13, 11, 10, 2, 3))
    ks = (5, 5, 5)
    st = (2, 2, 2)
    dil = (2, 1, 2)
    caus = (False, False, False)
    scale = 0.23

    logits_pure = pure.na3d_qk_forward(q, k, ks, st, dil, caus, scale)
    attn = mx.softmax(logits_pure, axis=-1)
    out_pure = pure.na3d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_pure, out_pure)

    def _no_qk_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_qk_forward")

    def _no_av_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_av_forward")

    monkeypatch.setattr(pure, "na3d_qk_forward", _no_qk_fallback)
    monkeypatch.setattr(pure, "na3d_av_forward", _no_av_fallback)

    logits_fast = fast_metal.na3d_qk_forward(q, k, ks, st, dil, caus, scale)
    out_fast = fast_metal.na3d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_fast, out_fast)

    np.testing.assert_allclose(np.array(logits_fast), np.array(logits_pure), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na1d_split_stride_k9_causal_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 29, 2, 4))
    k = mx.random.normal((1, 29, 2, 4))
    v = mx.random.normal((1, 29, 2, 4))
    ks = (9,)
    st = (2,)
    dil = (2,)
    caus = (True,)
    scale = 0.31

    logits_pure = pure.na1d_qk_forward(q, k, ks, st, dil, caus, scale)
    attn = mx.softmax(logits_pure, axis=-1)
    out_pure = pure.na1d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_pure, out_pure)

    def _no_qk_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_qk_forward")

    def _no_av_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_av_forward")

    monkeypatch.setattr(pure, "na1d_qk_forward", _no_qk_fallback)
    monkeypatch.setattr(pure, "na1d_av_forward", _no_av_fallback)

    logits_fast = fast_metal.na1d_qk_forward(q, k, ks, st, dil, caus, scale)
    out_fast = fast_metal.na1d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_fast, out_fast)

    np.testing.assert_allclose(np.array(logits_fast), np.array(logits_pure), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na1d_qk_vec4_uses_vec4_kernel_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 128, 4, 16))
    k = mx.random.normal((1, 128, 4, 16))
    ks = (7,)
    st = (1,)
    dil = (1,)
    caus = (True,)
    scale = 0.5

    logits_pure = pure.na1d_qk_forward(q, k, ks, st, dil, caus, scale)
    mx.eval(logits_pure)

    def _no_qk_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_qk_forward")

    def _unexpected_scalar_kernel(*_args, **_kwargs):
        raise AssertionError("unexpected scalar 1d qk kernel path")

    monkeypatch.setattr(pure, "na1d_qk_forward", _no_qk_fallback)
    monkeypatch.setattr(fast_metal, "_get_1d_qk_kernel", _unexpected_scalar_kernel)

    logits_fast = fast_metal.na1d_qk_forward(q, k, ks, st, dil, caus, scale)
    mx.eval(logits_fast)
    np.testing.assert_allclose(np.array(logits_fast), np.array(logits_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na2d_split_stride_k9_causal_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 21, 19, 2, 3))
    k = mx.random.normal((1, 21, 19, 2, 3))
    v = mx.random.normal((1, 21, 19, 2, 3))
    ks = (9, 9)
    st = (2, 3)
    dil = (2, 1)
    caus = (True, False)
    scale = 0.27

    logits_pure = pure.na2d_qk_forward(q, k, ks, st, dil, caus, scale)
    attn = mx.softmax(logits_pure, axis=-1)
    out_pure = pure.na2d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_pure, out_pure)

    def _no_qk_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na2d_qk_forward")

    def _no_av_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na2d_av_forward")

    monkeypatch.setattr(pure, "na2d_qk_forward", _no_qk_fallback)
    monkeypatch.setattr(pure, "na2d_av_forward", _no_av_fallback)

    logits_fast = fast_metal.na2d_qk_forward(q, k, ks, st, dil, caus, scale)
    out_fast = fast_metal.na2d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_fast, out_fast)

    np.testing.assert_allclose(np.array(logits_fast), np.array(logits_pure), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na3d_split_stride_k5_causal_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 13, 11, 10, 2, 3))
    k = mx.random.normal((1, 13, 11, 10, 2, 3))
    v = mx.random.normal((1, 13, 11, 10, 2, 3))
    ks = (5, 5, 5)
    st = (2, 2, 2)
    dil = (2, 1, 2)
    caus = (True, False, True)
    scale = 0.23

    logits_pure = pure.na3d_qk_forward(q, k, ks, st, dil, caus, scale)
    attn = mx.softmax(logits_pure, axis=-1)
    out_pure = pure.na3d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_pure, out_pure)

    def _no_qk_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_qk_forward")

    def _no_av_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_av_forward")

    monkeypatch.setattr(pure, "na3d_qk_forward", _no_qk_fallback)
    monkeypatch.setattr(pure, "na3d_av_forward", _no_av_fallback)

    logits_fast = fast_metal.na3d_qk_forward(q, k, ks, st, dil, caus, scale)
    out_fast = fast_metal.na3d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(logits_fast, out_fast)

    np.testing.assert_allclose(np.array(logits_fast), np.array(logits_pure), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na1d_av_vec4_uses_vec4_kernel_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 128, 4, 16))
    k = mx.random.normal((1, 128, 4, 16))
    v = mx.random.normal((1, 128, 4, 16))
    ks = (7,)
    st = (1,)
    dil = (1,)
    caus = (True,)
    scale = 0.5

    logits = pure.na1d_qk_forward(q, k, ks, st, dil, caus, scale)
    attn = mx.softmax(logits, axis=-1)
    out_pure = pure.na1d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(out_pure)

    def _no_av_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_av_forward")

    def _unexpected_scalar_kernel(*_args, **_kwargs):
        raise AssertionError("unexpected scalar 1d av kernel path")

    monkeypatch.setattr(pure, "na1d_av_forward", _no_av_fallback)
    monkeypatch.setattr(fast_metal, "_get_1d_av_kernel", _unexpected_scalar_kernel)

    out_fast = fast_metal.na1d_av_forward(attn, v, ks, st, dil, caus)
    mx.eval(out_fast)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na2d_qk_vec4_uses_vec4_kernel_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 24, 20, 4, 16))
    k = mx.random.normal((1, 24, 20, 4, 16))
    ks = (7, 7)
    st = (1, 1)
    dil = (1, 1)
    caus = (True, False)
    scale = 0.5

    logits_pure = pure.na2d_qk_forward(q, k, ks, st, dil, caus, scale)
    mx.eval(logits_pure)

    def _no_qk_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na2d_qk_forward")

    def _unexpected_scalar_kernel(*_args, **_kwargs):
        raise AssertionError("unexpected scalar 2d qk kernel path")

    monkeypatch.setattr(pure, "na2d_qk_forward", _no_qk_fallback)
    monkeypatch.setattr(fast_metal, "_get_2d_qk_kernel", _unexpected_scalar_kernel)

    logits_fast = fast_metal.na2d_qk_forward(q, k, ks, st, dil, caus, scale)
    mx.eval(logits_fast)
    np.testing.assert_allclose(np.array(logits_fast), np.array(logits_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na3d_qk_vec4_uses_vec4_kernel_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 10, 12, 14, 4, 16))
    k = mx.random.normal((1, 10, 12, 14, 4, 16))
    ks = (3, 3, 3)
    st = (1, 1, 1)
    dil = (1, 1, 1)
    caus = (True, False, True)
    scale = 0.5

    logits_pure = pure.na3d_qk_forward(q, k, ks, st, dil, caus, scale)
    mx.eval(logits_pure)

    def _no_qk_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_qk_forward")

    def _unexpected_scalar_kernel(*_args, **_kwargs):
        raise AssertionError("unexpected scalar 3d qk kernel path")

    monkeypatch.setattr(pure, "na3d_qk_forward", _no_qk_fallback)
    monkeypatch.setattr(fast_metal, "_get_3d_qk_kernel", _unexpected_scalar_kernel)

    logits_fast = fast_metal.na3d_qk_forward(q, k, ks, st, dil, caus, scale)
    mx.eval(logits_fast)
    np.testing.assert_allclose(np.array(logits_fast), np.array(logits_pure), rtol=1e-5, atol=1e-5)
