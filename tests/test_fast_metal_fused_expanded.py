import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx._core import fast_metal, pure


def test_fast_metal_na1d_fused_stride_causal_k9_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 25, 2, 4))
    k = mx.random.normal((1, 25, 2, 4))
    v = mx.random.normal((1, 25, 2, 4))
    ks = (9,)
    st = (2,)
    dil = (2,)
    caus = (True,)
    scale = 0.29

    out_pure = pure.na1d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_pure)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_forward")

    monkeypatch.setattr(pure, "na1d_forward", _no_fallback)
    out_fast = fast_metal.na1d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_fast)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na1d_fused_causal_stride1_uses_causal_kernel_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 128, 4, 32))
    k = mx.random.normal((1, 128, 4, 32))
    v = mx.random.normal((1, 128, 4, 32))
    ks = (7,)
    st = (1,)
    dil = (1,)
    caus = (True,)
    scale = 0.5

    out_pure = pure.na1d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_pure)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_forward")

    def _unexpected_noncausal_kernel(*_args, **_kwargs):
        raise AssertionError("unexpected non-causal fused kernel path")

    monkeypatch.setattr(pure, "na1d_forward", _no_fallback)
    monkeypatch.setattr(fast_metal, "_get_1d_fused_kernel", _unexpected_noncausal_kernel)
    out_fast = fast_metal.na1d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_fast)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na1d_fused_vec4_noncausal_uses_vec4_kernel_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 128, 4, 16))
    k = mx.random.normal((1, 128, 4, 16))
    v = mx.random.normal((1, 128, 4, 16))
    ks = (7,)
    st = (1,)
    dil = (1,)
    caus = (False,)
    scale = 0.5

    out_pure = pure.na1d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_pure)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_forward")

    def _unexpected_scalar_kernel(*_args, **_kwargs):
        raise AssertionError("unexpected scalar 1d fused kernel path")

    monkeypatch.setattr(pure, "na1d_forward", _no_fallback)
    monkeypatch.setattr(fast_metal, "_get_1d_fused_kernel", _unexpected_scalar_kernel)
    out_fast = fast_metal.na1d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_fast)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na1d_fused_vec4_causal_uses_vec4_kernel_without_fallback(monkeypatch):
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

    out_pure = pure.na1d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_pure)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na1d_forward")

    def _unexpected_scalar_kernel(*_args, **_kwargs):
        raise AssertionError("unexpected scalar 1d fused causal kernel path")

    monkeypatch.setattr(pure, "na1d_forward", _no_fallback)
    monkeypatch.setattr(fast_metal, "_get_1d_fused_causal_kernel", _unexpected_scalar_kernel)
    out_fast = fast_metal.na1d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_fast)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na2d_fused_stride_causal_k9_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 19, 17, 2, 3))
    k = mx.random.normal((1, 19, 17, 2, 3))
    v = mx.random.normal((1, 19, 17, 2, 3))
    ks = (9, 9)
    st = (2, 3)
    dil = (2, 1)
    caus = (True, False)
    scale = 0.33

    out_pure = pure.na2d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_pure)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na2d_forward")

    monkeypatch.setattr(pure, "na2d_forward", _no_fallback)
    out_fast = fast_metal.na2d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_fast)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na2d_fused_causal_uses_causal_kernel_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 24, 24, 4, 16))
    k = mx.random.normal((1, 24, 24, 4, 16))
    v = mx.random.normal((1, 24, 24, 4, 16))
    ks = (7, 7)
    st = (1, 1)
    dil = (1, 1)
    caus = (True, False)
    scale = 0.5

    out_pure = pure.na2d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_pure)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na2d_forward")

    def _unexpected_noncausal_kernel(*_args, **_kwargs):
        raise AssertionError("unexpected non-causal 2d fused kernel path")

    monkeypatch.setattr(pure, "na2d_forward", _no_fallback)
    monkeypatch.setattr(fast_metal, "_get_2d_fused_kernel", _unexpected_noncausal_kernel)
    out_fast = fast_metal.na2d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_fast)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na3d_fused_stride_causal_k5_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 13, 11, 9, 2, 3))
    k = mx.random.normal((1, 13, 11, 9, 2, 3))
    v = mx.random.normal((1, 13, 11, 9, 2, 3))
    ks = (5, 5, 5)
    st = (2, 3, 2)
    dil = (2, 1, 2)
    caus = (True, False, True)
    scale = 0.29

    out_pure = pure.na3d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_pure)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_forward")

    def _unexpected_split(*_args, **_kwargs):
        raise AssertionError("unexpected split path in na3d_forward")

    monkeypatch.setattr(pure, "na3d_forward", _no_fallback)
    monkeypatch.setattr(fast_metal, "na3d_qk_forward", _unexpected_split)
    monkeypatch.setattr(fast_metal, "na3d_av_forward", _unexpected_split)
    out_fast = fast_metal.na3d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_fast)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)


def test_fast_metal_na3d_fused_causal_uses_causal_kernel_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 10, 12, 14, 4, 16))
    k = mx.random.normal((1, 10, 12, 14, 4, 16))
    v = mx.random.normal((1, 10, 12, 14, 4, 16))
    ks = (3, 3, 3)
    st = (1, 1, 1)
    dil = (1, 1, 1)
    caus = (True, False, False)
    scale = 0.5

    out_pure = pure.na3d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_pure)

    def _no_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_forward")

    def _unexpected_noncausal_kernel(*_args, **_kwargs):
        raise AssertionError("unexpected non-causal 3d fused kernel path")

    def _unexpected_split(*_args, **_kwargs):
        raise AssertionError("unexpected split path in na3d_forward")

    monkeypatch.setattr(pure, "na3d_forward", _no_fallback)
    monkeypatch.setattr(fast_metal, "_get_3d_fused_kernel", _unexpected_noncausal_kernel)
    monkeypatch.setattr(fast_metal, "na3d_qk_forward", _unexpected_split)
    monkeypatch.setattr(fast_metal, "na3d_av_forward", _unexpected_split)
    out_fast = fast_metal.na3d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_fast)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)
