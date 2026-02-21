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
