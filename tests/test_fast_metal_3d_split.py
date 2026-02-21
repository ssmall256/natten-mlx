import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx._core import fast_metal, pure


def test_fast_metal_na3d_split_matches_pure_without_fallback(monkeypatch):
    if not fast_metal.is_available():
        pytest.skip("fast_metal unavailable")

    q = mx.random.normal((1, 6, 7, 8, 2, 3))
    k = mx.random.normal((1, 6, 7, 8, 2, 3))
    v = mx.random.normal((1, 6, 7, 8, 2, 3))
    ks = (3, 3, 3)
    st = (1, 1, 1)
    dil = (1, 1, 1)
    caus = (False, False, False)
    scale = 0.23

    out_pure = pure.na3d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_pure)
    out_probe = fast_metal.na3d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_probe)
    np.testing.assert_allclose(np.array(out_probe), np.array(out_pure), rtol=1e-5, atol=1e-5)
    if not fast_metal.is_available():
        pytest.skip("fast_metal 3D split kernel unavailable in this environment")

    def _no_qk_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_qk_forward")

    def _no_av_fallback(*_args, **_kwargs):
        raise AssertionError("unexpected fallback to pure.na3d_av_forward")

    monkeypatch.setattr(pure, "na3d_qk_forward", _no_qk_fallback)
    monkeypatch.setattr(pure, "na3d_av_forward", _no_av_fallback)
    out_fast = fast_metal.na3d_forward(q, k, v, ks, st, dil, caus, scale)
    mx.eval(out_fast)
    np.testing.assert_allclose(np.array(out_fast), np.array(out_pure), rtol=1e-5, atol=1e-5)
