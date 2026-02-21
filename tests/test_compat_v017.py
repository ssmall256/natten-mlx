import pytest

mx = pytest.importorskip("mlx.core")

import natten_mlx.compat.v017 as v017


def test_v017_reexports_v015_surface():
    q = mx.random.normal((1, 8, 2, 4))
    out = v017.na1d(q, q, q, kernel_size=3)
    assert out.shape == (1, 8, 2, 4)


def test_v017_has_3d_support():
    q = mx.random.normal((1, 6, 7, 8, 2, 4))
    out = v017.na3d(q, q, q, kernel_size=(3, 3, 3))
    assert out.shape == (1, 6, 7, 8, 2, 4)
