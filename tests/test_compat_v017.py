import pytest

mx = pytest.importorskip("mlx.core")

import natten_mlx.compat.v017 as v017


def test_v017_reexports_v015_surface():
    q = mx.random.normal((1, 8, 2, 4))
    out = v017.na1d(q, q, q, kernel_size=3)
    assert out.shape == (1, 8, 2, 4)


def test_v017_has_3d_stub():
    with pytest.raises(NotImplementedError):
        v017.NeighborhoodAttention3D()
