import pytest

mx = pytest.importorskip("mlx.core")

import natten_mlx.compat.v015 as v015


def test_v015_fused_na1d_shape():
    q = mx.random.normal((2, 16, 4, 8))
    out = v015.na1d(q, q, q, kernel_size=5, dilation=1, is_causal=False)
    assert out.shape == (2, 16, 4, 8)


def test_v015_fused_na3d_shape():
    q = mx.random.normal((1, 6, 7, 8, 2, 4))
    out = v015.na3d(q, q, q, kernel_size=(3, 3, 3), dilation=(1, 2, 1), is_causal=(True, False, False))
    assert out.shape == (1, 6, 7, 8, 2, 4)


def test_v015_neighborhood_attention_3d_module_shape():
    layer = v015.NeighborhoodAttention3D(dim=24, kernel_size=(3, 3, 3), num_heads=3)
    x = mx.random.normal((2, 5, 6, 7, 24))
    y = layer(x)
    assert y.shape == (2, 5, 6, 7, 24)


def test_v015_feature_flags():
    assert v015.has_mlx() is True
    assert v015.has_cuda() is False
