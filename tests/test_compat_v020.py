import pytest

mx = pytest.importorskip("mlx.core")

import natten_mlx.compat.v020 as v020


def test_v020_module_with_stride():
    layer = v020.NeighborhoodAttention1D(
        embed_dim=128,
        num_heads=4,
        kernel_size=7,
        stride=2,
    )
    x = mx.random.normal((2, 33, 128))
    y = layer(x)
    assert y.shape == (2, 17, 128)


def test_v020_na1d_stride_shape():
    q = mx.random.normal((2, 15, 2, 8))
    out = v020.na1d(q, q, q, kernel_size=3, stride=2)
    assert out.shape == (2, 8, 2, 8)


def test_v020_na3d_stride_shape():
    q = mx.random.normal((1, 9, 8, 7, 2, 4))
    out = v020.na3d(q, q, q, kernel_size=(3, 3, 3), stride=(2, 3, 2))
    assert out.shape == (1, 5, 3, 4, 2, 4)


def test_v020_feature_detection_functions_return_bool():
    assert isinstance(v020.has_cuda(), bool)
    assert isinstance(v020.has_mps(), bool)
    assert isinstance(v020.has_mlx(), bool)
    assert isinstance(v020.has_fna(), bool)
