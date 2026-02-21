import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.nn import NeighborhoodAttention3D


def test_neighborhood_attention_3d_module_shape():
    layer = NeighborhoodAttention3D(embed_dim=12, num_heads=3, kernel_size=(3, 3, 3))
    x = mx.random.normal((2, 5, 7, 6, 12))
    y = layer(x)
    assert y.shape == (2, 5, 7, 6, 12)


def test_neighborhood_attention_3d_module_stride_downsamples():
    layer = NeighborhoodAttention3D(
        embed_dim=12,
        num_heads=3,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 3),
    )
    x = mx.random.normal((2, 8, 7, 10, 12))
    y = layer(x)
    assert y.shape == (2, 4, 4, 4, 12)


def test_neighborhood_attention_3d_module_with_attn_drop():
    layer = NeighborhoodAttention3D(embed_dim=12, num_heads=3, kernel_size=(3, 3, 3), attn_drop=0.1)
    x = mx.random.normal((2, 5, 7, 6, 12))
    y = layer(x)
    assert y.shape == (2, 5, 7, 6, 12)
