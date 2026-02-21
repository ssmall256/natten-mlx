import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.nn import NeighborhoodAttention2D


def test_neighborhood_attention_2d_module_shape():
    layer = NeighborhoodAttention2D(embed_dim=12, num_heads=3, kernel_size=(3, 3))
    x = mx.random.normal((2, 7, 5, 12))
    y = layer(x)
    assert y.shape == (2, 7, 5, 12)


def test_neighborhood_attention_2d_module_stride_downsamples():
    layer = NeighborhoodAttention2D(
        embed_dim=12,
        num_heads=3,
        kernel_size=(3, 3),
        stride=(2, 2),
    )
    x = mx.random.normal((2, 8, 7, 12))
    y = layer(x)
    assert y.shape == (2, 4, 4, 12)


def test_neighborhood_attention_2d_module_with_attn_drop():
    layer = NeighborhoodAttention2D(embed_dim=12, num_heads=3, kernel_size=(3, 3), attn_drop=0.1)
    x = mx.random.normal((2, 7, 5, 12))
    y = layer(x)
    assert y.shape == (2, 7, 5, 12)


def test_neighborhood_attention_2d_attn_drop_causal_supported():
    layer = NeighborhoodAttention2D(
        embed_dim=12,
        num_heads=3,
        kernel_size=(3, 3),
        is_causal=(True, False),
        attn_drop=0.1,
    )
    x = mx.random.normal((2, 7, 5, 12))
    y = layer(x)
    assert y.shape == (2, 7, 5, 12)


def test_neighborhood_attention_2d_attn_drop_stride_supported():
    layer = NeighborhoodAttention2D(
        embed_dim=12,
        num_heads=3,
        kernel_size=(3, 3),
        stride=(2, 1),
        attn_drop=0.1,
    )
    x = mx.random.normal((2, 7, 5, 12))
    y = layer(x)
    assert y.shape == (2, 4, 5, 12)
