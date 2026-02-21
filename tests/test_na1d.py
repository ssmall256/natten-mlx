import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.nn import NeighborhoodAttention1D


def test_neighborhood_attention_1d_module_shape():
    layer = NeighborhoodAttention1D(embed_dim=16, num_heads=4, kernel_size=3)
    x = mx.random.normal((2, 10, 16))
    y = layer(x)
    assert y.shape == (2, 10, 16)


def test_neighborhood_attention_1d_module_stride_downsamples():
    layer = NeighborhoodAttention1D(embed_dim=16, num_heads=4, kernel_size=3, stride=2)
    x = mx.random.normal((2, 11, 16))
    y = layer(x)
    assert y.shape == (2, 6, 16)


def test_neighborhood_attention_1d_module_with_attn_drop():
    layer = NeighborhoodAttention1D(embed_dim=16, num_heads=4, kernel_size=3, attn_drop=0.1)
    x = mx.random.normal((2, 10, 16))
    y = layer(x)
    assert y.shape == (2, 10, 16)


def test_neighborhood_attention_1d_attn_drop_causal_not_supported():
    layer = NeighborhoodAttention1D(
        embed_dim=16,
        num_heads=4,
        kernel_size=3,
        is_causal=True,
        attn_drop=0.1,
    )
    x = mx.random.normal((2, 10, 16))
    with pytest.raises(NotImplementedError):
        layer(x)


def test_neighborhood_attention_1d_attn_drop_stride_not_supported():
    layer = NeighborhoodAttention1D(
        embed_dim=16,
        num_heads=4,
        kernel_size=3,
        stride=2,
        attn_drop=0.1,
    )
    x = mx.random.normal((2, 10, 16))
    with pytest.raises(NotImplementedError):
        layer(x)
