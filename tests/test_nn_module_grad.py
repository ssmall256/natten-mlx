"""Gradient tests for NN module wrappers across all dimensions."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.nn import (
    NeighborhoodAttention1D,
    NeighborhoodAttention2D,
    NeighborhoodAttention3D,
)


# ---------------------------------------------------------------------------
# 1D module gradients
# ---------------------------------------------------------------------------


def test_na1d_module_grad_flows():
    layer = NeighborhoodAttention1D(embed_dim=16, num_heads=4, kernel_size=3)
    x = mx.random.normal((1, 10, 16))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na1d_module_param_grad():
    layer = NeighborhoodAttention1D(embed_dim=16, num_heads=4, kernel_size=3)
    x = mx.random.normal((1, 10, 16))

    def loss_fn(params):
        layer.update(params)
        return mx.sum(layer(x))

    grads = mx.grad(loss_fn)(layer.parameters())
    # QKV weight gradient should exist and be finite
    assert "qkv" in grads
    assert "weight" in grads["qkv"]
    assert np.isfinite(np.array(grads["qkv"]["weight"])).all()


def test_na1d_module_grad_with_stride():
    layer = NeighborhoodAttention1D(embed_dim=16, num_heads=4, kernel_size=3, stride=2)
    x = mx.random.normal((1, 10, 16))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na1d_module_grad_with_causal():
    layer = NeighborhoodAttention1D(
        embed_dim=16, num_heads=4, kernel_size=3, is_causal=True
    )
    x = mx.random.normal((1, 10, 16))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na1d_module_grad_with_dilation():
    layer = NeighborhoodAttention1D(
        embed_dim=16, num_heads=4, kernel_size=3, dilation=2
    )
    x = mx.random.normal((1, 10, 16))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na1d_module_grad_with_qkv_bias_false():
    layer = NeighborhoodAttention1D(
        embed_dim=16, num_heads=4, kernel_size=3, qkv_bias=False
    )
    x = mx.random.normal((1, 10, 16))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape


def test_na1d_module_grad_with_custom_scale():
    layer = NeighborhoodAttention1D(
        embed_dim=16, num_heads=4, kernel_size=3, qk_scale=0.1
    )
    x = mx.random.normal((1, 10, 16))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape


# ---------------------------------------------------------------------------
# 2D module gradients
# ---------------------------------------------------------------------------


def test_na2d_module_grad_flows():
    layer = NeighborhoodAttention2D(embed_dim=12, num_heads=3, kernel_size=(3, 3))
    x = mx.random.normal((1, 7, 5, 12))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na2d_module_grad_with_stride():
    layer = NeighborhoodAttention2D(
        embed_dim=12, num_heads=3, kernel_size=(3, 3), stride=(2, 2)
    )
    x = mx.random.normal((1, 8, 8, 12))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na2d_module_grad_with_causal():
    layer = NeighborhoodAttention2D(
        embed_dim=12, num_heads=3, kernel_size=(3, 3), is_causal=(True, False)
    )
    x = mx.random.normal((1, 7, 5, 12))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na2d_module_param_grad():
    layer = NeighborhoodAttention2D(embed_dim=12, num_heads=3, kernel_size=(3, 3))
    x = mx.random.normal((1, 7, 5, 12))

    def loss_fn(params):
        layer.update(params)
        return mx.sum(layer(x))

    grads = mx.grad(loss_fn)(layer.parameters())
    assert "qkv" in grads
    assert np.isfinite(np.array(grads["qkv"]["weight"])).all()


# ---------------------------------------------------------------------------
# 3D module gradients
# ---------------------------------------------------------------------------


def test_na3d_module_grad_flows():
    layer = NeighborhoodAttention3D(embed_dim=12, num_heads=3, kernel_size=(3, 3, 3))
    x = mx.random.normal((1, 5, 5, 5, 12))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na3d_module_grad_with_stride():
    layer = NeighborhoodAttention3D(
        embed_dim=12, num_heads=3, kernel_size=(3, 3, 3), stride=(2, 2, 2)
    )
    x = mx.random.normal((1, 6, 6, 6, 12))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na3d_module_grad_with_causal():
    layer = NeighborhoodAttention3D(
        embed_dim=12, num_heads=3, kernel_size=(3, 3, 3), is_causal=(True, False, False)
    )
    x = mx.random.normal((1, 5, 5, 5, 12))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na3d_module_param_grad():
    layer = NeighborhoodAttention3D(embed_dim=12, num_heads=3, kernel_size=(3, 3, 3))
    x = mx.random.normal((1, 5, 5, 5, 12))

    def loss_fn(params):
        layer.update(params)
        return mx.sum(layer(x))

    grads = mx.grad(loss_fn)(layer.parameters())
    assert "qkv" in grads
    assert np.isfinite(np.array(grads["qkv"]["weight"])).all()


# ---------------------------------------------------------------------------
# Module validation
# ---------------------------------------------------------------------------


def test_na1d_module_rejects_bad_embed_dim():
    with pytest.raises(ValueError, match="embed_dim must be positive"):
        NeighborhoodAttention1D(embed_dim=0, num_heads=1, kernel_size=3)


def test_na1d_module_rejects_bad_num_heads():
    with pytest.raises(ValueError, match="num_heads must be positive"):
        NeighborhoodAttention1D(embed_dim=16, num_heads=0, kernel_size=3)


def test_na1d_module_rejects_indivisible():
    with pytest.raises(ValueError, match="divisible"):
        NeighborhoodAttention1D(embed_dim=15, num_heads=4, kernel_size=3)


def test_na1d_module_rejects_wrong_ndim():
    layer = NeighborhoodAttention1D(embed_dim=16, num_heads=4, kernel_size=3)
    with pytest.raises(ValueError, match="Expected input shape"):
        layer(mx.random.normal((2, 10, 16, 1)))


def test_na1d_module_rejects_wrong_channels():
    layer = NeighborhoodAttention1D(embed_dim=16, num_heads=4, kernel_size=3)
    with pytest.raises(ValueError, match="channel dim"):
        layer(mx.random.normal((2, 10, 8)))


# ---------------------------------------------------------------------------
# GQA / MQA nn module tests
# ---------------------------------------------------------------------------


def test_na1d_module_gqa_forward_shape():
    layer = NeighborhoodAttention1D(
        embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
    )
    x = mx.random.normal((1, 10, 16))
    out = layer(x)
    assert out.shape == x.shape


def test_na1d_module_mqa_forward_shape():
    layer = NeighborhoodAttention1D(
        embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=1
    )
    x = mx.random.normal((1, 10, 16))
    out = layer(x)
    assert out.shape == x.shape


def test_na1d_module_gqa_grad_flows():
    layer = NeighborhoodAttention1D(
        embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
    )
    x = mx.random.normal((1, 10, 16))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na1d_module_gqa_param_grad():
    layer = NeighborhoodAttention1D(
        embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=2
    )
    x = mx.random.normal((1, 10, 16))

    def loss_fn(params):
        layer.update(params)
        return mx.sum(layer(x))

    grads = mx.grad(loss_fn)(layer.parameters())
    assert "q_proj" in grads
    assert "kv_proj" in grads
    assert np.isfinite(np.array(grads["q_proj"]["weight"])).all()


def test_na1d_module_gqa_indivisible_raises():
    with pytest.raises(ValueError, match="divisible"):
        NeighborhoodAttention1D(
            embed_dim=16, num_heads=4, kernel_size=3, num_kv_heads=3
        )


def test_na2d_module_gqa_forward_shape():
    layer = NeighborhoodAttention2D(
        embed_dim=12, num_heads=3, kernel_size=(3, 3), num_kv_heads=1
    )
    x = mx.random.normal((1, 7, 5, 12))
    out = layer(x)
    assert out.shape == x.shape


def test_na2d_module_gqa_grad_flows():
    layer = NeighborhoodAttention2D(
        embed_dim=12, num_heads=3, kernel_size=(3, 3), num_kv_heads=1
    )
    x = mx.random.normal((1, 7, 5, 12))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()


def test_na2d_module_gqa_param_grad():
    layer = NeighborhoodAttention2D(
        embed_dim=12, num_heads=3, kernel_size=(3, 3), num_kv_heads=1
    )
    x = mx.random.normal((1, 7, 5, 12))

    def loss_fn(params):
        layer.update(params)
        return mx.sum(layer(x))

    grads = mx.grad(loss_fn)(layer.parameters())
    assert "q_proj" in grads
    assert "kv_proj" in grads


def test_na3d_module_gqa_forward_shape():
    layer = NeighborhoodAttention3D(
        embed_dim=12, num_heads=3, kernel_size=(3, 3, 3), num_kv_heads=1
    )
    x = mx.random.normal((1, 5, 5, 5, 12))
    out = layer(x)
    assert out.shape == x.shape


def test_na3d_module_gqa_grad_flows():
    layer = NeighborhoodAttention3D(
        embed_dim=12, num_heads=3, kernel_size=(3, 3, 3), num_kv_heads=1
    )
    x = mx.random.normal((1, 5, 5, 5, 12))

    def loss_fn(x_in):
        return mx.sum(layer(x_in))

    grad = mx.grad(loss_fn)(x)
    assert grad.shape == x.shape
    assert np.isfinite(np.array(grad)).all()
