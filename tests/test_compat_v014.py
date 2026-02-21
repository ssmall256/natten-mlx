import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import natten_mlx.compat.v014 as v014
from natten_mlx.utils.window import compute_pb_start


def test_v014_module_1d_shape():
    layer = v014.NeighborhoodAttention1D(dim=128, kernel_size=7, num_heads=4)
    x = mx.random.normal((2, 32, 128))
    y = layer(x)
    assert y.shape == (2, 32, 128)


def test_v014_qkrpb_and_av_shapes():
    q = mx.random.normal((2, 4, 16, 8))
    k = mx.random.normal((2, 4, 16, 8))
    v = mx.random.normal((2, 4, 16, 8))
    rpb = mx.random.normal((4, 2 * 3 - 1))

    logits = v014.natten1dqkrpb(q, k, rpb, kernel_size=3, dilation=1)
    assert logits.shape == (2, 4, 16, 3)

    attn = mx.softmax(logits, axis=-1)
    out = v014.natten1dav(attn, v, kernel_size=3, dilation=1)
    assert out.shape == (2, 4, 16, 8)


def test_v014_rpb_changes_logits():
    q = mx.zeros((1, 2, 8, 4))
    k = mx.zeros((1, 2, 8, 4))
    rpb = mx.ones((2, 2 * 3 - 1))

    logits = v014.natten1dqkrpb(q, k, rpb, kernel_size=3, dilation=1)
    logits_np = np.array(logits)
    assert not np.allclose(logits_np, 0.0)


def test_v014_rpb_uses_pb_start_coupling_for_dilation():
    batch, heads, length, dim = 1, 1, 8, 4
    kernel_size = 3
    dilation = 2

    q = mx.zeros((batch, heads, length, dim))
    k = mx.zeros((batch, heads, length, dim))

    rpb_values = np.arange(2 * kernel_size - 1, dtype=np.float32)
    rpb = mx.array(rpb_values.reshape(1, -1))

    logits = v014.natten1dqkrpb(q, k, rpb, kernel_size=kernel_size, dilation=dilation)
    logits_np = np.array(logits)[0, 0]  # [L, K]

    pb = compute_pb_start(
        np.arange(length, dtype=np.int32),
        length,
        kernel_size,
        dilation,
    )
    expected_idx = pb[:, None] + np.arange(kernel_size, dtype=np.int32)[None, :]
    expected_idx = np.clip(expected_idx, 0, 2 * kernel_size - 2)
    expected = rpb_values[expected_idx]

    assert np.allclose(logits_np, expected)
