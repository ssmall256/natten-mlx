"""Gradient tests for split QK/AV operations across all dimensions."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.functional import (
    na1d,
    na1d_av,
    na1d_qk,
    na2d,
    na2d_av,
    na2d_qk,
    na3d,
    na3d_av,
    na3d_qk,
)


# ---------------------------------------------------------------------------
# 1D split ops
# ---------------------------------------------------------------------------


def test_na1d_qk_grad_shape():
    q = mx.random.normal((1, 8, 2, 4))
    k = mx.random.normal((1, 8, 2, 4))

    def loss_fn(q_in):
        logits = na1d_qk(q_in, k, kernel_size=3)
        return mx.sum(logits)

    grad = mx.grad(loss_fn)(q)
    assert grad.shape == q.shape


def test_na1d_qk_grad_k_shape():
    q = mx.random.normal((1, 8, 2, 4))
    k = mx.random.normal((1, 8, 2, 4))

    def loss_fn(k_in):
        logits = na1d_qk(q, k_in, kernel_size=3)
        return mx.sum(logits)

    grad = mx.grad(loss_fn)(k)
    assert grad.shape == k.shape


def test_na1d_av_grad_shape():
    q = mx.random.normal((1, 8, 2, 4))
    k = mx.random.normal((1, 8, 2, 4))
    v = mx.random.normal((1, 8, 2, 4))

    logits = na1d_qk(q, k, kernel_size=3)
    attn = mx.softmax(logits, axis=-1)

    def loss_fn(v_in):
        out = na1d_av(attn, v_in, kernel_size=3)
        return mx.sum(out)

    grad = mx.grad(loss_fn)(v)
    assert grad.shape == v.shape


def test_na1d_split_grad_matches_fused():
    """Split QK->softmax->AV gradient should match fused gradient."""
    q = mx.random.normal((1, 8, 2, 4))
    k = mx.random.normal((1, 8, 2, 4))
    v = mx.random.normal((1, 8, 2, 4))

    def fused_loss(v_in):
        return mx.sum(na1d(q, k, v_in, kernel_size=3))

    def split_loss(v_in):
        logits = na1d_qk(q, k, kernel_size=3)
        attn = mx.softmax(logits, axis=-1)
        return mx.sum(na1d_av(attn, v_in, kernel_size=3))

    grad_fused = mx.grad(fused_loss)(v)
    grad_split = mx.grad(split_loss)(v)
    assert np.allclose(np.array(grad_fused), np.array(grad_split), atol=1e-4, rtol=1e-4)


def test_na1d_split_grad_with_stride_and_causal():
    q = mx.random.normal((1, 10, 2, 4))
    k = mx.random.normal((1, 10, 2, 4))
    v = mx.random.normal((1, 10, 2, 4))

    def loss_fn(q_in):
        logits = na1d_qk(q_in, k, kernel_size=3, stride=2, is_causal=True, scale=0.3)
        attn = mx.softmax(logits, axis=-1)
        return mx.sum(na1d_av(attn, v, kernel_size=3, stride=2, is_causal=True))

    grad = mx.grad(loss_fn)(q)
    assert grad.shape == q.shape
    assert np.isfinite(np.array(grad)).all()


# ---------------------------------------------------------------------------
# 2D split ops
# ---------------------------------------------------------------------------


def test_na2d_qk_grad_shape():
    q = mx.random.normal((1, 6, 6, 2, 4))
    k = mx.random.normal((1, 6, 6, 2, 4))

    def loss_fn(q_in):
        logits = na2d_qk(q_in, k, kernel_size=(3, 3))
        return mx.sum(logits)

    grad = mx.grad(loss_fn)(q)
    assert grad.shape == q.shape


def test_na2d_av_grad_shape():
    q = mx.random.normal((1, 6, 6, 2, 4))
    k = mx.random.normal((1, 6, 6, 2, 4))
    v = mx.random.normal((1, 6, 6, 2, 4))

    logits = na2d_qk(q, k, kernel_size=(3, 3))
    attn = mx.softmax(logits, axis=-1)

    def loss_fn(v_in):
        out = na2d_av(attn, v_in, kernel_size=(3, 3))
        return mx.sum(out)

    grad = mx.grad(loss_fn)(v)
    assert grad.shape == v.shape


def test_na2d_split_grad_matches_fused():
    q = mx.random.normal((1, 6, 6, 2, 4))
    k = mx.random.normal((1, 6, 6, 2, 4))
    v = mx.random.normal((1, 6, 6, 2, 4))

    def fused_loss(v_in):
        return mx.sum(na2d(q, k, v_in, kernel_size=(3, 3)))

    def split_loss(v_in):
        logits = na2d_qk(q, k, kernel_size=(3, 3))
        attn = mx.softmax(logits, axis=-1)
        return mx.sum(na2d_av(attn, v_in, kernel_size=(3, 3)))

    grad_fused = mx.grad(fused_loss)(v)
    grad_split = mx.grad(split_loss)(v)
    assert np.allclose(np.array(grad_fused), np.array(grad_split), atol=1e-4, rtol=1e-4)


def test_na2d_split_grad_with_stride_and_causal():
    q = mx.random.normal((1, 8, 8, 2, 4))
    k = mx.random.normal((1, 8, 8, 2, 4))
    v = mx.random.normal((1, 8, 8, 2, 4))

    def loss_fn(q_in):
        logits = na2d_qk(
            q_in, k, kernel_size=(3, 3), stride=(2, 2), is_causal=(True, False), scale=0.3
        )
        attn = mx.softmax(logits, axis=-1)
        return mx.sum(na2d_av(attn, v, kernel_size=(3, 3), stride=(2, 2), is_causal=(True, False)))

    grad = mx.grad(loss_fn)(q)
    assert grad.shape == q.shape
    assert np.isfinite(np.array(grad)).all()


# ---------------------------------------------------------------------------
# 3D split ops
# ---------------------------------------------------------------------------


def test_na3d_qk_grad_shape():
    q = mx.random.normal((1, 5, 5, 5, 2, 4))
    k = mx.random.normal((1, 5, 5, 5, 2, 4))

    def loss_fn(q_in):
        logits = na3d_qk(q_in, k, kernel_size=(3, 3, 3))
        return mx.sum(logits)

    grad = mx.grad(loss_fn)(q)
    assert grad.shape == q.shape


def test_na3d_av_grad_shape():
    q = mx.random.normal((1, 5, 5, 5, 2, 4))
    k = mx.random.normal((1, 5, 5, 5, 2, 4))
    v = mx.random.normal((1, 5, 5, 5, 2, 4))

    logits = na3d_qk(q, k, kernel_size=(3, 3, 3))
    attn = mx.softmax(logits, axis=-1)

    def loss_fn(v_in):
        out = na3d_av(attn, v_in, kernel_size=(3, 3, 3))
        return mx.sum(out)

    grad = mx.grad(loss_fn)(v)
    assert grad.shape == v.shape


def test_na3d_split_grad_matches_fused():
    q = mx.random.normal((1, 5, 5, 5, 2, 4))
    k = mx.random.normal((1, 5, 5, 5, 2, 4))
    v = mx.random.normal((1, 5, 5, 5, 2, 4))

    def fused_loss(v_in):
        return mx.sum(na3d(q, k, v_in, kernel_size=(3, 3, 3)))

    def split_loss(v_in):
        logits = na3d_qk(q, k, kernel_size=(3, 3, 3))
        attn = mx.softmax(logits, axis=-1)
        return mx.sum(na3d_av(attn, v_in, kernel_size=(3, 3, 3)))

    grad_fused = mx.grad(fused_loss)(v)
    grad_split = mx.grad(split_loss)(v)
    assert np.allclose(np.array(grad_fused), np.array(grad_split), atol=1e-4, rtol=1e-4)


def test_na3d_split_grad_with_causal():
    q = mx.random.normal((1, 5, 5, 5, 2, 4))
    k = mx.random.normal((1, 5, 5, 5, 2, 4))
    v = mx.random.normal((1, 5, 5, 5, 2, 4))

    def loss_fn(q_in):
        logits = na3d_qk(
            q_in, k, kernel_size=(3, 3, 3), is_causal=(True, False, False), scale=0.3
        )
        attn = mx.softmax(logits, axis=-1)
        return mx.sum(
            na3d_av(attn, v, kernel_size=(3, 3, 3), is_causal=(True, False, False))
        )

    grad = mx.grad(loss_fn)(q)
    assert grad.shape == q.shape
    assert np.isfinite(np.array(grad)).all()
