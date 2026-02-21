import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import natten_mlx.compat.v014 as v014
import natten_mlx.compat.v015 as v015
import natten_mlx.compat.v017 as v017
import natten_mlx.compat.v020 as v020


def test_v014_split_pipeline_gradients_exist():
    q = mx.random.normal((1, 2, 10, 4))
    k = mx.random.normal((1, 2, 10, 4))
    v = mx.random.normal((1, 2, 10, 4))
    rpb = mx.random.normal((2, 2 * 3 - 1))

    def _f(q_in, k_in, v_in):
        logits = v014.natten1dqkrpb(q_in, k_in, rpb, kernel_size=3, dilation=1)
        attn = mx.softmax(logits, axis=-1)
        return v014.natten1dav(attn, v_in, kernel_size=3, dilation=1)

    target = mx.ones((1, 2, 10, 4), dtype=q.dtype)

    def _loss_q(q_in):
        return mx.sum(_f(q_in, k, v) * target)

    def _loss_k(k_in):
        return mx.sum(_f(q, k_in, v) * target)

    def _loss_v(v_in):
        return mx.sum(_f(q, k, v_in) * target)

    grad_q = mx.grad(_loss_q)(q)
    grad_k = mx.grad(_loss_k)(k)
    grad_v = mx.grad(_loss_v)(v)
    mx.eval(grad_q, grad_k, grad_v)
    assert grad_q.shape == q.shape
    assert grad_k.shape == k.shape
    assert grad_v.shape == v.shape
    assert np.isfinite(np.array(grad_q)).all()
    assert np.isfinite(np.array(grad_k)).all()
    assert np.isfinite(np.array(grad_v)).all()


@pytest.mark.parametrize("compat_module", [v015, v017])
def test_v015_v017_fused_na1d_gradients_exist(compat_module):
    q = mx.random.normal((1, 11, 2, 4))
    k = mx.random.normal((1, 11, 2, 4))
    v = mx.random.normal((1, 11, 2, 4))

    def _f(q_in, k_in, v_in):
        return compat_module.na1d(
            q_in,
            k_in,
            v_in,
            kernel_size=3,
            dilation=1,
            is_causal=False,
            scale=0.37,
        )

    target = mx.ones((1, 11, 2, 4), dtype=q.dtype)

    def _loss_q(q_in):
        return mx.sum(_f(q_in, k, v) * target)

    def _loss_k(k_in):
        return mx.sum(_f(q, k_in, v) * target)

    def _loss_v(v_in):
        return mx.sum(_f(q, k, v_in) * target)

    grad_q = mx.grad(_loss_q)(q)
    grad_k = mx.grad(_loss_k)(k)
    grad_v = mx.grad(_loss_v)(v)
    mx.eval(grad_q, grad_k, grad_v)
    assert grad_q.shape == q.shape
    assert grad_k.shape == k.shape
    assert grad_v.shape == v.shape


def test_v020_fused_na1d_gradients_exist():
    q = mx.random.normal((1, 11, 2, 4))
    k = mx.random.normal((1, 11, 2, 4))
    v = mx.random.normal((1, 11, 2, 4))

    def _f(q_in, k_in, v_in):
        return v020.na1d(
            q_in,
            k_in,
            v_in,
            kernel_size=3,
            stride=1,
            dilation=1,
            is_causal=False,
            scale=0.37,
        )

    target = mx.ones((1, 11, 2, 4), dtype=q.dtype)

    def _loss_q(q_in):
        return mx.sum(_f(q_in, k, v) * target)

    def _loss_k(k_in):
        return mx.sum(_f(q, k_in, v) * target)

    def _loss_v(v_in):
        return mx.sum(_f(q, k, v_in) * target)

    grad_q = mx.grad(_loss_q)(q)
    grad_k = mx.grad(_loss_k)(k)
    grad_v = mx.grad(_loss_v)(v)
    mx.eval(grad_q, grad_k, grad_v)
    assert grad_q.shape == q.shape
    assert grad_k.shape == k.shape
    assert grad_v.shape == v.shape
