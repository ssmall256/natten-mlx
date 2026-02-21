import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.functional import na1d, na2d


def test_na1d_grad_shape():
    q = mx.random.normal((1, 5, 2, 3))
    k = mx.random.normal((1, 5, 2, 3))
    v = mx.random.normal((1, 5, 2, 3))

    def loss_fn(q_in):
        out = na1d(q_in, k, v, kernel_size=3)
        return mx.sum(out)

    grad = mx.grad(loss_fn)(q)
    assert grad.shape == q.shape


def test_na2d_value_and_grad():
    q = mx.random.normal((1, 4, 4, 2, 2))
    k = mx.random.normal((1, 4, 4, 2, 2))
    v = mx.random.normal((1, 4, 4, 2, 2))

    def loss_fn(v_in):
        out = na2d(q, k, v_in, kernel_size=(3, 3))
        return mx.sum(out)

    value, grad = mx.value_and_grad(loss_fn)(v)
    assert np.isscalar(np.array(value).item())
    assert grad.shape == v.shape
