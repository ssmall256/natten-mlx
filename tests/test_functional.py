import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.functional import na1d, na2d


def test_na1d_output_shape_with_stride():
    q = mx.random.normal((2, 9, 4, 8))
    out = na1d(q, q, q, kernel_size=3, stride=2, dilation=1, is_causal=False)
    assert out.shape == (2, 5, 4, 8)


def test_na2d_output_shape_with_stride():
    q = mx.random.normal((2, 8, 7, 2, 4))
    out = na2d(q, q, q, kernel_size=(3, 3), stride=(2, 3), dilation=1, is_causal=False)
    assert out.shape == (2, 4, 3, 2, 4)


def test_na1d_causal_first_position_attends_to_self_only():
    q = mx.ones((1, 5, 1, 1))
    k = mx.ones((1, 5, 1, 1))
    v = mx.arange(5, dtype=mx.float32).reshape((1, 5, 1, 1))

    out = na1d(q, k, v, kernel_size=3, is_causal=True, scale=1.0)
    out_np = np.array(out).reshape(5)

    assert np.isclose(out_np[0], 0.0)
    assert out_np[1] <= 1.0


def test_validation_errors():
    q = mx.random.normal((1, 4, 2, 2))
    with pytest.raises(ValueError):
        na1d(q, q, q, kernel_size=7)

    with pytest.raises(ValueError):
        na1d(q, q, q, kernel_size=3, stride=4)

    q2 = mx.random.normal((1, 4, 4, 2, 2))
    with pytest.raises(ValueError):
        na2d(q2, q2, q2, kernel_size=(5, 5), dilation=(2, 2))
