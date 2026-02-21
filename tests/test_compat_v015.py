import pytest

mx = pytest.importorskip("mlx.core")

import natten_mlx.compat.v015 as v015


def test_v015_fused_na1d_shape():
    q = mx.random.normal((2, 16, 4, 8))
    out = v015.na1d(q, q, q, kernel_size=5, dilation=1, is_causal=False)
    assert out.shape == (2, 16, 4, 8)


def test_v015_feature_flags():
    assert v015.has_mlx() is True
    assert v015.has_cuda() is False
