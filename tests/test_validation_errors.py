"""Input validation error tests for functional API across all dimensions."""

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
# 1D: wrong ndim
# ---------------------------------------------------------------------------


def test_na1d_rejects_3d_input():
    q = mx.random.normal((1, 5, 4))
    with pytest.raises(ValueError, match="na1d"):
        na1d(q, q, q, kernel_size=3)


def test_na1d_rejects_5d_input():
    q = mx.random.normal((1, 5, 2, 4, 1))
    with pytest.raises(ValueError, match="na1d"):
        na1d(q, q, q, kernel_size=3)


def test_na1d_qk_rejects_wrong_ndim():
    q = mx.random.normal((1, 5, 4))
    with pytest.raises(ValueError):
        na1d_qk(q, q, kernel_size=3)


# ---------------------------------------------------------------------------
# 2D: wrong ndim
# ---------------------------------------------------------------------------


def test_na2d_rejects_4d_input():
    q = mx.random.normal((1, 5, 5, 4))
    with pytest.raises(ValueError, match="na2d"):
        na2d(q, q, q, kernel_size=(3, 3))


def test_na2d_rejects_6d_input():
    q = mx.random.normal((1, 5, 5, 2, 4, 1))
    with pytest.raises(ValueError, match="na2d"):
        na2d(q, q, q, kernel_size=(3, 3))


# ---------------------------------------------------------------------------
# 3D: wrong ndim
# ---------------------------------------------------------------------------


def test_na3d_rejects_5d_input():
    q = mx.random.normal((1, 4, 4, 4, 4))
    with pytest.raises(ValueError, match="na3d"):
        na3d(q, q, q, kernel_size=(3, 3, 3))


def test_na3d_rejects_7d_input():
    q = mx.random.normal((1, 4, 4, 4, 2, 4, 1))
    with pytest.raises(ValueError, match="na3d"):
        na3d(q, q, q, kernel_size=(3, 3, 3))


# ---------------------------------------------------------------------------
# Shape mismatches between Q, K, V
# ---------------------------------------------------------------------------


def test_na1d_rejects_qk_shape_mismatch():
    q = mx.random.normal((1, 8, 2, 4))
    k = mx.random.normal((1, 6, 2, 4))
    v = mx.random.normal((1, 8, 2, 4))
    with pytest.raises(ValueError, match="Spatial dimensions must match"):
        na1d(q, k, v, kernel_size=3)


def test_na1d_rejects_qv_shape_mismatch():
    q = mx.random.normal((1, 8, 2, 4))
    k = mx.random.normal((1, 8, 2, 4))
    v = mx.random.normal((1, 8, 2, 8))
    with pytest.raises(ValueError, match="Head dim must match"):
        na1d(q, k, v, kernel_size=3)


def test_na2d_rejects_shape_mismatch():
    q = mx.random.normal((1, 6, 6, 2, 4))
    k = mx.random.normal((1, 5, 6, 2, 4))
    with pytest.raises(ValueError, match="Spatial dimensions must match"):
        na2d(q, k, q, kernel_size=(3, 3))


def test_na3d_rejects_shape_mismatch():
    q = mx.random.normal((1, 5, 5, 5, 2, 4))
    k = mx.random.normal((1, 5, 5, 4, 2, 4))
    with pytest.raises(ValueError, match="Spatial dimensions must match"):
        na3d(q, k, q, kernel_size=(3, 3, 3))


# ---------------------------------------------------------------------------
# Kernel size vs input size
# ---------------------------------------------------------------------------


def test_na1d_rejects_kernel_too_large():
    q = mx.random.normal((1, 4, 2, 2))
    with pytest.raises(ValueError):
        na1d(q, q, q, kernel_size=5)


def test_na2d_rejects_kernel_too_large():
    q = mx.random.normal((1, 4, 4, 2, 2))
    with pytest.raises(ValueError):
        na2d(q, q, q, kernel_size=(5, 3))


def test_na3d_rejects_kernel_too_large():
    q = mx.random.normal((1, 4, 4, 4, 2, 2))
    with pytest.raises(ValueError):
        na3d(q, q, q, kernel_size=(5, 3, 3))


# ---------------------------------------------------------------------------
# Stride vs kernel size
# ---------------------------------------------------------------------------


def test_na1d_rejects_stride_exceeding_kernel():
    q = mx.random.normal((1, 10, 2, 2))
    with pytest.raises(ValueError):
        na1d(q, q, q, kernel_size=3, stride=4)


def test_na2d_rejects_stride_exceeding_kernel():
    q = mx.random.normal((1, 10, 10, 2, 2))
    with pytest.raises(ValueError):
        na2d(q, q, q, kernel_size=(3, 3), stride=(4, 1))


def test_na3d_rejects_stride_exceeding_kernel():
    q = mx.random.normal((1, 10, 10, 10, 2, 2))
    with pytest.raises(ValueError):
        na3d(q, q, q, kernel_size=(3, 3, 3), stride=(4, 1, 1))


# ---------------------------------------------------------------------------
# Dilation * kernel > input
# ---------------------------------------------------------------------------


def test_na1d_rejects_dilation_kernel_exceeding_input():
    q = mx.random.normal((1, 5, 2, 2))
    with pytest.raises(ValueError):
        na1d(q, q, q, kernel_size=3, dilation=2)


def test_na2d_rejects_dilation_kernel_exceeding_input():
    q = mx.random.normal((1, 5, 5, 2, 2))
    with pytest.raises(ValueError):
        na2d(q, q, q, kernel_size=(3, 3), dilation=(2, 2))


def test_na3d_rejects_dilation_kernel_exceeding_input():
    q = mx.random.normal((1, 5, 5, 5, 2, 2))
    with pytest.raises(ValueError):
        na3d(q, q, q, kernel_size=(3, 3, 3), dilation=(2, 2, 2))


# ---------------------------------------------------------------------------
# Split ops: same validation
# ---------------------------------------------------------------------------


def test_na1d_qk_rejects_kernel_too_large():
    q = mx.random.normal((1, 4, 2, 2))
    with pytest.raises(ValueError):
        na1d_qk(q, q, kernel_size=5)


def test_na2d_qk_rejects_kernel_too_large():
    q = mx.random.normal((1, 4, 4, 2, 2))
    with pytest.raises(ValueError):
        na2d_qk(q, q, kernel_size=(5, 3))


def test_na3d_qk_rejects_kernel_too_large():
    q = mx.random.normal((1, 4, 4, 4, 2, 2))
    with pytest.raises(ValueError):
        na3d_qk(q, q, kernel_size=(5, 3, 3))
