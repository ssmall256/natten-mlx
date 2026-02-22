import mlx.core as mx
import pytest

import natten_mlx
from natten_mlx import na1d, na2d, set_backend
from natten_mlx.functional import na3d

_EXT = pytest.importorskip(
    "natten_mlx._core._nanobind_ext",
    reason="requires compiled nanobind extension",
)


@pytest.fixture
def _backend_nanobind():
    previous = natten_mlx.get_backend()
    set_backend("nanobind")
    _EXT._debug_clear_last_routes()
    _EXT._debug_force_fused_failure(False)
    _EXT._debug_force_split_failure(False)
    try:
        yield
    finally:
        _EXT._debug_force_fused_failure(False)
        _EXT._debug_force_split_failure(False)
        _EXT._debug_clear_last_routes()
        set_backend(previous)


@pytest.mark.parametrize("is_causal,stride,dim", [(False, 1, 16), (True, 2, 12)])
def test_nanobind_fused_1d_dispatch(_backend_nanobind, is_causal: bool, stride: int, dim: int):
    q = mx.random.normal((1, 19, 2, dim))
    k = mx.random.normal((1, 19, 2, dim))
    v = mx.random.normal((1, 19, 2, dim))
    out = na1d(q, k, v, kernel_size=3, stride=stride, dilation=1, is_causal=is_causal, scale=0.5)
    mx.eval(out)
    assert out.shape == (1, (19 + stride - 1) // stride, 2, dim)
    assert _EXT._debug_get_last_route("na1d_forward") == "fused"


@pytest.mark.parametrize(
    "is_causal,stride,dim",
    [((False, False), (1, 1), 16), ((True, False), (2, 1), 12)],
)
def test_nanobind_fused_2d_dispatch(_backend_nanobind, is_causal, stride, dim: int):
    q = mx.random.normal((1, 11, 9, 2, dim))
    k = mx.random.normal((1, 11, 9, 2, dim))
    v = mx.random.normal((1, 11, 9, 2, dim))
    out = na2d(q, k, v, kernel_size=(3, 3), stride=stride, dilation=(1, 1), is_causal=is_causal, scale=0.5)
    mx.eval(out)
    assert out.shape == (1, (11 + stride[0] - 1) // stride[0], (9 + stride[1] - 1) // stride[1], 2, dim)
    assert _EXT._debug_get_last_route("na2d_forward") == "fused"


@pytest.mark.parametrize(
    "is_causal,stride,dim",
    [((False, False, False), (1, 1, 1), 16), ((True, False, True), (2, 1, 1), 12)],
)
def test_nanobind_fused_3d_dispatch(_backend_nanobind, is_causal, stride, dim: int):
    q = mx.random.normal((1, 7, 6, 5, 2, dim))
    k = mx.random.normal((1, 7, 6, 5, 2, dim))
    v = mx.random.normal((1, 7, 6, 5, 2, dim))
    out = na3d(q, k, v, kernel_size=(3, 3, 3), stride=stride, dilation=(1, 1, 1), is_causal=is_causal, scale=0.5)
    mx.eval(out)
    assert out.shape == (
        1,
        (7 + stride[0] - 1) // stride[0],
        (6 + stride[1] - 1) // stride[1],
        (5 + stride[2] - 1) // stride[2],
        2,
        dim,
    )
    assert _EXT._debug_get_last_route("na3d_forward") == "fused"


def test_nanobind_fused_forward_fallback_chain(_backend_nanobind):
    q = mx.random.normal((1, 17, 2, 12))
    k = mx.random.normal((1, 17, 2, 12))
    v = mx.random.normal((1, 17, 2, 12))

    _EXT._debug_force_fused_failure(True)
    out = na1d(q, k, v, kernel_size=3, stride=1, dilation=1, is_causal=False, scale=0.5)
    mx.eval(out)
    assert _EXT._debug_get_last_route("na1d_forward") == "split"

    _EXT._debug_force_split_failure(True)
    out = na1d(q, k, v, kernel_size=3, stride=1, dilation=1, is_causal=False, scale=0.5)
    mx.eval(out)
    assert _EXT._debug_get_last_route("na1d_forward") == "pure"
