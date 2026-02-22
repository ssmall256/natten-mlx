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


def _grad_via_na1d(q, k, v):
    def loss_fn(q_in):
        return mx.sum(
            na1d(
                q_in,
                k,
                v,
                kernel_size=3,
                stride=1,
                dilation=1,
                is_causal=False,
                scale=0.5,
            )
        )

    return mx.grad(loss_fn)(q)


def _grad_via_na2d(q, k, v):
    def loss_fn(q_in):
        return mx.sum(
            na2d(
                q_in,
                k,
                v,
                kernel_size=(3, 3),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=(False, False),
                scale=0.5,
            )
        )

    return mx.grad(loss_fn)(q)


def _grad_via_na3d(q, k, v):
    def loss_fn(q_in):
        return mx.sum(
            na3d(
                q_in,
                k,
                v,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
                is_causal=(False, False, False),
                scale=0.5,
            )
        )

    return mx.grad(loss_fn)(q)


def test_nanobind_fused_backward_1d_dispatch(_backend_nanobind):
    q = mx.random.normal((1, 21, 2, 16))
    k = mx.random.normal((1, 21, 2, 16))
    v = mx.random.normal((1, 21, 2, 16))
    out = _grad_via_na1d(q, k, v)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na1d_backward") == "fused"


def test_nanobind_fused_backward_2d_dispatch(_backend_nanobind):
    q = mx.random.normal((1, 11, 9, 2, 16))
    k = mx.random.normal((1, 11, 9, 2, 16))
    v = mx.random.normal((1, 11, 9, 2, 16))
    out = _grad_via_na2d(q, k, v)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na2d_backward") == "fused"


def test_nanobind_fused_backward_3d_dispatch(_backend_nanobind):
    q = mx.random.normal((1, 5, 6, 7, 2, 12))
    k = mx.random.normal((1, 5, 6, 7, 2, 12))
    v = mx.random.normal((1, 5, 6, 7, 2, 12))
    out = _grad_via_na3d(q, k, v)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na3d_backward") == "fused"


def test_nanobind_fused_backward_fallback_chain(_backend_nanobind):
    q = mx.random.normal((1, 21, 2, 16))
    k = mx.random.normal((1, 21, 2, 16))
    v = mx.random.normal((1, 21, 2, 16))

    _EXT._debug_force_fused_failure(True)
    out = _grad_via_na1d(q, k, v)
    mx.eval(out)
    assert _EXT._debug_get_last_route("na1d_backward") == "split"

    _EXT._debug_force_split_failure(True)
    out = _grad_via_na1d(q, k, v)
    mx.eval(out)
    assert _EXT._debug_get_last_route("na1d_backward") == "pure"
