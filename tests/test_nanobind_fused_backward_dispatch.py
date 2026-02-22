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
    _EXT._debug_clear_last_kernels()
    _EXT._debug_force_fused_failure(False)
    _EXT._debug_force_split_failure(False)
    try:
        yield
    finally:
        _EXT._debug_force_fused_failure(False)
        _EXT._debug_force_split_failure(False)
        _EXT._debug_clear_last_routes()
        _EXT._debug_clear_last_kernels()
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
    assert _EXT._debug_get_last_route("na1d_backward") in ("fused", "v2_primitive")


def test_nanobind_fused_backward_2d_dispatch(_backend_nanobind):
    q = mx.random.normal((1, 11, 9, 2, 16))
    k = mx.random.normal((1, 11, 9, 2, 16))
    v = mx.random.normal((1, 11, 9, 2, 16))
    out = _grad_via_na2d(q, k, v)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na2d_backward") in ("fused", "v2_primitive")


def test_nanobind_fused_backward_3d_dispatch(_backend_nanobind):
    q = mx.random.normal((1, 5, 6, 7, 2, 12))
    k = mx.random.normal((1, 5, 6, 7, 2, 12))
    v = mx.random.normal((1, 5, 6, 7, 2, 12))
    out = _grad_via_na3d(q, k, v)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na3d_backward") in ("fused", "v2_primitive")


def test_nanobind_fused_backward_fallback_chain(_backend_nanobind, monkeypatch):
    monkeypatch.setenv("NATTEN_NANOBIND_DISABLE_V2", "1")
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


@pytest.mark.parametrize(
    ("kernel_size", "dilation", "is_causal", "expected_q_kernel", "expected_kv_kernel"),
    [
        (
            7,
            1,
            False,
            "na1d_fused_bwd_q_softmax_s1_vec4_fp32",
            "na1d_fused_bwd_kv_softmax_direct_u1d1_nc_vec4_fp32",
        ),
        (
            9,
            2,
            True,
            "na1d_fused_bwd_q_softmax_direct_s1_causal_k9_vec4_fp32",
            "na1d_fused_bwd_kv_softmax_direct_s1_causal_k9_vec4_fp32",
        ),
    ],
)
def test_nanobind_fused_backward_qkv_1d_direct_dispatch(
    _backend_nanobind, monkeypatch, kernel_size, dilation, is_causal, expected_q_kernel, expected_kv_kernel
):
    monkeypatch.setenv("NATTEN_NANOBIND_DISABLE_V2", "1")
    q = mx.random.normal((1, 64, 2, 16))
    k = mx.random.normal((1, 64, 2, 16))
    v = mx.random.normal((1, 64, 2, 16))

    def _loss(q_in):
        return mx.sum(
            na1d(
                q_in,
                k,
                v,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                is_causal=is_causal,
                scale=0.5,
            )
        )

    out = mx.grad(_loss)(q)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na1d_backward") == "fused"
    assert _EXT._debug_get_last_kernel("na1d_fused_backward_qk_grad_q") == expected_q_kernel
    assert _EXT._debug_get_last_kernel("na1d_fused_backward_qk") == expected_kv_kernel
    assert _EXT._debug_get_last_kernel("na1d_fused_backward_v") == expected_kv_kernel


def test_nanobind_fused_backward_qkv_1d_direct_dispatch_causal_k9_token_vec4(
    _backend_nanobind, monkeypatch,
):
    monkeypatch.setenv("NATTEN_NANOBIND_DISABLE_V2", "1")
    q = mx.random.normal((1, 128, 2, 64))
    k = mx.random.normal((1, 128, 2, 64))
    v = mx.random.normal((1, 128, 2, 64))

    def _loss(q_in):
        return mx.sum(
            na1d(
                q_in,
                k,
                v,
                kernel_size=9,
                stride=1,
                dilation=2,
                is_causal=True,
                scale=0.5,
            )
        )

    out = mx.grad(_loss)(q)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na1d_backward") == "fused"
    assert _EXT._debug_get_last_kernel("na1d_fused_backward_qk_grad_q") == (
        "na1d_fused_bwd_q_softmax_direct_s1_causal_k9_token_vec4_fp32"
    )
    assert _EXT._debug_get_last_kernel("na1d_fused_backward_qk") == (
        "na1d_fused_bwd_kv_softmax_direct_s1_causal_k9_vec4_fp32"
    )
    assert _EXT._debug_get_last_kernel("na1d_fused_backward_v") == (
        "na1d_fused_bwd_kv_softmax_direct_s1_causal_k9_vec4_fp32"
    )


def test_nanobind_fused_backward_qkv_tiled_2d_dispatch(_backend_nanobind, monkeypatch):
    monkeypatch.setenv("NATTEN_NANOBIND_DISABLE_V2", "1")
    monkeypatch.setenv("NATTEN_NANOBIND_FUSED_BWD_QKV_STAGE", "1")
    monkeypatch.setenv("NATTEN_NANOBIND_QKV_STAGE_MODE", "tiled")
    monkeypatch.setenv("NATTEN_NANOBIND_FUSED_BWD_2D_MODE", "fused")
    q = mx.random.normal((1, 32, 32, 8, 32))
    k = mx.random.normal((1, 32, 32, 8, 32))
    v = mx.random.normal((1, 32, 32, 8, 32))
    out = _grad_via_na2d(q, k, v)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na2d_backward") == "fused"
    assert _EXT._debug_get_last_kernel("na2d_fused_backward_qk_grad_q") == (
        "na2d_fused_bwd_q_softmax_u1d1_nc_vec4_fp32"
    )
    assert _EXT._debug_get_last_kernel("na2d_fused_backward_qk") == (
        "na2d_fused_bwd_kv_softmax_tiled_k3_vec4_fp32"
    )
    assert _EXT._debug_get_last_kernel("na2d_fused_backward_v") == (
        "na2d_fused_bwd_kv_softmax_tiled_k3_vec4_fp32"
    )


def test_nanobind_fused_backward_qkv_tiled_2d_k5_dispatch(_backend_nanobind, monkeypatch):
    monkeypatch.setenv("NATTEN_NANOBIND_DISABLE_V2", "1")
    monkeypatch.setenv("NATTEN_NANOBIND_FUSED_BWD_QKV_STAGE", "1")
    monkeypatch.setenv("NATTEN_NANOBIND_QKV_STAGE_MODE", "tiled")
    monkeypatch.setenv("NATTEN_NANOBIND_FUSED_BWD_2D_MODE", "fused")

    def loss_fn(q_in, k_in, v_in):
        return mx.sum(
            na2d(
                q_in,
                k_in,
                v_in,
                kernel_size=(5, 5),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=(False, False),
                scale=0.5,
            )
        )

    q = mx.random.normal((1, 24, 24, 4, 16))
    k = mx.random.normal((1, 24, 24, 4, 16))
    v = mx.random.normal((1, 24, 24, 4, 16))
    out = mx.grad(lambda q_in: loss_fn(q_in, k, v))(q)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na2d_backward") == "fused"
    assert _EXT._debug_get_last_kernel("na2d_fused_backward_qk_grad_q") == (
        "na2d_fused_bwd_q_softmax_u1d1_nc_vec4_fp32"
    )
    assert _EXT._debug_get_last_kernel("na2d_fused_backward_qk") == (
        "na2d_fused_bwd_kv_softmax_tiled_k5_vec4_fp32"
    )
    assert _EXT._debug_get_last_kernel("na2d_fused_backward_v") == (
        "na2d_fused_bwd_kv_softmax_tiled_k5_vec4_fp32"
    )


def test_nanobind_fused_backward_qkv_tiled_2d_k7_dispatch(_backend_nanobind, monkeypatch):
    monkeypatch.setenv("NATTEN_NANOBIND_DISABLE_V2", "1")
    monkeypatch.setenv("NATTEN_NANOBIND_FUSED_BWD_QKV_STAGE", "1")
    monkeypatch.setenv("NATTEN_NANOBIND_QKV_STAGE_MODE", "tiled")
    monkeypatch.setenv("NATTEN_NANOBIND_FUSED_BWD_2D_MODE", "fused")

    def loss_fn(q_in, k_in, v_in):
        return mx.sum(
            na2d(
                q_in,
                k_in,
                v_in,
                kernel_size=(7, 7),
                stride=(1, 1),
                dilation=(1, 1),
                is_causal=(False, False),
                scale=0.5,
            )
        )

    q = mx.random.normal((1, 24, 24, 4, 16))
    k = mx.random.normal((1, 24, 24, 4, 16))
    v = mx.random.normal((1, 24, 24, 4, 16))
    out = mx.grad(lambda q_in: loss_fn(q_in, k, v))(q)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na2d_backward") == "fused"
    assert _EXT._debug_get_last_kernel("na2d_fused_backward_qk_grad_q") == (
        "na2d_fused_bwd_q_softmax_u1d1_nc_vec4_fp32"
    )
    assert _EXT._debug_get_last_kernel("na2d_fused_backward_qk") == (
        "na2d_fused_bwd_kv_softmax_tiled_k7_vec4_fp32"
    )
    assert _EXT._debug_get_last_kernel("na2d_fused_backward_v") == (
        "na2d_fused_bwd_kv_softmax_tiled_k7_vec4_fp32"
    )


def test_nanobind_fused_backward_qkv_tiled_3d_dispatch(_backend_nanobind, monkeypatch):
    monkeypatch.setenv("NATTEN_NANOBIND_DISABLE_V2", "1")
    monkeypatch.setenv("NATTEN_NANOBIND_FUSED_BWD_QKV_STAGE", "1")
    monkeypatch.setenv("NATTEN_NANOBIND_QKV_STAGE_MODE", "tiled")
    monkeypatch.setenv("NATTEN_NANOBIND_FUSED_BWD_3D_MODE", "fused")
    q = mx.random.normal((1, 10, 12, 14, 4, 16))
    k = mx.random.normal((1, 10, 12, 14, 4, 16))
    v = mx.random.normal((1, 10, 12, 14, 4, 16))
    out = _grad_via_na3d(q, k, v)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na3d_backward") == "fused"
    assert _EXT._debug_get_last_kernel("na3d_fused_backward_qk_grad_q") in {
        "na3d_fused_bwd_q_softmax_k3_vec4_fp32",
        "na3d_fused_bwd_q_softmax_token_k3_vec4_fp32",
    }
    assert _EXT._debug_get_last_kernel("na3d_fused_backward_qk") == (
        "na3d_fused_bwd_kv_softmax_tiled_k3_vec4_fp32"
    )
    assert _EXT._debug_get_last_kernel("na3d_fused_backward_v") == (
        "na3d_fused_bwd_kv_softmax_tiled_k3_vec4_fp32"
    )


def test_nanobind_fused_backward_qkv_tiled_3d_k5_dispatch(_backend_nanobind, monkeypatch):
    monkeypatch.setenv("NATTEN_NANOBIND_DISABLE_V2", "1")
    monkeypatch.setenv("NATTEN_NANOBIND_FUSED_BWD_QKV_STAGE", "1")
    monkeypatch.setenv("NATTEN_NANOBIND_QKV_STAGE_MODE", "tiled")
    monkeypatch.setenv("NATTEN_NANOBIND_FUSED_BWD_3D_MODE", "fused")

    def loss_fn(q_in, k_in, v_in):
        return mx.sum(
            na3d(
                q_in,
                k_in,
                v_in,
                kernel_size=(5, 5, 5),
                stride=(1, 1, 1),
                dilation=(1, 1, 1),
                is_causal=(False, False, False),
                scale=0.5,
            )
        )

    q = mx.random.normal((1, 9, 10, 11, 2, 16))
    k = mx.random.normal((1, 9, 10, 11, 2, 16))
    v = mx.random.normal((1, 9, 10, 11, 2, 16))
    out = mx.grad(lambda q_in: loss_fn(q_in, k, v))(q)
    mx.eval(out)
    assert out.shape == q.shape
    assert _EXT._debug_get_last_route("na3d_backward") == "fused"
    assert _EXT._debug_get_last_kernel("na3d_fused_backward_qk_grad_q") == (
        "na3d_fused_bwd_q_softmax_k5_vec4_fp32"
    )
    assert _EXT._debug_get_last_kernel("na3d_fused_backward_qk") == (
        "na3d_fused_bwd_kv_softmax_tiled_k5_vec4_fp32"
    )
    assert _EXT._debug_get_last_kernel("na3d_fused_backward_v") == (
        "na3d_fused_bwd_kv_softmax_tiled_k5_vec4_fp32"
    )
