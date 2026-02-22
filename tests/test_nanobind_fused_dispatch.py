import os

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


@pytest.mark.parametrize("is_causal,stride,dim", [(False, 1, 16), (True, 2, 12)])
def test_nanobind_fused_1d_dispatch(_backend_nanobind, is_causal: bool, stride: int, dim: int):
    q = mx.random.normal((1, 19, 2, dim))
    k = mx.random.normal((1, 19, 2, dim))
    v = mx.random.normal((1, 19, 2, dim))
    out = na1d(q, k, v, kernel_size=3, stride=stride, dilation=1, is_causal=is_causal, scale=0.5)
    mx.eval(out)
    assert out.shape == (1, (19 + stride - 1) // stride, 2, dim)
    route = _EXT._debug_get_last_route("na1d_forward")
    assert route in ("fused", "v2_primitive")
    if route == "fused":
        kernel = _EXT._debug_get_last_kernel("na1d_forward")
        assert ("fp16" in kernel) or ("bf16" in kernel) or ("fp32" in kernel)


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
    assert _EXT._debug_get_last_route("na2d_forward") in ("fused", "v2_primitive")


def test_nanobind_auto_uses_bf16_strided_causal_h_fused_kernel(_backend_nanobind):
    bf16 = getattr(mx, "bfloat16", None)
    if bf16 is None:
        pytest.skip("bfloat16 not available")
    q = mx.random.normal((1, 20, 18, 8, 16)).astype(bf16)
    k = mx.random.normal((1, 20, 18, 8, 16)).astype(bf16)
    v = mx.random.normal((1, 20, 18, 8, 16)).astype(bf16)
    out = na2d(
        q,
        k,
        v,
        kernel_size=(7, 7),
        stride=(2, 1),
        dilation=(1, 2),
        is_causal=(True, False),
        scale=0.5,
    )
    mx.eval(out)
    route = _EXT._debug_get_last_route("na2d_forward")
    assert route in ("fused", "v2_primitive")
    if route == "fused":
        assert _EXT._debug_get_last_kernel("na2d_forward") == "na2d_fused_strided_causal_h_k7d16_bf16"


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
    assert _EXT._debug_get_last_route("na3d_forward") in ("fused", "v2_primitive")


def test_nanobind_fused_forward_fallback_chain(_backend_nanobind):
    q = mx.random.normal((1, 17, 2, 12))
    k = mx.random.normal((1, 17, 2, 12))
    v = mx.random.normal((1, 17, 2, 12))

    # Disable v2 to test legacy fallback chain
    prev_v2 = os.environ.get("NATTEN_NANOBIND_DISABLE_V2")
    os.environ["NATTEN_NANOBIND_DISABLE_V2"] = "1"
    try:
        _EXT._debug_force_fused_failure(True)
        out = na1d(q, k, v, kernel_size=3, stride=1, dilation=1, is_causal=False, scale=0.5)
        mx.eval(out)
        assert _EXT._debug_get_last_route("na1d_forward") == "split"

        _EXT._debug_force_split_failure(True)
        out = na1d(q, k, v, kernel_size=3, stride=1, dilation=1, is_causal=False, scale=0.5)
        mx.eval(out)
        assert _EXT._debug_get_last_route("na1d_forward") == "pure"
    finally:
        if prev_v2 is None:
            os.environ.pop("NATTEN_NANOBIND_DISABLE_V2", None)
        else:
            os.environ["NATTEN_NANOBIND_DISABLE_V2"] = prev_v2


def test_nanobind_forward_auto_prefers_split_for_large_noncausal_2d(_backend_nanobind):
    q = mx.random.normal((1, 32, 32, 8, 32))
    k = mx.random.normal((1, 32, 32, 8, 32))
    v = mx.random.normal((1, 32, 32, 8, 32))
    out = na2d(
        q,
        k,
        v,
        kernel_size=(7, 7),
        stride=(1, 1),
        dilation=(1, 1),
        is_causal=(False, False),
        scale=0.5,
    )
    mx.eval(out)
    assert _EXT._debug_get_last_route("na2d_forward") in ("split", "v2_primitive")


def test_nanobind_forward_auto_prefers_split_for_large_noncausal_3d(_backend_nanobind):
    q = mx.random.normal((1, 10, 12, 14, 4, 16))
    k = mx.random.normal((1, 10, 12, 14, 4, 16))
    v = mx.random.normal((1, 10, 12, 14, 4, 16))
    out = na3d(
        q,
        k,
        v,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        dilation=(1, 1, 1),
        is_causal=(False, False, False),
        scale=0.5,
    )
    mx.eval(out)
    assert _EXT._debug_get_last_route("na3d_forward") in ("split", "v2_primitive")


def test_nanobind_forward_mode_override_forces_fused_3d(_backend_nanobind):
    prev = os.environ.get("NATTEN_NANOBIND_FUSED_FWD_3D_MODE")
    os.environ["NATTEN_NANOBIND_FUSED_FWD_3D_MODE"] = "fused"
    try:
        q = mx.random.normal((1, 10, 12, 14, 4, 16))
        k = mx.random.normal((1, 10, 12, 14, 4, 16))
        v = mx.random.normal((1, 10, 12, 14, 4, 16))
        out = na3d(
            q,
            k,
            v,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            is_causal=(False, False, False),
            scale=0.5,
        )
        mx.eval(out)
        assert _EXT._debug_get_last_route("na3d_forward") in ("fused", "v2_primitive")
    finally:
        if prev is None:
            os.environ.pop("NATTEN_NANOBIND_FUSED_FWD_3D_MODE", None)
        else:
            os.environ["NATTEN_NANOBIND_FUSED_FWD_3D_MODE"] = prev


def test_nanobind_forward_auto_prefers_fused_for_medium_causal_1d(_backend_nanobind):
    q = mx.random.normal((1, 512, 8, 64))
    k = mx.random.normal((1, 512, 8, 64))
    v = mx.random.normal((1, 512, 8, 64))
    out = na1d(q, k, v, kernel_size=9, stride=1, dilation=1, is_causal=True, scale=0.5)
    mx.eval(out)
    assert _EXT._debug_get_last_route("na1d_forward") in ("fused", "v2_primitive")


def test_nanobind_forward_mode_override_forces_fused_1d(_backend_nanobind):
    prev = os.environ.get("NATTEN_NANOBIND_FUSED_FWD_1D_MODE")
    os.environ["NATTEN_NANOBIND_FUSED_FWD_1D_MODE"] = "fused"
    try:
        q = mx.random.normal((1, 512, 8, 64))
        k = mx.random.normal((1, 512, 8, 64))
        v = mx.random.normal((1, 512, 8, 64))
        out = na1d(q, k, v, kernel_size=9, stride=1, dilation=1, is_causal=True, scale=0.5)
        mx.eval(out)
        assert _EXT._debug_get_last_route("na1d_forward") in ("fused", "v2_primitive")
    finally:
        if prev is None:
            os.environ.pop("NATTEN_NANOBIND_FUSED_FWD_1D_MODE", None)
        else:
            os.environ["NATTEN_NANOBIND_FUSED_FWD_1D_MODE"] = prev


def test_nanobind_split_lowp_fp32_pipeline_preserves_dtype_1d(_backend_nanobind):
    prev_split = os.environ.get("NATTEN_NANOBIND_FUSED_FWD_1D_MODE")
    prev_dtype = os.environ.get("NATTEN_NANOBIND_SPLIT_FWD_DTYPE_MODE")
    os.environ["NATTEN_NANOBIND_FUSED_FWD_1D_MODE"] = "split"
    os.environ["NATTEN_NANOBIND_SPLIT_FWD_DTYPE_MODE"] = "fp32"
    try:
        dtype = mx.float16
        q = mx.random.normal((1, 512, 8, 64), dtype=dtype)
        k = mx.random.normal((1, 512, 8, 64), dtype=dtype)
        v = mx.random.normal((1, 512, 8, 64), dtype=dtype)
        out = na1d(q, k, v, kernel_size=9, stride=1, dilation=1, is_causal=True, scale=0.5)
        mx.eval(out)
        assert _EXT._debug_get_last_route("na1d_forward") in ("split", "v2_primitive")
        assert out.dtype == dtype
    finally:
        if prev_split is None:
            os.environ.pop("NATTEN_NANOBIND_FUSED_FWD_1D_MODE", None)
        else:
            os.environ["NATTEN_NANOBIND_FUSED_FWD_1D_MODE"] = prev_split
        if prev_dtype is None:
            os.environ.pop("NATTEN_NANOBIND_SPLIT_FWD_DTYPE_MODE", None)
        else:
            os.environ["NATTEN_NANOBIND_SPLIT_FWD_DTYPE_MODE"] = prev_dtype


@pytest.mark.parametrize("dtype_name", ["float16", "float32"])
def test_nanobind_fused_forward_records_dtype_kernel(_backend_nanobind, dtype_name: str):
    dtype = getattr(mx, dtype_name)
    q = mx.random.normal((1, 21, 2, 16), dtype=dtype)
    k = mx.random.normal((1, 21, 2, 16), dtype=dtype)
    v = mx.random.normal((1, 21, 2, 16), dtype=dtype)
    out = na1d(q, k, v, kernel_size=7, stride=1, dilation=1, is_causal=True, scale=0.5)
    mx.eval(out)
    route = _EXT._debug_get_last_route("na1d_forward")
    assert route in ("fused", "v2_primitive")
    if route == "fused":
        kernel = _EXT._debug_get_last_kernel("na1d_forward")
        assert kernel
        if dtype_name == "float16":
            # Parity-safe routing may use fp32 fused kernels for low-precision inputs.
            assert ("fp16" in kernel) or ("fp32" in kernel)
        else:
            assert "fp32" in kernel
