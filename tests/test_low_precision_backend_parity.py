import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx import na1d, na1d_qk, na2d, na3d, set_backend


def _run_backend(backend: str, fn):
    set_backend(backend)
    try:
        return fn()
    finally:
        set_backend("auto")


def _to_np32(x):
    # bf16 interop can be backend/build dependent; normalize via float32.
    if hasattr(mx, "astype"):
        x = mx.astype(x, mx.float32)
    else:
        x = x.astype(mx.float32)
    return np.array(x)


def _low_precision_dtypes():
    dtypes = [mx.float16]
    bf16 = getattr(mx, "bfloat16", None)
    if bf16 is not None:
        dtypes.append(bf16)
    return dtypes


def _tol(dtype):
    if dtype == mx.float16:
        return 1.2e-2, 1.2e-2
    return 1.6e-1, 1.6e-1


@pytest.mark.parametrize("backend", ["fast_metal", "nanobind"])
@pytest.mark.parametrize("dtype", _low_precision_dtypes(), ids=lambda d: str(d))
@pytest.mark.parametrize("causal", [False, True])
def test_low_precision_backend_parity_na1d_fused(backend: str, dtype, causal: bool):
    rng = np.random.default_rng(101)
    q = mx.array(rng.standard_normal((2, 256, 8, 16), dtype=np.float32), dtype=dtype)
    k = mx.array(rng.standard_normal((2, 256, 8, 16), dtype=np.float32), dtype=dtype)
    v = mx.array(rng.standard_normal((2, 256, 8, 16), dtype=np.float32), dtype=dtype)

    out_pure = _run_backend(
        "pure",
        lambda: na1d(q, k, v, kernel_size=7, stride=1, dilation=1, is_causal=causal, scale=0.5),
    )
    out_backend = _run_backend(
        backend,
        lambda: na1d(q, k, v, kernel_size=7, stride=1, dilation=1, is_causal=causal, scale=0.5),
    )
    mx.eval(out_pure, out_backend)
    assert out_backend.dtype == dtype

    atol, rtol = _tol(dtype)
    np.testing.assert_allclose(_to_np32(out_backend), _to_np32(out_pure), atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend", ["fast_metal", "nanobind"])
@pytest.mark.parametrize("dtype", _low_precision_dtypes(), ids=lambda d: str(d))
def test_low_precision_backend_parity_na2d_fused_causal(backend: str, dtype):
    rng = np.random.default_rng(102)
    q = mx.array(rng.standard_normal((1, 24, 24, 4, 16), dtype=np.float32), dtype=dtype)
    k = mx.array(rng.standard_normal((1, 24, 24, 4, 16), dtype=np.float32), dtype=dtype)
    v = mx.array(rng.standard_normal((1, 24, 24, 4, 16), dtype=np.float32), dtype=dtype)

    out_pure = _run_backend(
        "pure",
        lambda: na2d(
            q,
            k,
            v,
            kernel_size=(7, 7),
            stride=(1, 1),
            dilation=(1, 1),
            is_causal=(True, False),
            scale=0.5,
        ),
    )
    out_backend = _run_backend(
        backend,
        lambda: na2d(
            q,
            k,
            v,
            kernel_size=(7, 7),
            stride=(1, 1),
            dilation=(1, 1),
            is_causal=(True, False),
            scale=0.5,
        ),
    )
    mx.eval(out_pure, out_backend)
    assert out_backend.dtype == dtype

    atol, rtol = _tol(dtype)
    np.testing.assert_allclose(_to_np32(out_backend), _to_np32(out_pure), atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend", ["fast_metal", "nanobind"])
@pytest.mark.parametrize("dtype", _low_precision_dtypes(), ids=lambda d: str(d))
def test_low_precision_backend_parity_na3d_fused_causal(backend: str, dtype):
    rng = np.random.default_rng(103)
    q = mx.array(rng.standard_normal((1, 10, 12, 14, 4, 16), dtype=np.float32), dtype=dtype)
    k = mx.array(rng.standard_normal((1, 10, 12, 14, 4, 16), dtype=np.float32), dtype=dtype)
    v = mx.array(rng.standard_normal((1, 10, 12, 14, 4, 16), dtype=np.float32), dtype=dtype)

    out_pure = _run_backend(
        "pure",
        lambda: na3d(
            q,
            k,
            v,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            is_causal=(True, False, False),
            scale=0.5,
        ),
    )
    out_backend = _run_backend(
        backend,
        lambda: na3d(
            q,
            k,
            v,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            is_causal=(True, False, False),
            scale=0.5,
        ),
    )
    mx.eval(out_pure, out_backend)
    assert out_backend.dtype == dtype

    atol, rtol = _tol(dtype)
    np.testing.assert_allclose(_to_np32(out_backend), _to_np32(out_pure), atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend", ["fast_metal", "nanobind"])
@pytest.mark.parametrize("dtype", _low_precision_dtypes(), ids=lambda d: str(d))
@pytest.mark.parametrize("causal", [False, True])
def test_low_precision_backend_parity_na1d_qk_inf_mask(backend: str, dtype, causal: bool):
    rng = np.random.default_rng(104)
    q = mx.array(rng.standard_normal((2, 256, 8, 16), dtype=np.float32), dtype=dtype)
    k = mx.array(rng.standard_normal((2, 256, 8, 16), dtype=np.float32), dtype=dtype)

    logits_pure = _run_backend(
        "pure",
        lambda: na1d_qk(q, k, kernel_size=7, stride=1, dilation=1, is_causal=causal, scale=0.5),
    )
    logits_backend = _run_backend(
        backend,
        lambda: na1d_qk(q, k, kernel_size=7, stride=1, dilation=1, is_causal=causal, scale=0.5),
    )
    mx.eval(logits_pure, logits_backend)
    assert logits_backend.dtype == dtype

    pure_np = _to_np32(logits_pure)
    back_np = _to_np32(logits_backend)
    assert np.array_equal(np.isneginf(back_np), np.isneginf(pure_np))

    finite = np.isfinite(pure_np) & np.isfinite(back_np)
    atol, rtol = _tol(dtype)
    np.testing.assert_allclose(back_np[finite], pure_np[finite], atol=atol, rtol=rtol)
