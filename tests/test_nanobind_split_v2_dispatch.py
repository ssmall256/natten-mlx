import mlx.core as mx
import pytest

import natten_mlx
from natten_mlx import na2d_qk, na2d_av, set_backend

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
    try:
        yield
    finally:
        _EXT._debug_clear_last_routes()
        _EXT._debug_clear_last_kernels()
        set_backend(previous)


def _ref_na2d_qk(q, k, kernel_size, stride, dilation, is_causal, scale):
    """Compute reference via pure backend."""
    prev = natten_mlx.get_backend()
    set_backend("pure")
    try:
        out = na2d_qk(q, k, kernel_size, dilation=dilation, stride=stride,
                       is_causal=is_causal, scale=scale)
        mx.eval(out)
        return out
    finally:
        set_backend(prev)


def _ref_na2d_av(attn, v, kernel_size, stride, dilation, is_causal):
    """Compute reference via pure backend."""
    prev = natten_mlx.get_backend()
    set_backend("pure")
    try:
        out = na2d_av(attn, v, kernel_size, dilation=dilation, stride=stride,
                       is_causal=is_causal)
        mx.eval(out)
        return out
    finally:
        set_backend(prev)


@pytest.mark.parametrize("ks", [3, 7])
@pytest.mark.parametrize("is_causal", [(False, False), (True, False)])
def test_split_qk_v2_correctness(_backend_nanobind, ks, is_causal):
    B, IH, IW, H, D = 1, 14, 12, 2, 16
    q = mx.random.normal((B, IH, IW, H, D))
    k = mx.random.normal((B, IH, IW, H, D))
    scale = 0.25

    ref = _ref_na2d_qk(q, k, (ks, ks), (1, 1), (1, 1), is_causal, scale)
    out = na2d_qk(q, k, (ks, ks), dilation=(1, 1), stride=(1, 1),
                   is_causal=is_causal, scale=scale)
    mx.eval(out)

    assert out.shape == ref.shape
    assert mx.allclose(out, ref, atol=1e-5).item()


@pytest.mark.parametrize("ks", [3, 7])
@pytest.mark.parametrize("is_causal", [(False, False), (True, False)])
def test_split_av_v2_correctness(_backend_nanobind, ks, is_causal):
    B, IH, IW, H, D = 1, 14, 12, 2, 16
    K2 = ks * ks
    attn = mx.softmax(mx.random.normal((B, IH, IW, H, K2)), axis=-1)
    v = mx.random.normal((B, IH, IW, H, D))

    ref = _ref_na2d_av(attn, v, (ks, ks), (1, 1), (1, 1), is_causal)
    out = na2d_av(attn, v, (ks, ks), dilation=(1, 1), stride=(1, 1),
                   is_causal=is_causal)
    mx.eval(out)

    assert out.shape == ref.shape
    assert mx.allclose(out, ref, atol=1e-5).item()


def test_split_qk_v2_vec4_path(_backend_nanobind):
    """Head dim divisible by 4 should use vec4 kernel."""
    B, IH, IW, H, D = 1, 10, 10, 2, 32
    q = mx.random.normal((B, IH, IW, H, D))
    k = mx.random.normal((B, IH, IW, H, D))

    ref = _ref_na2d_qk(q, k, (3, 3), (1, 1), (1, 1), (False, False), 0.5)
    out = na2d_qk(q, k, (3, 3), dilation=(1, 1), stride=(1, 1),
                   is_causal=(False, False), scale=0.5)
    mx.eval(out)

    assert out.shape == ref.shape
    assert mx.allclose(out, ref, atol=1e-5).item()


def test_split_av_v2_vec4_path(_backend_nanobind):
    """Head dim divisible by 4 should use vec4 kernel."""
    B, IH, IW, H, D = 1, 10, 10, 2, 32
    K2 = 9
    attn = mx.softmax(mx.random.normal((B, IH, IW, H, K2)), axis=-1)
    v = mx.random.normal((B, IH, IW, H, D))

    ref = _ref_na2d_av(attn, v, (3, 3), (1, 1), (1, 1), (False, False))
    out = na2d_av(attn, v, (3, 3), dilation=(1, 1), stride=(1, 1),
                   is_causal=(False, False))
    mx.eval(out)

    assert out.shape == ref.shape
    assert mx.allclose(out, ref, atol=1e-5).item()
