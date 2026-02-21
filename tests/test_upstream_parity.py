import os
import pytest

if os.environ.get("NATTEN_UPSTREAM_PARITY") != "1":
    pytest.skip(
        "Upstream parity tests are opt-in. Set NATTEN_UPSTREAM_PARITY=1 to run.",
        allow_module_level=True,
    )

np = pytest.importorskip("numpy", exc_type=ImportError)
mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch", exc_type=ImportError)
upstream_natten = pytest.importorskip("natten", exc_type=ImportError)

import natten_mlx.compat.v014 as mlx_v014


def _resolve_v014_symbol(name: str):
    if hasattr(upstream_natten, name):
        return getattr(upstream_natten, name)
    functional = getattr(upstream_natten, "functional", None)
    if functional is not None and hasattr(functional, name):
        return getattr(functional, name)
    pytest.skip(f"Official natten symbol {name} is unavailable in installed version.")


def _np_to_torch(array: np.ndarray):
    return torch.from_numpy(array.copy())


def _np_to_mx(array: np.ndarray):
    return mx.array(array)


def _assert_close(mx_out, torch_out, atol=1e-5, rtol=1e-5):
    mx_np = np.array(mx_out)
    torch_np = torch_out.detach().cpu().numpy()
    assert np.allclose(mx_np, torch_np, atol=atol, rtol=rtol)


def test_upstream_v014_natten1dqkrpb_matches_official():
    rng = np.random.default_rng(123)
    q = rng.standard_normal((2, 4, 12, 8), dtype=np.float32)
    k = rng.standard_normal((2, 4, 12, 8), dtype=np.float32)
    rpb = rng.standard_normal((4, 2 * 3 - 1), dtype=np.float32)

    fn = _resolve_v014_symbol("natten1dqkrpb")
    torch_out = fn(_np_to_torch(q), _np_to_torch(k), _np_to_torch(rpb), kernel_size=3, dilation=2)
    mlx_out = mlx_v014.natten1dqkrpb(_np_to_mx(q), _np_to_mx(k), _np_to_mx(rpb), kernel_size=3, dilation=2)
    _assert_close(mlx_out, torch_out)


def test_upstream_v014_natten1dav_matches_official():
    rng = np.random.default_rng(124)
    logits = rng.standard_normal((2, 4, 10, 5), dtype=np.float32)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    attn = probs / probs.sum(axis=-1, keepdims=True)
    value = rng.standard_normal((2, 4, 10, 16), dtype=np.float32)

    fn = _resolve_v014_symbol("natten1dav")
    torch_out = fn(_np_to_torch(attn), _np_to_torch(value), kernel_size=5, dilation=1)
    mlx_out = mlx_v014.natten1dav(_np_to_mx(attn), _np_to_mx(value), kernel_size=5, dilation=1)
    _assert_close(mlx_out, torch_out)


def test_upstream_v014_natten2dqkrpb_matches_official():
    rng = np.random.default_rng(125)
    q = rng.standard_normal((1, 3, 7, 8, 6), dtype=np.float32)
    k = rng.standard_normal((1, 3, 7, 8, 6), dtype=np.float32)
    rpb = rng.standard_normal((3, 2 * 3 - 1, 2 * 3 - 1), dtype=np.float32)

    fn = _resolve_v014_symbol("natten2dqkrpb")
    torch_out = fn(
        _np_to_torch(q),
        _np_to_torch(k),
        _np_to_torch(rpb),
        kernel_size=3,
        dilation=2,
    )
    mlx_out = mlx_v014.natten2dqkrpb(
        _np_to_mx(q),
        _np_to_mx(k),
        _np_to_mx(rpb),
        kernel_size=3,
        dilation=2,
    )
    _assert_close(mlx_out, torch_out)


def test_upstream_v014_natten2dav_matches_official():
    rng = np.random.default_rng(126)
    logits = rng.standard_normal((1, 2, 6, 7, 9), dtype=np.float32)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    attn = probs / probs.sum(axis=-1, keepdims=True)
    value = rng.standard_normal((1, 2, 6, 7, 5), dtype=np.float32)

    fn = _resolve_v014_symbol("natten2dav")
    torch_out = fn(_np_to_torch(attn), _np_to_torch(value), kernel_size=3, dilation=1)
    mlx_out = mlx_v014.natten2dav(_np_to_mx(attn), _np_to_mx(value), kernel_size=3, dilation=1)
    _assert_close(mlx_out, torch_out)
