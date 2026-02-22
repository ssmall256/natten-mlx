import importlib
import importlib.util
import os
from pathlib import Path

import mlx.core as mx
import pytest


def _load_compiled_nanobind_extension_direct():
    ext_dir = Path(__file__).resolve().parents[1] / "src" / "natten_mlx" / "_core"
    candidates = sorted(ext_dir.glob("_nanobind_ext*.so"))
    if not candidates:
        raise ImportError("compiled nanobind extension artifact not found")
    spec = importlib.util.spec_from_file_location("_nanobind_ext", str(candidates[0]))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create import spec for {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nanobind_loader_has_module_name():
    import natten_mlx._core.nanobind as nb_backend

    name = nb_backend.loaded_module_name()
    assert name in {
        "natten_mlx._core._nanobind_ext",
        "natten_mlx._core._nanobind_impl",
        None,
    }
    assert isinstance(nb_backend.is_available(), bool)


def test_nanobind_override_falls_back_to_in_tree(monkeypatch):
    import natten_mlx._core.nanobind as nb_backend

    baseline = nb_backend.loaded_module_name()
    monkeypatch.setenv("NATTEN_MLX_NANOBIND_MODULE", "natten_mlx._core._does_not_exist")
    reloaded = importlib.reload(nb_backend)
    try:
        assert reloaded.loaded_module_name() == "natten_mlx._core._nanobind_impl"
        assert reloaded.is_available() is False
    finally:
        monkeypatch.delenv("NATTEN_MLX_NANOBIND_MODULE", raising=False)
        restored = importlib.reload(nb_backend)
        if baseline is not None:
            assert restored.loaded_module_name() in {
                baseline,
                "natten_mlx._core._nanobind_ext",
                "natten_mlx._core._nanobind_impl",
            }


def test_nanobind_override_accepts_explicit_in_tree_module(monkeypatch):
    import natten_mlx._core.nanobind as nb_backend

    baseline = nb_backend.loaded_module_name()
    monkeypatch.setenv("NATTEN_MLX_NANOBIND_MODULE", "natten_mlx._core._nanobind_impl")
    reloaded = importlib.reload(nb_backend)
    try:
        assert reloaded.loaded_module_name() == "natten_mlx._core._nanobind_impl"
        assert reloaded.is_available() is False
    finally:
        monkeypatch.delenv("NATTEN_MLX_NANOBIND_MODULE", raising=False)
        restored = importlib.reload(nb_backend)
        if baseline is not None:
            assert restored.loaded_module_name() in {
                baseline,
                "natten_mlx._core._nanobind_ext",
                "natten_mlx._core._nanobind_impl",
            }


def test_compiled_nanobind_extension_symbols_if_built():
    if os.environ.get("NATTEN_REQUIRE_NANOBIND_EXT") == "1":
        ext = _load_compiled_nanobind_extension_direct()
    else:
        ext = pytest.importorskip("natten_mlx._core._nanobind_ext", exc_type=ImportError)
    expected = [
        "na1d_forward",
        "na2d_forward",
        "na3d_forward",
        "na1d_qk_forward",
        "na1d_av_forward",
        "na2d_qk_forward",
        "na2d_av_forward",
        "na3d_qk_forward",
        "na3d_av_forward",
        "na1d_backward",
        "na2d_backward",
        "na3d_backward",
        "na1d_qk_backward",
        "na1d_av_backward",
        "na2d_qk_backward",
        "na2d_av_backward",
        "na3d_qk_backward",
        "na3d_av_backward",
    ]
    for name in expected:
        assert hasattr(ext, name), name


def test_compiled_nanobind_extension_required_gate():
    if os.environ.get("NATTEN_REQUIRE_NANOBIND_EXT") != "1":
        pytest.skip("compiled nanobind extension requirement is disabled")
    ext = _load_compiled_nanobind_extension_direct()
    assert ext is not None


def test_nanobind_runtime_does_not_delegate_to_fast_metal(monkeypatch):
    import natten_mlx
    from natten_mlx import na1d, set_backend
    from natten_mlx._core import fast_metal

    previous = natten_mlx.get_backend()

    def _unexpected_fast_call(*_args, **_kwargs):
        raise AssertionError("nanobind backend called fast_metal runtime")

    monkeypatch.setattr(fast_metal, "na1d_forward", _unexpected_fast_call)
    try:
        set_backend("nanobind")
        q = mx.random.normal((1, 8, 2, 4))
        k = mx.random.normal((1, 8, 2, 4))
        v = mx.random.normal((1, 8, 2, 4))
        out = na1d(q, k, v, kernel_size=3, stride=1, dilation=1, is_causal=False)
        mx.eval(out)
    finally:
        set_backend(previous)
