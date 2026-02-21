import importlib

import pytest


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
        assert reloaded.is_available() is True
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
        assert reloaded.is_available() is True
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
