from __future__ import annotations

import os

from setuptools import setup


def _build_enabled() -> bool:
    return os.environ.get("NATTEN_MLX_BUILD_NANOBIND", "1") != "0"


def _cmake_ext_config():
    if not _build_enabled():
        return [], {}

    try:
        from mlx import extension
    except Exception:
        return [], {}

    return (
        [extension.CMakeExtension("natten_mlx._core._nanobind_ext")],
        {"build_ext": extension.CMakeBuild},
    )


ext_modules, cmdclass = _cmake_ext_config()

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
