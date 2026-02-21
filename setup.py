from __future__ import annotations

import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Build extension opportunistically; fall back to Python impl on failure."""

    def run(self):
        try:
            super().run()
        except Exception as exc:  # pragma: no cover - exercised in build envs
            self.announce(f"Skipping optional nanobind extension build: {exc}", level=3)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover - exercised in build envs
            self.announce(f"Skipping extension {ext.name}: {exc}", level=3)


def _nanobind_extensions() -> list[Extension]:
    if os.environ.get("NATTEN_MLX_BUILD_NANOBIND", "1") == "0":
        return []

    try:
        import nanobind  # type: ignore
    except Exception:
        return []

    extra_compile_args = ["-std=c++17"]
    if os.name != "nt":
        extra_compile_args.append("-fvisibility=hidden")

    project_root = os.path.dirname(__file__)
    source_dir = nanobind.source_dir()
    robin_map_include = os.path.join(source_dir, "..", "ext", "robin_map", "include")
    nb_combined = os.path.relpath(os.path.join(source_dir, "nb_combined.cpp"), start=project_root)

    return [
        Extension(
            "natten_mlx._core._nanobind_ext",
            sources=[
                "src/natten_mlx/_core/_nanobind_ext.cpp",
                nb_combined,
            ],
            include_dirs=[nanobind.include_dir(), robin_map_include],
            language="c++",
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    ext_modules=_nanobind_extensions(),
    cmdclass={"build_ext": OptionalBuildExt},
)
