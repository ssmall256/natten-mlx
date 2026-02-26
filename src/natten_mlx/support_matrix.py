"""Backend capability matrix for natten-mlx."""

from __future__ import annotations

from natten_mlx._core import fast_metal, nanobind, pure


def get_support_matrix() -> dict[str, dict]:
    """Return capability matrix for each backend tier.

    Notes:
    - "backward" is defined for end-to-end `na1d` / `na2d` module use.
    - Non-pure backends expose explicit backward entrypoints with pure fallback safety.
    """

    return {
        "pure": {
            "available": pure.is_available(),
            "forward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "backward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "varlen": {"na1d": True, "na2d": True, "na3d": True},
            "fusion": {"na1d": False, "na2d": False, "na3d": False},
            "constraints": [],
        },
        "fast_metal": {
            "available": fast_metal.is_available(),
            "forward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "backward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "varlen": {"na1d": True, "na2d": True, "na3d": True},
            "fusion": {"na1d": True, "na2d": True, "na3d": True},
            "constraints": [
                "Fused 1D fast path: odd K, stride>=1, dilation>=1, causal/non-causal supported.",
                "Fused 2D fast path: square odd K, stride>=1 per axis, dilation>=1 per axis, causal/non-causal per axis.",
                "Fused 3D fast path: cubic odd K, stride>=1 per axis, dilation>=1 per axis, causal/non-causal per axis.",
                "Split 1D fast path: odd K, stride>=1, dilation>=1, causal/non-causal supported.",
                "Split 2D fast path: square odd K, stride>=1 per axis, dilation>=1 per axis, causal/non-causal per axis supported.",
                "Split 3D fast path: cubic odd K, stride>=1 per axis, dilation>=1 per axis, causal/non-causal per axis supported.",
                "Unsupported configs automatically fall back to pure backend.",
            ],
        },
        "nanobind": {
            "available": nanobind.is_available(),
            "forward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "backward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "varlen": {"na1d": False, "na2d": False, "na3d": False},
            "fusion": {"na1d": True, "na2d": True, "na3d": True},
            "constraints": [
                "Ships with an in-tree implementation that uses nanobind-owned backend kernels where available, otherwise pure fallback.",
                "Compiled extension Stage B: na*d forward is fused-first from C++ entrypoints with fallback chain fused -> split -> pure.",
                "Compiled extension Stage B: na*d backward is fused-first staged pipeline from C++ entrypoints with fallback chain fused -> split -> pure.",
                "Can be overridden with an external extension via NATTEN_MLX_NANOBIND_MODULE.",
                "Compiled extension availability is reported independently from pure fallback behavior.",
            ],
        },
    }
