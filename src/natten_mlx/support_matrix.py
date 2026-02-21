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
            "fusion": {"na1d": False, "na2d": False, "na3d": False},
            "constraints": [],
        },
        "fast_metal": {
            "available": fast_metal.is_available(),
            "forward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "backward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "fusion": {"na1d": True, "na2d": True, "na3d": False},
            "constraints": [
                "Fused 1D fast path: odd K, stride>=1, dilation>=1, causal/non-causal supported.",
                "Fused 2D fast path: square odd K, stride>=1 per axis, dilation>=1 per axis, causal/non-causal per axis.",
                "Split 1D fast path: K in {3,5,7}, stride=1, non-causal.",
                "Split 2D fast path: square K in {3,5,7}, stride=(1,1), equal dilations, non-causal on both axes.",
                "Split 3D fast path: cubic K in {3,5,7}, stride=(1,1,1), equal dilations, non-causal on all axes.",
                "Unsupported configs automatically fall back to pure backend.",
            ],
        },
        "nanobind": {
            "available": nanobind.is_available(),
            "forward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "backward": {"na1d": True, "na2d": True, "na3d": True, "split_qk_av": True},
            "fusion": {"na1d": True, "na2d": True, "na3d": False},
            "constraints": [
                "Ships with an in-tree implementation that delegates to fast_metal where available, otherwise pure.",
                "Can be overridden with an external extension via NATTEN_MLX_NANOBIND_MODULE.",
                "When delegated to fast_metal, the exact same fused/split eligibility constraints apply.",
            ],
        },
    }
