"""Backend capability matrix for natten-mlx."""

from __future__ import annotations

from natten_mlx._core import fast_metal, nanobind, pure


def get_support_matrix() -> dict[str, dict]:
    """Return capability matrix for each backend tier.

    Notes:
    - "backward" is defined for end-to-end `na1d` / `na2d` module use.
    - For non-pure backends, gradients are computed with pure semantics in custom VJP.
    """

    return {
        "pure": {
            "available": pure.is_available(),
            "forward": {"na1d": True, "na2d": True, "split_qk_av": True},
            "backward": {"na1d": True, "na2d": True},
            "fusion": {"na1d": False, "na2d": False},
            "constraints": [],
        },
        "fast_metal": {
            "available": fast_metal.is_available(),
            "forward": {"na1d": True, "na2d": True, "split_qk_av": True},
            "backward": {"na1d": True, "na2d": True},
            "fusion": {"na1d": True, "na2d": True},
            "constraints": [
                "Fused fast path currently targets non-causal, stride=1, K in {3,5,7}.",
                "2D fused fast path requires square kernel and equal dilations.",
                "Unsupported configs automatically fall back to pure backend.",
            ],
        },
        "nanobind": {
            "available": nanobind.is_available(),
            "forward": {"na1d": True, "na2d": True, "split_qk_av": True},
            "backward": {"na1d": True, "na2d": True},
            "fusion": {"na1d": True, "na2d": True},
            "constraints": [
                "Ships with an in-tree implementation that delegates to fast_metal where available, otherwise pure.",
                "Can be overridden with an external extension via NATTEN_MLX_NANOBIND_MODULE.",
                "When delegated to fast_metal, fused-path constraints match fast_metal constraints.",
            ],
        },
    }
