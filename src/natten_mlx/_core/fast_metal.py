"""Tier 1: MLX fast Metal kernel backend (stub)."""

_AVAILABLE = False


def is_available() -> bool:
    return _AVAILABLE


def _not_available() -> None:
    raise NotImplementedError(
        "Fast Metal kernel backend is not yet available. "
        "Use set_backend('pure') or install natten-mlx with Metal kernel support."
    )


def na1d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    _not_available()


def na2d_forward(q, k, v, kernel_size, stride, dilation, is_causal, scale):
    _not_available()


def na1d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    _not_available()


def na1d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    _not_available()


def na2d_qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale):
    _not_available()


def na2d_av_forward(attn, v, kernel_size, stride, dilation, is_causal):
    _not_available()
