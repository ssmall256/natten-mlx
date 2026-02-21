"""Tier 2: nanobind pure-Metal backend (stub)."""

_AVAILABLE = False


def is_available() -> bool:
    return _AVAILABLE


def _not_available() -> None:
    raise NotImplementedError(
        "Nanobind backend is not yet available. "
        "Use set_backend('pure') or install natten-mlx with nanobind support."
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
