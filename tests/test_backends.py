import pytest
import numpy as np
import mlx.core as mx

from natten_mlx import get_backend, na1d, set_backend


def test_backend_switching():
    set_backend("pure")
    assert get_backend() == "pure"

    set_backend("auto")
    assert get_backend() in {"pure", "fast_metal", "nanobind"}


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        set_backend("invalid-backend")


def test_nanobind_backend_matches_pure_for_supported_case():
    q = mx.random.normal((1, 8, 2, 4))
    k = mx.random.normal((1, 8, 2, 4))
    v = mx.random.normal((1, 8, 2, 4))

    prev = get_backend()
    try:
        set_backend("pure")
        out_pure = na1d(q, k, v, kernel_size=3, stride=1, dilation=1, is_causal=False)
        mx.eval(out_pure)

        set_backend("nanobind")
        out_nb = na1d(q, k, v, kernel_size=3, stride=1, dilation=1, is_causal=False)
        mx.eval(out_nb)

        np.testing.assert_allclose(np.array(out_nb), np.array(out_pure), rtol=1e-5, atol=1e-5)
    finally:
        set_backend(prev)
