import pytest

from natten_mlx import get_backend, set_backend


def test_backend_switching():
    set_backend("pure")
    assert get_backend() == "pure"

    set_backend("auto")
    assert get_backend() in {"pure", "fast_metal", "nanobind"}


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        set_backend("invalid-backend")
