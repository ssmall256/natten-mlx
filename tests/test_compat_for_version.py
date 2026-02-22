"""Tests for compat.for_version() version routing."""

import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.compat import for_version


def test_for_version_014():
    mod = for_version("0.14.6")
    assert mod.__name__ == "natten_mlx.compat.v014"
    assert hasattr(mod, "natten1dqkrpb")
    assert hasattr(mod, "NeighborhoodAttention1D")


def test_for_version_014_exact():
    mod = for_version("0.14.0")
    assert mod.__name__ == "natten_mlx.compat.v014"


def test_for_version_015():
    mod = for_version("0.15.0")
    assert mod.__name__ == "natten_mlx.compat.v015"
    assert hasattr(mod, "na1d")


def test_for_version_015_patch():
    mod = for_version("0.15.2")
    assert mod.__name__ == "natten_mlx.compat.v015"


def test_for_version_017():
    mod = for_version("0.17.0")
    assert mod.__name__ == "natten_mlx.compat.v017"


def test_for_version_017_patch():
    mod = for_version("0.17.4")
    assert mod.__name__ == "natten_mlx.compat.v017"


def test_for_version_020():
    mod = for_version("0.20.0")
    assert mod.__name__ == "natten_mlx.compat.v020"
    assert hasattr(mod, "na1d")


def test_for_version_future_gets_v020():
    mod = for_version("1.0.0")
    assert mod.__name__ == "natten_mlx.compat.v020"


def test_for_version_very_old_gets_v014():
    mod = for_version("0.10.0")
    assert mod.__name__ == "natten_mlx.compat.v014"


def test_for_version_016_gets_v015():
    """0.16 falls in the (0.15, 0.17] range, so v015 is wrong — it should be v017.
    Actually per the code: target <= (0,15) -> v015, target <= (0,17) -> v017.
    0.16 > 0.15, so it should get v017."""
    mod = for_version("0.16.0")
    assert mod.__name__ == "natten_mlx.compat.v017"


def test_for_version_invalid_raises():
    with pytest.raises(Exception):
        for_version("not_a_version")
