"""Parameter normalization and validation helpers."""

from __future__ import annotations

from typing import Any, Iterable


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (tuple, list))


def normalize_tuple_param(param: Any, rank: int, name: str) -> tuple[Any, ...]:
    """Normalize scalar/bool/sequence parameter to a tuple of length ``rank``."""
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")

    if _is_sequence(param):
        result = tuple(param)
        if len(result) != rank:
            raise ValueError(
                f"{name} must have length {rank}, got {len(result)}: {param}"
            )
        return result

    return tuple(param for _ in range(rank))


def normalize_kernel_size(kernel_size: Any, rank: int) -> tuple[int, ...]:
    """Normalize and validate kernel size."""
    ks = normalize_tuple_param(kernel_size, rank, "kernel_size")
    for value in ks:
        if not isinstance(value, int):
            raise ValueError(f"kernel_size values must be int, got {type(value)!r}")
        if value <= 0:
            raise ValueError(f"kernel_size values must be positive, got {value}")
    return ks


def check_kernel_size_vs_input(
    kernel_size: Iterable[int], input_spatial_shape: Iterable[int]
) -> None:
    """Validate ``kernel_size <= input_spatial_shape`` per dimension."""
    for dim, (k, size) in enumerate(zip(kernel_size, input_spatial_shape)):
        if k > size:
            raise ValueError(
                f"kernel_size[{dim}]={k} must be <= input spatial size {size}"
            )


def check_stride_vs_kernel(stride: Iterable[int], kernel_size: Iterable[int]) -> None:
    """Validate ``stride <= kernel_size`` per dimension."""
    for dim, (s, k) in enumerate(zip(stride, kernel_size)):
        if not isinstance(s, int):
            raise ValueError(f"stride[{dim}] must be int, got {type(s)!r}")
        if s <= 0:
            raise ValueError(f"stride[{dim}] must be positive, got {s}")
        if s > k:
            raise ValueError(f"stride[{dim}]={s} must be <= kernel_size[{dim}]={k}")


def check_dilation_kernel_vs_input(
    dilation: Iterable[int], kernel_size: Iterable[int], input_spatial_shape: Iterable[int]
) -> None:
    """Validate dilation * kernel_size <= input size in each dimension."""
    for dim, (d, k, size) in enumerate(zip(dilation, kernel_size, input_spatial_shape)):
        if not isinstance(d, int):
            raise ValueError(f"dilation[{dim}] must be int, got {type(d)!r}")
        if d <= 0:
            raise ValueError(f"dilation[{dim}] must be positive, got {d}")
        required = d * k
        if required > size:
            raise ValueError(
                f"dilation[{dim}] * kernel_size[{dim}] = {required} exceeds input size {size}"
            )
