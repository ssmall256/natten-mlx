#include "na_fused_forward.h"

#include <typeinfo>

#include "metal_runtime.h"
#include "py_dispatch.h"

namespace nb = nanobind;

namespace natten_mlx::nanobind_fused_forward {

nb::object na1d_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    if (!natten_mlx::nanobind_backend::use_native_runtime()) {
        if (natten_mlx::nanobind_metal_runtime::debug_forced_fused_failure()) {
            throw std::runtime_error("forced fused failure");
        }
        return natten_mlx::nanobind_backend::call_backend(
            "na1d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
    }
    try {
        return natten_mlx::nanobind_metal_runtime::na1d_fused_forward(
            q, k, v, kernel_size, stride, dilation, is_causal, scale);
    } catch (const std::bad_cast&) {
        return natten_mlx::nanobind_backend::call_backend(
            "na1d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
    }
}

nb::object na2d_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    if (!natten_mlx::nanobind_backend::use_native_runtime()) {
        if (natten_mlx::nanobind_metal_runtime::debug_forced_fused_failure()) {
            throw std::runtime_error("forced fused failure");
        }
        return natten_mlx::nanobind_backend::call_backend(
            "na2d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
    }
    try {
        return natten_mlx::nanobind_metal_runtime::na2d_fused_forward(
            q, k, v, kernel_size, stride, dilation, is_causal, scale);
    } catch (const std::bad_cast&) {
        return natten_mlx::nanobind_backend::call_backend(
            "na2d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
    }
}

nb::object na3d_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    if (!natten_mlx::nanobind_backend::use_native_runtime()) {
        if (natten_mlx::nanobind_metal_runtime::debug_forced_fused_failure()) {
            throw std::runtime_error("forced fused failure");
        }
        return natten_mlx::nanobind_backend::call_backend(
            "na3d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
    }
    try {
        return natten_mlx::nanobind_metal_runtime::na3d_fused_forward(
            q, k, v, kernel_size, stride, dilation, is_causal, scale);
    } catch (const std::bad_cast&) {
        return natten_mlx::nanobind_backend::call_backend(
            "na3d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
    }
}

}  // namespace natten_mlx::nanobind_fused_forward
