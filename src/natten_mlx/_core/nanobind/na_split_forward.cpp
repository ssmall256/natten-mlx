#include "na_split_forward.h"

#include <typeinfo>

#include "metal_runtime.h"
#include "py_dispatch.h"

namespace natten_mlx::nanobind_split_forward {

nb::object na1d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    if (!natten_mlx::nanobind_backend::use_native_runtime()) {
        if (natten_mlx::nanobind_metal_runtime::debug_forced_split_failure()) {
            throw std::runtime_error("forced split failure");
        }
        return natten_mlx::nanobind_backend::call_backend(
            "na1d_qk_forward", q, k, kernel_size, stride, dilation, is_causal, scale);
    }
    try {
        return natten_mlx::nanobind_metal_runtime::na1d_qk_forward(
            q, k, kernel_size, stride, dilation, is_causal, scale);
    } catch (const std::bad_cast&) {
        return natten_mlx::nanobind_backend::call_backend(
            "na1d_qk_forward", q, k, kernel_size, stride, dilation, is_causal, scale);
    }
}

nb::object na1d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
    if (!natten_mlx::nanobind_backend::use_native_runtime()) {
        if (natten_mlx::nanobind_metal_runtime::debug_forced_split_failure()) {
            throw std::runtime_error("forced split failure");
        }
        return natten_mlx::nanobind_backend::call_backend(
            "na1d_av_forward", attn, v, kernel_size, stride, dilation, is_causal);
    }
    try {
        return natten_mlx::nanobind_metal_runtime::na1d_av_forward(
            attn, v, kernel_size, stride, dilation, is_causal);
    } catch (const std::bad_cast&) {
        return natten_mlx::nanobind_backend::call_backend(
            "na1d_av_forward", attn, v, kernel_size, stride, dilation, is_causal);
    }
}

nb::object na2d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    if (!natten_mlx::nanobind_backend::use_native_runtime()) {
        if (natten_mlx::nanobind_metal_runtime::debug_forced_split_failure()) {
            throw std::runtime_error("forced split failure");
        }
        return natten_mlx::nanobind_backend::call_backend(
            "na2d_qk_forward", q, k, kernel_size, stride, dilation, is_causal, scale);
    }
    try {
        return natten_mlx::nanobind_metal_runtime::na2d_qk_forward(
            q, k, kernel_size, stride, dilation, is_causal, scale);
    } catch (const std::bad_cast&) {
        return natten_mlx::nanobind_backend::call_backend(
            "na2d_qk_forward", q, k, kernel_size, stride, dilation, is_causal, scale);
    }
}

nb::object na2d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
    if (!natten_mlx::nanobind_backend::use_native_runtime()) {
        if (natten_mlx::nanobind_metal_runtime::debug_forced_split_failure()) {
            throw std::runtime_error("forced split failure");
        }
        return natten_mlx::nanobind_backend::call_backend(
            "na2d_av_forward", attn, v, kernel_size, stride, dilation, is_causal);
    }
    try {
        return natten_mlx::nanobind_metal_runtime::na2d_av_forward(
            attn, v, kernel_size, stride, dilation, is_causal);
    } catch (const std::bad_cast&) {
        return natten_mlx::nanobind_backend::call_backend(
            "na2d_av_forward", attn, v, kernel_size, stride, dilation, is_causal);
    }
}

nb::object na3d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    if (!natten_mlx::nanobind_backend::use_native_runtime()) {
        if (natten_mlx::nanobind_metal_runtime::debug_forced_split_failure()) {
            throw std::runtime_error("forced split failure");
        }
        return natten_mlx::nanobind_backend::call_backend(
            "na3d_qk_forward", q, k, kernel_size, stride, dilation, is_causal, scale);
    }
    try {
        return natten_mlx::nanobind_metal_runtime::na3d_qk_forward(
            q, k, kernel_size, stride, dilation, is_causal, scale);
    } catch (const std::bad_cast&) {
        return natten_mlx::nanobind_backend::call_backend(
            "na3d_qk_forward", q, k, kernel_size, stride, dilation, is_causal, scale);
    }
}

nb::object na3d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
    if (!natten_mlx::nanobind_backend::use_native_runtime()) {
        if (natten_mlx::nanobind_metal_runtime::debug_forced_split_failure()) {
            throw std::runtime_error("forced split failure");
        }
        return natten_mlx::nanobind_backend::call_backend(
            "na3d_av_forward", attn, v, kernel_size, stride, dilation, is_causal);
    }
    try {
        return natten_mlx::nanobind_metal_runtime::na3d_av_forward(
            attn, v, kernel_size, stride, dilation, is_causal);
    } catch (const std::bad_cast&) {
        return natten_mlx::nanobind_backend::call_backend(
            "na3d_av_forward", attn, v, kernel_size, stride, dilation, is_causal);
    }
}

}  // namespace natten_mlx::nanobind_split_forward
