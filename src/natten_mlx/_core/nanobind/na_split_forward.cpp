#include "na_split_forward.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <typeinfo>

#include <mlx/array.h>

#include "metal_runtime.h"
#include "na2d_split_primitive.h"
#include "py_dispatch.h"

namespace {

bool is_sequence(const nb::object& obj) {
    return nb::isinstance<nb::sequence>(obj) && !nb::isinstance<nb::str>(obj);
}

int scalar_or_index_int(const nb::object& obj, size_t idx) {
    if (!is_sequence(obj)) return nb::cast<int>(obj);
    nb::sequence seq = nb::cast<nb::sequence>(obj);
    return nb::cast<int>(seq[idx]);
}

bool scalar_or_index_bool(const nb::object& obj, size_t idx) {
    if (!is_sequence(obj)) return nb::cast<bool>(obj);
    nb::sequence seq = nb::cast<nb::sequence>(obj);
    return nb::cast<bool>(seq[idx]);
}

}  // namespace

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
    // Try v2 primitive path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                mx::array k_arr = nb::cast<mx::array>(k);
                int ks = scalar_or_index_int(kernel_size, 0);
                int sh = scalar_or_index_int(stride, 0);
                int sw = scalar_or_index_int(stride, 1);
                int dh = scalar_or_index_int(dilation, 0);
                int dw = scalar_or_index_int(dilation, 1);
                bool ch = scalar_or_index_bool(is_causal, 0);
                bool cw = scalar_or_index_bool(is_causal, 1);
                int head_dim = q_arr.shape(4);
                float sc;
                if (scale.is_none()) {
                    sc = std::pow(static_cast<float>(head_dim), -0.5f);
                } else {
                    sc = nb::cast<float>(scale);
                }
                if (ks > 0 && ks % 2 == 1 && ks <= 13) {
                    mx::array out = natten_mlx::na2d_qk_forward_v2(
                        q_arr, k_arr, ks, sh, sw, dh, dw, ch, cw, sc);
                    return nb::cast(out);
                }
            } catch (...) {
                // Fall through to legacy
            }
        }
    }
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
    // Try v2 primitive path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array attn_arr = nb::cast<mx::array>(attn);
                mx::array v_arr = nb::cast<mx::array>(v);
                int ks = scalar_or_index_int(kernel_size, 0);
                int sh = scalar_or_index_int(stride, 0);
                int sw = scalar_or_index_int(stride, 1);
                int dh = scalar_or_index_int(dilation, 0);
                int dw = scalar_or_index_int(dilation, 1);
                bool ch = scalar_or_index_bool(is_causal, 0);
                bool cw = scalar_or_index_bool(is_causal, 1);
                if (ks > 0 && ks % 2 == 1 && ks <= 13) {
                    mx::array out = natten_mlx::na2d_av_forward_v2(
                        attn_arr, v_arr, ks, sh, sw, dh, dw, ch, cw);
                    return nb::cast(out);
                }
            } catch (...) {
                // Fall through to legacy
            }
        }
    }
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
