#include "na_fused_forward.h"

#include <cstdlib>
#include <string>
#include <typeinfo>

#include <mlx/array.h>

#include "metal_runtime.h"
#include "py_dispatch.h"

namespace nb = nanobind;
namespace mx = mlx::core;

namespace {

bool is_sequence(const nb::object& obj) {
    return nb::isinstance<nb::tuple>(obj) || nb::isinstance<nb::list>(obj);
}

int scalar_or_index_int(const nb::object& obj, size_t idx) {
    if (!is_sequence(obj)) {
        return nb::cast<int>(obj);
    }
    nb::sequence seq = nb::cast<nb::sequence>(obj);
    if (idx >= static_cast<size_t>(nb::len(seq))) {
        throw std::runtime_error("invalid parameter rank");
    }
    return nb::cast<int>(seq[idx]);
}

bool scalar_or_index_bool(const nb::object& obj, size_t idx) {
    if (!is_sequence(obj)) {
        return nb::cast<bool>(obj);
    }
    nb::sequence seq = nb::cast<nb::sequence>(obj);
    if (idx >= static_cast<size_t>(nb::len(seq))) {
        throw std::runtime_error("invalid parameter rank");
    }
    return nb::cast<bool>(seq[idx]);
}

bool env_force_split(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::string(value) == "split";
}

bool env_force_fused(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::string(value) == "fused";
}

bool prefer_split_composed_fwd_1d(
    const nb::object& q,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
    if (env_force_fused("NATTEN_NANOBIND_FUSED_FWD_1D_MODE")) {
        return false;
    }
    if (env_force_split("NATTEN_NANOBIND_FUSED_FWD_1D_MODE")) {
        return true;
    }
    auto q_arr = nb::cast<mx::array>(q);
    int tokens = q_arr.shape(1);
    int head_dim = q_arr.shape(3);
    int k = scalar_or_index_int(kernel_size, 0);
    int s = scalar_or_index_int(stride, 0);
    int d = scalar_or_index_int(dilation, 0);
    bool causal = scalar_or_index_bool(is_causal, 0);
    bool lowp = (q_arr.dtype() == mx::float16 || q_arr.dtype() == mx::bfloat16);
    bool decode_like = tokens >= 1536;
    // Retuned: split wins for medium-length low-precision causal bands; decode-like stays fused.
    return lowp && causal && s == 1 && d == 1 && k >= 9 && tokens >= 384 && tokens <= 1024 &&
        head_dim >= 32 && !decode_like;
}

bool prefer_split_composed_fwd_2d(
    const nb::object& q,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
    if (env_force_fused("NATTEN_NANOBIND_FUSED_FWD_2D_MODE")) {
        return false;
    }
    if (env_force_split("NATTEN_NANOBIND_FUSED_FWD_2D_MODE")) {
        return true;
    }
    auto q_arr = nb::cast<mx::array>(q);
    int ih = q_arr.shape(1);
    int iw = q_arr.shape(2);
    int k = scalar_or_index_int(kernel_size, 0);
    int sh = scalar_or_index_int(stride, 0);
    int sw = scalar_or_index_int(stride, 1);
    int dh = scalar_or_index_int(dilation, 0);
    int dw = scalar_or_index_int(dilation, 1);
    bool ch = scalar_or_index_bool(is_causal, 0);
    bool cw = scalar_or_index_bool(is_causal, 1);
    int d = q_arr.shape(4);
    int causal_rank = (ch ? 1 : 0) + (cw ? 1 : 0);
    bool unit_step = (sh == 1 && sw == 1 && dh == 1 && dw == 1);
    int tokens = ih * iw;
    // Keep fused eligible for the optimized bf16 strided causal-H kernel family.
    if (q_arr.dtype() == mx::bfloat16 && ch && !cw && d == 16 && k == 7 && sh == 2 && sw == 1 &&
        dh == 1 && dw == 2) {
        return false;
    }
    return k >= 7 && tokens >= 256 && (causal_rank > 0 || !unit_step || tokens >= 512);
}

bool prefer_split_composed_fwd_3d(
    const nb::object& q,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
    if (env_force_fused("NATTEN_NANOBIND_FUSED_FWD_3D_MODE")) {
        return false;
    }
    if (env_force_split("NATTEN_NANOBIND_FUSED_FWD_3D_MODE")) {
        return true;
    }
    auto q_arr = nb::cast<mx::array>(q);
    int id = q_arr.shape(1);
    int ih = q_arr.shape(2);
    int iw = q_arr.shape(3);
    int k = scalar_or_index_int(kernel_size, 0);
    int sd = scalar_or_index_int(stride, 0);
    int sh = scalar_or_index_int(stride, 1);
    int sw = scalar_or_index_int(stride, 2);
    int dd = scalar_or_index_int(dilation, 0);
    int dh = scalar_or_index_int(dilation, 1);
    int dw = scalar_or_index_int(dilation, 2);
    bool cd = scalar_or_index_bool(is_causal, 0);
    bool ch = scalar_or_index_bool(is_causal, 1);
    bool cw = scalar_or_index_bool(is_causal, 2);
    int causal_rank = (cd ? 1 : 0) + (ch ? 1 : 0) + (cw ? 1 : 0);
    bool unit_step = (sd == 1 && sh == 1 && sw == 1 && dd == 1 && dh == 1 && dw == 1);
    int tokens = id * ih * iw;
    // Keep fused for unit-stride causal 3D; split wins for strided/dilated or large noncausal.
    if (!unit_step && tokens >= 512 && k >= 3) {
        return true;
    }
    return causal_rank == 0 && unit_step && k >= 3 && tokens >= 1024;
}

}  // namespace

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
    if (prefer_split_composed_fwd_1d(q, kernel_size, stride, dilation, is_causal)) {
        throw std::runtime_error("nanobind fused 1D forward prefers split-composed path");
    }
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
    if (prefer_split_composed_fwd_2d(q, kernel_size, stride, dilation, is_causal)) {
        throw std::runtime_error("nanobind fused 2D forward prefers split-composed path");
    }
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
    if (prefer_split_composed_fwd_3d(q, kernel_size, stride, dilation, is_causal)) {
        throw std::runtime_error("nanobind fused 3D forward prefers split-composed path");
    }
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
