#include "na_composed.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <nanobind/stl/tuple.h>

#include "metal_runtime.h"
#include "na_fused_backward.h"
#include "na_fused_forward.h"
#include "na_split_backward.h"
#include "na_split_forward.h"
#include "na1d_primitive.h"
#include "na2d_primitive.h"
#include "na3d_primitive.h"
#include "na1d_bwd_primitive.h"
#include "na2d_bwd_primitive.h"
#include "na3d_bwd_primitive.h"
#include "py_dispatch.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

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

enum class SplitForwardDtypeMode {
    Auto,
    Native,
    FP32,
};

SplitForwardDtypeMode parse_split_forward_dtype_mode() {
    const char* value = std::getenv("NATTEN_NANOBIND_SPLIT_FWD_DTYPE_MODE");
    if (value == nullptr) {
        return SplitForwardDtypeMode::Auto;
    }
    std::string mode(value);
    if (mode == "native") {
        return SplitForwardDtypeMode::Native;
    }
    if (mode == "fp32") {
        return SplitForwardDtypeMode::FP32;
    }
    return SplitForwardDtypeMode::Auto;
}

bool is_low_precision_dtype(mx::Dtype dtype) {
    return dtype == mx::float16 || dtype == mx::bfloat16;
}

bool prefer_fp32_split_pipeline(
    const nb::object& q,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
    (void)stride;
    (void)dilation;
    (void)is_causal;
    SplitForwardDtypeMode mode = parse_split_forward_dtype_mode();
    if (mode == SplitForwardDtypeMode::Native) {
        return false;
    }
    mx::array q_arr = nb::cast<mx::array>(q);
    if (!is_low_precision_dtype(q_arr.dtype())) {
        return false;
    }
    if (mode == SplitForwardDtypeMode::FP32) {
        return true;
    }
    // Auto mode defaults to native low-precision split kernels.
    return false;
}

enum class FusedForwardMode {
    Auto,
    Fused,
    Split,
};

FusedForwardMode parse_fused_forward_mode(const char* env_name) {
    const char* value = std::getenv(env_name);
    if (value == nullptr) {
        return FusedForwardMode::Auto;
    }
    std::string mode(value);
    if (mode == "fused") {
        return FusedForwardMode::Fused;
    }
    if (mode == "split") {
        return FusedForwardMode::Split;
    }
    return FusedForwardMode::Auto;
}

bool prefer_split_forward_2d_fastpath(
    const nb::object& q,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
    FusedForwardMode mode = parse_fused_forward_mode("NATTEN_NANOBIND_FUSED_FWD_2D_MODE");
    if (mode == FusedForwardMode::Fused) {
        return false;
    }
    if (mode == FusedForwardMode::Split) {
        return true;
    }
    mx::array q_arr = nb::cast<mx::array>(q);
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

nb::object softmax_last_dim(const nb::object& x) {
    mx::array xa = nb::cast<mx::array>(x);
    return nb::cast(mx::softmax(xa, -1));
}

nb::object grad_logits_from_softmax(
    const nb::object& attn,
    const nb::object& grad_attn) {
    mx::array attn_arr = nb::cast<mx::array>(attn);
    mx::array grad_attn_arr = nb::cast<mx::array>(grad_attn);
    mx::array prod = mx::multiply(grad_attn_arr, attn_arr);
    mx::array inner = mx::sum(prod, -1, true);
    mx::array centered = mx::subtract(grad_attn_arr, inner);
    return nb::cast(mx::multiply(attn_arr, centered));
}

nb::object backward_composed(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale,
    nb::object (*qk_forward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&),
    nb::object (*av_forward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&),
    nb::object (*qk_backward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&),
    nb::object (*av_backward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&)) {
    nb::object logits = qk_forward(q, k, kernel_size, stride, dilation, is_causal, scale);
    nb::object attn = softmax_last_dim(logits);
    nb::tuple av_bw = nb::cast<nb::tuple>(
        av_backward(attn, v, grad_out, kernel_size, stride, dilation, is_causal));
    nb::object grad_attn = av_bw[0];
    nb::object grad_v = av_bw[1];
    nb::object grad_logits = grad_logits_from_softmax(attn, grad_attn);

    nb::tuple qk_bw = nb::cast<nb::tuple>(
        qk_backward(q, k, grad_logits, kernel_size, stride, dilation, is_causal, scale));
    nb::object grad_q = qk_bw[0];
    nb::object grad_k = qk_bw[1];
    return nb::make_tuple(grad_q, grad_k, grad_v);
}

nb::object forward_split_composed(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale,
    nb::object (*qk_forward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&),
    nb::object (*av_forward)(
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&,
        const nb::object&)) {
    nb::object mx_mod = natten_mlx::nanobind_backend::mx_module();
    nb::object qx = q;
    nb::object kx = k;
    nb::object vx = v;
    bool cast_back_out = false;
    if (prefer_fp32_split_pipeline(q, stride, dilation, is_causal)) {
        nb::object fp32 = mx_mod.attr("float32");
        qx = q.attr("astype")(fp32);
        kx = k.attr("astype")(fp32);
        vx = v.attr("astype")(fp32);
        cast_back_out = true;
    }

    nb::object logits = qk_forward(qx, kx, kernel_size, stride, dilation, is_causal, scale);
    nb::object attn = softmax_last_dim(logits);
    nb::object out = av_forward(attn, vx, kernel_size, stride, dilation, is_causal);
    if (cast_back_out) {
        out = out.attr("astype")(v.attr("dtype"));
    }
    return out;
}

}  // namespace

namespace natten_mlx::nanobind_composed {

nb::object na1d_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
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
                int ks = scalar_or_index_int(kernel_size, 0);
                int s = scalar_or_index_int(stride, 0);
                int d = scalar_or_index_int(dilation, 0);
                bool c = scalar_or_index_bool(is_causal, 0);
                int head_dim = q_arr.shape(3);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 63) &&
                    s >= 1 && d >= 1;
                if (eligible) {
                    mx::array k_arr = nb::cast<mx::array>(k);
                    mx::array v_arr = nb::cast<mx::array>(v);
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    mx::array out = natten_mlx::na1d_fused_forward_v2(
                        q_arr, k_arr, v_arr, ks, s, d, c, sc);
                    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_forward", "v2_primitive");
                    return nb::cast(out);
                }
            } catch (...) {
                // Fall through to legacy paths
            }
        }
    }
    try {
        nb::object out = natten_mlx::nanobind_fused_forward::na1d_forward(
            q, k, v, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_forward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = forward_split_composed(
            q,
            k,
            v,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na1d_qk_forward,
            natten_mlx::nanobind_split_forward::na1d_av_forward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_forward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_forward", "pure");
    return natten_mlx::nanobind_backend::call_pure(
        "na1d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
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
    // Try v2 primitive path first (lazy graph, heads-first layout, optimized kernels)
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                mx::array k_arr = nb::cast<mx::array>(k);
                mx::array v_arr = nb::cast<mx::array>(v);
                int ks = scalar_or_index_int(kernel_size, 0);
                int sh = scalar_or_index_int(stride, 0);
                int sw = scalar_or_index_int(stride, 1);
                int dh = scalar_or_index_int(dilation, 0);
                int dw = scalar_or_index_int(dilation, 1);
                bool ch = scalar_or_index_bool(is_causal, 0);
                bool cw = scalar_or_index_bool(is_causal, 1);
                int head_dim = q_arr.shape(4);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 13) &&
                    sh >= 1 && sw >= 1 && dh >= 1 && dw >= 1;
                if (eligible) {
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    mx::array out = natten_mlx::na2d_fused_forward_v2(
                        q_arr, k_arr, v_arr,
                        ks, sh, sw, dh, dw, ch, cw, sc);
                    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_forward", "v2_primitive");
                    return nb::cast(out);
                }
            } catch (...) {
                // Fall through to legacy paths
            }
        }
    }
    if (prefer_split_forward_2d_fastpath(q, kernel_size, stride, dilation, is_causal)) {
        try {
            nb::object out = forward_split_composed(
                q,
                k,
                v,
                kernel_size,
                stride,
                dilation,
                is_causal,
                scale,
                natten_mlx::nanobind_split_forward::na2d_qk_forward,
                natten_mlx::nanobind_split_forward::na2d_av_forward);
            natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_forward", "split");
            return out;
        } catch (...) {
        }
    }
    try {
        nb::object out = natten_mlx::nanobind_fused_forward::na2d_forward(
            q, k, v, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_forward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = forward_split_composed(
            q,
            k,
            v,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na2d_qk_forward,
            natten_mlx::nanobind_split_forward::na2d_av_forward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_forward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_forward", "pure");
    return natten_mlx::nanobind_backend::call_pure(
        "na2d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
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
    // Try v2 primitive path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                int ks = scalar_or_index_int(kernel_size, 0);
                int sd = scalar_or_index_int(stride, 0);
                int sh = scalar_or_index_int(stride, 1);
                int sw = scalar_or_index_int(stride, 2);
                int dd = scalar_or_index_int(dilation, 0);
                int dh = scalar_or_index_int(dilation, 1);
                int dw = scalar_or_index_int(dilation, 2);
                bool cd = scalar_or_index_bool(is_causal, 0);
                bool ch = scalar_or_index_bool(is_causal, 1);
                bool cw = scalar_or_index_bool(is_causal, 2);
                int head_dim = q_arr.shape(5);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 7) &&
                    sd >= 1 && sh >= 1 && sw >= 1 &&
                    dd >= 1 && dh >= 1 && dw >= 1;
                if (eligible) {
                    mx::array k_arr = nb::cast<mx::array>(k);
                    mx::array v_arr = nb::cast<mx::array>(v);
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    mx::array out = natten_mlx::na3d_fused_forward_v2(
                        q_arr, k_arr, v_arr, ks,
                        sd, sh, sw, dd, dh, dw, cd, ch, cw, sc);
                    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_forward", "v2_primitive");
                    return nb::cast(out);
                }
            } catch (...) {
                // Fall through to legacy paths
            }
        }
    }
    try {
        nb::object out = natten_mlx::nanobind_fused_forward::na3d_forward(
            q, k, v, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_forward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = forward_split_composed(
            q,
            k,
            v,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na3d_qk_forward,
            natten_mlx::nanobind_split_forward::na3d_av_forward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_forward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_forward", "pure");
    return natten_mlx::nanobind_backend::call_pure(
        "na3d_forward", q, k, v, kernel_size, stride, dilation, is_causal, scale);
}

nb::object na1d_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    // Try v2 primitive backward path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                mx::array k_arr = nb::cast<mx::array>(k);
                mx::array v_arr = nb::cast<mx::array>(v);
                mx::array go_arr = nb::cast<mx::array>(grad_out);
                int ks = scalar_or_index_int(kernel_size, 0);
                int s = scalar_or_index_int(stride, 0);
                int d = scalar_or_index_int(dilation, 0);
                bool c = scalar_or_index_bool(is_causal, 0);
                int head_dim = q_arr.shape(3);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 63) &&
                    s >= 1 && d >= 1;
                if (eligible) {
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    auto grads = natten_mlx::na1d_backward_v2(
                        q_arr, k_arr, v_arr, go_arr,
                        ks, s, d, c, sc);
                    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_backward", "v2_primitive");
                    return nb::cast(nb::make_tuple(
                        nb::cast(grads[0]), nb::cast(grads[1]), nb::cast(grads[2])));
                }
            } catch (...) {
                // Fall through to legacy paths
            }
        }
    }
    try {
        nb::object out = natten_mlx::nanobind_fused_backward::na1d_backward(
            q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_backward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = backward_composed(
            q,
            k,
            v,
            grad_out,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na1d_qk_forward,
            natten_mlx::nanobind_split_forward::na1d_av_forward,
            natten_mlx::nanobind_split_backward::na1d_qk_backward,
            natten_mlx::nanobind_split_backward::na1d_av_backward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_backward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na1d_backward", "pure");
    return natten_mlx::nanobind_backend::call_pure(
        "na1d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
}

nb::object na2d_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    // Try v2 primitive backward path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                mx::array k_arr = nb::cast<mx::array>(k);
                mx::array v_arr = nb::cast<mx::array>(v);
                mx::array go_arr = nb::cast<mx::array>(grad_out);
                int ks = scalar_or_index_int(kernel_size, 0);
                int sh = scalar_or_index_int(stride, 0);
                int sw = scalar_or_index_int(stride, 1);
                int dh = scalar_or_index_int(dilation, 0);
                int dw = scalar_or_index_int(dilation, 1);
                bool ch = scalar_or_index_bool(is_causal, 0);
                bool cw = scalar_or_index_bool(is_causal, 1);
                int head_dim = q_arr.shape(4);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 13) &&
                    sh >= 1 && sw >= 1 && dh >= 1 && dw >= 1;
                if (eligible) {
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    auto grads = natten_mlx::na2d_backward_v2(
                        q_arr, k_arr, v_arr, go_arr,
                        ks, sh, sw, dh, dw, ch, cw, sc);
                    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_backward", "v2_primitive");
                    return nb::cast(nb::make_tuple(
                        nb::cast(grads[0]), nb::cast(grads[1]), nb::cast(grads[2])));
                }
            } catch (...) {
                // Fall through to legacy paths
            }
        }
    }
    try {
        nb::object out = natten_mlx::nanobind_fused_backward::na2d_backward(
            q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_backward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = backward_composed(
            q,
            k,
            v,
            grad_out,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na2d_qk_forward,
            natten_mlx::nanobind_split_forward::na2d_av_forward,
            natten_mlx::nanobind_split_backward::na2d_qk_backward,
            natten_mlx::nanobind_split_backward::na2d_av_backward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_backward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na2d_backward", "pure");
    return natten_mlx::nanobind_backend::call_pure(
        "na2d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
}

nb::object na3d_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
    // Try v2 primitive backward path first
    {
        const char* v2_env = std::getenv("NATTEN_NANOBIND_DISABLE_V2");
        bool v2_disabled = v2_env != nullptr && std::string(v2_env) == "1";
        if (!v2_disabled) {
            try {
                mx::array q_arr = nb::cast<mx::array>(q);
                mx::array k_arr = nb::cast<mx::array>(k);
                mx::array v_arr = nb::cast<mx::array>(v);
                mx::array go_arr = nb::cast<mx::array>(grad_out);
                int ks = scalar_or_index_int(kernel_size, 0);
                int sd = scalar_or_index_int(stride, 0);
                int sh = scalar_or_index_int(stride, 1);
                int sw = scalar_or_index_int(stride, 2);
                int dd = scalar_or_index_int(dilation, 0);
                int dh = scalar_or_index_int(dilation, 1);
                int dw = scalar_or_index_int(dilation, 2);
                bool cd = scalar_or_index_bool(is_causal, 0);
                bool ch = scalar_or_index_bool(is_causal, 1);
                bool cw = scalar_or_index_bool(is_causal, 2);
                int head_dim = q_arr.shape(5);
                bool eligible = (ks > 0) && (ks % 2 == 1) && (ks <= 7) &&
                    sd >= 1 && sh >= 1 && sw >= 1 &&
                    dd >= 1 && dh >= 1 && dw >= 1;
                if (eligible) {
                    float sc;
                    if (scale.is_none()) {
                        sc = std::pow(static_cast<float>(head_dim), -0.5f);
                    } else {
                        sc = nb::cast<float>(scale);
                    }
                    auto grads = natten_mlx::na3d_backward_v2(
                        q_arr, k_arr, v_arr, go_arr,
                        ks, sd, sh, sw, dd, dh, dw, cd, ch, cw, sc);
                    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_backward", "v2_primitive");
                    return nb::cast(nb::make_tuple(
                        nb::cast(grads[0]), nb::cast(grads[1]), nb::cast(grads[2])));
                }
            } catch (...) {
                // Fall through to legacy paths
            }
        }
    }
    try {
        nb::object out = natten_mlx::nanobind_fused_backward::na3d_backward(
            q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_backward", "fused");
        return out;
    } catch (...) {
    }
    try {
        nb::object out = backward_composed(
            q,
            k,
            v,
            grad_out,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            natten_mlx::nanobind_split_forward::na3d_qk_forward,
            natten_mlx::nanobind_split_forward::na3d_av_forward,
            natten_mlx::nanobind_split_backward::na3d_qk_backward,
            natten_mlx::nanobind_split_backward::na3d_av_backward);
        natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_backward", "split");
        return out;
    } catch (...) {
    }
    natten_mlx::nanobind_metal_runtime::debug_set_last_route("na3d_backward", "pure");
    return natten_mlx::nanobind_backend::call_pure(
        "na3d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
}

}  // namespace natten_mlx::nanobind_composed
