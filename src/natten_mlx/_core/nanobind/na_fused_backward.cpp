#include "na_fused_backward.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <nanobind/stl/tuple.h>

#include "metal_runtime.h"
#include "py_dispatch.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {

bool is_sequence(const nb::object& obj) {
  return nb::isinstance<nb::tuple>(obj) || nb::isinstance<nb::list>(obj);
}

bool enable_qkv_stage_fusion() {
  const char* v = std::getenv("NATTEN_NANOBIND_FUSED_BWD_QKV_STAGE");
  if (v == nullptr) {
    return true;
  }
  std::string mode(v);
  if (mode == "0" || mode == "false" || mode == "off") {
    return false;
  }
  return true;
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

int shape_index(const nb::object& x, size_t idx) {
  nb::sequence shape = nb::cast<nb::sequence>(x.attr("shape"));
  if (idx >= static_cast<size_t>(nb::len(shape))) {
    throw std::runtime_error("invalid tensor rank");
  }
  return nb::cast<int>(shape[idx]);
}

bool prefer_split_composed_bwd_2d(
    const nb::object& q,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  if (const char* mode = std::getenv("NATTEN_NANOBIND_FUSED_BWD_2D_MODE")) {
    std::string m(mode);
    if (m == "split") {
      return true;
    }
    if (m == "fused") {
      return false;
    }
  }
  (void)q;
  (void)kernel_size;
  (void)stride;
  (void)dilation;
  (void)is_causal;
  // Default to fused staged backward now that native q/kv paths are optimized.
  // Split remains available via explicit env override.
  return false;
}

bool prefer_split_composed_bwd_1d(
    const nb::object& q,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  if (const char* mode = std::getenv("NATTEN_NANOBIND_FUSED_BWD_1D_MODE")) {
    std::string m(mode);
    if (m == "split") {
      return true;
    }
    if (m == "fused") {
      return false;
    }
  }
  (void)q;
  (void)kernel_size;
  (void)stride;
  (void)dilation;
  (void)is_causal;
  // Prefer fused staged backward by default; split remains available
  // via explicit env override.
  return false;
}

bool prefer_split_composed_bwd_3d(
    const nb::object& q,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  if (const char* mode = std::getenv("NATTEN_NANOBIND_FUSED_BWD_3D_MODE")) {
    std::string m(mode);
    if (m == "split") {
      return true;
    }
    if (m == "fused") {
      return false;
    }
  }
  int id = shape_index(q, 1);
  int ih = shape_index(q, 2);
  int iw = shape_index(q, 3);
  int head_dim = shape_index(q, 5);
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

  // Guarded auto-route: for large, noncausal unit-step 3D shapes,
  // split-composed backward is currently more efficient than fused-staged.
  int tokens = id * ih * iw;
  bool noncausal = !(cd || ch || cw);
  bool unit_step = (sd == 1 && sh == 1 && sw == 1 && dd == 1 && dh == 1 && dw == 1);
  if (noncausal && unit_step && k == 3 && tokens >= 768 && head_dim <= 64) {
    return true;
  }

  // Otherwise default to fused staged backward.
  return false;
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

nb::object fused_backward_1d(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  if (prefer_split_composed_bwd_1d(q, kernel_size, stride, dilation, is_causal)) {
    nb::object logits = natten_mlx::nanobind_metal_runtime::na1d_qk_forward(
        q, k, kernel_size, stride, dilation, is_causal, scale);
    nb::object attn = softmax_last_dim(logits);
    nb::tuple av_bw = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na1d_av_backward(
        attn, v, grad_out, kernel_size, stride, dilation, is_causal));
    nb::object grad_attn = av_bw[0];
    nb::object grad_v = av_bw[1];
    nb::object grad_logits = grad_logits_from_softmax(attn, grad_attn);
    nb::tuple qk_bw = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na1d_qk_backward(
        q, k, grad_logits, kernel_size, stride, dilation, is_causal, scale));
    return nb::make_tuple(qk_bw[0], qk_bw[1], grad_v);
  }
  nb::tuple pair = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na1d_fused_backward_attn(
      q,
      k,
      v,
      grad_out,
      kernel_size,
      stride,
      dilation,
      is_causal,
      scale));
  nb::object attn = pair[0];
  nb::object grad_attn = pair[1];

  // Prefer a native fused qkv stage for supported 1D unit-stride shapes.
  // This cuts one stage and avoids materializing grad_logits.
  try {
    nb::tuple qkv = nb::cast<nb::tuple>(
        natten_mlx::nanobind_metal_runtime::na1d_fused_backward_qkv_from_softmax(
            q,
            k,
            v,
            attn,
            grad_attn,
            grad_out,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale));
    return nb::make_tuple(qkv[0], qkv[1], qkv[2]);
  } catch (const std::exception&) {
  }

  nb::object grad_logits = grad_logits_from_softmax(attn, grad_attn);

  nb::tuple qk = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na1d_fused_backward_qk(
      q,
      k,
      grad_logits,
      kernel_size,
      stride,
      dilation,
      is_causal,
      scale));
  nb::object grad_v = natten_mlx::nanobind_metal_runtime::na1d_fused_backward_v(
      attn,
      v,
      grad_out,
      kernel_size,
      stride,
      dilation,
      is_causal);

  return nb::make_tuple(qk[0], qk[1], grad_v);
}

nb::object fused_backward_2d(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  if (prefer_split_composed_bwd_2d(q, kernel_size, stride, dilation, is_causal)) {
    nb::object logits = natten_mlx::nanobind_metal_runtime::na2d_qk_forward(
        q, k, kernel_size, stride, dilation, is_causal, scale);
    nb::object attn = softmax_last_dim(logits);
    nb::tuple av_bw = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na2d_av_backward(
        attn, v, grad_out, kernel_size, stride, dilation, is_causal));
    nb::object grad_attn = av_bw[0];
    nb::object grad_v = av_bw[1];
    nb::object grad_logits = grad_logits_from_softmax(attn, grad_attn);
    nb::tuple qk_bw = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na2d_qk_backward(
        q, k, grad_logits, kernel_size, stride, dilation, is_causal, scale));
    return nb::make_tuple(qk_bw[0], qk_bw[1], grad_v);
  }
  nb::tuple pair = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na2d_fused_backward_attn(
      q,
      k,
      v,
      grad_out,
      kernel_size,
      stride,
      dilation,
      is_causal,
      scale));
  nb::object attn = pair[0];
  nb::object grad_attn = pair[1];
  if (enable_qkv_stage_fusion()) {
    try {
      nb::tuple qkv = nb::cast<nb::tuple>(
          natten_mlx::nanobind_metal_runtime::na2d_fused_backward_qkv_from_softmax(
              q,
              k,
              v,
              attn,
              grad_attn,
              grad_out,
              kernel_size,
              stride,
              dilation,
              is_causal,
              scale));
      return nb::make_tuple(qkv[0], qkv[1], qkv[2]);
    } catch (const std::exception&) {
    }
  }
  nb::tuple qk =
      nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na2d_fused_backward_qk_from_softmax(
          q,
          k,
          attn,
          grad_attn,
          kernel_size,
          stride,
          dilation,
          is_causal,
          scale));
  nb::object grad_v = natten_mlx::nanobind_metal_runtime::na2d_fused_backward_v(
      attn,
      v,
      grad_out,
      kernel_size,
      stride,
      dilation,
      is_causal);
  return nb::make_tuple(qk[0], qk[1], grad_v);
}

nb::object fused_backward_3d(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  if (prefer_split_composed_bwd_3d(q, kernel_size, stride, dilation, is_causal)) {
    bool use_qk_from_softmax = true;
    if (const char* mode = std::getenv("NATTEN_NANOBIND_SPLIT3D_QK_SOFTMAX_MODE")) {
      std::string m(mode);
      if (m == "on" || m == "1" || m == "true") {
        use_qk_from_softmax = true;
      } else if (m == "off" || m == "0" || m == "false") {
        use_qk_from_softmax = false;
      }
    }

    // Fast path for split-composed 3D: compute attn/grad_attn in one fused stage
    // and then consume those directly in fused qk/qkv softmax backward kernels.
    // This reduces split-chain materialization overhead (qk logits + explicit softmax
    // + av grad_attn pass) on the main noncausal unit-step hot shape band.
    if (use_qk_from_softmax) {
      try {
        nb::tuple pair = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na3d_fused_backward_attn(
            q,
            k,
            v,
            grad_out,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale));
        nb::object attn = pair[0];
        nb::object grad_attn = pair[1];
        if (enable_qkv_stage_fusion()) {
          try {
            nb::tuple qkv = nb::cast<nb::tuple>(
                natten_mlx::nanobind_metal_runtime::na3d_fused_backward_qkv_from_softmax(
                    q,
                    k,
                    v,
                    attn,
                    grad_attn,
                    grad_out,
                    kernel_size,
                    stride,
                    dilation,
                    is_causal,
                    scale));
            return nb::make_tuple(qkv[0], qkv[1], qkv[2]);
          } catch (const std::exception&) {
          }
        }
        nb::tuple qk_from_softmax = nb::cast<nb::tuple>(
            natten_mlx::nanobind_metal_runtime::na3d_fused_backward_qk_from_softmax(
                q,
                k,
                attn,
                grad_attn,
                kernel_size,
                stride,
                dilation,
                is_causal,
                scale));
        nb::object grad_v = natten_mlx::nanobind_metal_runtime::na3d_fused_backward_v(
            attn,
            v,
            grad_out,
            kernel_size,
            stride,
            dilation,
            is_causal);
        return nb::make_tuple(qk_from_softmax[0], qk_from_softmax[1], grad_v);
      } catch (const std::exception&) {
      }
    }

    nb::object logits = natten_mlx::nanobind_metal_runtime::na3d_qk_forward(
        q, k, kernel_size, stride, dilation, is_causal, scale);
    nb::object attn = softmax_last_dim(logits);
    nb::tuple av_bw = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na3d_av_backward(
        attn, v, grad_out, kernel_size, stride, dilation, is_causal));
    nb::object grad_attn = av_bw[0];
    nb::object grad_v = av_bw[1];
    // Reduce split-composed materialization overhead: consume attn+grad_attn
    // directly in fused qk-from-softmax when available.
    if (use_qk_from_softmax) {
      try {
        nb::tuple qk_from_softmax = nb::cast<nb::tuple>(
            natten_mlx::nanobind_metal_runtime::na3d_fused_backward_qk_from_softmax(
                q,
                k,
                attn,
                grad_attn,
                kernel_size,
                stride,
                dilation,
                is_causal,
                scale));
        return nb::make_tuple(qk_from_softmax[0], qk_from_softmax[1], grad_v);
      } catch (const std::exception&) {
      }
    }
    nb::object grad_logits = grad_logits_from_softmax(attn, grad_attn);
    nb::tuple qk_bw = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na3d_qk_backward(
        q, k, grad_logits, kernel_size, stride, dilation, is_causal, scale));
    return nb::make_tuple(qk_bw[0], qk_bw[1], grad_v);
  }
  nb::tuple pair = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na3d_fused_backward_attn(
      q,
      k,
      v,
      grad_out,
      kernel_size,
      stride,
      dilation,
      is_causal,
      scale));
  nb::object attn = pair[0];
  nb::object grad_attn = pair[1];
  if (enable_qkv_stage_fusion()) {
    try {
      nb::tuple qkv = nb::cast<nb::tuple>(
          natten_mlx::nanobind_metal_runtime::na3d_fused_backward_qkv_from_softmax(
              q,
              k,
              v,
              attn,
              grad_attn,
              grad_out,
              kernel_size,
              stride,
              dilation,
              is_causal,
              scale));
      return nb::make_tuple(qkv[0], qkv[1], qkv[2]);
    } catch (const std::exception&) {
    }
  }
  nb::tuple qk =
      nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na3d_fused_backward_qk_from_softmax(
          q,
          k,
          attn,
          grad_attn,
          kernel_size,
          stride,
          dilation,
          is_causal,
          scale));
  nb::object grad_v = natten_mlx::nanobind_metal_runtime::na3d_fused_backward_v(
      attn,
      v,
      grad_out,
      kernel_size,
      stride,
      dilation,
      is_causal);
  return nb::make_tuple(qk[0], qk[1], grad_v);
}

}  // namespace

namespace natten_mlx::nanobind_fused_backward {

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
  if (!natten_mlx::nanobind_backend::use_native_runtime()) {
    if (natten_mlx::nanobind_metal_runtime::debug_forced_fused_failure()) {
      throw std::runtime_error("forced fused failure");
    }
    return natten_mlx::nanobind_backend::call_backend(
        "na1d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
  }
  if (natten_mlx::nanobind_metal_runtime::debug_forced_fused_failure()) {
    throw std::runtime_error("forced fused failure");
  }
  if (!natten_mlx::nanobind_metal_runtime::supports_1d_fused(kernel_size, stride, dilation)) {
    throw std::runtime_error("nanobind fused backward 1D unsupported configuration");
  }
  try {
    return fused_backward_1d(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
  } catch (const std::bad_cast&) {
    return natten_mlx::nanobind_backend::call_backend(
        "na1d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
  }
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
  if (!natten_mlx::nanobind_backend::use_native_runtime()) {
    if (natten_mlx::nanobind_metal_runtime::debug_forced_fused_failure()) {
      throw std::runtime_error("forced fused failure");
    }
    return natten_mlx::nanobind_backend::call_backend(
        "na2d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
  }
  if (natten_mlx::nanobind_metal_runtime::debug_forced_fused_failure()) {
    throw std::runtime_error("forced fused failure");
  }
  if (!natten_mlx::nanobind_metal_runtime::supports_2d_fused(kernel_size, stride, dilation)) {
    throw std::runtime_error("nanobind fused backward 2D unsupported configuration");
  }
  try {
    return fused_backward_2d(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
  } catch (const std::bad_cast&) {
    return natten_mlx::nanobind_backend::call_backend(
        "na2d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
  }
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
  if (!natten_mlx::nanobind_backend::use_native_runtime()) {
    if (natten_mlx::nanobind_metal_runtime::debug_forced_fused_failure()) {
      throw std::runtime_error("forced fused failure");
    }
    return natten_mlx::nanobind_backend::call_backend(
        "na3d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
  }
  if (natten_mlx::nanobind_metal_runtime::debug_forced_fused_failure()) {
    throw std::runtime_error("forced fused failure");
  }
  if (!natten_mlx::nanobind_metal_runtime::supports_3d_fused(kernel_size, stride, dilation)) {
    throw std::runtime_error("nanobind fused backward 3D unsupported configuration");
  }
  try {
    return fused_backward_3d(q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
  } catch (const std::bad_cast&) {
    return natten_mlx::nanobind_backend::call_backend(
        "na3d_backward", q, k, v, grad_out, kernel_size, stride, dilation, is_causal, scale);
  }
}

}  // namespace natten_mlx::nanobind_fused_backward
