#include "na_fused_backward.h"

#include <typeinfo>

#include <nanobind/stl/tuple.h>

#include "metal_runtime.h"
#include "py_dispatch.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

nb::object softmax_last_dim(const nb::object& x) {
  return natten_mlx::nanobind_backend::mx_module().attr("softmax")(x, "axis"_a = -1);
}

nb::object grad_logits_from_softmax(
    const nb::object& attn,
    const nb::object& grad_attn) {
  nb::object mx = natten_mlx::nanobind_backend::mx_module();
  nb::object prod = mx.attr("multiply")(grad_attn, attn);
  nb::object inner = mx.attr("sum")(prod, "axis"_a = -1, "keepdims"_a = true);
  nb::object centered = mx.attr("subtract")(grad_attn, inner);
  return mx.attr("multiply")(attn, centered);
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
  nb::object logits = natten_mlx::nanobind_metal_runtime::na2d_qk_forward(
      q, k, kernel_size, stride, dilation, is_causal, scale);
  nb::object attn = softmax_last_dim(logits);
  nb::tuple av = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na2d_av_backward(
      attn,
      v,
      grad_out,
      kernel_size,
      stride,
      dilation,
      is_causal));
  nb::object grad_attn = av[0];

  nb::object grad_logits = grad_logits_from_softmax(attn, grad_attn);
  nb::tuple qk = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na2d_fused_backward_qk(
      q,
      k,
      grad_logits,
      kernel_size,
      stride,
      dilation,
      is_causal,
      scale));
  nb::object grad_v = natten_mlx::nanobind_metal_runtime::na2d_fused_backward_v(
      attn,
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
  nb::object logits = natten_mlx::nanobind_metal_runtime::na3d_qk_forward(
      q, k, kernel_size, stride, dilation, is_causal, scale);
  nb::object attn = softmax_last_dim(logits);
  nb::tuple av = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na3d_av_backward(
      attn,
      v,
      grad_out,
      kernel_size,
      stride,
      dilation,
      is_causal));
  nb::object grad_attn = av[0];

  nb::object grad_logits = grad_logits_from_softmax(attn, grad_attn);
  nb::tuple qk = nb::cast<nb::tuple>(natten_mlx::nanobind_metal_runtime::na3d_fused_backward_qk(
      q,
      k,
      grad_logits,
      kernel_size,
      stride,
      dilation,
      is_causal,
      scale));
  nb::object grad_v = natten_mlx::nanobind_metal_runtime::na3d_fused_backward_v(
      attn,
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
