#include "metal_runtime.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/stream.h>
#include <mlx/backend/metal/device.h>

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace mx = mlx::core;
namespace {

struct NA1DParams {
  int B;
  int L;
  int H;
  int D;
  int K;
  int S;
  int DIL;
  int CAUSAL;
  float SCALE;
};

struct NA2DParams {
  int B;
  int IH;
  int IW;
  int H;
  int D;
  int K;
  int SH;
  int SW;
  int DH;
  int DW;
  int CH;
  int CW;
  float SCALE;
};

struct NA3DParams {
  int B;
  int ID;
  int IH;
  int IW;
  int H;
  int D;
  int K;
  int SD;
  int SH;
  int SW;
  int DD;
  int DH;
  int DW;
  int CD;
  int CH;
  int CW;
  float SCALE;
};

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

int first_kernel_size(const nb::object& kernel_size) {
  return scalar_or_index_int(kernel_size, 0);
}

bool valid_kernel(int k) {
  return k > 0 && (k % 2 == 1);
}

bool valid_stride_1d(const nb::object& stride) {
  return scalar_or_index_int(stride, 0) >= 1;
}

bool valid_dilation_1d(const nb::object& dilation) {
  return scalar_or_index_int(dilation, 0) >= 1;
}

bool valid_stride_2d(const nb::object& stride) {
  return scalar_or_index_int(stride, 0) >= 1 && scalar_or_index_int(stride, 1) >= 1;
}

bool valid_dilation_2d(const nb::object& dilation) {
  return scalar_or_index_int(dilation, 0) >= 1 && scalar_or_index_int(dilation, 1) >= 1;
}

bool valid_stride_3d(const nb::object& stride) {
  return scalar_or_index_int(stride, 0) >= 1 && scalar_or_index_int(stride, 1) >= 1 &&
      scalar_or_index_int(stride, 2) >= 1;
}

bool valid_dilation_3d(const nb::object& dilation) {
  return scalar_or_index_int(dilation, 0) >= 1 && scalar_or_index_int(dilation, 1) >= 1 &&
      scalar_or_index_int(dilation, 2) >= 1;
}

bool square_kernel_2d(const nb::object& kernel_size) {
  return scalar_or_index_int(kernel_size, 0) == scalar_or_index_int(kernel_size, 1);
}

bool cubic_kernel_3d(const nb::object& kernel_size) {
  int k0 = scalar_or_index_int(kernel_size, 0);
  return k0 == scalar_or_index_int(kernel_size, 1) && k0 == scalar_or_index_int(kernel_size, 2);
}

int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

mx::Shape to_shape(const std::vector<int>& dims) {
  mx::Shape s;
  s.reserve(dims.size());
  for (int d : dims) {
    s.push_back(static_cast<mx::ShapeElem>(d));
  }
  return s;
}

size_t numel(const mx::Shape& shape) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(std::max<int>(d, 1));
  }
  return n;
}

mx::array to_float32(const mx::array& x) {
  if (x.dtype() == mx::float32) {
    return x;
  }
  return mx::astype(x, mx::float32);
}

mx::array cast_to_dtype(const mx::array& x, mx::Dtype dtype) {
  if (x.dtype() == dtype) {
    return x;
  }
  return mx::astype(x, dtype);
}

mx::array as_array(const nb::object& obj) {
  return nb::cast<mx::array>(obj);
}

float resolve_scale(const nb::object& scale, int head_dim) {
  if (scale.is_none()) {
    return std::pow(static_cast<float>(head_dim), -0.5f);
  }
  return nb::cast<float>(scale);
}

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to resolve current binary path");
    }
    std::string path(info.dli_fname);
    auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
      return std::string(".");
    }
    return path.substr(0, pos);
  }();
  return binary_dir;
}

std::mutex& route_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, std::string>& route_map() {
  static std::unordered_map<std::string, std::string> m;
  return m;
}

bool& force_fused_failure_flag() {
  static bool v = false;
  return v;
}

bool& force_split_failure_flag() {
  static bool v = false;
  return v;
}

int& python_bridge_calls() {
  static int v = 0;
  return v;
}

void throw_if_forced_split_failure() {
  if (natten_mlx::nanobind_metal_runtime::debug_forced_split_failure()) {
    throw std::runtime_error("forced split failure");
  }
}

void throw_if_forced_fused_failure() {
  if (natten_mlx::nanobind_metal_runtime::debug_forced_fused_failure()) {
    throw std::runtime_error("forced fused failure");
  }
}

MTL::ComputePipelineState* get_kernel(const std::string& name) {
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* lib = dev.get_library("natten_nb", current_binary_dir());
  if (lib == nullptr) {
    throw std::runtime_error("failed to load natten_nb metallib");
  }
  auto* kernel = dev.get_kernel(name, lib);
  if (kernel == nullptr) {
    throw std::runtime_error("failed to resolve kernel: " + name);
  }
  return kernel;
}

template <typename Params>
mx::array launch_one(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const Params& params,
    const mx::Shape& out_shape,
    bool zero_init) {
  auto stream = mx::default_stream(mx::Device::gpu);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* kernel = get_kernel(kernel_name);

  mx::array out = mx::zeros(out_shape, mx::float32, stream);
  if (!zero_init) {
    out = mx::zeros(out_shape, mx::float32, stream);
  }

  auto& enc = dev.get_command_encoder(stream.index);
  enc.set_compute_pipeline_state(kernel);
  int arg = 0;
  for (const auto& in : inputs) {
    enc.set_input_array(in, arg++);
  }
  enc.set_output_array(out, arg++);
  enc.set_bytes(params, arg++);

  size_t threads = std::max<size_t>(1, numel(out_shape));
  size_t tg = std::min<size_t>(256, threads);
  enc.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(tg, 1, 1));
  dev.end_encoding(stream.index);
  return out;
}

template <typename Params>
nb::tuple launch_two(
    const std::string& kernel_name,
    const std::vector<mx::array>& inputs,
    const Params& params,
    const mx::Shape& out_shape0,
    const mx::Shape& out_shape1,
    bool zero_init) {
  auto stream = mx::default_stream(mx::Device::gpu);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* kernel = get_kernel(kernel_name);

  mx::array out0 = mx::zeros(out_shape0, mx::float32, stream);
  mx::array out1 = mx::zeros(out_shape1, mx::float32, stream);
  if (!zero_init) {
    out0 = mx::zeros(out_shape0, mx::float32, stream);
    out1 = mx::zeros(out_shape1, mx::float32, stream);
  }

  auto& enc = dev.get_command_encoder(stream.index);
  enc.set_compute_pipeline_state(kernel);
  int arg = 0;
  for (const auto& in : inputs) {
    enc.set_input_array(in, arg++);
  }
  enc.set_output_array(out0, arg++);
  enc.set_output_array(out1, arg++);
  enc.set_bytes(params, arg++);

  size_t threads = std::max(numel(out_shape0), numel(out_shape1));
  threads = std::max<size_t>(1, threads);
  size_t tg = std::min<size_t>(256, threads);
  enc.dispatch_threads(MTL::Size(threads, 1, 1), MTL::Size(tg, 1, 1));
  dev.end_encoding(stream.index);
  return nb::make_tuple(out0, out1);
}

}  // namespace

namespace natten_mlx::nanobind_metal_runtime {

bool supports_1d_fused(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation) {
  return valid_kernel(first_kernel_size(kernel_size)) && valid_stride_1d(stride) &&
      valid_dilation_1d(dilation);
}

bool supports_2d_fused(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation) {
  return square_kernel_2d(kernel_size) && valid_kernel(first_kernel_size(kernel_size)) &&
      valid_stride_2d(stride) && valid_dilation_2d(dilation);
}

bool supports_3d_fused(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation) {
  return cubic_kernel_3d(kernel_size) && valid_kernel(first_kernel_size(kernel_size)) &&
      valid_stride_3d(stride) && valid_dilation_3d(dilation);
}

bool supports_1d_split(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object&) {
  return supports_1d_fused(kernel_size, stride, dilation);
}

bool supports_2d_split(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object&) {
  return supports_2d_fused(kernel_size, stride, dilation);
}

bool supports_3d_split(
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object&) {
  return supports_3d_fused(kernel_size, stride, dilation);
}

nb::object na1d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_split_failure();
  if (!supports_1d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 1D QK unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);

  int B = qf.shape(0);
  int L = qf.shape(1);
  int H = qf.shape(2);
  int D = qf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = ceil_div(L, S);

  NA1DParams p{B, L, H, D, K, S, Dil, C, resolve_scale(scale, D)};
  auto out = launch_one("na1d_qk_fp32", {qf, kf}, p, to_shape({B, out_l, H, K}), false);
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na1d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_1d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 1D AV unsupported configuration");
  }

  auto attn_arr = as_array(attn);
  auto v_arr = as_array(v);
  auto af = to_float32(attn_arr);
  auto vf = to_float32(v_arr);

  int B = vf.shape(0);
  int L = vf.shape(1);
  int H = vf.shape(2);
  int D = vf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = af.shape(1);

  NA1DParams p{B, L, H, D, K, S, Dil, C, 1.0f};
  auto out = launch_one("na1d_av_fp32", {af, vf}, p, to_shape({B, out_l, H, D}), false);
  return nb::cast(cast_to_dtype(out, v_arr.dtype()));
}

nb::object na2d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_split_failure();
  if (!supports_2d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 2D QK unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);

  int B = qf.shape(0);
  int IH = qf.shape(1);
  int IW = qf.shape(2);
  int H = qf.shape(3);
  int D = qf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, resolve_scale(scale, D)};
  auto out = launch_one(
      "na2d_qk_fp32", {qf, kf}, p, to_shape({B, out_h, out_w, H, K * K}), false);
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na2d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_2d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 2D AV unsupported configuration");
  }

  auto attn_arr = as_array(attn);
  auto v_arr = as_array(v);
  auto af = to_float32(attn_arr);
  auto vf = to_float32(v_arr);

  int B = vf.shape(0);
  int IH = vf.shape(1);
  int IW = vf.shape(2);
  int H = vf.shape(3);
  int D = vf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;

  int out_h = af.shape(1);
  int out_w = af.shape(2);

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, 1.0f};
  auto out = launch_one("na2d_av_fp32", {af, vf}, p, to_shape({B, out_h, out_w, H, D}), false);
  return nb::cast(cast_to_dtype(out, v_arr.dtype()));
}

nb::object na3d_qk_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_split_failure();
  if (!supports_3d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 3D QK unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);

  int B = qf.shape(0);
  int ID = qf.shape(1);
  int IH = qf.shape(2);
  int IW = qf.shape(3);
  int H = qf.shape(4);
  int D = qf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;
  int out_d = ceil_div(ID, SD);
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);

  NA3DParams p{
      B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, resolve_scale(scale, D)};
  auto out = launch_one(
      "na3d_qk_fp32", {qf, kf}, p, to_shape({B, out_d, out_h, out_w, H, K * K * K}), false);
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na3d_av_forward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_3d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 3D AV unsupported configuration");
  }

  auto attn_arr = as_array(attn);
  auto v_arr = as_array(v);
  auto af = to_float32(attn_arr);
  auto vf = to_float32(v_arr);

  int B = vf.shape(0);
  int ID = vf.shape(1);
  int IH = vf.shape(2);
  int IW = vf.shape(3);
  int H = vf.shape(4);
  int D = vf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;

  int out_d = af.shape(1);
  int out_h = af.shape(2);
  int out_w = af.shape(3);

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, 1.0f};
  auto out =
      launch_one("na3d_av_fp32", {af, vf}, p, to_shape({B, out_d, out_h, out_w, H, D}), false);
  return nb::cast(cast_to_dtype(out, v_arr.dtype()));
}

nb::object na1d_qk_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_split_failure();
  if (!supports_1d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 1D QK backward unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto g_arr = as_array(grad_attn);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto gf = to_float32(g_arr);

  int B = qf.shape(0);
  int L = qf.shape(1);
  int H = qf.shape(2);
  int D = qf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;

  NA1DParams p{B, L, H, D, K, S, Dil, C, resolve_scale(scale, D)};
  auto grad_q = launch_one("na1d_qk_bwd_q_fp32", {gf, kf}, p, qf.shape(), false);
  auto grad_k = launch_one("na1d_qk_bwd_k_accum_fp32", {gf, qf}, p, kf.shape(), true);
  return nb::make_tuple(cast_to_dtype(grad_q, q_arr.dtype()), cast_to_dtype(grad_k, k_arr.dtype()));
}

nb::object na1d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_1d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 1D AV backward unsupported configuration");
  }

  auto a_arr = as_array(attn);
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto af = to_float32(a_arr);
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = vf.shape(0);
  int L = vf.shape(1);
  int H = vf.shape(2);
  int D = vf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;

  NA1DParams p{B, L, H, D, K, S, Dil, C, 1.0f};
  auto grad_attn = launch_one("na1d_av_bwd_attn_fp32", {gof, vf}, p, af.shape(), false);
  auto grad_v = launch_one("na1d_av_bwd_v_accum_fp32", {af, gof}, p, vf.shape(), true);
  return nb::make_tuple(cast_to_dtype(grad_attn, a_arr.dtype()), cast_to_dtype(grad_v, v_arr.dtype()));
}

nb::object na2d_qk_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_split_failure();
  if (!supports_2d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 2D QK backward unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto g_arr = as_array(grad_attn);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto gf = to_float32(g_arr);

  int B = qf.shape(0);
  int IH = qf.shape(1);
  int IW = qf.shape(2);
  int H = qf.shape(3);
  int D = qf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, resolve_scale(scale, D)};
  auto grad_q = launch_one("na2d_qk_bwd_q_fp32", {gf, kf}, p, qf.shape(), false);
  auto grad_k = launch_one("na2d_qk_bwd_k_accum_fp32", {gf, qf}, p, kf.shape(), true);
  return nb::make_tuple(cast_to_dtype(grad_q, q_arr.dtype()), cast_to_dtype(grad_k, k_arr.dtype()));
}

nb::object na2d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_2d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 2D AV backward unsupported configuration");
  }

  auto a_arr = as_array(attn);
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto af = to_float32(a_arr);
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = vf.shape(0);
  int IH = vf.shape(1);
  int IW = vf.shape(2);
  int H = vf.shape(3);
  int D = vf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, 1.0f};
  auto grad_attn = launch_one("na2d_av_bwd_attn_fp32", {gof, vf}, p, af.shape(), false);
  auto grad_v = launch_one("na2d_av_bwd_v_accum_fp32", {af, gof}, p, vf.shape(), true);
  return nb::make_tuple(cast_to_dtype(grad_attn, a_arr.dtype()), cast_to_dtype(grad_v, v_arr.dtype()));
}

nb::object na3d_qk_backward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_attn,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_split_failure();
  if (!supports_3d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 3D QK backward unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto g_arr = as_array(grad_attn);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto gf = to_float32(g_arr);

  int B = qf.shape(0);
  int ID = qf.shape(1);
  int IH = qf.shape(2);
  int IW = qf.shape(3);
  int H = qf.shape(4);
  int D = qf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, resolve_scale(scale, D)};
  auto grad_q = launch_one("na3d_qk_bwd_q_fp32", {gf, kf}, p, qf.shape(), false);
  auto grad_k = launch_one("na3d_qk_bwd_k_accum_fp32", {gf, qf}, p, kf.shape(), true);
  return nb::make_tuple(cast_to_dtype(grad_q, q_arr.dtype()), cast_to_dtype(grad_k, k_arr.dtype()));
}

nb::object na3d_av_backward(
    const nb::object& attn,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  throw_if_forced_split_failure();
  if (!supports_3d_split(kernel_size, stride, dilation, is_causal)) {
    throw std::runtime_error("nanobind split 3D AV backward unsupported configuration");
  }

  auto a_arr = as_array(attn);
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto af = to_float32(a_arr);
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = vf.shape(0);
  int ID = vf.shape(1);
  int IH = vf.shape(2);
  int IW = vf.shape(3);
  int H = vf.shape(4);
  int D = vf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, 1.0f};
  auto grad_attn = launch_one("na3d_av_bwd_attn_fp32", {gof, vf}, p, af.shape(), false);
  auto grad_v = launch_one("na3d_av_bwd_v_accum_fp32", {af, gof}, p, vf.shape(), true);
  return nb::make_tuple(cast_to_dtype(grad_attn, a_arr.dtype()), cast_to_dtype(grad_v, v_arr.dtype()));
}

nb::object na1d_fused_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_fused_failure();
  if (!supports_1d_fused(kernel_size, stride, dilation)) {
    throw std::runtime_error("nanobind fused 1D unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto vf = to_float32(v_arr);

  int B = qf.shape(0);
  int L = qf.shape(1);
  int H = qf.shape(2);
  int D = qf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = ceil_div(L, S);

  NA1DParams p{B, L, H, D, K, S, Dil, C, resolve_scale(scale, D)};
  std::string kname = (D % 4 == 0) ? "na1d_fused_vec4_fp32" : "na1d_fused_fp32";
  auto out = launch_one(kname, {qf, kf, vf}, p, to_shape({B, out_l, H, D}), false);
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na2d_fused_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_fused_failure();
  if (!supports_2d_fused(kernel_size, stride, dilation)) {
    throw std::runtime_error("nanobind fused 2D unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto vf = to_float32(v_arr);

  int B = qf.shape(0);
  int IH = qf.shape(1);
  int IW = qf.shape(2);
  int H = qf.shape(3);
  int D = qf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, resolve_scale(scale, D)};
  std::string kname = (D % 4 == 0) ? "na2d_fused_vec4_fp32" : "na2d_fused_fp32";
  auto out = launch_one(kname, {qf, kf, vf}, p, to_shape({B, out_h, out_w, H, D}), false);
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na3d_fused_forward(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  throw_if_forced_fused_failure();
  if (!supports_3d_fused(kernel_size, stride, dilation)) {
    throw std::runtime_error("nanobind fused 3D unsupported configuration");
  }

  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto vf = to_float32(v_arr);

  int B = qf.shape(0);
  int ID = qf.shape(1);
  int IH = qf.shape(2);
  int IW = qf.shape(3);
  int H = qf.shape(4);
  int D = qf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;
  int out_d = ceil_div(ID, SD);
  int out_h = ceil_div(IH, SH);
  int out_w = ceil_div(IW, SW);

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, resolve_scale(scale, D)};
  std::string kname = (D % 4 == 0) ? "na3d_fused_vec4_fp32" : "na3d_fused_fp32";
  auto out = launch_one(kname, {qf, kf, vf}, p, to_shape({B, out_d, out_h, out_w, H, D}), false);
  return nb::cast(cast_to_dtype(out, q_arr.dtype()));
}

nb::object na1d_fused_backward_attn(
    const nb::object& q,
    const nb::object& k,
    const nb::object& v,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto v_arr = as_array(v);
  auto go_arr = as_array(grad_out);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto vf = to_float32(v_arr);
  auto gof = to_float32(go_arr);

  int B = qf.shape(0);
  int L = qf.shape(1);
  int H = qf.shape(2);
  int D = qf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int out_l = ceil_div(L, S);

  NA1DParams p{B, L, H, D, K, S, Dil, C, resolve_scale(scale, D)};
  return launch_two(
      "na1d_fused_bwd_attn_fp32",
      {qf, kf, vf, gof},
      p,
      to_shape({B, out_l, H, K}),
      to_shape({B, out_l, H, K}),
      false);
}

nb::object na1d_fused_backward_qk(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_logits,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object& scale) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto gl_arr = as_array(grad_logits);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto glf = to_float32(gl_arr);

  int B = qf.shape(0);
  int L = qf.shape(1);
  int H = qf.shape(2);
  int D = qf.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;

  NA1DParams p{B, L, H, D, K, S, Dil, C, 1.0f};
  auto out = launch_two("na1d_fused_bwd_qk_fp32", {glf, qf, kf}, p, qf.shape(), kf.shape(), true);
  auto gq = nb::cast<mx::array>(out[0]);
  auto gk = nb::cast<mx::array>(out[1]);
  return nb::make_tuple(cast_to_dtype(gq, q_arr.dtype()), cast_to_dtype(gk, k_arr.dtype()));
}

nb::object na1d_fused_backward_v(
    const nb::object& attn,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  auto a_arr = as_array(attn);
  auto go_arr = as_array(grad_out);
  auto af = to_float32(a_arr);
  auto gof = to_float32(go_arr);

  int B = gof.shape(0);
  int out_l = gof.shape(1);
  int H = gof.shape(2);
  int D = gof.shape(3);
  int K = scalar_or_index_int(kernel_size, 0);
  int S = scalar_or_index_int(stride, 0);
  int Dil = scalar_or_index_int(dilation, 0);
  int C = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int L = out_l * S;

  NA1DParams p{B, L, H, D, K, S, Dil, C, 1.0f};
  auto gv = launch_one("na1d_fused_bwd_v_fp32", {af, gof}, p, to_shape({B, L, H, D}), true);
  return nb::cast(cast_to_dtype(gv, go_arr.dtype()));
}

nb::object na2d_fused_backward_qk(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_logits,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object&) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto gl_arr = as_array(grad_logits);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto glf = to_float32(gl_arr);

  int B = qf.shape(0);
  int IH = qf.shape(1);
  int IW = qf.shape(2);
  int H = qf.shape(3);
  int D = qf.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, 1.0f};
  auto out = launch_two("na2d_fused_bwd_qk_fp32", {glf, qf, kf}, p, qf.shape(), kf.shape(), true);
  auto gq = nb::cast<mx::array>(out[0]);
  auto gk = nb::cast<mx::array>(out[1]);
  return nb::make_tuple(cast_to_dtype(gq, q_arr.dtype()), cast_to_dtype(gk, k_arr.dtype()));
}

nb::object na2d_fused_backward_v(
    const nb::object& attn,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  auto go_arr = as_array(grad_out);
  auto af = to_float32(as_array(attn));
  auto gof = to_float32(go_arr);

  int B = gof.shape(0);
  int out_h = gof.shape(1);
  int out_w = gof.shape(2);
  int H = gof.shape(3);
  int D = gof.shape(4);
  int K = scalar_or_index_int(kernel_size, 0);
  int SH = scalar_or_index_int(stride, 0);
  int SW = scalar_or_index_int(stride, 1);
  int DH = scalar_or_index_int(dilation, 0);
  int DW = scalar_or_index_int(dilation, 1);
  int CH = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int IH = out_h * SH;
  int IW = out_w * SW;

  NA2DParams p{B, IH, IW, H, D, K, SH, SW, DH, DW, CH, CW, 1.0f};
  auto gv = launch_one("na2d_fused_bwd_v_fp32", {af, gof}, p, to_shape({B, IH, IW, H, D}), true);
  return nb::cast(cast_to_dtype(gv, go_arr.dtype()));
}

nb::object na3d_fused_backward_qk(
    const nb::object& q,
    const nb::object& k,
    const nb::object& grad_logits,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal,
    const nb::object&) {
  auto q_arr = as_array(q);
  auto k_arr = as_array(k);
  auto gl_arr = as_array(grad_logits);
  auto qf = to_float32(q_arr);
  auto kf = to_float32(k_arr);
  auto glf = to_float32(gl_arr);

  int B = qf.shape(0);
  int ID = qf.shape(1);
  int IH = qf.shape(2);
  int IW = qf.shape(3);
  int H = qf.shape(4);
  int D = qf.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, 1.0f};
  auto out = launch_two("na3d_fused_bwd_qk_fp32", {glf, qf, kf}, p, qf.shape(), kf.shape(), true);
  auto gq = nb::cast<mx::array>(out[0]);
  auto gk = nb::cast<mx::array>(out[1]);
  return nb::make_tuple(cast_to_dtype(gq, q_arr.dtype()), cast_to_dtype(gk, k_arr.dtype()));
}

nb::object na3d_fused_backward_v(
    const nb::object& attn,
    const nb::object& grad_out,
    const nb::object& kernel_size,
    const nb::object& stride,
    const nb::object& dilation,
    const nb::object& is_causal) {
  auto go_arr = as_array(grad_out);
  auto af = to_float32(as_array(attn));
  auto gof = to_float32(go_arr);

  int B = gof.shape(0);
  int out_d = gof.shape(1);
  int out_h = gof.shape(2);
  int out_w = gof.shape(3);
  int H = gof.shape(4);
  int D = gof.shape(5);
  int K = scalar_or_index_int(kernel_size, 0);
  int SD = scalar_or_index_int(stride, 0);
  int SH = scalar_or_index_int(stride, 1);
  int SW = scalar_or_index_int(stride, 2);
  int DD = scalar_or_index_int(dilation, 0);
  int DH = scalar_or_index_int(dilation, 1);
  int DW = scalar_or_index_int(dilation, 2);
  int CD = scalar_or_index_bool(is_causal, 0) ? 1 : 0;
  int CH = scalar_or_index_bool(is_causal, 1) ? 1 : 0;
  int CW = scalar_or_index_bool(is_causal, 2) ? 1 : 0;
  int ID = out_d * SD;
  int IH = out_h * SH;
  int IW = out_w * SW;

  NA3DParams p{B, ID, IH, IW, H, D, K, SD, SH, SW, DD, DH, DW, CD, CH, CW, 1.0f};
  auto gv = launch_one("na3d_fused_bwd_v_fp32", {af, gof}, p, to_shape({B, ID, IH, IW, H, D}), true);
  return nb::cast(cast_to_dtype(gv, go_arr.dtype()));
}

void debug_set_last_route(const std::string& op, const std::string& route) {
  std::lock_guard<std::mutex> lock(route_mutex());
  route_map()[op] = route;
}

std::string debug_get_last_route(const std::string& op) {
  std::lock_guard<std::mutex> lock(route_mutex());
  auto it = route_map().find(op);
  return (it == route_map().end()) ? std::string() : it->second;
}

void debug_clear_last_routes() {
  std::lock_guard<std::mutex> lock(route_mutex());
  route_map().clear();
}

void debug_force_fused_failure(bool enabled) {
  force_fused_failure_flag() = enabled;
}

void debug_force_split_failure(bool enabled) {
  force_split_failure_flag() = enabled;
}

bool debug_forced_fused_failure() {
  return force_fused_failure_flag();
}

bool debug_forced_split_failure() {
  return force_split_failure_flag();
}

void debug_inc_python_bridge_calls() {
  python_bridge_calls() += 1;
}

int debug_get_python_bridge_calls() {
  return python_bridge_calls();
}

void debug_clear_python_bridge_calls() {
  python_bridge_calls() = 0;
}

}  // namespace natten_mlx::nanobind_metal_runtime
