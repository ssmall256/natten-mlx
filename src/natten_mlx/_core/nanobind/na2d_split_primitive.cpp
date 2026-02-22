#include "nanobind/na2d_split_primitive.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>

#include <dlfcn.h>

#include <mlx/allocator.h>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/stream.h>
#include <mlx/backend/metal/device.h>

namespace mx = mlx::core;

namespace natten_mlx {

namespace {

struct NA2DParamsV2 {
  int B;
  int H;
  int IH;
  int IW;
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

std::mutex& split_cache_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, MTL::ComputePipelineState*>& split_cache() {
  static std::unordered_map<std::string, MTL::ComputePipelineState*> cache;
  return cache;
}

std::string split_binary_dir() {
  static std::string dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&split_binary_dir), &info)) {
      return std::string(".");
    }
    std::string path(info.dli_fname);
    auto pos = path.find_last_of('/');
    return (pos == std::string::npos) ? std::string(".") : path.substr(0, pos);
  }();
  return dir;
}

MTL::ComputePipelineState* get_split_kernel(const std::string& name, int kernel_size) {
  std::string cache_key = name + "_k" + std::to_string(kernel_size);
  {
    std::lock_guard<std::mutex> lock(split_cache_mutex());
    auto it = split_cache().find(cache_key);
    if (it != split_cache().end()) return it->second;
  }
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* lib = dev.get_library("natten_nb", split_binary_dir());
  if (!lib) throw std::runtime_error("Failed to load natten_nb metallib");
  mx::metal::MTLFCList fc = {
      {&kernel_size, MTL::DataType::DataTypeInt, 0}};
  auto* k = dev.get_kernel(name, lib, cache_key, fc);
  if (!k) throw std::runtime_error("Failed to resolve split kernel: " + name);
  {
    std::lock_guard<std::mutex> lock(split_cache_mutex());
    split_cache()[cache_key] = k;
  }
  return k;
}

std::string dtype_suffix(mx::Dtype dtype) {
  if (dtype == mx::float32) return "fp32";
  if (dtype == mx::float16) return "fp16";
  if (dtype == mx::bfloat16) return "bf16";
  return "fp32";
}

int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

}  // namespace

// ---- NA2DSplitQK ----

void NA2DSplitQK::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& q = inputs[0];
  auto& k = inputs[1];

  int B = q.shape(0);
  int IH = q.shape(1);
  int IW = q.shape(2);
  int H = q.shape(3);
  int D = q.shape(4);
  int out_h = ceil_div(IH, stride_h_);
  int out_w = ceil_div(IW, stride_w_);
  int K2 = kernel_size_ * kernel_size_;

  bool vec4 = use_vec4_ && (D % 4 == 0);
  std::string suffix = dtype_suffix(q.dtype());
  std::string kname = vec4
      ? ("na2d_qk_v2_vec4_" + suffix)
      : ("na2d_qk_v2_" + suffix);

  NA2DParamsV2 params{
      B, H, IH, IW, D, kernel_size_,
      stride_h_, stride_w_, dilation_h_, dilation_w_,
      causal_h_ ? 1 : 0, causal_w_ ? 1 : 0, scale_};

  size_t out_bytes = static_cast<size_t>(B) * out_h * out_w * H * K2 *
      mx::size_of(q.dtype());
  outputs[0].set_data(mx::allocator::malloc(out_bytes));

  auto* kernel_fn = get_split_kernel(kname, kernel_size_);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto& enc = dev.get_command_encoder(stream().index);

  enc.set_compute_pipeline_state(kernel_fn);
  enc.set_input_array(q, 0);
  enc.set_input_array(k, 1);
  enc.set_output_array(outputs[0], 2);
  enc.set_bytes(params, 3);

  // 3D grid: (K², out_w, out_h * B * H)
  size_t gx = static_cast<size_t>(K2);
  size_t gy = static_cast<size_t>(out_w);
  size_t gz = static_cast<size_t>(out_h) * B * H;
  enc.dispatch_threads(
      MTL::Size(gx, gy, gz),
      MTL::Size(std::min<size_t>(K2, 64), 1, 1));
}

// ---- NA2DSplitAV ----

void NA2DSplitAV::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& attn = inputs[0];
  auto& v = inputs[1];

  // attn: [B, OH, OW, H, K²], v: [B, IH, IW, H, D]
  int B = v.shape(0);
  int IH = v.shape(1);
  int IW = v.shape(2);
  int H = v.shape(3);
  int D = v.shape(4);
  int out_h = attn.shape(1);
  int out_w = attn.shape(2);

  bool vec4 = use_vec4_ && (D % 4 == 0);
  std::string suffix = dtype_suffix(v.dtype());
  std::string kname = vec4
      ? ("na2d_av_v2_vec4_" + suffix)
      : ("na2d_av_v2_" + suffix);

  // AV kernel doesn't use SCALE but params struct includes it
  NA2DParamsV2 params{
      B, H, IH, IW, D, kernel_size_,
      stride_h_, stride_w_, dilation_h_, dilation_w_,
      causal_h_ ? 1 : 0, causal_w_ ? 1 : 0, 0.0f};

  size_t out_bytes = static_cast<size_t>(B) * out_h * out_w * H * D *
      mx::size_of(v.dtype());
  outputs[0].set_data(mx::allocator::malloc(out_bytes));

  auto* kernel_fn = get_split_kernel(kname, kernel_size_);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto& enc = dev.get_command_encoder(stream().index);

  enc.set_compute_pipeline_state(kernel_fn);
  enc.set_input_array(attn, 0);
  enc.set_input_array(v, 1);
  enc.set_output_array(outputs[0], 2);
  enc.set_bytes(params, 3);

  // 3D grid: (D or D/4, out_w, out_h * B * H)
  size_t gx = vec4 ? static_cast<size_t>(D / 4) : static_cast<size_t>(D);
  size_t gy = static_cast<size_t>(out_w);
  size_t gz = static_cast<size_t>(out_h) * B * H;
  size_t tgx = std::min<size_t>(gx, 64);
  enc.dispatch_threads(
      MTL::Size(gx, gy, gz),
      MTL::Size(tgx, 1, 1));
}

// ---- Factory functions ----

mx::array na2d_qk_forward_v2(
    const mx::array& q,
    const mx::array& k,
    int kernel_size,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    bool causal_h, bool causal_w,
    float scale,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  int D = q.shape(4);
  bool use_vec4 = (D % 4 == 0);

  auto qc = mx::contiguous(q, false, stream);
  auto kc = mx::contiguous(k, false, stream);

  int B = q.shape(0);
  int IH = q.shape(1);
  int IW = q.shape(2);
  int H = q.shape(3);
  int out_h = ceil_div(IH, stride_h);
  int out_w = ceil_div(IW, stride_w);
  int K2 = kernel_size * kernel_size;

  auto prim = std::make_shared<NA2DSplitQK>(
      stream, kernel_size, stride_h, stride_w,
      dilation_h, dilation_w, causal_h, causal_w,
      scale, use_vec4);

  mx::Shape out_shape = {B, out_h, out_w, H, K2};
  return mx::array(out_shape, q.dtype(), std::move(prim), {qc, kc});
}

mx::array na2d_av_forward_v2(
    const mx::array& attn,
    const mx::array& v,
    int kernel_size,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    bool causal_h, bool causal_w,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  int D = v.shape(4);
  bool use_vec4 = (D % 4 == 0);

  auto ac = mx::contiguous(attn, false, stream);
  auto vc = mx::contiguous(v, false, stream);

  int B = v.shape(0);
  int H = v.shape(3);
  int out_h = attn.shape(1);
  int out_w = attn.shape(2);

  auto prim = std::make_shared<NA2DSplitAV>(
      stream, kernel_size, stride_h, stride_w,
      dilation_h, dilation_w, causal_h, causal_w,
      use_vec4);

  mx::Shape out_shape = {B, out_h, out_w, H, D};
  return mx::array(out_shape, v.dtype(), std::move(prim), {ac, vc});
}

}  // namespace natten_mlx
