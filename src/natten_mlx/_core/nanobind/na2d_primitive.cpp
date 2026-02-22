#include "nanobind/na2d_primitive.h"

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

// Param struct must exactly match the Metal side.
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

std::mutex& v2_kernel_cache_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<std::string, MTL::ComputePipelineState*>& v2_kernel_cache() {
  static std::unordered_map<std::string, MTL::ComputePipelineState*> cache;
  return cache;
}

std::string v2_binary_dir() {
  static std::string dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&v2_binary_dir), &info)) {
      return std::string(".");
    }
    std::string path(info.dli_fname);
    auto pos = path.find_last_of('/');
    return (pos == std::string::npos) ? std::string(".") : path.substr(0, pos);
  }();
  return dir;
}

MTL::ComputePipelineState* get_v2_kernel(const std::string& name, int kernel_size) {
  std::string cache_key = name + "_k" + std::to_string(kernel_size);
  {
    std::lock_guard<std::mutex> lock(v2_kernel_cache_mutex());
    auto it = v2_kernel_cache().find(cache_key);
    if (it != v2_kernel_cache().end()) {
      return it->second;
    }
  }
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto* lib = dev.get_library("natten_nb", v2_binary_dir());
  if (!lib) {
    throw std::runtime_error("Failed to load natten_nb metallib for v2 kernels");
  }
  mx::metal::MTLFCList fc = {
      {&kernel_size, MTL::DataType::DataTypeInt, 0}};
  auto* kernel = dev.get_kernel(name, lib, cache_key, fc);
  if (!kernel) {
    throw std::runtime_error("Failed to resolve v2 kernel: " + name);
  }
  {
    std::lock_guard<std::mutex> lock(v2_kernel_cache_mutex());
    v2_kernel_cache()[cache_key] = kernel;
  }
  return kernel;
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

void NA2DFusedForward::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs: [q, k, v] in spatial-first layout [B, IH, IW, H, D]
  // contiguous (ensured by factory function)
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];

  int B = q.shape(0);
  int IH = q.shape(1);
  int IW = q.shape(2);
  int H = q.shape(3);
  int D = q.shape(4);
  int out_h = ceil_div(IH, stride_h_);
  int out_w = ceil_div(IW, stride_w_);

  // Determine dtype and kernel variant
  mx::Dtype compute_dtype = q.dtype();

  bool vec4 = use_vec4_ && (D % 4 == 0);
  std::string suffix = dtype_suffix(compute_dtype);
  std::string kname = vec4
      ? ("na2d_fused_v2_stored_vec4_" + suffix)
      : ("na2d_fused_v2_stored_" + suffix);

  // Build params
  NA2DParamsV2 params{
      B, H, IH, IW, D, kernel_size_,
      stride_h_, stride_w_, dilation_h_, dilation_w_,
      causal_h_, causal_w_, scale_};

  // Allocate output: [B, out_h, out_w, H, D] (spatial-first)
  size_t out_bytes = static_cast<size_t>(B) * out_h * out_w * H * D *
      mx::size_of(compute_dtype);
  outputs[0].set_data(mx::allocator::malloc(out_bytes));

  // Get kernel and dispatch
  auto* kernel_fn = get_v2_kernel(kname, kernel_size_);
  auto& dev = mx::metal::device(mx::Device::gpu);
  auto& enc = dev.get_command_encoder(stream().index);

  enc.set_compute_pipeline_state(kernel_fn);
  enc.set_input_array(q, 0);
  enc.set_input_array(k, 1);
  enc.set_input_array(v, 2);
  enc.set_output_array(outputs[0], 3);
  enc.set_bytes(params, 4);

  // 3D grid: (out_w, out_h, B*H) with tuned threadgroup
  size_t gx = static_cast<size_t>(out_w);
  size_t gy = static_cast<size_t>(out_h);
  size_t gz = static_cast<size_t>(B * H);
  size_t tgx = std::min<size_t>(16, gx);
  size_t tgy = std::min<size_t>(8, gy);
  size_t tgz = 1;

  enc.dispatch_threads(
      MTL::Size(gx, gy, gz),
      MTL::Size(tgx, tgy, tgz));
}

mx::array na2d_fused_forward_v2(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    int kernel_size,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    bool causal_h,
    bool causal_w,
    float scale,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);

  int D = q.shape(q.ndim() - 1);
  bool use_vec4 = (D % 4 == 0);

  // Input is spatial-first: [B, IH, IW, H, D]
  int B = q.shape(0);
  int IH = q.shape(1);
  int IW = q.shape(2);
  int H = q.shape(3);
  int out_h = (IH + stride_h - 1) / stride_h;
  int out_w = (IW + stride_w - 1) / stride_w;

  // Determine compute dtype
  mx::Dtype compute_dtype = q.dtype();
  if (compute_dtype != mx::float32 && compute_dtype != mx::float16 &&
      compute_dtype != mx::bfloat16) {
    compute_dtype = mx::float32;
  }

  // Ensure contiguous (no transpose needed — kernel uses spatial-first layout)
  auto qc = mx::contiguous(q, false, stream);
  auto kc = mx::contiguous(k, false, stream);
  auto vc = mx::contiguous(v, false, stream);

  // Cast if needed
  if (qc.dtype() != compute_dtype) {
    qc = mx::astype(qc, compute_dtype, stream);
    kc = mx::astype(kc, compute_dtype, stream);
    vc = mx::astype(vc, compute_dtype, stream);
  }

  auto prim = std::make_shared<NA2DFusedForward>(
      stream,
      kernel_size,
      stride_h,
      stride_w,
      dilation_h,
      dilation_w,
      causal_h ? 1 : 0,
      causal_w ? 1 : 0,
      scale,
      use_vec4);

  // Output in spatial-first: [B, out_h, out_w, H, D]
  mx::Shape out_shape = {B, out_h, out_w, H, D};
  auto out = mx::array(out_shape, compute_dtype, std::move(prim), {qc, kc, vc});

  // Cast back if needed
  if (out.dtype() != q.dtype()) {
    out = mx::astype(out, q.dtype(), stream);
  }

  return out;
}

}  // namespace natten_mlx
