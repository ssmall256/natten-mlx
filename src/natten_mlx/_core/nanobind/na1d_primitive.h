#pragma once

#include <memory>
#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/primitives.h>
#include <mlx/stream.h>
#include <mlx/utils.h>

namespace mx = mlx::core;

namespace natten_mlx {

class NA1DFusedForward : public mx::Primitive {
 public:
  NA1DFusedForward(
      mx::Stream stream,
      int kernel_size,
      int stride,
      int dilation,
      int causal,
      float scale,
      bool use_vec4)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_(stride),
        dilation_(dilation),
        causal_(causal),
        scale_(scale),
        use_vec4_(use_vec4) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NA1DFusedForward: CPU not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  DEFINE_NAME(NA1DFusedForward)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA1DFusedForward*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ && stride_ == o->stride_ &&
        dilation_ == o->dilation_ && causal_ == o->causal_ &&
        scale_ == o->scale_ && use_vec4_ == o->use_vec4_;
  }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    auto& q = inputs[0];
    int B = q.shape(0);
    int L = q.shape(1);
    int H = q.shape(2);
    int D = q.shape(3);
    int out_len = (L + stride_ - 1) / stride_;
    return {{B, out_len, H, D}};
  }

  auto state() const {
    return std::make_tuple(
        nullptr, kernel_size_, stride_, dilation_, causal_, scale_, use_vec4_);
  }

 private:
  int kernel_size_;
  int stride_;
  int dilation_;
  int causal_;
  float scale_;
  bool use_vec4_;
};

mx::array na1d_fused_forward_v2(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    int kernel_size,
    int stride,
    int dilation,
    bool causal,
    float scale,
    mx::StreamOrDevice s = {});

}  // namespace natten_mlx
