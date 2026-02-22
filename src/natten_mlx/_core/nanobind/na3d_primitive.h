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

class NA3DFusedForward : public mx::Primitive {
 public:
  NA3DFusedForward(
      mx::Stream stream,
      int kernel_size,
      int stride_d, int stride_h, int stride_w,
      int dilation_d, int dilation_h, int dilation_w,
      int causal_d, int causal_h, int causal_w,
      float scale,
      bool use_vec4)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
        dilation_d_(dilation_d), dilation_h_(dilation_h), dilation_w_(dilation_w),
        causal_d_(causal_d), causal_h_(causal_h), causal_w_(causal_w),
        scale_(scale),
        use_vec4_(use_vec4) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NA3DFusedForward: CPU not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  DEFINE_NAME(NA3DFusedForward)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA3DFusedForward*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ &&
        stride_d_ == o->stride_d_ && stride_h_ == o->stride_h_ && stride_w_ == o->stride_w_ &&
        dilation_d_ == o->dilation_d_ && dilation_h_ == o->dilation_h_ && dilation_w_ == o->dilation_w_ &&
        causal_d_ == o->causal_d_ && causal_h_ == o->causal_h_ && causal_w_ == o->causal_w_ &&
        scale_ == o->scale_ && use_vec4_ == o->use_vec4_;
  }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    // Input: [B, ID, IH, IW, H, D]
    auto& q = inputs[0];
    int B = q.shape(0);
    int ID = q.shape(1);
    int IH = q.shape(2);
    int IW = q.shape(3);
    int H = q.shape(4);
    int D = q.shape(5);
    int out_d = (ID + stride_d_ - 1) / stride_d_;
    int out_h = (IH + stride_h_ - 1) / stride_h_;
    int out_w = (IW + stride_w_ - 1) / stride_w_;
    return {{B, out_d, out_h, out_w, H, D}};
  }

  auto state() const {
    return std::make_tuple(
        nullptr, kernel_size_,
        stride_d_, stride_h_, stride_w_,
        dilation_d_, dilation_h_, dilation_w_,
        causal_d_, causal_h_, causal_w_,
        scale_, use_vec4_);
  }

 private:
  int kernel_size_;
  int stride_d_, stride_h_, stride_w_;
  int dilation_d_, dilation_h_, dilation_w_;
  int causal_d_, causal_h_, causal_w_;
  float scale_;
  bool use_vec4_;
};

mx::array na3d_fused_forward_v2(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    int kernel_size,
    int stride_d, int stride_h, int stride_w,
    int dilation_d, int dilation_h, int dilation_w,
    bool causal_d, bool causal_h, bool causal_w,
    float scale,
    mx::StreamOrDevice s = {});

}  // namespace natten_mlx
