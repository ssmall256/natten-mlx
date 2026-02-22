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

// Split QK: compute neighborhood attention logits
// Inputs: (q, k) spatial-first [B, IH, IW, H, D]
// Output: logits [B, OH, OW, H, K²] in same dtype
class NA2DSplitQK : public mx::Primitive {
 public:
  NA2DSplitQK(
      mx::Stream stream,
      int kernel_size,
      int stride_h, int stride_w,
      int dilation_h, int dilation_w,
      bool causal_h, bool causal_w,
      float scale,
      bool use_vec4)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_h_(stride_h), stride_w_(stride_w),
        dilation_h_(dilation_h), dilation_w_(dilation_w),
        causal_h_(causal_h), causal_w_(causal_w),
        scale_(scale), use_vec4_(use_vec4) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NA2DSplitQK: CPU not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  DEFINE_NAME(NA2DSplitQK)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA2DSplitQK*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ && stride_h_ == o->stride_h_ &&
        stride_w_ == o->stride_w_ && dilation_h_ == o->dilation_h_ &&
        dilation_w_ == o->dilation_w_ && causal_h_ == o->causal_h_ &&
        causal_w_ == o->causal_w_ && scale_ == o->scale_ &&
        use_vec4_ == o->use_vec4_;
  }

 private:
  int kernel_size_;
  int stride_h_, stride_w_;
  int dilation_h_, dilation_w_;
  bool causal_h_, causal_w_;
  float scale_;
  bool use_vec4_;
};

// Split AV: apply attention weights to values
// Inputs: (attn, v) where attn [B, OH, OW, H, K²], v [B, IH, IW, H, D]
// Output: [B, OH, OW, H, D] in v's dtype
class NA2DSplitAV : public mx::Primitive {
 public:
  NA2DSplitAV(
      mx::Stream stream,
      int kernel_size,
      int stride_h, int stride_w,
      int dilation_h, int dilation_w,
      bool causal_h, bool causal_w,
      bool use_vec4)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_h_(stride_h), stride_w_(stride_w),
        dilation_h_(dilation_h), dilation_w_(dilation_w),
        causal_h_(causal_h), causal_w_(causal_w),
        use_vec4_(use_vec4) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    throw std::runtime_error("NA2DSplitAV: CPU not supported");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  DEFINE_NAME(NA2DSplitAV)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA2DSplitAV*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ && stride_h_ == o->stride_h_ &&
        stride_w_ == o->stride_w_ && dilation_h_ == o->dilation_h_ &&
        dilation_w_ == o->dilation_w_ && causal_h_ == o->causal_h_ &&
        causal_w_ == o->causal_w_ && use_vec4_ == o->use_vec4_;
  }

 private:
  int kernel_size_;
  int stride_h_, stride_w_;
  int dilation_h_, dilation_w_;
  bool causal_h_, causal_w_;
  bool use_vec4_;
};

// Factory functions
mx::array na2d_qk_forward_v2(
    const mx::array& q,
    const mx::array& k,
    int kernel_size,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    bool causal_h, bool causal_w,
    float scale,
    mx::StreamOrDevice s = {});

mx::array na2d_av_forward_v2(
    const mx::array& attn,
    const mx::array& v,
    int kernel_size,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    bool causal_h, bool causal_w,
    mx::StreamOrDevice s = {});

}  // namespace natten_mlx
