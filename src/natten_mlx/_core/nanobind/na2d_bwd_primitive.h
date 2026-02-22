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

// ---- Stage 1: Fused attn recompute + grad_logits ----
// Inputs: (q, k, v, grad_out) spatial-first [B, IH, IW, H, D]
// Outputs: (attn, grad_logits) both [B, OH, OW, H, K²] in fp32
class NA2DBwdAttn : public mx::Primitive {
 public:
  NA2DBwdAttn(
      mx::Stream stream, int kernel_size,
      int stride_h, int stride_w,
      int dilation_h, int dilation_w,
      int causal_h, int causal_w,
      float scale, bool use_vec4)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_h_(stride_h), stride_w_(stride_w),
        dilation_h_(dilation_h), dilation_w_(dilation_w),
        causal_h_(causal_h), causal_w_(causal_w),
        scale_(scale), use_vec4_(use_vec4) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA2DBwdAttn: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA2DBwdAttn)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA2DBwdAttn*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ && stride_h_ == o->stride_h_ &&
        stride_w_ == o->stride_w_ && dilation_h_ == o->dilation_h_ &&
        dilation_w_ == o->dilation_w_ && causal_h_ == o->causal_h_ &&
        causal_w_ == o->causal_w_ && scale_ == o->scale_ && use_vec4_ == o->use_vec4_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    auto& q = inputs[0];
    int B = q.shape(0), IH = q.shape(1), IW = q.shape(2), H = q.shape(3);
    int oh = (IH + stride_h_ - 1) / stride_h_;
    int ow = (IW + stride_w_ - 1) / stride_w_;
    int K2 = kernel_size_ * kernel_size_;
    mx::Shape s = {B, oh, ow, H, K2};
    return {s, s};
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_, stride_h_, stride_w_,
        dilation_h_, dilation_w_, causal_h_, causal_w_, scale_, use_vec4_);
  }

 private:
  int kernel_size_, stride_h_, stride_w_, dilation_h_, dilation_w_;
  int causal_h_, causal_w_;
  float scale_;
  bool use_vec4_;
};

// ---- Stage 3: grad_q ----
// Inputs: (grad_logits [B,OH,OW,H,K²] fp32, key_fp32 [B,IH,IW,H,D])
// Output: grad_q [B,IH,IW,H,D] in output dtype
class NA2DBwdGradQ : public mx::Primitive {
 public:
  NA2DBwdGradQ(
      mx::Stream stream, int kernel_size,
      int stride_h, int stride_w,
      int dilation_h, int dilation_w,
      int causal_h, int causal_w, float scale,
      mx::Dtype out_dtype)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_h_(stride_h), stride_w_(stride_w),
        dilation_h_(dilation_h), dilation_w_(dilation_w),
        causal_h_(causal_h), causal_w_(causal_w),
        scale_(scale), out_dtype_(out_dtype) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA2DBwdGradQ: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA2DBwdGradQ)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA2DBwdGradQ*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ && stride_h_ == o->stride_h_ &&
        stride_w_ == o->stride_w_ && dilation_h_ == o->dilation_h_ &&
        dilation_w_ == o->dilation_w_ && causal_h_ == o->causal_h_ &&
        causal_w_ == o->causal_w_ && scale_ == o->scale_ && out_dtype_ == o->out_dtype_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {inputs[1].shape()};  // same shape as key
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_, stride_h_, stride_w_,
        dilation_h_, dilation_w_, causal_h_, causal_w_, scale_);
  }

 private:
  int kernel_size_, stride_h_, stride_w_, dilation_h_, dilation_w_;
  int causal_h_, causal_w_;
  float scale_;
  mx::Dtype out_dtype_;
};

// ---- Stage 3b: grad_k ----
class NA2DBwdGradK : public mx::Primitive {
 public:
  NA2DBwdGradK(
      mx::Stream stream, int kernel_size,
      int stride_h, int stride_w,
      int dilation_h, int dilation_w,
      int causal_h, int causal_w, float scale,
      mx::Dtype out_dtype)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_h_(stride_h), stride_w_(stride_w),
        dilation_h_(dilation_h), dilation_w_(dilation_w),
        causal_h_(causal_h), causal_w_(causal_w),
        scale_(scale), out_dtype_(out_dtype) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA2DBwdGradK: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA2DBwdGradK)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA2DBwdGradK*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ && stride_h_ == o->stride_h_ &&
        stride_w_ == o->stride_w_ && dilation_h_ == o->dilation_h_ &&
        dilation_w_ == o->dilation_w_ && causal_h_ == o->causal_h_ &&
        causal_w_ == o->causal_w_ && scale_ == o->scale_ && out_dtype_ == o->out_dtype_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {inputs[1].shape()};  // same shape as query
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_, stride_h_, stride_w_,
        dilation_h_, dilation_w_, causal_h_, causal_w_, scale_);
  }

 private:
  int kernel_size_, stride_h_, stride_w_, dilation_h_, dilation_w_;
  int causal_h_, causal_w_;
  float scale_;
  mx::Dtype out_dtype_;
};

// ---- Stage 4: grad_v ----
// Inputs: (attn [B,OH,OW,H,K²] fp32, grad_out [B,OH,OW,H,D])
// Output: grad_v [B,IH,IW,H,D]
class NA2DBwdGradV : public mx::Primitive {
 public:
  NA2DBwdGradV(
      mx::Stream stream, int kernel_size,
      int stride_h, int stride_w,
      int dilation_h, int dilation_w,
      int causal_h, int causal_w,
      int IH, int IW, mx::Dtype out_dtype)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_h_(stride_h), stride_w_(stride_w),
        dilation_h_(dilation_h), dilation_w_(dilation_w),
        causal_h_(causal_h), causal_w_(causal_w),
        IH_(IH), IW_(IW), out_dtype_(out_dtype) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA2DBwdGradV: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA2DBwdGradV)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA2DBwdGradV*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ && stride_h_ == o->stride_h_ &&
        stride_w_ == o->stride_w_ && dilation_h_ == o->dilation_h_ &&
        dilation_w_ == o->dilation_w_ && causal_h_ == o->causal_h_ &&
        causal_w_ == o->causal_w_ && IH_ == o->IH_ && IW_ == o->IW_ &&
        out_dtype_ == o->out_dtype_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    auto& attn = inputs[0];
    int B = attn.shape(0), H = attn.shape(3);
    int D = inputs[1].shape(4);
    return {{B, IH_, IW_, H, D}};
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_, stride_h_, stride_w_,
        dilation_h_, dilation_w_, causal_h_, causal_w_, IH_, IW_);
  }

 private:
  int kernel_size_, stride_h_, stride_w_, dilation_h_, dilation_w_;
  int causal_h_, causal_w_;
  int IH_, IW_;
  mx::Dtype out_dtype_;
};

// Factory: creates lazy graph returning (grad_q, grad_k, grad_v).
std::vector<mx::array> na2d_backward_v2(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& grad_out,
    int kernel_size,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    bool causal_h, bool causal_w,
    float scale,
    mx::StreamOrDevice s = {});

}  // namespace natten_mlx
