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

// ---- Stage 1: Fused attn recompute + grad_attn ----
// Inputs: (q, k, v, grad_out) [B, ID, IH, IW, H, D]
// Outputs: (attn, grad_attn) both [B, OD, OH, OW, H, K³] in fp32
class NA3DBwdAttn : public mx::Primitive {
 public:
  NA3DBwdAttn(
      mx::Stream stream, int kernel_size,
      int stride_d, int stride_h, int stride_w,
      int dilation_d, int dilation_h, int dilation_w,
      int causal_d, int causal_h, int causal_w,
      float scale, bool use_vec4)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
        dilation_d_(dilation_d), dilation_h_(dilation_h), dilation_w_(dilation_w),
        causal_d_(causal_d), causal_h_(causal_h), causal_w_(causal_w),
        scale_(scale), use_vec4_(use_vec4) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA3DBwdAttn: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA3DBwdAttn)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA3DBwdAttn*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ &&
        stride_d_ == o->stride_d_ && stride_h_ == o->stride_h_ && stride_w_ == o->stride_w_ &&
        dilation_d_ == o->dilation_d_ && dilation_h_ == o->dilation_h_ && dilation_w_ == o->dilation_w_ &&
        causal_d_ == o->causal_d_ && causal_h_ == o->causal_h_ && causal_w_ == o->causal_w_ &&
        scale_ == o->scale_ && use_vec4_ == o->use_vec4_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    auto& q = inputs[0];
    int B = q.shape(0), ID = q.shape(1), IH = q.shape(2), IW = q.shape(3), H = q.shape(4);
    int od = (ID + stride_d_ - 1) / stride_d_;
    int oh = (IH + stride_h_ - 1) / stride_h_;
    int ow = (IW + stride_w_ - 1) / stride_w_;
    int K3 = kernel_size_ * kernel_size_ * kernel_size_;
    mx::Shape s = {B, od, oh, ow, H, K3};
    return {s, s};
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_,
        stride_d_, stride_h_, stride_w_,
        dilation_d_, dilation_h_, dilation_w_,
        causal_d_, causal_h_, causal_w_, scale_, use_vec4_);
  }

 private:
  int kernel_size_;
  int stride_d_, stride_h_, stride_w_;
  int dilation_d_, dilation_h_, dilation_w_;
  int causal_d_, causal_h_, causal_w_;
  float scale_;
  bool use_vec4_;
};

// ---- Stage 3: grad_q ----
class NA3DBwdGradQ : public mx::Primitive {
 public:
  NA3DBwdGradQ(
      mx::Stream stream, int kernel_size,
      int stride_d, int stride_h, int stride_w,
      int dilation_d, int dilation_h, int dilation_w,
      int causal_d, int causal_h, int causal_w,
      float scale, mx::Dtype out_dtype)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
        dilation_d_(dilation_d), dilation_h_(dilation_h), dilation_w_(dilation_w),
        causal_d_(causal_d), causal_h_(causal_h), causal_w_(causal_w),
        scale_(scale), out_dtype_(out_dtype) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA3DBwdGradQ: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA3DBwdGradQ)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA3DBwdGradQ*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ &&
        stride_d_ == o->stride_d_ && stride_h_ == o->stride_h_ && stride_w_ == o->stride_w_ &&
        dilation_d_ == o->dilation_d_ && dilation_h_ == o->dilation_h_ && dilation_w_ == o->dilation_w_ &&
        causal_d_ == o->causal_d_ && causal_h_ == o->causal_h_ && causal_w_ == o->causal_w_ &&
        scale_ == o->scale_ && out_dtype_ == o->out_dtype_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {inputs[1].shape()};
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_,
        stride_d_, stride_h_, stride_w_,
        dilation_d_, dilation_h_, dilation_w_,
        causal_d_, causal_h_, causal_w_, scale_);
  }

 private:
  int kernel_size_;
  int stride_d_, stride_h_, stride_w_;
  int dilation_d_, dilation_h_, dilation_w_;
  int causal_d_, causal_h_, causal_w_;
  float scale_;
  mx::Dtype out_dtype_;
};

// ---- Stage 3b: grad_k ----
class NA3DBwdGradK : public mx::Primitive {
 public:
  NA3DBwdGradK(
      mx::Stream stream, int kernel_size,
      int stride_d, int stride_h, int stride_w,
      int dilation_d, int dilation_h, int dilation_w,
      int causal_d, int causal_h, int causal_w,
      float scale, mx::Dtype out_dtype)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
        dilation_d_(dilation_d), dilation_h_(dilation_h), dilation_w_(dilation_w),
        causal_d_(causal_d), causal_h_(causal_h), causal_w_(causal_w),
        scale_(scale), out_dtype_(out_dtype) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA3DBwdGradK: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA3DBwdGradK)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA3DBwdGradK*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ &&
        stride_d_ == o->stride_d_ && stride_h_ == o->stride_h_ && stride_w_ == o->stride_w_ &&
        dilation_d_ == o->dilation_d_ && dilation_h_ == o->dilation_h_ && dilation_w_ == o->dilation_w_ &&
        causal_d_ == o->causal_d_ && causal_h_ == o->causal_h_ && causal_w_ == o->causal_w_ &&
        scale_ == o->scale_ && out_dtype_ == o->out_dtype_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {inputs[1].shape()};
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_,
        stride_d_, stride_h_, stride_w_,
        dilation_d_, dilation_h_, dilation_w_,
        causal_d_, causal_h_, causal_w_, scale_);
  }

 private:
  int kernel_size_;
  int stride_d_, stride_h_, stride_w_;
  int dilation_d_, dilation_h_, dilation_w_;
  int causal_d_, causal_h_, causal_w_;
  float scale_;
  mx::Dtype out_dtype_;
};

// ---- Stage 4: grad_v ----
class NA3DBwdGradV : public mx::Primitive {
 public:
  NA3DBwdGradV(
      mx::Stream stream, int kernel_size,
      int stride_d, int stride_h, int stride_w,
      int dilation_d, int dilation_h, int dilation_w,
      int causal_d, int causal_h, int causal_w,
      int ID, int IH, int IW, mx::Dtype out_dtype)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_d_(stride_d), stride_h_(stride_h), stride_w_(stride_w),
        dilation_d_(dilation_d), dilation_h_(dilation_h), dilation_w_(dilation_w),
        causal_d_(causal_d), causal_h_(causal_h), causal_w_(causal_w),
        ID_(ID), IH_(IH), IW_(IW), out_dtype_(out_dtype) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA3DBwdGradV: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA3DBwdGradV)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA3DBwdGradV*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ &&
        stride_d_ == o->stride_d_ && stride_h_ == o->stride_h_ && stride_w_ == o->stride_w_ &&
        dilation_d_ == o->dilation_d_ && dilation_h_ == o->dilation_h_ && dilation_w_ == o->dilation_w_ &&
        causal_d_ == o->causal_d_ && causal_h_ == o->causal_h_ && causal_w_ == o->causal_w_ &&
        ID_ == o->ID_ && IH_ == o->IH_ && IW_ == o->IW_ && out_dtype_ == o->out_dtype_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    auto& attn = inputs[0];
    int B = attn.shape(0), H = attn.shape(4);
    int D = inputs[1].shape(5);
    return {{B, ID_, IH_, IW_, H, D}};
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_,
        stride_d_, stride_h_, stride_w_,
        dilation_d_, dilation_h_, dilation_w_,
        causal_d_, causal_h_, causal_w_, ID_, IH_, IW_);
  }

 private:
  int kernel_size_;
  int stride_d_, stride_h_, stride_w_;
  int dilation_d_, dilation_h_, dilation_w_;
  int causal_d_, causal_h_, causal_w_;
  int ID_, IH_, IW_;
  mx::Dtype out_dtype_;
};

// Factory
std::vector<mx::array> na3d_backward_v2(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& grad_out,
    int kernel_size,
    int stride_d, int stride_h, int stride_w,
    int dilation_d, int dilation_h, int dilation_w,
    bool causal_d, bool causal_h, bool causal_w,
    float scale,
    mx::StreamOrDevice s = {});

}  // namespace natten_mlx
