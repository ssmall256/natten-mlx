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
// Inputs: (q, k, v, grad_out) spatial-first [B, L, H, D]
// Outputs: (attn, grad_attn) both [B, OL, H, K] in fp32
class NA1DBwdAttn : public mx::Primitive {
 public:
  NA1DBwdAttn(
      mx::Stream stream, int kernel_size,
      int stride, int dilation, int causal,
      float scale, bool use_vec4)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_(stride), dilation_(dilation),
        causal_(causal), scale_(scale), use_vec4_(use_vec4) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA1DBwdAttn: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA1DBwdAttn)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA1DBwdAttn*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ && stride_ == o->stride_ &&
        dilation_ == o->dilation_ && causal_ == o->causal_ &&
        scale_ == o->scale_ && use_vec4_ == o->use_vec4_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    auto& q = inputs[0];
    int B = q.shape(0), L = q.shape(1), H = q.shape(2);
    int ol = (L + stride_ - 1) / stride_;
    mx::Shape s = {B, ol, H, kernel_size_};
    return {s, s};
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_, stride_, dilation_, causal_, scale_, use_vec4_);
  }

 private:
  int kernel_size_, stride_, dilation_, causal_;
  float scale_;
  bool use_vec4_;
};

// ---- Stage 3: grad_q ----
class NA1DBwdGradQ : public mx::Primitive {
 public:
  NA1DBwdGradQ(
      mx::Stream stream, int kernel_size,
      int stride, int dilation, int causal,
      float scale, mx::Dtype out_dtype)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_(stride), dilation_(dilation),
        causal_(causal), scale_(scale), out_dtype_(out_dtype) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA1DBwdGradQ: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA1DBwdGradQ)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA1DBwdGradQ*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ && stride_ == o->stride_ &&
        dilation_ == o->dilation_ && causal_ == o->causal_ &&
        scale_ == o->scale_ && out_dtype_ == o->out_dtype_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {inputs[1].shape()};  // same shape as key
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_, stride_, dilation_, causal_, scale_);
  }

 private:
  int kernel_size_, stride_, dilation_, causal_;
  float scale_;
  mx::Dtype out_dtype_;
};

// ---- Stage 3b: grad_k ----
class NA1DBwdGradK : public mx::Primitive {
 public:
  NA1DBwdGradK(
      mx::Stream stream, int kernel_size,
      int stride, int dilation, int causal,
      float scale, mx::Dtype out_dtype)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_(stride), dilation_(dilation),
        causal_(causal), scale_(scale), out_dtype_(out_dtype) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA1DBwdGradK: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA1DBwdGradK)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA1DBwdGradK*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ && stride_ == o->stride_ &&
        dilation_ == o->dilation_ && causal_ == o->causal_ &&
        scale_ == o->scale_ && out_dtype_ == o->out_dtype_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {inputs[1].shape()};  // same shape as query
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_, stride_, dilation_, causal_, scale_);
  }

 private:
  int kernel_size_, stride_, dilation_, causal_;
  float scale_;
  mx::Dtype out_dtype_;
};

// ---- Stage 4: grad_v ----
class NA1DBwdGradV : public mx::Primitive {
 public:
  NA1DBwdGradV(
      mx::Stream stream, int kernel_size,
      int stride, int dilation, int causal,
      int IL, mx::Dtype out_dtype)
      : mx::Primitive(stream),
        kernel_size_(kernel_size),
        stride_(stride), dilation_(dilation),
        causal_(causal), IL_(IL), out_dtype_(out_dtype) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("NA1DBwdGradV: CPU not supported");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  DEFINE_NAME(NA1DBwdGradV)

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* o = dynamic_cast<const NA1DBwdGradV*>(&other);
    if (!o) return false;
    return kernel_size_ == o->kernel_size_ && stride_ == o->stride_ &&
        dilation_ == o->dilation_ && causal_ == o->causal_ &&
        IL_ == o->IL_ && out_dtype_ == o->out_dtype_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    auto& attn = inputs[0];
    int B = attn.shape(0), H = attn.shape(2);
    int D = inputs[1].shape(3);
    return {{B, IL_, H, D}};
  }

  auto state() const {
    return std::make_tuple(nullptr, kernel_size_, stride_, dilation_, causal_, IL_);
  }

 private:
  int kernel_size_, stride_, dilation_, causal_;
  int IL_;
  mx::Dtype out_dtype_;
};

// Factory: creates lazy graph returning (grad_q, grad_k, grad_v).
std::vector<mx::array> na1d_backward_v2(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    const mx::array& grad_out,
    int kernel_size,
    int stride, int dilation,
    bool causal,
    float scale,
    mx::StreamOrDevice s = {});

}  // namespace natten_mlx
