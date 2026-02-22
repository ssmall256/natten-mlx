#include <metal_stdlib>

using namespace metal;

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

inline int ceil_div_int(int a, int b) {
  return (a + b - 1) / b;
}

inline int natten_get_window_start(
    int index,
    int length,
    int kernel_size,
    int neighborhood_size,
    int dilation) {
  if (dilation <= 1) {
    return max(index - neighborhood_size, 0) +
        ((index + neighborhood_size >= length) ? (length - index - neighborhood_size - 1) : 0);
  }

  int ni = index - neighborhood_size * dilation;
  if (ni < 0) {
    return index % dilation;
  }

  if (index + neighborhood_size * dilation >= length) {
    int imod = index % dilation;
    int a = (length / dilation) * dilation;
    int b = length - a;
    if (imod < b) {
      return length - b + imod - 2 * neighborhood_size * dilation;
    }
    return a + imod - kernel_size * dilation;
  }

  return ni;
}

inline bool valid_1d(int idx, int length) {
  return idx >= 0 && idx < length;
}

inline int base_1d(int b, int i, int h, int d, int L, int H, int D) {
  return (((b * L + i) * H + h) * D + d);
}

inline int base_2d(int b, int i, int j, int h, int d, int IH, int IW, int H, int D) {
  return ((((b * IH + i) * IW + j) * H + h) * D + d);
}

inline int base_3d(int b, int z, int i, int j, int h, int d, int ID, int IH, int IW, int H, int D) {
  return (((((b * ID + z) * IH + i) * IW + j) * H + h) * D + d);
}

[[kernel]] void na1d_qk_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H * p.K;
  if (idx >= total) {
    return;
  }

  int kpos = idx % p.K;
  int t = idx / p.K;
  int h = t % p.H;
  t /= p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  if (qidx >= p.L) {
    out[idx] = -INFINITY;
    return;
  }

  int nh = p.K / 2;
  int kidx = p.CAUSAL ? (qidx - kpos * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (!valid_1d(kidx, p.L)) {
    out[idx] = -INFINITY;
    return;
  }

  float acc = 0.0f;
  int q_base = base_1d(b, qidx, h, 0, p.L, p.H, p.D);
  int k_base = base_1d(b, kidx, h, 0, p.L, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += query[q_base + d] * key[k_base + d];
  }
  out[idx] = acc * p.SCALE;
}

[[kernel]] void na1d_av_fp32(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  if (qidx >= p.L) {
    out[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  float acc = 0.0f;
  int attn_base = (((b * out_l + oq) * p.H + h) * p.K);
  for (int kpos = 0; kpos < p.K; ++kpos) {
    int kidx = p.CAUSAL ? (qidx - kpos * p.DIL)
                        : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
    if (!valid_1d(kidx, p.L)) {
      continue;
    }
    float w = attn[attn_base + kpos];
    acc += w * value[base_1d(b, kidx, h, d, p.L, p.H, p.D)];
  }
  out[idx] = acc;
}

[[kernel]] void na1d_qk_bwd_q_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* grad_q [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * p.L * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int i = t % p.L;
  int b = t / p.L;

  if ((i % p.S) != 0) {
    grad_q[idx] = 0.0f;
    return;
  }
  int oq = i / p.S;
  if (oq >= out_l) {
    grad_q[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int attn_base = (((b * out_l + oq) * p.H + h) * p.K);
  float acc = 0.0f;
  for (int kpos = 0; kpos < p.K; ++kpos) {
    int kidx = p.CAUSAL ? (i - kpos * p.DIL)
                        : (natten_get_window_start(i, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
    if (!valid_1d(kidx, p.L)) {
      continue;
    }
    acc += grad_attn[attn_base + kpos] * key[base_1d(b, kidx, h, d, p.L, p.H, p.D)];
  }
  grad_q[idx] = acc * p.SCALE;
}

[[kernel]] void na1d_qk_bwd_k_accum_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device atomic<float>* grad_k [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H * p.K * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % p.K;
  t /= p.K;
  int h = t % p.H;
  t /= p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  if (qidx >= p.L) {
    return;
  }

  int nh = p.K / 2;
  int kidx = p.CAUSAL ? (qidx - kpos * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (!valid_1d(kidx, p.L)) {
    return;
  }

  int attn_idx = (((b * out_l + oq) * p.H + h) * p.K + kpos);
  float g = grad_attn[attn_idx] * query[base_1d(b, qidx, h, d, p.L, p.H, p.D)] * p.SCALE;
  atomic_fetch_add_explicit(&grad_k[base_1d(b, kidx, h, d, p.L, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na1d_av_bwd_attn_fp32(
    device const float* grad_out [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* grad_attn [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H * p.K;
  if (idx >= total) {
    return;
  }

  int kpos = idx % p.K;
  int t = idx / p.K;
  int h = t % p.H;
  t /= p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  if (qidx >= p.L) {
    grad_attn[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int kidx = p.CAUSAL ? (qidx - kpos * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (!valid_1d(kidx, p.L)) {
    grad_attn[idx] = 0.0f;
    return;
  }

  float acc = 0.0f;
  int go_base = (((b * out_l + oq) * p.H + h) * p.D);
  int v_base = (((b * p.L + kidx) * p.H + h) * p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += grad_out[go_base + d] * value[v_base + d];
  }
  grad_attn[idx] = acc;
}

[[kernel]] void na1d_av_bwd_v_accum_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic<float>* grad_v [[buffer(2)]],
    constant NA1DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_l = ceil_div_int(p.L, p.S);
  int idx = static_cast<int>(tid);
  int total = p.B * out_l * p.H * p.K * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % p.K;
  t /= p.K;
  int h = t % p.H;
  t /= p.H;
  int oq = t % out_l;
  int b = t / out_l;

  int qidx = oq * p.S;
  if (qidx >= p.L) {
    return;
  }

  int nh = p.K / 2;
  int kidx = p.CAUSAL ? (qidx - kpos * p.DIL)
                      : (natten_get_window_start(qidx, p.L, p.K, nh, p.DIL) + kpos * p.DIL);
  if (!valid_1d(kidx, p.L)) {
    return;
  }

  int attn_idx = (((b * out_l + oq) * p.H + h) * p.K + kpos);
  int go_idx = (((b * out_l + oq) * p.H + h) * p.D + d);
  float g = attn[attn_idx] * grad_out[go_idx];
  atomic_fetch_add_explicit(&grad_v[base_1d(b, kidx, h, d, p.L, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na2d_qk_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * k2;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k2;
  int kh = kpos / p.K;
  int kw = kpos % p.K;
  int t = idx / k2;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    out[idx] = -INFINITY;
    return;
  }

  int nh = p.K / 2;
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    out[idx] = -INFINITY;
    return;
  }

  float acc = 0.0f;
  int q_base = base_2d(b, qh, qw, h, 0, p.IH, p.IW, p.H, p.D);
  int k_base = base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += query[q_base + d] * key[k_base + d];
  }
  out[idx] = acc * p.SCALE;
}

[[kernel]] void na2d_av_fp32(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    out[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  float acc = 0.0f;
  int attn_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * (p.K * p.K));
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH ? (qh - kh * p.DH)
                  : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW ? (qw - kw * p.DW)
                    : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      float w = attn[attn_base + kh * p.K + kw];
      acc += w * value[base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)];
    }
  }
  out[idx] = acc;
}

[[kernel]] void na2d_qk_bwd_q_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* grad_q [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * p.IH * p.IW * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int w = t % p.IW;
  t /= p.IW;
  int i = t % p.IH;
  int b = t / p.IH;

  if ((i % p.SH) != 0 || (w % p.SW) != 0) {
    grad_q[idx] = 0.0f;
    return;
  }
  int oh = i / p.SH;
  int ow = w / p.SW;
  if (oh >= out_h || ow >= out_w) {
    grad_q[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int attn_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * (p.K * p.K));
  float acc = 0.0f;
  for (int kh = 0; kh < p.K; ++kh) {
    int ih = p.CH ? (i - kh * p.DH)
                  : (natten_get_window_start(i, p.IH, p.K, nh, p.DH) + kh * p.DH);
    if (ih < 0 || ih >= p.IH) {
      continue;
    }
    for (int kw = 0; kw < p.K; ++kw) {
      int iw = p.CW ? (w - kw * p.DW)
                    : (natten_get_window_start(w, p.IW, p.K, nh, p.DW) + kw * p.DW);
      if (iw < 0 || iw >= p.IW) {
        continue;
      }
      float g = grad_attn[attn_base + kh * p.K + kw];
      acc += g * key[base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)];
    }
  }
  grad_q[idx] = acc * p.SCALE;
}

[[kernel]] void na2d_qk_bwd_k_accum_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device atomic<float>* grad_k [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * k2 * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % k2;
  int kh = kpos / p.K;
  int kw = kpos % p.K;
  t /= k2;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    return;
  }

  int nh = p.K / 2;
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int attn_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * k2 + kpos);
  float g = grad_attn[attn_idx] * query[base_2d(b, qh, qw, h, d, p.IH, p.IW, p.H, p.D)] * p.SCALE;
  atomic_fetch_add_explicit(&grad_k[base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na2d_av_bwd_attn_fp32(
    device const float* grad_out [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* grad_attn [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * k2;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k2;
  int kh = kpos / p.K;
  int kw = kpos % p.K;
  int t = idx / k2;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    grad_attn[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    grad_attn[idx] = 0.0f;
    return;
  }

  float acc = 0.0f;
  int go_base = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D);
  int v_base = base_2d(b, ih, iw, h, 0, p.IH, p.IW, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += grad_out[go_base + d] * value[v_base + d];
  }
  grad_attn[idx] = acc;
}

[[kernel]] void na2d_av_bwd_v_accum_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic<float>* grad_v [[buffer(2)]],
    constant NA2DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k2 = p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_h * out_w * p.H * k2 * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % k2;
  int kh = kpos / p.K;
  int kw = kpos % p.K;
  t /= k2;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  int b = t / out_h;

  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qh >= p.IH || qw >= p.IW) {
    return;
  }

  int nh = p.K / 2;
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int attn_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * k2 + kpos);
  int go_idx = ((((b * out_h + oh) * out_w + ow) * p.H + h) * p.D + d);
  float g = attn[attn_idx] * grad_out[go_idx];
  atomic_fetch_add_explicit(&grad_v[base_2d(b, ih, iw, h, d, p.IH, p.IW, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na3d_qk_fp32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * k3;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k3;
  int kd = kpos / (p.K * p.K);
  int rem = kpos % (p.K * p.K);
  int kh = rem / p.K;
  int kw = rem % p.K;
  int t = idx / k3;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  t /= out_h;
  int od = t % out_d;
  int b = t / out_d;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    out[idx] = -INFINITY;
    return;
  }

  int nh = p.K / 2;
  int id = p.CD ? (qd - kd * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    out[idx] = -INFINITY;
    return;
  }

  float acc = 0.0f;
  int q_base = base_3d(b, qd, qh, qw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  int k_base = base_3d(b, id, ih, iw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += query[q_base + d] * key[k_base + d];
  }
  out[idx] = acc * p.SCALE;
}

[[kernel]] void na3d_av_fp32(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  t /= out_h;
  int od = t % out_d;
  int b = t / out_d;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    out[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  float acc = 0.0f;
  int k3 = p.K * p.K * p.K;
  int attn_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3);
  for (int kd = 0; kd < p.K; ++kd) {
    int id = p.CD ? (qd - kd * p.DD)
                  : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH ? (qh - kh * p.DH)
                    : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW ? (qw - kw * p.DW)
                      : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        float w = attn[attn_base + kpos];
        acc += w * value[base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
      }
    }
  }
  out[idx] = acc;
}

[[kernel]] void na3d_qk_bwd_q_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* grad_q [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int idx = static_cast<int>(tid);
  int total = p.B * p.ID * p.IH * p.IW * p.H * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int h = t % p.H;
  t /= p.H;
  int w = t % p.IW;
  t /= p.IW;
  int i = t % p.IH;
  t /= p.IH;
  int z = t % p.ID;
  int b = t / p.ID;

  if ((z % p.SD) != 0 || (i % p.SH) != 0 || (w % p.SW) != 0) {
    grad_q[idx] = 0.0f;
    return;
  }
  int od = z / p.SD;
  int oh = i / p.SH;
  int ow = w / p.SW;
  if (od >= out_d || oh >= out_h || ow >= out_w) {
    grad_q[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int k3 = p.K * p.K * p.K;
  int attn_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3);
  float acc = 0.0f;
  for (int kd = 0; kd < p.K; ++kd) {
    int id = p.CD ? (z - kd * p.DD)
                  : (natten_get_window_start(z, p.ID, p.K, nh, p.DD) + kd * p.DD);
    if (id < 0 || id >= p.ID) {
      continue;
    }
    for (int kh = 0; kh < p.K; ++kh) {
      int ih = p.CH ? (i - kh * p.DH)
                    : (natten_get_window_start(i, p.IH, p.K, nh, p.DH) + kh * p.DH);
      if (ih < 0 || ih >= p.IH) {
        continue;
      }
      for (int kw = 0; kw < p.K; ++kw) {
        int iw = p.CW ? (w - kw * p.DW)
                      : (natten_get_window_start(w, p.IW, p.K, nh, p.DW) + kw * p.DW);
        if (iw < 0 || iw >= p.IW) {
          continue;
        }
        int kpos = (kd * p.K + kh) * p.K + kw;
        float g = grad_attn[attn_base + kpos];
        acc += g * key[base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)];
      }
    }
  }
  grad_q[idx] = acc * p.SCALE;
}

[[kernel]] void na3d_qk_bwd_k_accum_fp32(
    device const float* grad_attn [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device atomic<float>* grad_k [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * k3 * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % k3;
  int kd = kpos / (p.K * p.K);
  int rem = kpos % (p.K * p.K);
  int kh = rem / p.K;
  int kw = rem % p.K;
  t /= k3;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  t /= out_h;
  int od = t % out_d;
  int b = t / out_d;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    return;
  }

  int nh = p.K / 2;
  int id = p.CD ? (qd - kd * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int attn_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3 + kpos);
  float g = grad_attn[attn_idx] * query[base_3d(b, qd, qh, qw, h, d, p.ID, p.IH, p.IW, p.H, p.D)] * p.SCALE;
  atomic_fetch_add_explicit(&grad_k[base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], g, memory_order_relaxed);
}

[[kernel]] void na3d_av_bwd_attn_fp32(
    device const float* grad_out [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* grad_attn [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * k3;
  if (idx >= total) {
    return;
  }

  int kpos = idx % k3;
  int kd = kpos / (p.K * p.K);
  int rem = kpos % (p.K * p.K);
  int kh = rem / p.K;
  int kw = rem % p.K;
  int t = idx / k3;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  t /= out_h;
  int od = t % out_d;
  int b = t / out_d;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    grad_attn[idx] = 0.0f;
    return;
  }

  int nh = p.K / 2;
  int id = p.CD ? (qd - kd * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    grad_attn[idx] = 0.0f;
    return;
  }

  float acc = 0.0f;
  int go_base = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D);
  int v_base = base_3d(b, id, ih, iw, h, 0, p.ID, p.IH, p.IW, p.H, p.D);
  for (int d = 0; d < p.D; ++d) {
    acc += grad_out[go_base + d] * value[v_base + d];
  }
  grad_attn[idx] = acc;
}

[[kernel]] void na3d_av_bwd_v_accum_fp32(
    device const float* attn [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device atomic<float>* grad_v [[buffer(2)]],
    constant NA3DParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  int out_d = ceil_div_int(p.ID, p.SD);
  int out_h = ceil_div_int(p.IH, p.SH);
  int out_w = ceil_div_int(p.IW, p.SW);
  int k3 = p.K * p.K * p.K;
  int idx = static_cast<int>(tid);
  int total = p.B * out_d * out_h * out_w * p.H * k3 * p.D;
  if (idx >= total) {
    return;
  }

  int d = idx % p.D;
  int t = idx / p.D;
  int kpos = t % k3;
  int kd = kpos / (p.K * p.K);
  int rem = kpos % (p.K * p.K);
  int kh = rem / p.K;
  int kw = rem % p.K;
  t /= k3;
  int h = t % p.H;
  t /= p.H;
  int ow = t % out_w;
  t /= out_w;
  int oh = t % out_h;
  t /= out_h;
  int od = t % out_d;
  int b = t / out_d;

  int qd = od * p.SD;
  int qh = oh * p.SH;
  int qw = ow * p.SW;
  if (qd >= p.ID || qh >= p.IH || qw >= p.IW) {
    return;
  }

  int nh = p.K / 2;
  int id = p.CD ? (qd - kd * p.DD)
                : (natten_get_window_start(qd, p.ID, p.K, nh, p.DD) + kd * p.DD);
  int ih = p.CH ? (qh - kh * p.DH)
                : (natten_get_window_start(qh, p.IH, p.K, nh, p.DH) + kh * p.DH);
  int iw = p.CW ? (qw - kw * p.DW)
                : (natten_get_window_start(qw, p.IW, p.K, nh, p.DW) + kw * p.DW);
  if (id < 0 || id >= p.ID || ih < 0 || ih >= p.IH || iw < 0 || iw >= p.IW) {
    return;
  }

  int attn_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * k3 + kpos);
  int go_idx = (((((b * out_d + od) * out_h + oh) * out_w + ow) * p.H + h) * p.D + d);
  float g = attn[attn_idx] * grad_out[go_idx];
  atomic_fetch_add_explicit(&grad_v[base_3d(b, id, ih, iw, h, d, p.ID, p.IH, p.IW, p.H, p.D)], g, memory_order_relaxed);
}
