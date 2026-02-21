"""Metal kernel sources for fast neighborhood attention paths."""

from __future__ import annotations


_HELPERS = r"""
// NATTEN helper macros (shifted-window semantics)

#define NATTEN_GET_WINDOW_START(OUT, IDX, LEN, K, NH, DIL) do {                 \
    int _idx = (IDX);                                                            \
    int _len = (LEN);                                                            \
    int _K = (K);                                                                \
    int _nh = (NH);                                                              \
    int _d = (DIL);                                                              \
    int _dilation_idx = _idx % _d;                                               \
    int _index_pdp = _idx / _d;                                                  \
    int _length_pdp = (_len + _d - 1) / _d;                                      \
    int _num_padded = (_length_pdp * _d) - _len;                                 \
    if (_dilation_idx >= (_d - _num_padded)) {                                   \
        _length_pdp -= 1;                                                        \
    }                                                                            \
    int _start_idx = _index_pdp - _nh;                                           \
    if (_start_idx < 0) _start_idx = 0;                                          \
    if (_index_pdp + _nh >= _length_pdp) {                                       \
        _start_idx += (_length_pdp - _index_pdp - _nh - 1);                      \
    }                                                                            \
    (OUT) = _start_idx * _d + _dilation_idx;                                     \
} while(0)

#define NATTEN_GET_WINDOW_END(OUT, START, LEN, K, DIL) do {                      \
    int _end = (START) + (K) * (DIL);                                            \
    if (_end > (LEN)) _end = (LEN);                                              \
    (OUT) = _end;                                                                \
} while(0)

#define NATTEN_GET_PB_START(OUT, IDX, LEN, K, NH, DIL) do {                      \
    int _idx = (IDX);                                                            \
    int _len = (LEN);                                                            \
    int _K = (K);                                                                \
    int _nh = (NH);                                                              \
    int _d = (DIL);                                                              \
    int _pb;                                                                     \
    if (_d <= 1) {                                                               \
        _pb = _nh;                                                               \
        if (_idx < _nh) _pb += (_nh - _idx);                                     \
        if (_idx + _nh >= _len) _pb += (_len - _idx - 1 - _nh);                  \
    } else {                                                                     \
        if (_idx - _nh * _d < 0) {                                               \
            _pb = (_K - 1) - (_idx / _d);                                        \
        } else if (_idx + _nh * _d >= _len) {                                    \
            _pb = (_len - _idx - 1) / _d;                                        \
        } else {                                                                 \
            _pb = _nh;                                                           \
        }                                                                        \
    }                                                                            \
    (OUT) = _pb;                                                                 \
} while(0)
"""


def _nh(kernel_size: int) -> int:
    return kernel_size // 2


def _area(kernel_size: int) -> int:
    return kernel_size * kernel_size


def source_1d_qk(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const int K = {kernel_size};
const int NH = {nh};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni, ei, pi;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
NATTEN_GET_PB_START(pi, i, length, K, NH, dilation);

for (int ki = 0; ki < K; ki++) {{
    int key_i = ni + ki * dilation;
    float score;
    if (key_i >= 0 && key_i < ei) {{
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {{
            int q_idx = (((b * heads + h) * length + i) * dim + d);
            int k_idx = (((b * heads + h) * length + key_i) * dim + d);
            sum += query[q_idx] * key[k_idx];
        }}
        int rpb_idx = h * (2 * K - 1) + (pi + ki);
        score = sum + rpb[rpb_idx];
    }} else {{
        score = -INFINITY;
    }}
    int out_idx = (((b * heads + h) * length + i) * K + ki);
    out[out_idx] = score;
}}
"""


def source_1d_av(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = attention_probs_shape[0];
const int heads = attention_probs_shape[1];
const int length = attention_probs_shape[2];
const int dim = value_shape[3];
const int dilation = (int)dilation_param[0];
const int K = {kernel_size};
const int NH = {nh};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni, ei;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);

for (int d = 0; d < dim; d++) {{
    float sum = 0.0f;
    for (int ki = 0; ki < K; ki++) {{
        int val_i = ni + ki * dilation;
        if (val_i >= 0 && val_i < ei) {{
            int attn_idx = (((b * heads + h) * length + i) * K + ki);
            int val_idx = (((b * heads + h) * length + val_i) * dim + d);
            sum += attention_probs[attn_idx] * value[val_idx];
        }}
    }}
    int out_idx = (((b * heads + h) * length + i) * dim + d);
    out[out_idx] = sum;
}}
"""


def source_2d_qk(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    area = _area(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];
const int dilation = (int)dilation_param[0];
const int K = {kernel_size};
const int NH = {nh};
const int L = {area};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;

int ni, nj, ei, ej, pi, pj;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);
NATTEN_GET_PB_START(pi, i, height, K, NH, dilation);
NATTEN_GET_PB_START(pj, j, width, K, NH, dilation);

int neighbor_idx = 0;
for (int ki = 0; ki < K; ki++) {{
    for (int kj = 0; kj < K; kj++) {{
        int key_i = ni + ki * dilation;
        int key_j = nj + kj * dilation;
        float score;
        if (key_i >= 0 && key_i < ei && key_j >= 0 && key_j < ej) {{
            float sum = 0.0f;
            for (int d = 0; d < dim; d++) {{
                int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
                int k_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                sum += query[q_idx] * key[k_idx];
            }}
            int rpb_size = (2 * K - 1);
            int rpb_idx = h * rpb_size * rpb_size + (pi + ki) * rpb_size + (pj + kj);
            score = sum + rpb[rpb_idx];
        }} else {{
            score = -INFINITY;
        }}
        int out_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
        out[out_idx] = score;
        neighbor_idx++;
    }}
}}
"""


def source_2d_av(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    area = _area(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = attention_probs_shape[0];
const int heads = attention_probs_shape[1];
const int height = attention_probs_shape[2];
const int width = attention_probs_shape[3];
const int dim = value_shape[4];
const int dilation = (int)dilation_param[0];
const int K = {kernel_size};
const int NH = {nh};
const int L = {area};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;

int ni, nj, ei, ej;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);

for (int d = 0; d < dim; d++) {{
    float sum = 0.0f;
    int neighbor_idx = 0;
    for (int ki = 0; ki < K; ki++) {{
        for (int kj = 0; kj < K; kj++) {{
            int val_i = ni + ki * dilation;
            int val_j = nj + kj * dilation;
            if (val_i >= 0 && val_i < ei && val_j >= 0 && val_j < ej) {{
                int attn_idx = (((b * heads + h) * height + i) * width + j) * L + neighbor_idx;
                int val_idx = (((b * heads + h) * height + val_i) * width + val_j) * dim + d;
                sum += attention_probs[attn_idx] * value[val_idx];
            }}
            neighbor_idx++;
        }}
    }}
    int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
    out[out_idx] = sum;
}}
"""


def source_1d_fused(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const float scale = scale_param[0];
const int K = {kernel_size};
const int NH = {nh};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni, ei;
NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);

float logits[K];
float probs[K];
int key_pos[K];

float max_logit = -INFINITY;
for (int ki = 0; ki < K; ki++) {{
    int key_i = ni + ki * dilation;
    key_pos[ki] = key_i;
    float score = -INFINITY;
    if (key_i >= 0 && key_i < ei) {{
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {{
            int q_idx = (((b * heads + h) * length + i) * dim + d);
            int k_idx = (((b * heads + h) * length + key_i) * dim + d);
            sum += query[q_idx] * key[k_idx];
        }}
        score = sum * scale;
    }}
    logits[ki] = score;
    max_logit = fmax(max_logit, score);
}}

float denom = 0.0f;
for (int ki = 0; ki < K; ki++) {{
    float p = exp(logits[ki] - max_logit);
    probs[ki] = p;
    denom += p;
}}
float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;

for (int d = 0; d < dim; d++) {{
    float out_sum = 0.0f;
    for (int ki = 0; ki < K; ki++) {{
        int key_i = key_pos[ki];
        if (key_i >= 0 && key_i < ei) {{
            float w = probs[ki] * inv_denom;
            int v_idx = (((b * heads + h) * length + key_i) * dim + d);
            out_sum += w * value[v_idx];
        }}
    }}
    int out_idx = (((b * heads + h) * length + i) * dim + d);
    out[out_idx] = out_sum;
}}
"""


def source_2d_fused(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    area = _area(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];
const int dilation = (int)dilation_param[0];
const float scale = scale_param[0];
const int K = {kernel_size};
const int NH = {nh};
const int L = {area};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;

int ni, nj, ei, ej;
NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation);
NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation);
NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation);
NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation);

float logits[L];
float probs[L];
int key_i_arr[L];
int key_j_arr[L];

float max_logit = -INFINITY;
int neighbor_idx = 0;
for (int ki = 0; ki < K; ki++) {{
    for (int kj = 0; kj < K; kj++) {{
        int key_i = ni + ki * dilation;
        int key_j = nj + kj * dilation;
        key_i_arr[neighbor_idx] = key_i;
        key_j_arr[neighbor_idx] = key_j;
        float score = -INFINITY;
        if (key_i >= 0 && key_i < ei && key_j >= 0 && key_j < ej) {{
            float sum = 0.0f;
            for (int d = 0; d < dim; d++) {{
                int q_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
                int k_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
                sum += query[q_idx] * key[k_idx];
            }}
            score = sum * scale;
        }}
        logits[neighbor_idx] = score;
        max_logit = fmax(max_logit, score);
        neighbor_idx++;
    }}
}}

float denom = 0.0f;
for (int n = 0; n < L; n++) {{
    float p = exp(logits[n] - max_logit);
    probs[n] = p;
    denom += p;
}}
float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;

for (int d = 0; d < dim; d++) {{
    float out_sum = 0.0f;
    for (int n = 0; n < L; n++) {{
        int key_i = key_i_arr[n];
        int key_j = key_j_arr[n];
        if (key_i >= 0 && key_i < ei && key_j >= 0 && key_j < ej) {{
            float w = probs[n] * inv_denom;
            int v_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
            out_sum += w * value[v_idx];
        }}
    }}
    int out_idx = (((b * heads + h) * height + i) * width + j) * dim + d;
    out[out_idx] = out_sum;
}}
"""
