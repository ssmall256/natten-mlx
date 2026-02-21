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


def _volume(kernel_size: int) -> int:
    return kernel_size * kernel_size * kernel_size


def source_1d_qk(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int dilation = (int)dilation_param[0];
const bool causal = ((int)causal_param[0]) != 0;
const int K = {kernel_size};
const int NH = {nh};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni = 0;
int ei = length;
int pi = NH;
if (!causal) {{
    NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
    NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
    NATTEN_GET_PB_START(pi, i, length, K, NH, dilation);
}}

for (int ki = 0; ki < K; ki++) {{
    int key_i = (causal ? (i - (K - 1) * dilation) : ni) + ki * dilation;
    float score;
    bool valid = causal
        ? (key_i >= 0 && key_i <= i && key_i < length)
        : (key_i >= 0 && key_i < ei);
    if (valid) {{
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
const bool causal = ((int)causal_param[0]) != 0;
const int K = {kernel_size};
const int NH = {nh};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

int ni = 0;
int ei = length;
if (!causal) {{
    NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
    NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
}}

for (int d = 0; d < dim; d++) {{
    float sum = 0.0f;
    for (int ki = 0; ki < K; ki++) {{
        int val_i = (causal ? (i - (K - 1) * dilation) : ni) + ki * dilation;
        bool valid = causal
            ? (val_i >= 0 && val_i <= i && val_i < length)
            : (val_i >= 0 && val_i < ei);
        if (valid) {{
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
const int dilation_h = (int)dilation_param[0];
const int dilation_w = (int)dilation_param[1];
const bool causal_h = ((int)causal_param[0]) != 0;
const bool causal_w = ((int)causal_param[1]) != 0;
const int K = {kernel_size};
const int NH = {nh};
const int L = {area};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;

int ni = 0;
int nj = 0;
int ei = height;
int ej = width;
int pi = NH;
int pj = NH;
if (!causal_h) {{
    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
    NATTEN_GET_PB_START(pi, i, height, K, NH, dilation_h);
}}
if (!causal_w) {{
    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
    NATTEN_GET_PB_START(pj, j, width, K, NH, dilation_w);
}}

int neighbor_idx = 0;
for (int ki = 0; ki < K; ki++) {{
    for (int kj = 0; kj < K; kj++) {{
        int key_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
        int key_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
        float score;
        bool valid_i = causal_h
            ? (key_i >= 0 && key_i <= i && key_i < height)
            : (key_i >= 0 && key_i < ei);
        bool valid_j = causal_w
            ? (key_j >= 0 && key_j <= j && key_j < width)
            : (key_j >= 0 && key_j < ej);
        if (valid_i && valid_j) {{
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
const int dilation_h = (int)dilation_param[0];
const int dilation_w = (int)dilation_param[1];
const bool causal_h = ((int)causal_param[0]) != 0;
const bool causal_w = ((int)causal_param[1]) != 0;
const int K = {kernel_size};
const int NH = {nh};
const int L = {area};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;

int ni = 0;
int nj = 0;
int ei = height;
int ej = width;
if (!causal_h) {{
    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
}}
if (!causal_w) {{
    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
}}

for (int d = 0; d < dim; d++) {{
    float sum = 0.0f;
    int neighbor_idx = 0;
    for (int ki = 0; ki < K; ki++) {{
        for (int kj = 0; kj < K; kj++) {{
            int val_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
            int val_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
            bool valid_i = causal_h
                ? (val_i >= 0 && val_i <= i && val_i < height)
                : (val_i >= 0 && val_i < ei);
            bool valid_j = causal_w
                ? (val_j >= 0 && val_j <= j && val_j < width)
                : (val_j >= 0 && val_j < ej);
            if (valid_i && valid_j) {{
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


def source_3d_qk(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    volume = _volume(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int depth = query_shape[2];
const int height = query_shape[3];
const int width = query_shape[4];
const int dim = query_shape[5];
const int dilation_d = (int)dilation_param[0];
const int dilation_h = (int)dilation_param[1];
const int dilation_w = (int)dilation_param[2];
const bool causal_d = ((int)causal_param[0]) != 0;
const bool causal_h = ((int)causal_param[1]) != 0;
const bool causal_w = ((int)causal_param[2]) != 0;
const int K = {kernel_size};
const int NH = {nh};
const int L = {volume};

int z = gid.z % depth;
int bh = gid.z / depth;
int b = bh / heads;
int h = bh % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || z >= depth || i >= height || j >= width) return;

int nz = 0;
int ni = 0;
int nj = 0;
int ez = depth;
int ei = height;
int ej = width;
int pz = NH;
int pi = NH;
int pj = NH;
if (!causal_d) {{
    NATTEN_GET_WINDOW_START(nz, z, depth, K, NH, dilation_d);
    NATTEN_GET_WINDOW_END(ez, nz, depth, K, dilation_d);
    NATTEN_GET_PB_START(pz, z, depth, K, NH, dilation_d);
}}
if (!causal_h) {{
    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
    NATTEN_GET_PB_START(pi, i, height, K, NH, dilation_h);
}}
if (!causal_w) {{
    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
    NATTEN_GET_PB_START(pj, j, width, K, NH, dilation_w);
}}

int neighbor_idx = 0;
for (int kz = 0; kz < K; kz++) {{
    for (int ki = 0; ki < K; ki++) {{
        for (int kj = 0; kj < K; kj++) {{
            int key_z = (causal_d ? (z - (K - 1) * dilation_d) : nz) + kz * dilation_d;
            int key_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
            int key_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
            float score;
            bool valid_z = causal_d
                ? (key_z >= 0 && key_z <= z && key_z < depth)
                : (key_z >= 0 && key_z < ez);
            bool valid_i = causal_h
                ? (key_i >= 0 && key_i <= i && key_i < height)
                : (key_i >= 0 && key_i < ei);
            bool valid_j = causal_w
                ? (key_j >= 0 && key_j <= j && key_j < width)
                : (key_j >= 0 && key_j < ej);
            if (valid_z && valid_i && valid_j) {{
                float sum = 0.0f;
                for (int d = 0; d < dim; d++) {{
                    int q_idx = ((((b * heads + h) * depth + z) * height + i) * width + j) * dim + d;
                    int k_idx = ((((b * heads + h) * depth + key_z) * height + key_i) * width + key_j) * dim + d;
                    sum += query[q_idx] * key[k_idx];
                }}
                int rpb_size = (2 * K - 1);
                int rpb_idx = ((h * rpb_size + (pz + kz)) * rpb_size + (pi + ki)) * rpb_size + (pj + kj);
                score = sum + rpb[rpb_idx];
            }} else {{
                score = -INFINITY;
            }}
            int out_idx = ((((b * heads + h) * depth + z) * height + i) * width + j) * L + neighbor_idx;
            out[out_idx] = score;
            neighbor_idx++;
        }}
    }}
}}
"""


def source_3d_av(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    volume = _volume(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = attention_probs_shape[0];
const int heads = attention_probs_shape[1];
const int depth = attention_probs_shape[2];
const int height = attention_probs_shape[3];
const int width = attention_probs_shape[4];
const int dim = value_shape[5];
const int dilation_d = (int)dilation_param[0];
const int dilation_h = (int)dilation_param[1];
const int dilation_w = (int)dilation_param[2];
const bool causal_d = ((int)causal_param[0]) != 0;
const bool causal_h = ((int)causal_param[1]) != 0;
const bool causal_w = ((int)causal_param[2]) != 0;
const int K = {kernel_size};
const int NH = {nh};
const int L = {volume};

int z = gid.z % depth;
int bh = gid.z / depth;
int b = bh / heads;
int h = bh % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || z >= depth || i >= height || j >= width) return;

int nz = 0;
int ni = 0;
int nj = 0;
int ez = depth;
int ei = height;
int ej = width;
if (!causal_d) {{
    NATTEN_GET_WINDOW_START(nz, z, depth, K, NH, dilation_d);
    NATTEN_GET_WINDOW_END(ez, nz, depth, K, dilation_d);
}}
if (!causal_h) {{
    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
}}
if (!causal_w) {{
    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
}}

for (int d = 0; d < dim; d++) {{
    float sum = 0.0f;
    int neighbor_idx = 0;
    for (int kz = 0; kz < K; kz++) {{
        for (int ki = 0; ki < K; ki++) {{
            for (int kj = 0; kj < K; kj++) {{
                int val_z = (causal_d ? (z - (K - 1) * dilation_d) : nz) + kz * dilation_d;
                int val_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
                int val_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
                bool valid_z = causal_d
                    ? (val_z >= 0 && val_z <= z && val_z < depth)
                    : (val_z >= 0 && val_z < ez);
                bool valid_i = causal_h
                    ? (val_i >= 0 && val_i <= i && val_i < height)
                    : (val_i >= 0 && val_i < ei);
                bool valid_j = causal_w
                    ? (val_j >= 0 && val_j <= j && val_j < width)
                    : (val_j >= 0 && val_j < ej);
                if (valid_z && valid_i && valid_j) {{
                    int attn_idx = ((((b * heads + h) * depth + z) * height + i) * width + j) * L + neighbor_idx;
                    int val_idx = ((((b * heads + h) * depth + val_z) * height + val_i) * width + val_j) * dim + d;
                    sum += attention_probs[attn_idx] * value[val_idx];
                }}
                neighbor_idx++;
            }}
        }}
    }}
    int out_idx = ((((b * heads + h) * depth + z) * height + i) * width + j) * dim + d;
    out[out_idx] = sum;
}}
"""


def source_3d_fused(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    volume = _volume(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int depth = query_shape[2];
const int height = query_shape[3];
const int width = query_shape[4];
const int dim = query_shape[5];
const int stride_d = (int)stride_param[0];
const int stride_h = (int)stride_param[1];
const int stride_w = (int)stride_param[2];
const int out_depth = (depth + stride_d - 1) / stride_d;
const int out_height = (height + stride_h - 1) / stride_h;
const int out_width = (width + stride_w - 1) / stride_w;
const int dilation_d = (int)dilation_param[0];
const int dilation_h = (int)dilation_param[1];
const int dilation_w = (int)dilation_param[2];
const bool causal_d = ((int)causal_param[0]) != 0;
const bool causal_h = ((int)causal_param[1]) != 0;
const bool causal_w = ((int)causal_param[2]) != 0;
const float scale = scale_param[0];
const int K = {kernel_size};
const int NH = {nh};
const int L = {volume};

int out_z = gid.z % out_depth;
int bh = gid.z / out_depth;
int b = bh / heads;
int h = bh % heads;
int out_i = gid.y;
int out_j = gid.x;
if (b >= batch_size || h >= heads || out_z >= out_depth || out_i >= out_height || out_j >= out_width) return;

int z = out_z * stride_d;
int i = out_i * stride_h;
int j = out_j * stride_w;
if (z >= depth || i >= height || j >= width) return;

int nz = 0;
int ni = 0;
int nj = 0;
int ez = depth;
int ei = height;
int ej = width;
if (!causal_d) {{
    NATTEN_GET_WINDOW_START(nz, z, depth, K, NH, dilation_d);
    NATTEN_GET_WINDOW_END(ez, nz, depth, K, dilation_d);
}}
if (!causal_h) {{
    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
}}
if (!causal_w) {{
    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
}}

float logits[L];
float probs[L];
int key_z_arr[L];
int key_i_arr[L];
int key_j_arr[L];
bool key_valid_arr[L];

float max_logit = -INFINITY;
int neighbor_idx = 0;
for (int kz = 0; kz < K; kz++) {{
    for (int ki = 0; ki < K; ki++) {{
        for (int kj = 0; kj < K; kj++) {{
            int key_z = (causal_d ? (z - (K - 1) * dilation_d) : nz) + kz * dilation_d;
            int key_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
            int key_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
            bool valid_z = causal_d
                ? (key_z >= 0 && key_z <= z && key_z < depth)
                : (key_z >= 0 && key_z < ez);
            bool valid_i = causal_h
                ? (key_i >= 0 && key_i <= i && key_i < height)
                : (key_i >= 0 && key_i < ei);
            bool valid_j = causal_w
                ? (key_j >= 0 && key_j <= j && key_j < width)
                : (key_j >= 0 && key_j < ej);
            bool valid = valid_z && valid_i && valid_j;
            key_z_arr[neighbor_idx] = key_z;
            key_i_arr[neighbor_idx] = key_i;
            key_j_arr[neighbor_idx] = key_j;
            key_valid_arr[neighbor_idx] = valid;
            float score = -INFINITY;
            if (valid) {{
                float sum = 0.0f;
                for (int d = 0; d < dim; d++) {{
                    int q_idx = ((((b * heads + h) * depth + z) * height + i) * width + j) * dim + d;
                    int k_idx = ((((b * heads + h) * depth + key_z) * height + key_i) * width + key_j) * dim + d;
                    sum += query[q_idx] * key[k_idx];
                }}
                score = sum * scale;
            }}
            logits[neighbor_idx] = score;
            max_logit = fmax(max_logit, score);
            neighbor_idx++;
        }}
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
        if (key_valid_arr[n]) {{
            int key_z = key_z_arr[n];
            int key_i = key_i_arr[n];
            int key_j = key_j_arr[n];
            float w = probs[n] * inv_denom;
            int v_idx = ((((b * heads + h) * depth + key_z) * height + key_i) * width + key_j) * dim + d;
            out_sum += w * value[v_idx];
        }}
    }}
    int out_idx = ((((b * heads + h) * out_depth + out_z) * out_height + out_i) * out_width + out_j) * dim + d;
    out[out_idx] = out_sum;
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
const int stride = (int)stride_param[0];
const int out_length = (length + stride - 1) / stride;
const int dilation = (int)dilation_param[0];
const bool causal = ((int)causal_param[0]) != 0;
const float scale = scale_param[0];
const int K = {kernel_size};
const int NH = {nh};

int b = gid.z / heads;
int h = gid.z % heads;
int i_out = gid.x;
if (b >= batch_size || h >= heads || i_out >= out_length) return;
int i = i_out * stride;
if (i >= length) return;

int ni = 0;
int ei = length;
if (!causal) {{
    NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
    NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
}}

float logits[K];
float probs[K];
int key_pos[K];
bool key_valid[K];

float max_logit = -INFINITY;
for (int ki = 0; ki < K; ki++) {{
    int key_i = (causal ? (i - (K - 1) * dilation) : ni) + ki * dilation;
    bool valid = causal
        ? (key_i >= 0 && key_i <= i && key_i < length)
        : (key_i >= 0 && key_i < ei);
    key_pos[ki] = key_i;
    key_valid[ki] = valid;
    float score = -INFINITY;
    if (valid) {{
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
        if (key_valid[ki]) {{
            float w = probs[ki] * inv_denom;
            int v_idx = (((b * heads + h) * length + key_i) * dim + d);
            out_sum += w * value[v_idx];
        }}
    }}
    int out_idx = (((b * heads + h) * out_length + i_out) * dim + d);
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
const int stride_h = (int)stride_param[0];
const int stride_w = (int)stride_param[1];
const int out_height = (height + stride_h - 1) / stride_h;
const int out_width = (width + stride_w - 1) / stride_w;
const int dilation_h = (int)dilation_param[0];
const int dilation_w = (int)dilation_param[1];
const bool causal_h = ((int)causal_param[0]) != 0;
const bool causal_w = ((int)causal_param[1]) != 0;
const float scale = scale_param[0];
const int K = {kernel_size};
const int NH = {nh};
const int L = {area};

int b = gid.z / heads;
int h = gid.z % heads;
int i_out = gid.y;
int j_out = gid.x;
if (b >= batch_size || h >= heads || i_out >= out_height || j_out >= out_width) return;
int i = i_out * stride_h;
int j = j_out * stride_w;
if (i >= height || j >= width) return;

int ni = 0;
int nj = 0;
int ei = height;
int ej = width;
if (!causal_h) {{
    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
}}
if (!causal_w) {{
    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
}}

float logits[L];
float probs[L];
int key_i_arr[L];
int key_j_arr[L];
bool key_valid_arr[L];

float max_logit = -INFINITY;
int neighbor_idx = 0;
for (int ki = 0; ki < K; ki++) {{
    for (int kj = 0; kj < K; kj++) {{
        int key_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
        int key_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
        bool valid_i = causal_h
            ? (key_i >= 0 && key_i <= i && key_i < height)
            : (key_i >= 0 && key_i < ei);
        bool valid_j = causal_w
            ? (key_j >= 0 && key_j <= j && key_j < width)
            : (key_j >= 0 && key_j < ej);
        bool valid = valid_i && valid_j;
        key_i_arr[neighbor_idx] = key_i;
        key_j_arr[neighbor_idx] = key_j;
        key_valid_arr[neighbor_idx] = valid;
        float score = -INFINITY;
        if (valid) {{
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
        if (key_valid_arr[n]) {{
            float w = probs[n] * inv_denom;
            int v_idx = (((b * heads + h) * height + key_i) * width + key_j) * dim + d;
            out_sum += w * value[v_idx];
        }}
    }}
    int out_idx = (((b * heads + h) * out_height + i_out) * out_width + j_out) * dim + d;
    out[out_idx] = out_sum;
}}
"""


def source_1d_qk_backward_k(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int out_length = grad_attn_shape[2];
const int stride = (int)stride_param[0];
const int dilation = (int)dilation_param[0];
const bool causal = ((int)causal_param[0]) != 0;
const float scale = scale_param[0];
const int K = {kernel_size};
const int NH = {nh};

int b = gid.z / heads;
int h = gid.z % heads;
int key_i = gid.x;
if (b >= batch_size || h >= heads || key_i >= length) return;

for (int d = 0; d < dim; d++) {{
    float acc = 0.0f;
    for (int out_i = 0; out_i < out_length; out_i++) {{
        int i = out_i * stride;
        if (i >= length) continue;

        int ni = 0;
        int ei = length;
        if (!causal) {{
            NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        }}

        for (int ki = 0; ki < K; ki++) {{
            int candidate = (causal ? (i - (K - 1) * dilation) : ni) + ki * dilation;
            bool valid = causal
                ? (candidate >= 0 && candidate <= i && candidate < length)
                : (candidate >= 0 && candidate < ei);
            if (valid && candidate == key_i) {{
                int g_idx = (((b * heads + h) * out_length + out_i) * K + ki);
                int q_idx = (((b * heads + h) * length + i) * dim + d);
                acc += grad_attn[g_idx] * query[q_idx] * scale;
            }}
        }}
    }}
    int out_idx = (((b * heads + h) * length + key_i) * dim + d);
out[out_idx] = acc;
}}
"""


def source_1d_qk_backward_k_inverse(kernel_size: int) -> str:
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int length = query_shape[2];
const int dim = query_shape[3];
const int out_length = grad_attn_shape[2];
const float scale = scale_param[0];
const int K = {kernel_size};

int b = gid.z / heads;
int h = gid.z % heads;
int d = gid.x;
int key_i = gid.y;
if (b >= batch_size || h >= heads || key_i >= length || d >= dim) return;

int bh = b * heads + h;
int attn_bh_base = bh * out_length * K;
int query_bh_base = bh * length * dim;
int start = (int)inv_offsets[key_i];
int end = (int)inv_offsets[key_i + 1];

float acc = 0.0f;
for (int edge = start; edge < end; edge++) {{
    int g_idx = attn_bh_base + (int)inv_attn_base[edge];
    int q_idx = query_bh_base + (int)inv_query_base[edge] + d;
    acc += grad_attn[g_idx] * query[q_idx] * scale;
}}
int out_idx = (((b * heads + h) * length + key_i) * dim + d);
out[out_idx] = acc;
"""


def source_1d_qk_backward_q(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int length = key_shape[2];
const int dim = key_shape[3];
const int out_length = grad_attn_shape[2];
const int stride = (int)stride_param[0];
const int dilation = (int)dilation_param[0];
const bool causal = ((int)causal_param[0]) != 0;
const float scale = scale_param[0];
const int K = {kernel_size};
const int NH = {nh};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.x;
if (b >= batch_size || h >= heads || i >= length) return;

bool query_valid = (i % stride) == 0;
int out_i = i / stride;
if (!query_valid || out_i < 0 || out_i >= out_length) {{
    for (int d = 0; d < dim; d++) {{
        int out_idx = (((b * heads + h) * length + i) * dim + d);
        out[out_idx] = 0.0f;
    }}
    return;
}}

int ni = 0;
int ei = length;
if (!causal) {{
    NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
    NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
}}

for (int d = 0; d < dim; d++) {{
    float acc = 0.0f;
    for (int ki = 0; ki < K; ki++) {{
        int key_i = (causal ? (i - (K - 1) * dilation) : ni) + ki * dilation;
        bool valid = causal
            ? (key_i >= 0 && key_i <= i && key_i < length)
            : (key_i >= 0 && key_i < ei);
        if (valid) {{
            int g_idx = (((b * heads + h) * out_length + out_i) * K + ki);
            int k_idx = (((b * heads + h) * length + key_i) * dim + d);
            acc += grad_attn[g_idx] * key[k_idx] * scale;
        }}
    }}
    int out_idx = (((b * heads + h) * length + i) * dim + d);
    out[out_idx] = acc;
}}
"""


def source_1d_av_backward_v(kernel_size: int) -> str:
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_length = grad_out_shape[2];
const int dim = grad_out_shape[3];
const int length = (int)target_shape_param[0];
const int K = {kernel_size};

int b = gid.z / heads;
int h = gid.z % heads;
int d = gid.x;
int val_i = gid.y;
if (b >= batch_size || h >= heads || val_i >= length || d >= dim) return;
int bh = b * heads + h;
int attn_bh_base = bh * out_length * K;
int grad_bh_base = bh * out_length * dim;

int start = (int)inv_offsets[val_i];
int end = (int)inv_offsets[val_i + 1];

float acc = 0.0f;
for (int edge = start; edge < end; edge++) {{
    int a_idx = attn_bh_base + (int)inv_attn_base[edge];
    int g_idx = grad_bh_base + (int)inv_grad_base[edge] + d;
    acc += attention_probs[a_idx] * grad_out[g_idx];
}}
int out_idx = (((b * heads + h) * length + val_i) * dim + d);
out[out_idx] = acc;
"""


def source_1d_av_backward_v_vec4(kernel_size: int) -> str:
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_length = grad_out_shape[2];
const int dim = grad_out_shape[3];
const int length = (int)target_shape_param[0];
const int K = {kernel_size};

int b = gid.z / heads;
int h = gid.z % heads;
int d4 = gid.x;
int val_i = gid.y;
if (b >= batch_size || h >= heads || val_i >= length) return;
int d0 = d4 * 4;
if (d0 + 3 >= dim) return;
int bh = b * heads + h;
int attn_bh_base = bh * out_length * K;
int grad_bh_base = bh * out_length * dim;

int start = (int)inv_offsets[val_i];
int end = (int)inv_offsets[val_i + 1];

float acc0 = 0.0f;
float acc1 = 0.0f;
float acc2 = 0.0f;
float acc3 = 0.0f;
for (int edge = start; edge < end; edge++) {{
    int a_idx = attn_bh_base + (int)inv_attn_base[edge];
    int g_base = grad_bh_base + (int)inv_grad_base[edge] + d0;
    float a = attention_probs[a_idx];
    acc0 += a * grad_out[g_base];
    acc1 += a * grad_out[g_base + 1];
    acc2 += a * grad_out[g_base + 2];
    acc3 += a * grad_out[g_base + 3];
}}
int out_base = (((b * heads + h) * length + val_i) * dim + d0);
out[out_base] = acc0;
out[out_base + 1] = acc1;
out[out_base + 2] = acc2;
out[out_base + 3] = acc3;
"""


def source_1d_av_backward_attn(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_length = grad_out_shape[2];
const int dim = grad_out_shape[3];
const int length = (int)target_shape_param[0];
const int stride = (int)stride_param[0];
const int dilation = (int)dilation_param[0];
const bool causal = ((int)causal_param[0]) != 0;
const int K = {kernel_size};
const int NH = {nh};

int b = gid.z / heads;
int h = gid.z % heads;
int out_i = gid.x;
if (b >= batch_size || h >= heads || out_i >= out_length) return;

int i = out_i * stride;
if (i >= length) {{
    for (int ki = 0; ki < K; ki++) {{
        int out_idx = (((b * heads + h) * out_length + out_i) * K + ki);
        out[out_idx] = 0.0f;
    }}
    return;
}}

int ni = 0;
int ei = length;
if (!causal) {{
    NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
    NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
}}

for (int ki = 0; ki < K; ki++) {{
    int val_i = (causal ? (i - (K - 1) * dilation) : ni) + ki * dilation;
    bool valid = causal
        ? (val_i >= 0 && val_i <= i && val_i < length)
        : (val_i >= 0 && val_i < ei);
    float acc = 0.0f;
    if (valid) {{
        for (int d = 0; d < dim; d++) {{
            int g_idx = (((b * heads + h) * out_length + out_i) * dim + d);
            int v_idx = (((b * heads + h) * length + val_i) * dim + d);
            acc += grad_out[g_idx] * value[v_idx];
        }}
    }}
    int out_idx = (((b * heads + h) * out_length + out_i) * K + ki);
    out[out_idx] = acc;
}}
"""


def source_1d_av_backward_fused(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_length = grad_out_shape[2];
const int dim = grad_out_shape[3];
const int length = (int)target_shape_param[0];
const int stride = (int)stride_param[0];
const int dilation = (int)dilation_param[0];
const bool causal = ((int)causal_param[0]) != 0;
const int K = {kernel_size};
const int NH = {nh};

int b = gid.z / heads;
int h = gid.z % heads;
int x = gid.x;
if (b >= batch_size || h >= heads) return;

if (x < out_length) {{
    int out_i = x;
    int i = out_i * stride;
    if (i >= length) {{
        for (int ki = 0; ki < K; ki++) {{
            int out_idx = (((b * heads + h) * out_length + out_i) * K + ki);
            grad_attn[out_idx] = 0.0f;
        }}
    }} else {{
        int ni = 0;
        int ei = length;
        if (!causal) {{
            NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
            NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
        }}

        for (int ki = 0; ki < K; ki++) {{
            int val_i = (causal ? (i - (K - 1) * dilation) : ni) + ki * dilation;
            bool valid = causal
                ? (val_i >= 0 && val_i <= i && val_i < length)
                : (val_i >= 0 && val_i < ei);
            float acc = 0.0f;
            if (valid) {{
                for (int d = 0; d < dim; d++) {{
                    int g_idx = (((b * heads + h) * out_length + out_i) * dim + d);
                    int v_idx = (((b * heads + h) * length + val_i) * dim + d);
                    acc += grad_out[g_idx] * value[v_idx];
                }}
            }}
            int out_idx = (((b * heads + h) * out_length + out_i) * K + ki);
            grad_attn[out_idx] = acc;
        }}
    }}
}}

if (x < length) {{
    int val_i = x;
    for (int d = 0; d < dim; d++) {{
        float acc = 0.0f;
        for (int out_i = 0; out_i < out_length; out_i++) {{
            int i = out_i * stride;
            if (i >= length) continue;

            int ni = 0;
            int ei = length;
            if (!causal) {{
                NATTEN_GET_WINDOW_START(ni, i, length, K, NH, dilation);
                NATTEN_GET_WINDOW_END(ei, ni, length, K, dilation);
            }}

            for (int ki = 0; ki < K; ki++) {{
                int candidate = (causal ? (i - (K - 1) * dilation) : ni) + ki * dilation;
                bool valid = causal
                    ? (candidate >= 0 && candidate <= i && candidate < length)
                    : (candidate >= 0 && candidate < ei);
                if (valid && candidate == val_i) {{
                    int a_idx = (((b * heads + h) * out_length + out_i) * K + ki);
                    int g_idx = (((b * heads + h) * out_length + out_i) * dim + d);
                    acc += attention_probs[a_idx] * grad_out[g_idx];
                }}
            }}
        }}
        int out_idx = (((b * heads + h) * length + val_i) * dim + d);
        grad_v[out_idx] = acc;
    }}
}}
"""


def source_2d_qk_backward_k(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    area = _area(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int height = query_shape[2];
const int width = query_shape[3];
const int dim = query_shape[4];
const int out_height = grad_attn_shape[2];
const int out_width = grad_attn_shape[3];
const int stride_h = (int)stride_param[0];
const int stride_w = (int)stride_param[1];
const int dilation_h = (int)dilation_param[0];
const int dilation_w = (int)dilation_param[1];
const bool causal_h = ((int)causal_param[0]) != 0;
const bool causal_w = ((int)causal_param[1]) != 0;
const float scale = scale_param[0];
const int K = {kernel_size};
const int NH = {nh};
const int L = {area};

int b = gid.z / heads;
int h = gid.z % heads;
int key_i = gid.y;
int key_j = gid.x;
if (b >= batch_size || h >= heads || key_i >= height || key_j >= width) return;

for (int d = 0; d < dim; d++) {{
    float acc = 0.0f;
    for (int out_i = 0; out_i < out_height; out_i++) {{
        int i = out_i * stride_h;
        if (i >= height) continue;
        for (int out_j = 0; out_j < out_width; out_j++) {{
            int j = out_j * stride_w;
            if (j >= width) continue;

            int ni = 0, nj = 0, ei = height, ej = width;
            if (!causal_h) {{
                NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
                NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
            }}
            if (!causal_w) {{
                NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
                NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
            }}

            int neighbor_idx = 0;
            for (int ki = 0; ki < K; ki++) {{
                for (int kj = 0; kj < K; kj++) {{
                    int cand_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
                    int cand_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
                    bool valid_i = causal_h
                        ? (cand_i >= 0 && cand_i <= i && cand_i < height)
                        : (cand_i >= 0 && cand_i < ei);
                    bool valid_j = causal_w
                        ? (cand_j >= 0 && cand_j <= j && cand_j < width)
                        : (cand_j >= 0 && cand_j < ej);
                    if (valid_i && valid_j && cand_i == key_i && cand_j == key_j) {{
                        int g_idx = ((((b * heads + h) * out_height + out_i) * out_width + out_j) * L + neighbor_idx);
                        int q_idx = ((((b * heads + h) * height + i) * width + j) * dim + d);
                        acc += grad_attn[g_idx] * query[q_idx] * scale;
                    }}
                    neighbor_idx++;
                }}
            }}
        }}
    }}
    int out_idx = ((((b * heads + h) * height + key_i) * width + key_j) * dim + d);
    out[out_idx] = acc;
}}
"""


def source_2d_qk_backward_q(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    area = _area(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int height = key_shape[2];
const int width = key_shape[3];
const int dim = key_shape[4];
const int out_height = grad_attn_shape[2];
const int out_width = grad_attn_shape[3];
const int stride_h = (int)stride_param[0];
const int stride_w = (int)stride_param[1];
const int dilation_h = (int)dilation_param[0];
const int dilation_w = (int)dilation_param[1];
const bool causal_h = ((int)causal_param[0]) != 0;
const bool causal_w = ((int)causal_param[1]) != 0;
const float scale = scale_param[0];
const int K = {kernel_size};
const int NH = {nh};
const int L = {area};

int b = gid.z / heads;
int h = gid.z % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || i >= height || j >= width) return;

bool query_valid = ((i % stride_h) == 0) && ((j % stride_w) == 0);
int out_i = i / stride_h;
int out_j = j / stride_w;
if (!query_valid || out_i < 0 || out_j < 0 || out_i >= out_height || out_j >= out_width) {{
    for (int d = 0; d < dim; d++) {{
        int out_idx = ((((b * heads + h) * height + i) * width + j) * dim + d);
        out[out_idx] = 0.0f;
    }}
    return;
}}

int ni = 0, nj = 0, ei = height, ej = width;
if (!causal_h) {{
    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
}}
if (!causal_w) {{
    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
}}

for (int d = 0; d < dim; d++) {{
    float acc = 0.0f;
    int neighbor_idx = 0;
    for (int ki = 0; ki < K; ki++) {{
        for (int kj = 0; kj < K; kj++) {{
            int key_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
            int key_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
            bool valid_i = causal_h
                ? (key_i >= 0 && key_i <= i && key_i < height)
                : (key_i >= 0 && key_i < ei);
            bool valid_j = causal_w
                ? (key_j >= 0 && key_j <= j && key_j < width)
                : (key_j >= 0 && key_j < ej);
            if (valid_i && valid_j) {{
                int g_idx = ((((b * heads + h) * out_height + out_i) * out_width + out_j) * L + neighbor_idx);
                int k_idx = ((((b * heads + h) * height + key_i) * width + key_j) * dim + d);
                acc += grad_attn[g_idx] * key[k_idx] * scale;
            }}
            neighbor_idx++;
        }}
    }}
    int out_idx = ((((b * heads + h) * height + i) * width + j) * dim + d);
    out[out_idx] = acc;
}}
"""


def source_2d_av_backward_v(kernel_size: int) -> str:
    area = _area(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_height = grad_out_shape[2];
const int out_width = grad_out_shape[3];
const int dim = grad_out_shape[4];
const int height = (int)target_shape_param[0];
const int width = (int)target_shape_param[1];
const int K = {kernel_size};
const int L = {area};
const int out_count = out_height * out_width;
const int value_count = height * width;

int b = gid.z / heads;
int h = gid.z % heads;
int d = gid.x;
int val_linear = gid.y;
if (b >= batch_size || h >= heads || val_linear >= value_count || d >= dim) return;
int val_i = val_linear / width;
int val_j = val_linear - val_i * width;
int bh = b * heads + h;
int attn_bh_base = bh * out_count * L;
int grad_bh_base = bh * out_count * dim;
int start = (int)inv_offsets[val_linear];
int end = (int)inv_offsets[val_linear + 1];

float acc = 0.0f;
for (int edge = start; edge < end; edge++) {{
    int a_idx = attn_bh_base + (int)inv_attn_base[edge];
    int g_idx = grad_bh_base + (int)inv_grad_base[edge] + d;
    acc += attention_probs[a_idx] * grad_out[g_idx];
}}
int out_idx = ((((b * heads + h) * height + val_i) * width + val_j) * dim + d);
out[out_idx] = acc;
"""


def source_2d_av_backward_v_vec4(kernel_size: int) -> str:
    area = _area(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_height = grad_out_shape[2];
const int out_width = grad_out_shape[3];
const int dim = grad_out_shape[4];
const int height = (int)target_shape_param[0];
const int width = (int)target_shape_param[1];
const int L = {area};
const int out_count = out_height * out_width;
const int value_count = height * width;

int b = gid.z / heads;
int h = gid.z % heads;
int d4 = gid.x;
int val_linear = gid.y;
if (b >= batch_size || h >= heads || val_linear >= value_count) return;
int d0 = d4 * 4;
if (d0 + 3 >= dim) return;
int val_i = val_linear / width;
int val_j = val_linear - val_i * width;
int bh = b * heads + h;
int attn_bh_base = bh * out_count * L;
int grad_bh_base = bh * out_count * dim;
int start = (int)inv_offsets[val_linear];
int end = (int)inv_offsets[val_linear + 1];

float acc0 = 0.0f;
float acc1 = 0.0f;
float acc2 = 0.0f;
float acc3 = 0.0f;
for (int edge = start; edge < end; edge++) {{
    int a_idx = attn_bh_base + (int)inv_attn_base[edge];
    int g_base = grad_bh_base + (int)inv_grad_base[edge] + d0;
    float a = attention_probs[a_idx];
    acc0 += a * grad_out[g_base];
    acc1 += a * grad_out[g_base + 1];
    acc2 += a * grad_out[g_base + 2];
    acc3 += a * grad_out[g_base + 3];
}}
int out_base = ((((b * heads + h) * height + val_i) * width + val_j) * dim + d0);
out[out_base] = acc0;
out[out_base + 1] = acc1;
out[out_base + 2] = acc2;
out[out_base + 3] = acc3;
"""


def source_2d_av_backward_attn(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    area = _area(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_height = grad_out_shape[2];
const int out_width = grad_out_shape[3];
const int dim = grad_out_shape[4];
const int height = (int)target_shape_param[0];
const int width = (int)target_shape_param[1];
const int stride_h = (int)stride_param[0];
const int stride_w = (int)stride_param[1];
const int dilation_h = (int)dilation_param[0];
const int dilation_w = (int)dilation_param[1];
const bool causal_h = ((int)causal_param[0]) != 0;
const bool causal_w = ((int)causal_param[1]) != 0;
const int K = {kernel_size};
const int NH = {nh};
const int L = {area};

int b = gid.z / heads;
int h = gid.z % heads;
int out_i = gid.y;
int out_j = gid.x;
if (b >= batch_size || h >= heads || out_i >= out_height || out_j >= out_width) return;

int i = out_i * stride_h;
int j = out_j * stride_w;
if (i >= height || j >= width) {{
    for (int n = 0; n < L; n++) {{
        int out_idx = ((((b * heads + h) * out_height + out_i) * out_width + out_j) * L + n);
        out[out_idx] = 0.0f;
    }}
    return;
}}

int ni = 0, nj = 0, ei = height, ej = width;
if (!causal_h) {{
    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
}}
if (!causal_w) {{
    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
}}

int neighbor_idx = 0;
for (int ki = 0; ki < K; ki++) {{
    for (int kj = 0; kj < K; kj++) {{
        int val_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
        int val_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
        bool valid_i = causal_h
            ? (val_i >= 0 && val_i <= i && val_i < height)
            : (val_i >= 0 && val_i < ei);
        bool valid_j = causal_w
            ? (val_j >= 0 && val_j <= j && val_j < width)
            : (val_j >= 0 && val_j < ej);
        float acc = 0.0f;
        if (valid_i && valid_j) {{
            for (int d = 0; d < dim; d++) {{
                int g_idx = ((((b * heads + h) * out_height + out_i) * out_width + out_j) * dim + d);
                int v_idx = ((((b * heads + h) * height + val_i) * width + val_j) * dim + d);
                acc += grad_out[g_idx] * value[v_idx];
            }}
        }}
        int out_idx = ((((b * heads + h) * out_height + out_i) * out_width + out_j) * L + neighbor_idx);
        out[out_idx] = acc;
        neighbor_idx++;
    }}
}}
"""


def source_2d_av_backward_fused(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    area = _area(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_height = grad_out_shape[2];
const int out_width = grad_out_shape[3];
const int dim = grad_out_shape[4];
const int height = (int)target_shape_param[0];
const int width = (int)target_shape_param[1];
const int stride_h = (int)stride_param[0];
const int stride_w = (int)stride_param[1];
const int dilation_h = (int)dilation_param[0];
const int dilation_w = (int)dilation_param[1];
const bool causal_h = ((int)causal_param[0]) != 0;
const bool causal_w = ((int)causal_param[1]) != 0;
const int K = {kernel_size};
const int NH = {nh};
const int L = {area};

int b = gid.z / heads;
int h = gid.z % heads;
int y = gid.y;
int x = gid.x;
if (b >= batch_size || h >= heads) return;

if (y < out_height && x < out_width) {{
    int out_i = y;
    int out_j = x;
    int i = out_i * stride_h;
    int j = out_j * stride_w;
    if (i >= height || j >= width) {{
        for (int n = 0; n < L; n++) {{
            int out_idx = ((((b * heads + h) * out_height + out_i) * out_width + out_j) * L + n);
            grad_attn[out_idx] = 0.0f;
        }}
    }} else {{
        int ni = 0, nj = 0, ei = height, ej = width;
        if (!causal_h) {{
            NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
            NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
        }}
        if (!causal_w) {{
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
        }}

        int neighbor_idx = 0;
        for (int ki = 0; ki < K; ki++) {{
            for (int kj = 0; kj < K; kj++) {{
                int val_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
                int val_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
                bool valid_i = causal_h
                    ? (val_i >= 0 && val_i <= i && val_i < height)
                    : (val_i >= 0 && val_i < ei);
                bool valid_j = causal_w
                    ? (val_j >= 0 && val_j <= j && val_j < width)
                    : (val_j >= 0 && val_j < ej);
                float acc = 0.0f;
                if (valid_i && valid_j) {{
                    for (int d = 0; d < dim; d++) {{
                        int g_idx = ((((b * heads + h) * out_height + out_i) * out_width + out_j) * dim + d);
                        int v_idx = ((((b * heads + h) * height + val_i) * width + val_j) * dim + d);
                        acc += grad_out[g_idx] * value[v_idx];
                    }}
                }}
                int out_idx = ((((b * heads + h) * out_height + out_i) * out_width + out_j) * L + neighbor_idx);
                grad_attn[out_idx] = acc;
                neighbor_idx++;
            }}
        }}
    }}
}}

if (y < height && x < width) {{
    int val_i = y;
    int val_j = x;
    for (int d = 0; d < dim; d++) {{
        float acc = 0.0f;
        for (int out_i = 0; out_i < out_height; out_i++) {{
            int i = out_i * stride_h;
            if (i >= height) continue;
            for (int out_j = 0; out_j < out_width; out_j++) {{
                int j = out_j * stride_w;
                if (j >= width) continue;

                int ni = 0, nj = 0, ei = height, ej = width;
                if (!causal_h) {{
                    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
                    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
                }}
                if (!causal_w) {{
                    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
                    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
                }}

                int neighbor_idx = 0;
                for (int ki = 0; ki < K; ki++) {{
                    for (int kj = 0; kj < K; kj++) {{
                        int cand_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
                        int cand_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
                        bool valid_i = causal_h
                            ? (cand_i >= 0 && cand_i <= i && cand_i < height)
                            : (cand_i >= 0 && cand_i < ei);
                        bool valid_j = causal_w
                            ? (cand_j >= 0 && cand_j <= j && cand_j < width)
                            : (cand_j >= 0 && cand_j < ej);
                        if (valid_i && valid_j && cand_i == val_i && cand_j == val_j) {{
                            int a_idx = ((((b * heads + h) * out_height + out_i) * out_width + out_j) * L + neighbor_idx);
                            int g_idx = ((((b * heads + h) * out_height + out_i) * out_width + out_j) * dim + d);
                            acc += attention_probs[a_idx] * grad_out[g_idx];
                        }}
                        neighbor_idx++;
                    }}
                }}
            }}
        }}
        int out_idx = ((((b * heads + h) * height + val_i) * width + val_j) * dim + d);
        grad_v[out_idx] = acc;
    }}
}}
"""


def source_3d_qk_backward_k(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    volume = _volume(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = query_shape[0];
const int heads = query_shape[1];
const int depth = query_shape[2];
const int height = query_shape[3];
const int width = query_shape[4];
const int dim = query_shape[5];
const int out_depth = grad_attn_shape[2];
const int out_height = grad_attn_shape[3];
const int out_width = grad_attn_shape[4];
const int stride_d = (int)stride_param[0];
const int stride_h = (int)stride_param[1];
const int stride_w = (int)stride_param[2];
const int dilation_d = (int)dilation_param[0];
const int dilation_h = (int)dilation_param[1];
const int dilation_w = (int)dilation_param[2];
const bool causal_d = ((int)causal_param[0]) != 0;
const bool causal_h = ((int)causal_param[1]) != 0;
const bool causal_w = ((int)causal_param[2]) != 0;
const float scale = scale_param[0];
const int K = {kernel_size};
const int NH = {nh};
const int L = {volume};

int key_z = gid.z % depth;
int bh = gid.z / depth;
int b = bh / heads;
int h = bh % heads;
int key_i = gid.y;
int key_j = gid.x;
if (b >= batch_size || h >= heads || key_z >= depth || key_i >= height || key_j >= width) return;

for (int d = 0; d < dim; d++) {{
    float acc = 0.0f;
    for (int out_z = 0; out_z < out_depth; out_z++) {{
        int z = out_z * stride_d;
        if (z >= depth) continue;
        for (int out_i = 0; out_i < out_height; out_i++) {{
            int i = out_i * stride_h;
            if (i >= height) continue;
            for (int out_j = 0; out_j < out_width; out_j++) {{
                int j = out_j * stride_w;
                if (j >= width) continue;

                int nz = 0, ni = 0, nj = 0, ez = depth, ei = height, ej = width;
                if (!causal_d) {{
                    NATTEN_GET_WINDOW_START(nz, z, depth, K, NH, dilation_d);
                    NATTEN_GET_WINDOW_END(ez, nz, depth, K, dilation_d);
                }}
                if (!causal_h) {{
                    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
                    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
                }}
                if (!causal_w) {{
                    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
                    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
                }}

                int neighbor_idx = 0;
                for (int kz = 0; kz < K; kz++) {{
                    for (int ki = 0; ki < K; ki++) {{
                        for (int kj = 0; kj < K; kj++) {{
                            int cand_z = (causal_d ? (z - (K - 1) * dilation_d) : nz) + kz * dilation_d;
                            int cand_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
                            int cand_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
                            bool valid_z = causal_d
                                ? (cand_z >= 0 && cand_z <= z && cand_z < depth)
                                : (cand_z >= 0 && cand_z < ez);
                            bool valid_i = causal_h
                                ? (cand_i >= 0 && cand_i <= i && cand_i < height)
                                : (cand_i >= 0 && cand_i < ei);
                            bool valid_j = causal_w
                                ? (cand_j >= 0 && cand_j <= j && cand_j < width)
                                : (cand_j >= 0 && cand_j < ej);
                            if (valid_z && valid_i && valid_j
                                && cand_z == key_z && cand_i == key_i && cand_j == key_j) {{
                                int g_idx = (((((b * heads + h) * out_depth + out_z) * out_height + out_i) * out_width + out_j) * L + neighbor_idx);
                                int q_idx = (((((b * heads + h) * depth + z) * height + i) * width + j) * dim + d);
                                acc += grad_attn[g_idx] * query[q_idx] * scale;
                            }}
                            neighbor_idx++;
                        }}
                    }}
                }}
            }}
        }}
    }}
    int out_idx = (((((b * heads + h) * depth + key_z) * height + key_i) * width + key_j) * dim + d);
    out[out_idx] = acc;
}}
"""


def source_3d_qk_backward_q(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    volume = _volume(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = key_shape[0];
const int heads = key_shape[1];
const int depth = key_shape[2];
const int height = key_shape[3];
const int width = key_shape[4];
const int dim = key_shape[5];
const int out_depth = grad_attn_shape[2];
const int out_height = grad_attn_shape[3];
const int out_width = grad_attn_shape[4];
const int stride_d = (int)stride_param[0];
const int stride_h = (int)stride_param[1];
const int stride_w = (int)stride_param[2];
const int dilation_d = (int)dilation_param[0];
const int dilation_h = (int)dilation_param[1];
const int dilation_w = (int)dilation_param[2];
const bool causal_d = ((int)causal_param[0]) != 0;
const bool causal_h = ((int)causal_param[1]) != 0;
const bool causal_w = ((int)causal_param[2]) != 0;
const float scale = scale_param[0];
const int K = {kernel_size};
const int NH = {nh};
const int L = {volume};

int z = gid.z % depth;
int bh = gid.z / depth;
int b = bh / heads;
int h = bh % heads;
int i = gid.y;
int j = gid.x;
if (b >= batch_size || h >= heads || z >= depth || i >= height || j >= width) return;

bool query_valid = ((z % stride_d) == 0) && ((i % stride_h) == 0) && ((j % stride_w) == 0);
int out_z = z / stride_d;
int out_i = i / stride_h;
int out_j = j / stride_w;
if (!query_valid || out_z < 0 || out_i < 0 || out_j < 0 || out_z >= out_depth || out_i >= out_height || out_j >= out_width) {{
    for (int d = 0; d < dim; d++) {{
        int out_idx = (((((b * heads + h) * depth + z) * height + i) * width + j) * dim + d);
        out[out_idx] = 0.0f;
    }}
    return;
}}

int nz = 0, ni = 0, nj = 0, ez = depth, ei = height, ej = width;
if (!causal_d) {{
    NATTEN_GET_WINDOW_START(nz, z, depth, K, NH, dilation_d);
    NATTEN_GET_WINDOW_END(ez, nz, depth, K, dilation_d);
}}
if (!causal_h) {{
    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
}}
if (!causal_w) {{
    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
}}

for (int d = 0; d < dim; d++) {{
    float acc = 0.0f;
    int neighbor_idx = 0;
    for (int kz = 0; kz < K; kz++) {{
        for (int ki = 0; ki < K; ki++) {{
            for (int kj = 0; kj < K; kj++) {{
                int key_z = (causal_d ? (z - (K - 1) * dilation_d) : nz) + kz * dilation_d;
                int key_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
                int key_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
                bool valid_z = causal_d
                    ? (key_z >= 0 && key_z <= z && key_z < depth)
                    : (key_z >= 0 && key_z < ez);
                bool valid_i = causal_h
                    ? (key_i >= 0 && key_i <= i && key_i < height)
                    : (key_i >= 0 && key_i < ei);
                bool valid_j = causal_w
                    ? (key_j >= 0 && key_j <= j && key_j < width)
                    : (key_j >= 0 && key_j < ej);
                if (valid_z && valid_i && valid_j) {{
                    int g_idx = (((((b * heads + h) * out_depth + out_z) * out_height + out_i) * out_width + out_j) * L + neighbor_idx);
                    int k_idx = (((((b * heads + h) * depth + key_z) * height + key_i) * width + key_j) * dim + d);
                    acc += grad_attn[g_idx] * key[k_idx] * scale;
                }}
                neighbor_idx++;
            }}
        }}
    }}
    int out_idx = (((((b * heads + h) * depth + z) * height + i) * width + j) * dim + d);
    out[out_idx] = acc;
}}
"""


def source_3d_av_backward_v(kernel_size: int) -> str:
    volume = _volume(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_depth = grad_out_shape[2];
const int out_height = grad_out_shape[3];
const int out_width = grad_out_shape[4];
const int dim = grad_out_shape[5];
const int depth = (int)target_shape_param[0];
const int height = (int)target_shape_param[1];
const int width = (int)target_shape_param[2];
const int L = {volume};
const int out_count = out_depth * out_height * out_width;
const int hw = height * width;
const int value_count = depth * hw;

int bh = gid.z;
int b = bh / heads;
int h = bh % heads;
int d = gid.x;
int val_linear = gid.y;
if (b >= batch_size || h >= heads || val_linear >= value_count || d >= dim) return;
int val_z = val_linear / hw;
int rem = val_linear - val_z * hw;
int val_i = rem / width;
int val_j = rem - val_i * width;
int bh_idx = b * heads + h;
int attn_bh_base = bh_idx * out_count * L;
int grad_bh_base = bh_idx * out_count * dim;
int start = (int)inv_offsets[val_linear];
int end = (int)inv_offsets[val_linear + 1];

float acc = 0.0f;
for (int edge = start; edge < end; edge++) {{
    int a_idx = attn_bh_base + (int)inv_attn_base[edge];
    int g_idx = grad_bh_base + (int)inv_grad_base[edge] + d;
    acc += attention_probs[a_idx] * grad_out[g_idx];
}}
int out_idx = (((((b * heads + h) * depth + val_z) * height + val_i) * width + val_j) * dim + d);
out[out_idx] = acc;
"""


def source_3d_av_backward_v_vec4(kernel_size: int) -> str:
    volume = _volume(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_depth = grad_out_shape[2];
const int out_height = grad_out_shape[3];
const int out_width = grad_out_shape[4];
const int dim = grad_out_shape[5];
const int depth = (int)target_shape_param[0];
const int height = (int)target_shape_param[1];
const int width = (int)target_shape_param[2];
const int L = {volume};
const int out_count = out_depth * out_height * out_width;
const int hw = height * width;
const int value_count = depth * hw;

int bh = gid.z;
int b = bh / heads;
int h = bh % heads;
int d4 = gid.x;
int val_linear = gid.y;
if (b >= batch_size || h >= heads || val_linear >= value_count) return;
int d0 = d4 * 4;
if (d0 + 3 >= dim) return;
int val_z = val_linear / hw;
int rem = val_linear - val_z * hw;
int val_i = rem / width;
int val_j = rem - val_i * width;
int bh_idx = b * heads + h;
int attn_bh_base = bh_idx * out_count * L;
int grad_bh_base = bh_idx * out_count * dim;
int start = (int)inv_offsets[val_linear];
int end = (int)inv_offsets[val_linear + 1];

float acc0 = 0.0f;
float acc1 = 0.0f;
float acc2 = 0.0f;
float acc3 = 0.0f;
for (int edge = start; edge < end; edge++) {{
    int a_idx = attn_bh_base + (int)inv_attn_base[edge];
    int g_base = grad_bh_base + (int)inv_grad_base[edge] + d0;
    float a = attention_probs[a_idx];
    acc0 += a * grad_out[g_base];
    acc1 += a * grad_out[g_base + 1];
    acc2 += a * grad_out[g_base + 2];
    acc3 += a * grad_out[g_base + 3];
}}
int out_base = (((((b * heads + h) * depth + val_z) * height + val_i) * width + val_j) * dim + d0);
out[out_base] = acc0;
out[out_base + 1] = acc1;
out[out_base + 2] = acc2;
out[out_base + 3] = acc3;
"""


def source_3d_av_backward_attn(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    volume = _volume(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_depth = grad_out_shape[2];
const int out_height = grad_out_shape[3];
const int out_width = grad_out_shape[4];
const int dim = grad_out_shape[5];
const int depth = (int)target_shape_param[0];
const int height = (int)target_shape_param[1];
const int width = (int)target_shape_param[2];
const int stride_d = (int)stride_param[0];
const int stride_h = (int)stride_param[1];
const int stride_w = (int)stride_param[2];
const int dilation_d = (int)dilation_param[0];
const int dilation_h = (int)dilation_param[1];
const int dilation_w = (int)dilation_param[2];
const bool causal_d = ((int)causal_param[0]) != 0;
const bool causal_h = ((int)causal_param[1]) != 0;
const bool causal_w = ((int)causal_param[2]) != 0;
const int K = {kernel_size};
const int NH = {nh};
const int L = {volume};

int out_z = gid.z % out_depth;
int bh = gid.z / out_depth;
int b = bh / heads;
int h = bh % heads;
int out_i = gid.y;
int out_j = gid.x;
if (b >= batch_size || h >= heads || out_z >= out_depth || out_i >= out_height || out_j >= out_width) return;

int z = out_z * stride_d;
int i = out_i * stride_h;
int j = out_j * stride_w;
if (z >= depth || i >= height || j >= width) {{
    for (int n = 0; n < L; n++) {{
        int out_idx = (((((b * heads + h) * out_depth + out_z) * out_height + out_i) * out_width + out_j) * L + n);
        out[out_idx] = 0.0f;
    }}
    return;
}}

int nz = 0, ni = 0, nj = 0, ez = depth, ei = height, ej = width;
if (!causal_d) {{
    NATTEN_GET_WINDOW_START(nz, z, depth, K, NH, dilation_d);
    NATTEN_GET_WINDOW_END(ez, nz, depth, K, dilation_d);
}}
if (!causal_h) {{
    NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
    NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
}}
if (!causal_w) {{
    NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
    NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
}}

int neighbor_idx = 0;
for (int kz = 0; kz < K; kz++) {{
    for (int ki = 0; ki < K; ki++) {{
        for (int kj = 0; kj < K; kj++) {{
            int val_z = (causal_d ? (z - (K - 1) * dilation_d) : nz) + kz * dilation_d;
            int val_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
            int val_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
            bool valid_z = causal_d
                ? (val_z >= 0 && val_z <= z && val_z < depth)
                : (val_z >= 0 && val_z < ez);
            bool valid_i = causal_h
                ? (val_i >= 0 && val_i <= i && val_i < height)
                : (val_i >= 0 && val_i < ei);
            bool valid_j = causal_w
                ? (val_j >= 0 && val_j <= j && val_j < width)
                : (val_j >= 0 && val_j < ej);
            float acc = 0.0f;
            if (valid_z && valid_i && valid_j) {{
                for (int d = 0; d < dim; d++) {{
                    int g_idx = (((((b * heads + h) * out_depth + out_z) * out_height + out_i) * out_width + out_j) * dim + d);
                    int v_idx = (((((b * heads + h) * depth + val_z) * height + val_i) * width + val_j) * dim + d);
                    acc += grad_out[g_idx] * value[v_idx];
                }}
            }}
            int out_idx = (((((b * heads + h) * out_depth + out_z) * out_height + out_i) * out_width + out_j) * L + neighbor_idx);
            out[out_idx] = acc;
            neighbor_idx++;
        }}
    }}
}}
"""


def source_3d_av_backward_fused(kernel_size: int) -> str:
    nh = _nh(kernel_size)
    volume = _volume(kernel_size)
    return _HELPERS + f"""
uint3 gid = thread_position_in_grid;
const int batch_size = grad_out_shape[0];
const int heads = grad_out_shape[1];
const int out_depth = grad_out_shape[2];
const int out_height = grad_out_shape[3];
const int out_width = grad_out_shape[4];
const int dim = grad_out_shape[5];
const int depth = (int)target_shape_param[0];
const int height = (int)target_shape_param[1];
const int width = (int)target_shape_param[2];
const int stride_d = (int)stride_param[0];
const int stride_h = (int)stride_param[1];
const int stride_w = (int)stride_param[2];
const int dilation_d = (int)dilation_param[0];
const int dilation_h = (int)dilation_param[1];
const int dilation_w = (int)dilation_param[2];
const bool causal_d = ((int)causal_param[0]) != 0;
const bool causal_h = ((int)causal_param[1]) != 0;
const bool causal_w = ((int)causal_param[2]) != 0;
const int K = {kernel_size};
const int NH = {nh};
const int L = {volume};

const int max_depth = depth > out_depth ? depth : out_depth;
int plane = gid.z % max_depth;
int bh = gid.z / max_depth;
int b = bh / heads;
int h = bh % heads;
int y = gid.y;
int x = gid.x;
if (b >= batch_size || h >= heads) return;

if (plane < out_depth && y < out_height && x < out_width) {{
    int out_z = plane;
    int out_i = y;
    int out_j = x;
    int z = out_z * stride_d;
    int i = out_i * stride_h;
    int j = out_j * stride_w;
    if (z >= depth || i >= height || j >= width) {{
        for (int n = 0; n < L; n++) {{
            int out_idx = (((((b * heads + h) * out_depth + out_z) * out_height + out_i) * out_width + out_j) * L + n);
            grad_attn[out_idx] = 0.0f;
        }}
    }} else {{
        int nz = 0, ni = 0, nj = 0, ez = depth, ei = height, ej = width;
        if (!causal_d) {{
            NATTEN_GET_WINDOW_START(nz, z, depth, K, NH, dilation_d);
            NATTEN_GET_WINDOW_END(ez, nz, depth, K, dilation_d);
        }}
        if (!causal_h) {{
            NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
            NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
        }}
        if (!causal_w) {{
            NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
            NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
        }}

        int neighbor_idx = 0;
        for (int kz = 0; kz < K; kz++) {{
            for (int ki = 0; ki < K; ki++) {{
                for (int kj = 0; kj < K; kj++) {{
                    int val_z = (causal_d ? (z - (K - 1) * dilation_d) : nz) + kz * dilation_d;
                    int val_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
                    int val_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
                    bool valid_z = causal_d
                        ? (val_z >= 0 && val_z <= z && val_z < depth)
                        : (val_z >= 0 && val_z < ez);
                    bool valid_i = causal_h
                        ? (val_i >= 0 && val_i <= i && val_i < height)
                        : (val_i >= 0 && val_i < ei);
                    bool valid_j = causal_w
                        ? (val_j >= 0 && val_j <= j && val_j < width)
                        : (val_j >= 0 && val_j < ej);
                    float acc = 0.0f;
                    if (valid_z && valid_i && valid_j) {{
                        for (int d = 0; d < dim; d++) {{
                            int g_idx = (((((b * heads + h) * out_depth + out_z) * out_height + out_i) * out_width + out_j) * dim + d);
                            int v_idx = (((((b * heads + h) * depth + val_z) * height + val_i) * width + val_j) * dim + d);
                            acc += grad_out[g_idx] * value[v_idx];
                        }}
                    }}
                    int out_idx = (((((b * heads + h) * out_depth + out_z) * out_height + out_i) * out_width + out_j) * L + neighbor_idx);
                    grad_attn[out_idx] = acc;
                    neighbor_idx++;
                }}
            }}
        }}
    }}
}}

if (plane < depth && y < height && x < width) {{
    int val_z = plane;
    int val_i = y;
    int val_j = x;
    for (int d = 0; d < dim; d++) {{
        float acc = 0.0f;
        for (int out_z = 0; out_z < out_depth; out_z++) {{
            int z = out_z * stride_d;
            if (z >= depth) continue;
            for (int out_i = 0; out_i < out_height; out_i++) {{
                int i = out_i * stride_h;
                if (i >= height) continue;
                for (int out_j = 0; out_j < out_width; out_j++) {{
                    int j = out_j * stride_w;
                    if (j >= width) continue;

                    int nz = 0, ni = 0, nj = 0, ez = depth, ei = height, ej = width;
                    if (!causal_d) {{
                        NATTEN_GET_WINDOW_START(nz, z, depth, K, NH, dilation_d);
                        NATTEN_GET_WINDOW_END(ez, nz, depth, K, dilation_d);
                    }}
                    if (!causal_h) {{
                        NATTEN_GET_WINDOW_START(ni, i, height, K, NH, dilation_h);
                        NATTEN_GET_WINDOW_END(ei, ni, height, K, dilation_h);
                    }}
                    if (!causal_w) {{
                        NATTEN_GET_WINDOW_START(nj, j, width, K, NH, dilation_w);
                        NATTEN_GET_WINDOW_END(ej, nj, width, K, dilation_w);
                    }}

                    int neighbor_idx = 0;
                    for (int kz = 0; kz < K; kz++) {{
                        for (int ki = 0; ki < K; ki++) {{
                            for (int kj = 0; kj < K; kj++) {{
                                int cand_z = (causal_d ? (z - (K - 1) * dilation_d) : nz) + kz * dilation_d;
                                int cand_i = (causal_h ? (i - (K - 1) * dilation_h) : ni) + ki * dilation_h;
                                int cand_j = (causal_w ? (j - (K - 1) * dilation_w) : nj) + kj * dilation_w;
                                bool valid_z = causal_d
                                    ? (cand_z >= 0 && cand_z <= z && cand_z < depth)
                                    : (cand_z >= 0 && cand_z < ez);
                                bool valid_i = causal_h
                                    ? (cand_i >= 0 && cand_i <= i && cand_i < height)
                                    : (cand_i >= 0 && cand_i < ei);
                                bool valid_j = causal_w
                                    ? (cand_j >= 0 && cand_j <= j && cand_j < width)
                                    : (cand_j >= 0 && cand_j < ej);
                                if (valid_z && valid_i && valid_j
                                    && cand_z == val_z && cand_i == val_i && cand_j == val_j) {{
                                    int a_idx = (((((b * heads + h) * out_depth + out_z) * out_height + out_i) * out_width + out_j) * L + neighbor_idx);
                                    int g_idx = (((((b * heads + h) * out_depth + out_z) * out_height + out_i) * out_width + out_j) * dim + d);
                                    acc += attention_probs[a_idx] * grad_out[g_idx];
                                }}
                                neighbor_idx++;
                            }}
                        }}
                    }}
                }}
            }}
        }}
        int out_idx = (((((b * heads + h) * depth + val_z) * height + val_i) * width + val_j) * dim + d);
        grad_v[out_idx] = acc;
    }}
}}
"""
