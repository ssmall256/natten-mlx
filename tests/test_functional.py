import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.functional import na1d, na1d_qk, na2d, na2d_qk
from natten_mlx.utils.window import compute_window_start_end


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _global_attention_1d(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> np.ndarray:
    scores = np.einsum("blhd,bshd->blhs", q, k) * scale
    attn = _softmax(scores, axis=-1)
    return np.einsum("blhs,bshd->blhd", attn, v)


def _global_attention_2d(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> np.ndarray:
    bsz, height, width, heads, dim = q.shape
    qf = q.reshape(bsz, height * width, heads, dim)
    kf = k.reshape(bsz, height * width, heads, dim)
    vf = v.reshape(bsz, height * width, heads, dim)

    scores = np.einsum("bshd,bthd->bsht", qf, kf) * scale
    attn = _softmax(scores, axis=-1)
    out = np.einsum("bsht,bthd->bshd", attn, vf)
    return out.reshape(bsz, height, width, heads, dim)


def _na1d_reference_shifted(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, kernel_size: int, dilation: int
) -> np.ndarray:
    bsz, length, heads, dim = q.shape
    scale = dim ** -0.5
    starts, ends = compute_window_start_end(
        np.arange(length, dtype=np.int32), length, kernel_size, dilation
    )

    offsets = np.arange(kernel_size, dtype=np.int32) * dilation
    out = np.empty_like(q)

    for i in range(length):
        raw_idx = starts[i] + offsets
        valid = raw_idx < ends[i]
        idx = np.clip(raw_idx, 0, length - 1)

        k_neighborhood = k[:, idx]  # [B, K, H, D]
        v_neighborhood = v[:, idx]
        logits = np.einsum("bhd,bkhd->bhk", q[:, i], k_neighborhood) * scale
        logits = np.where(valid[None, None, :], logits, -np.inf)
        attn = _softmax(logits, axis=-1)
        out[:, i] = np.einsum("bhk,bkhd->bhd", attn, v_neighborhood)

    return out


def _na2d_reference_shifted(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, kernel_size: tuple[int, int], dilation: tuple[int, int]
) -> np.ndarray:
    bsz, height, width, heads, dim = q.shape
    kh, kw = kernel_size
    dh, dw = dilation
    scale = dim ** -0.5

    starts_h, ends_h = compute_window_start_end(
        np.arange(height, dtype=np.int32), height, kh, dh
    )
    starts_w, ends_w = compute_window_start_end(
        np.arange(width, dtype=np.int32), width, kw, dw
    )

    offs_h = np.arange(kh, dtype=np.int32) * dh
    offs_w = np.arange(kw, dtype=np.int32) * dw

    out = np.empty_like(q)
    for i in range(height):
        raw_h = starts_h[i] + offs_h
        valid_h = raw_h < ends_h[i]
        idx_h = np.clip(raw_h, 0, height - 1)
        for j in range(width):
            raw_w = starts_w[j] + offs_w
            valid_w = raw_w < ends_w[j]
            idx_w = np.clip(raw_w, 0, width - 1)

            k_neighborhood = []
            v_neighborhood = []
            valid = []
            for hi in range(kh):
                for wi in range(kw):
                    k_neighborhood.append(k[:, idx_h[hi], idx_w[wi]])  # [B, heads, dim]
                    v_neighborhood.append(v[:, idx_h[hi], idx_w[wi]])
                    valid.append(bool(valid_h[hi] and valid_w[wi]))

            k_neighborhood = np.stack(k_neighborhood, axis=1)  # [B, Kh*Kw, heads, dim]
            v_neighborhood = np.stack(v_neighborhood, axis=1)
            valid = np.array(valid, dtype=bool)

            logits = np.einsum("bhd,bkhd->bhk", q[:, i, j], k_neighborhood) * scale
            logits = np.where(valid[None, None, :], logits, -np.inf)
            attn = _softmax(logits, axis=-1)
            out[:, i, j] = np.einsum("bhk,bkhd->bhd", attn, v_neighborhood)

    return out


def test_na1d_output_shape_with_stride():
    q = mx.random.normal((2, 9, 4, 8))
    out = na1d(q, q, q, kernel_size=3, stride=2, dilation=1, is_causal=False)
    assert out.shape == (2, 5, 4, 8)


def test_na2d_output_shape_with_stride():
    q = mx.random.normal((2, 8, 7, 2, 4))
    out = na2d(q, q, q, kernel_size=(3, 3), stride=(2, 3), dilation=1, is_causal=False)
    assert out.shape == (2, 4, 3, 2, 4)


def test_na1d_equals_global_attention_when_kernel_covers_sequence():
    q = mx.random.normal((1, 7, 2, 4))
    k = mx.random.normal((1, 7, 2, 4))
    v = mx.random.normal((1, 7, 2, 4))

    out = np.array(na1d(q, k, v, kernel_size=7))
    ref = _global_attention_1d(np.array(q), np.array(k), np.array(v), scale=4 ** -0.5)
    assert np.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_na2d_equals_global_attention_when_kernel_covers_spatial_extent():
    q = mx.random.normal((1, 5, 5, 2, 4))
    k = mx.random.normal((1, 5, 5, 2, 4))
    v = mx.random.normal((1, 5, 5, 2, 4))

    out = np.array(na2d(q, k, v, kernel_size=(5, 5)))
    ref = _global_attention_2d(np.array(q), np.array(k), np.array(v), scale=4 ** -0.5)
    assert np.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_na1d_matches_shifted_boundary_reference_with_dilation():
    q = mx.random.normal((1, 7, 2, 3))
    k = mx.random.normal((1, 7, 2, 3))
    v = mx.random.normal((1, 7, 2, 3))

    out = np.array(na1d(q, k, v, kernel_size=3, dilation=2))
    ref = _na1d_reference_shifted(np.array(q), np.array(k), np.array(v), kernel_size=3, dilation=2)
    assert np.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_na2d_matches_shifted_boundary_reference_with_dilation():
    q = mx.random.normal((1, 7, 8, 2, 3))
    k = mx.random.normal((1, 7, 8, 2, 3))
    v = mx.random.normal((1, 7, 8, 2, 3))

    out = np.array(na2d(q, k, v, kernel_size=(3, 3), dilation=(2, 2)))
    ref = _na2d_reference_shifted(
        np.array(q), np.array(k), np.array(v), kernel_size=(3, 3), dilation=(2, 2)
    )
    assert np.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_na1d_qk_matches_scaled_reference():
    q = mx.random.normal((1, 8, 2, 3))
    k = mx.random.normal((1, 8, 2, 3))
    logits = np.array(na1d_qk(q, k, kernel_size=3, dilation=1))

    q_np = np.array(q)
    k_np = np.array(k)
    starts, _ = compute_window_start_end(np.arange(8, dtype=np.int32), 8, 3, 1)
    idx = starts[:, None] + np.arange(3, dtype=np.int32)[None, :]
    kn = k_np[:, idx]  # [B, L, K, H, D]
    ref = np.einsum("blhd,blkhd->blhk", q_np, kn) * (3 ** -0.5)

    assert np.allclose(logits, ref, atol=1e-6, rtol=1e-6)


def test_na1d_causal_first_position_attends_to_self_only():
    q = mx.ones((1, 5, 1, 1))
    k = mx.ones((1, 5, 1, 1))
    v = mx.arange(5, dtype=mx.float32).reshape((1, 5, 1, 1))

    out = na1d(q, k, v, kernel_size=3, is_causal=True, scale=1.0)
    out_np = np.array(out).reshape(5)

    assert np.isclose(out_np[0], 0.0)
    assert out_np[1] <= 1.0


def test_int_vs_tuple_parameter_normalization_equivalence():
    q = mx.random.normal((1, 10, 2, 4))
    out_int = np.array(na1d(q, q, q, kernel_size=3, stride=1, dilation=1, is_causal=False))
    out_tuple = np.array(na1d(q, q, q, kernel_size=(3,), stride=(1,), dilation=(1,), is_causal=(False,)))
    assert np.allclose(out_int, out_tuple, atol=1e-6, rtol=1e-6)


def test_validation_errors():
    q = mx.random.normal((1, 4, 2, 2))
    with pytest.raises(ValueError):
        na1d(q, q, q, kernel_size=7)

    with pytest.raises(ValueError):
        na1d(q, q, q, kernel_size=3, stride=4)

    q_strict = mx.random.normal((1, 5, 2, 2))
    with pytest.raises(ValueError):
        na1d(q_strict, q_strict, q_strict, kernel_size=3, dilation=2)
    with pytest.raises(ValueError):
        na1d_qk(q_strict, q_strict, kernel_size=3, dilation=2)

    q2 = mx.random.normal((1, 4, 4, 2, 2))
    with pytest.raises(ValueError):
        na2d(q2, q2, q2, kernel_size=(5, 5), dilation=(2, 2))

    q2_strict = mx.random.normal((1, 5, 5, 2, 2))
    with pytest.raises(ValueError):
        na2d(q2_strict, q2_strict, q2_strict, kernel_size=(3, 3), dilation=(2, 2))
    with pytest.raises(ValueError):
        na2d_qk(q2_strict, q2_strict, kernel_size=(3, 3), dilation=(2, 2))
