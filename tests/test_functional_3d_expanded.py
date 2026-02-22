"""Expanded 3D functional tests with reference implementation and edge cases."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.functional import na3d, na3d_av, na3d_qk
from natten_mlx.utils.window import compute_window_start_end


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _global_attention_3d(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float
) -> np.ndarray:
    bsz, depth, height, width, heads, dim = q.shape
    seq = depth * height * width
    qf = q.reshape(bsz, seq, heads, dim)
    kf = k.reshape(bsz, seq, heads, dim)
    vf = v.reshape(bsz, seq, heads, dim)
    scores = np.einsum("bshd,bthd->bsht", qf, kf) * scale
    attn = _softmax(scores, axis=-1)
    out = np.einsum("bsht,bthd->bshd", attn, vf)
    return out.reshape(bsz, depth, height, width, heads, dim)


def _na3d_reference_shifted(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    kernel_size: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> np.ndarray:
    bsz, depth, height, width, heads, dim = q.shape
    kd, kh, kw = kernel_size
    dd, dh, dw = dilation
    scale = dim**-0.5

    starts_d, ends_d = compute_window_start_end(
        np.arange(depth, dtype=np.int32), depth, kd, dd
    )
    starts_h, ends_h = compute_window_start_end(
        np.arange(height, dtype=np.int32), height, kh, dh
    )
    starts_w, ends_w = compute_window_start_end(
        np.arange(width, dtype=np.int32), width, kw, dw
    )

    offs_d = np.arange(kd, dtype=np.int32) * dd
    offs_h = np.arange(kh, dtype=np.int32) * dh
    offs_w = np.arange(kw, dtype=np.int32) * dw

    out = np.empty_like(q)
    for di in range(depth):
        raw_d = starts_d[di] + offs_d
        valid_d = raw_d < ends_d[di]
        idx_d = np.clip(raw_d, 0, depth - 1)
        for hi in range(height):
            raw_h = starts_h[hi] + offs_h
            valid_h = raw_h < ends_h[hi]
            idx_h = np.clip(raw_h, 0, height - 1)
            for wi in range(width):
                raw_w = starts_w[wi] + offs_w
                valid_w = raw_w < ends_w[wi]
                idx_w = np.clip(raw_w, 0, width - 1)

                k_neigh = []
                v_neigh = []
                valid = []
                for ddi in range(kd):
                    for hhi in range(kh):
                        for wwi in range(kw):
                            k_neigh.append(k[:, idx_d[ddi], idx_h[hhi], idx_w[wwi]])
                            v_neigh.append(v[:, idx_d[ddi], idx_h[hhi], idx_w[wwi]])
                            valid.append(
                                bool(valid_d[ddi] and valid_h[hhi] and valid_w[wwi])
                            )

                k_neigh = np.stack(k_neigh, axis=1)  # [B, K^3, H, D]
                v_neigh = np.stack(v_neigh, axis=1)
                valid_mask = np.array(valid, dtype=bool)

                logits = (
                    np.einsum("bhd,bkhd->bhk", q[:, di, hi, wi], k_neigh) * scale
                )
                logits = np.where(valid_mask[None, None, :], logits, -np.inf)
                attn = _softmax(logits, axis=-1)
                out[:, di, hi, wi] = np.einsum("bhk,bkhd->bhd", attn, v_neigh)

    return out


# ---------------------------------------------------------------------------
# 3D reference implementation tests
# ---------------------------------------------------------------------------


def test_na3d_equals_global_when_kernel_covers_volume():
    q = mx.random.normal((1, 3, 3, 3, 2, 4))
    k = mx.random.normal((1, 3, 3, 3, 2, 4))
    v = mx.random.normal((1, 3, 3, 3, 2, 4))

    out = np.array(na3d(q, k, v, kernel_size=(3, 3, 3)))
    ref = _global_attention_3d(
        np.array(q), np.array(k), np.array(v), scale=4**-0.5
    )
    assert np.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_na3d_matches_shifted_boundary_reference():
    q = mx.random.normal((1, 5, 5, 5, 2, 3))
    k = mx.random.normal((1, 5, 5, 5, 2, 3))
    v = mx.random.normal((1, 5, 5, 5, 2, 3))

    out = np.array(na3d(q, k, v, kernel_size=(3, 3, 3)))
    ref = _na3d_reference_shifted(
        np.array(q),
        np.array(k),
        np.array(v),
        kernel_size=(3, 3, 3),
        dilation=(1, 1, 1),
    )
    assert np.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_na3d_matches_reference_with_dilation():
    q = mx.random.normal((1, 7, 7, 7, 2, 3))
    k = mx.random.normal((1, 7, 7, 7, 2, 3))
    v = mx.random.normal((1, 7, 7, 7, 2, 3))

    out = np.array(na3d(q, k, v, kernel_size=(3, 3, 3), dilation=(2, 2, 2)))
    ref = _na3d_reference_shifted(
        np.array(q),
        np.array(k),
        np.array(v),
        kernel_size=(3, 3, 3),
        dilation=(2, 2, 2),
    )
    assert np.allclose(out, ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3D causal correctness
# ---------------------------------------------------------------------------


def test_na3d_causal_first_position_attends_to_self_only():
    q = mx.ones((1, 3, 3, 3, 1, 1))
    k = mx.ones((1, 3, 3, 3, 1, 1))
    v_vals = mx.arange(27, dtype=mx.float32).reshape((1, 3, 3, 3, 1, 1))

    out = na3d(
        q, k, v_vals,
        kernel_size=(3, 3, 3),
        is_causal=(True, True, True),
        scale=1.0,
    )
    # Position (0,0,0) with all-causal should only see itself
    assert np.isclose(np.array(out)[0, 0, 0, 0, 0, 0], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 3D gradient tests (expanded)
# ---------------------------------------------------------------------------


def test_na3d_grad_all_inputs():
    """Gradient w.r.t. q, k, and v simultaneously."""
    q = mx.random.normal((1, 5, 5, 5, 2, 4))
    k = mx.random.normal((1, 5, 5, 5, 2, 4))
    v = mx.random.normal((1, 5, 5, 5, 2, 4))

    def loss_fn(q_in, k_in, v_in):
        return mx.sum(na3d(q_in, k_in, v_in, kernel_size=(3, 3, 3)))

    grads = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
    assert len(grads) == 3
    for g, ref in zip(grads, [q, k, v]):
        assert g.shape == ref.shape
        assert np.isfinite(np.array(g)).all()


def test_na3d_grad_with_stride_and_dilation():
    q = mx.random.normal((1, 7, 7, 7, 2, 4))
    k = mx.random.normal((1, 7, 7, 7, 2, 4))
    v = mx.random.normal((1, 7, 7, 7, 2, 4))

    def loss_fn(q_in):
        return mx.sum(
            na3d(
                q_in, k, v,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                dilation=(2, 2, 2),
            )
        )

    grad = mx.grad(loss_fn)(q)
    assert grad.shape == q.shape
    assert np.isfinite(np.array(grad)).all()


def test_na3d_grad_with_causal():
    q = mx.random.normal((1, 5, 5, 5, 2, 4))
    k = mx.random.normal((1, 5, 5, 5, 2, 4))
    v = mx.random.normal((1, 5, 5, 5, 2, 4))

    def loss_fn(q_in):
        return mx.sum(
            na3d(q_in, k, v, kernel_size=(3, 3, 3), is_causal=(True, False, True))
        )

    grad = mx.grad(loss_fn)(q)
    assert grad.shape == q.shape
    assert np.isfinite(np.array(grad)).all()
