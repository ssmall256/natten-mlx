import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from natten_mlx.functional import na3d, na3d_av, na3d_qk


def test_na3d_output_shape_with_stride():
    q = mx.random.normal((2, 9, 8, 7, 2, 4))
    out = na3d(
        q,
        q,
        q,
        kernel_size=(3, 3, 3),
        stride=(2, 3, 2),
        dilation=(1, 1, 1),
        is_causal=(False, False, False),
    )
    assert out.shape == (2, 5, 3, 4, 2, 4)


def test_na3d_split_qk_av_matches_fused_with_stride_and_causal():
    q = mx.random.normal((1, 9, 8, 7, 2, 3))
    k = mx.random.normal((1, 9, 8, 7, 2, 3))
    v = mx.random.normal((1, 9, 8, 7, 2, 3))

    logits = na3d_qk(
        q,
        k,
        kernel_size=(3, 3, 3),
        dilation=(2, 1, 1),
        stride=(2, 3, 2),
        is_causal=(True, False, True),
        scale=0.29,
    )
    attn = mx.softmax(logits, axis=-1)
    out_split = na3d_av(
        attn,
        v,
        kernel_size=(3, 3, 3),
        dilation=(2, 1, 1),
        stride=(2, 3, 2),
        is_causal=(True, False, True),
    )
    out_fused = na3d(
        q,
        k,
        v,
        kernel_size=(3, 3, 3),
        stride=(2, 3, 2),
        dilation=(2, 1, 1),
        is_causal=(True, False, True),
        scale=0.29,
    )

    assert np.allclose(np.array(out_split), np.array(out_fused), atol=1e-5, rtol=1e-5)


def test_na3d_validation_errors():
    q = mx.random.normal((1, 4, 4, 4, 2, 2))
    with pytest.raises(ValueError):
        na3d(q, q, q, kernel_size=(5, 3, 3))

    with pytest.raises(ValueError):
        na3d(q, q, q, kernel_size=(3, 3, 3), stride=(4, 1, 1))
