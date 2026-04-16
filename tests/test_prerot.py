"""Pre-rotated query kernels (metal_kernels_v4).

Anchors the math of the v4 fused attention path. Every score produced by
prerot_fused_qk_scores(prerotate_query(q, signs), ...) must match the
per-K-butterfly path packed_fused_qk_scores(q, ..., signs, ...) up to
float32 numerical noise — otherwise softmax downstream diverges.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.kernels import packed_fused_qk_scores
from turboquant_mlx.metal_kernels_v4 import (
    prerot_fused_qk_scores,
    prerotate_query,
)
from turboquant_mlx.packing import pack_indices
from turboquant_mlx.quantizer import PolarQuantizer


def _raw_wht_np(x: np.ndarray) -> np.ndarray:
    """Un-normalized Walsh-Hadamard Transform along the last axis.

    Matches metal_kernels_v4.PREROTATE_QUERY_KERNEL — no 1/sqrt(d) factor.
    """
    x = x.astype(np.float32).copy()
    d = x.shape[-1]
    h = 1
    while h < d:
        re = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        even = re[..., 0, :].copy()
        odd = re[..., 1, :].copy()
        re[..., 0, :] = even + odd
        re[..., 1, :] = even - odd
        x = re.reshape(*x.shape[:-1], d)
        h *= 2
    return x


def test_prerotate_query_matches_python_reference():
    """prerotate_query ≡ raw_WHT(signs * q) element-wise."""
    dim = 128
    mx.random.seed(0)
    q = mx.random.normal(shape=(4, dim))
    signs = mx.where(
        mx.random.bernoulli(p=0.5, shape=(dim,), key=mx.random.key(1)),
        mx.array(1.0),
        mx.array(-1.0),
    )
    out = prerotate_query(q, signs)
    mx.eval(out)
    ref = _raw_wht_np(np.array(q) * np.array(signs))
    assert np.allclose(np.array(out), ref, atol=1e-4), (
        f"prerotate_query diverges from raw WHT: max diff "
        f"{np.abs(np.array(out) - ref).max()}"
    )


@pytest.mark.parametrize(
    "n_heads,seq_len,dim,bits",
    [
        (4, 64, 128, 3),
        (8, 256, 128, 3),
        (2, 16, 64, 4),
        (1, 128, 32, 2),
    ],
)
def test_prerot_scores_match_full_kernel(n_heads, seq_len, dim, bits):
    """prerot_fused_qk_scores(prerotate_query(q), ...) == packed_fused_qk_scores(q, ...)."""
    mx.random.seed(0)
    pq = PolarQuantizer(dim=dim, bits=bits)

    raw = mx.random.randint(0, 2**bits, shape=(n_heads, seq_len, dim))
    k_packed = pack_indices(raw.reshape(-1, dim), bits).reshape(n_heads, seq_len, -1)
    k_norms = mx.random.uniform(shape=(n_heads, seq_len)) * 10.0 + 1.0
    query = mx.random.normal(shape=(n_heads, dim))

    scores_full = packed_fused_qk_scores(
        query, k_packed, k_norms, pq.centroids, pq.signs, dim, bits
    )
    q_rot = prerotate_query(query, pq.signs)
    scores_prerot = prerot_fused_qk_scores(
        q_rot, k_packed, k_norms, pq.centroids, dim, bits
    )
    mx.eval(scores_full, scores_prerot)

    a = np.array(scores_full)
    b = np.array(scores_prerot)
    max_abs = np.abs(a - b).max()
    # float32 accumulation noise grows with dim * seq_len; 1e-4 is safe
    assert max_abs < 1e-4, (
        f"prerot diverges from packed_fused_qk_scores at "
        f"H={n_heads} S={seq_len} D={dim} B={bits}: max_abs={max_abs}"
    )


@pytest.mark.parametrize(
    "n_kv_heads,n_rep",
    [(4, 2), (2, 4), (4, 7)],
)
def test_prerot_qk_gqa_matches_mx_repeat_reference(n_kv_heads, n_rep):
    """GQA: kv_head = q_head / n_rep in kernel == mx.repeat reference."""
    seq_len, dim, bits = 128, 128, 3
    n_q_heads = n_kv_heads * n_rep
    mx.random.seed(0)
    pq = PolarQuantizer(dim=dim, bits=bits)

    raw = mx.random.randint(0, 2**bits, shape=(n_kv_heads, seq_len, dim))
    k_packed = pack_indices(raw.reshape(-1, dim), bits).reshape(
        n_kv_heads, seq_len, -1
    )
    k_norms = mx.random.uniform(shape=(n_kv_heads, seq_len)) * 10.0 + 1.0
    query = mx.random.normal(shape=(n_q_heads, dim))
    q_rot = prerotate_query(query, pq.signs)

    # GQA path (kernel broadcasts)
    gqa = prerot_fused_qk_scores(
        q_rot, k_packed, k_norms, pq.centroids, dim, bits, n_rep=n_rep
    )
    # Reference: expand KV to q_heads via mx.repeat, use n_rep=1.
    kp_rep = mx.repeat(k_packed, n_rep, axis=0)
    kn_rep = mx.repeat(k_norms, n_rep, axis=0)
    ref = prerot_fused_qk_scores(
        q_rot, kp_rep, kn_rep, pq.centroids, dim, bits, n_rep=1
    )
    mx.eval(gqa, ref)
    max_abs = float(np.abs(np.array(gqa) - np.array(ref)).max())
    assert max_abs < 1e-4, (
        f"GQA QK (n_kv={n_kv_heads}, n_rep={n_rep}) diverges: max_abs={max_abs}"
    )


def test_prerot_scales_as_one_over_dim():
    """Sanity: doubling the query dimension (same signs/K construction)
    should not produce wild magnitude changes in score stats, because the
    kernel normalizes by 1/dim internally. This guards against an
    accidental missing scale factor.
    """
    mx.random.seed(0)
    dims = [64, 128]
    stds = []
    for D in dims:
        pq = PolarQuantizer(dim=D, bits=3)
        n_heads, seq_len = 2, 64
        raw = mx.random.randint(0, 8, shape=(n_heads, seq_len, D))
        k_packed = pack_indices(raw.reshape(-1, D), 3).reshape(n_heads, seq_len, -1)
        k_norms = mx.ones((n_heads, seq_len))
        q = mx.random.normal(shape=(n_heads, D))
        q_rot = prerotate_query(q, pq.signs)
        s = prerot_fused_qk_scores(q_rot, k_packed, k_norms, pq.centroids, D, 3)
        mx.eval(s)
        stds.append(float(np.array(s).std()))
    # With proper 1/dim scaling, stds should be on the same order; unscaled
    # kernel would give a 4x gap (ratio-of-dims) here.
    ratio = max(stds) / min(stds)
    assert ratio < 3.0, f"Score std ratio across dims suggests missing scale: {ratio}"
