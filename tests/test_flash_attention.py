"""Single-kernel fused TurboQuant SDPA correctness.

Pins that flash_attention_turboquant produces the same output as the
separate prerot_fused_qk_scores + softmax + sparse_v_matvec(threshold=0)
pipeline — i.e. that the fused kernel is mathematically equivalent to
the dense composed path, to float32 accumulation noise.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.flash_attention import flash_attention_turboquant
from turboquant_mlx.metal_kernels_v4 import (
    prerot_fused_qk_scores,
    prerotate_query,
)
from turboquant_mlx.packing import pack_indices
from turboquant_mlx.quantizer import PolarQuantizer
from turboquant_mlx.sparse_v import sparse_v_matvec


def _reference_softmax_v(q_rot, k_packed, k_norms, v_packed, v_norms,
                          k_pq, v_pq, dim, bits, attn_scale, n_rep):
    """Composed reference: prerot_fused_qk_scores → softmax → sparse_v_matvec(0)."""
    scores = prerot_fused_qk_scores(
        q_rot, k_packed, k_norms, k_pq.centroids, dim, bits, n_rep=n_rep
    )
    weights = mx.softmax(scores * attn_scale, axis=-1)
    return sparse_v_matvec(
        weights, v_packed, v_norms, v_pq.centroids, v_pq.signs, dim, bits,
        threshold=0.0, n_rep=n_rep,
    )


@pytest.mark.parametrize(
    "n_heads,seq_len,dim,bits,block_size",
    [
        (4, 32, 128, 3, 8),
        (4, 128, 128, 3, 16),
        (8, 256, 128, 3, 16),
        (2, 64, 64, 4, 8),
        (4, 96, 128, 3, 32),
    ],
)
def test_flash_attn_matches_composed_path_mha(
    n_heads, seq_len, dim, bits, block_size
):
    """MHA (n_rep=1): fused kernel == softmax(QK * scale) @ V."""
    mx.random.seed(0)
    k_pq = PolarQuantizer(dim=dim, bits=bits, seed=42)
    v_pq = PolarQuantizer(dim=dim, bits=bits, seed=43)
    attn_scale = 1.0 / math.sqrt(dim)

    k_raw = mx.random.randint(0, 2**bits, shape=(n_heads, seq_len, dim))
    v_raw = mx.random.randint(0, 2**bits, shape=(n_heads, seq_len, dim))
    k_packed = pack_indices(k_raw.reshape(-1, dim), bits).reshape(
        n_heads, seq_len, -1
    )
    v_packed = pack_indices(v_raw.reshape(-1, dim), bits).reshape(
        n_heads, seq_len, -1
    )
    k_norms = mx.random.uniform(shape=(n_heads, seq_len)) * 5.0 + 0.1
    v_norms = mx.random.uniform(shape=(n_heads, seq_len)) * 5.0 + 0.1
    q = mx.random.normal(shape=(n_heads, dim))
    q_rot = prerotate_query(q, k_pq.signs)

    ref = _reference_softmax_v(
        q_rot, k_packed, k_norms, v_packed, v_norms,
        k_pq, v_pq, dim, bits, attn_scale, n_rep=1,
    )
    flash = flash_attention_turboquant(
        q_rot, k_packed, k_norms, v_packed, v_norms,
        k_pq.centroids, v_pq.centroids, v_pq.signs,
        dim, bits, attn_scale, block_size=block_size, n_rep=1,
    )
    mx.eval(ref, flash)

    ref_np = np.array(ref)
    flash_np = np.array(flash)
    max_abs = float(np.abs(ref_np - flash_np).max())
    max_ref = float(np.abs(ref_np).max())
    rel = max_abs / max_ref if max_ref else max_abs
    assert rel < 1e-3, (
        f"flash vs composed: rel={rel:.3e} abs={max_abs:.3e} "
        f"(H={n_heads} S={seq_len} D={dim} B={bits} B_c={block_size})"
    )


@pytest.mark.parametrize(
    "n_kv_heads,n_rep",
    [(4, 2), (2, 4)],
)
def test_flash_attn_gqa(n_kv_heads, n_rep):
    """GQA: Q heads read KV at kv_head = q_head / n_rep."""
    seq_len, dim, bits = 128, 128, 3
    n_q_heads = n_kv_heads * n_rep
    mx.random.seed(0)
    k_pq = PolarQuantizer(dim=dim, bits=bits, seed=42)
    v_pq = PolarQuantizer(dim=dim, bits=bits, seed=43)
    attn_scale = 1.0 / math.sqrt(dim)

    k_raw = mx.random.randint(0, 2**bits, shape=(n_kv_heads, seq_len, dim))
    v_raw = mx.random.randint(0, 2**bits, shape=(n_kv_heads, seq_len, dim))
    k_packed = pack_indices(k_raw.reshape(-1, dim), bits).reshape(
        n_kv_heads, seq_len, -1
    )
    v_packed = pack_indices(v_raw.reshape(-1, dim), bits).reshape(
        n_kv_heads, seq_len, -1
    )
    k_norms = mx.random.uniform(shape=(n_kv_heads, seq_len)) * 5.0 + 0.1
    v_norms = mx.random.uniform(shape=(n_kv_heads, seq_len)) * 5.0 + 0.1
    q = mx.random.normal(shape=(n_q_heads, dim))
    q_rot = prerotate_query(q, k_pq.signs)

    ref = _reference_softmax_v(
        q_rot, k_packed, k_norms, v_packed, v_norms,
        k_pq, v_pq, dim, bits, attn_scale, n_rep=n_rep,
    )
    flash = flash_attention_turboquant(
        q_rot, k_packed, k_norms, v_packed, v_norms,
        k_pq.centroids, v_pq.centroids, v_pq.signs,
        dim, bits, attn_scale, block_size=16, n_rep=n_rep,
    )
    mx.eval(ref, flash)

    ref_np = np.array(ref)
    flash_np = np.array(flash)
    max_abs = float(np.abs(ref_np - flash_np).max())
    max_ref = float(np.abs(ref_np).max())
    rel = max_abs / max_ref if max_ref else max_abs
    assert rel < 1e-3, (
        f"GQA flash vs composed: rel={rel} abs={max_abs} "
        f"(kv={n_kv_heads}, n_rep={n_rep})"
    )
