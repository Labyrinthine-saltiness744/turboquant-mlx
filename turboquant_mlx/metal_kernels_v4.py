"""Metal kernels v4: pre-rotated query path.

The core insight (see fused_attention.py docstring):

    <Q, dequant(K_i)> = norm_i / sqrt(d) * <WHT(signs * Q), centroids[K_i]>

So if we pre-rotate the query ONCE per decode step (O(d log d)), we can
compute Q@K scores for the whole sequence without running the WHT butterfly
on every cached K (saves O(seq_len * d log d) work in the hot path).

Convention matches turboquant_mlx.metal.FUSED_QUANTIZE_KERNEL: the WHT
butterfly inside our Metal kernels is the "raw" (un-normalized) butterfly.
The 1/sqrt(d) factor is applied explicitly as `scale[0]` where needed so
the code stays self-consistent across encode/decode/attention.

Three public functions:
  prerotate_query            — one-shot: Q -> WHT_raw(signs * Q)
  prerot_fused_qk_scores     — Q_rot @ K_packed without per-K butterfly
  prerot_packed_dequantize   — full V dequant (just re-exports the existing
                                packed_dequantize, since V always needs the
                                inverse WHT butterfly).
"""

import mlx.core as mx
import math

from turboquant_mlx.kernels import packed_dequantize as prerot_packed_dequantize

__all__ = [
    "prerotate_query",
    "prerot_fused_qk_scores",
    "prerot_packed_dequantize",
]

# --- Pre-rotate query: signs * Q, then raw WHT butterfly (no 1/sqrt(d)).
# One threadgroup per head, `dim` threads cooperating on the butterfly.
PREROTATE_QUERY_KERNEL = """
    uint head = threadgroup_position_in_grid.x;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];

    threadgroup T shared[256];
    shared[elem] = q_in[head * dim + elem] * signs[elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint h = 1;
    while (h < dim) {
        uint block = elem / (2 * h);
        uint offset = elem % (2 * h);
        if (offset < h) {
            uint j = block * 2 * h + offset;
            T a = shared[j];
            T b = shared[j + h];
            shared[j]     = a + b;
            shared[j + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h *= 2;
    }

    q_out[head * dim + elem] = shared[elem];
"""

# --- Pre-rotated Q @ K: no per-K butterfly, no per-K signs.
# scores[head, pos] = (norms[head, pos] / sqrt(d)) * <Q_rot[head], centroids[K_idx[head, pos]]>
PREROT_FUSED_QK_KERNEL = """
    uint pos = threadgroup_position_in_grid.x;
    uint head = threadgroup_position_in_grid.y;
    uint elem = thread_position_in_threadgroup.x;
    uint dim = dims[0];
    uint seq_len = dims[1];
    uint bits = dims[2];
    uint vals_per_word = dims[3];
    uint packed_dim = dims[4];
    uint bit_mask = (1u << bits) - 1u;

    // Unpack one codebook index for this (head, pos, elem).
    uint kv_base = head * seq_len * packed_dim + pos * packed_dim;
    uint word_idx = elem / vals_per_word;
    uint pos_in_word = elem % vals_per_word;
    uint word = packed[kv_base + word_idx];
    uint idx = (word >> (pos_in_word * bits)) & bit_mask;

    // Partial product with the pre-rotated query — no butterfly here.
    T partial = centroids[idx] * q_rot[head * dim + elem];

    // Tree reduction across the `dim` threads of this threadgroup.
    threadgroup T shared[256];
    shared[elem] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = dim / 2; stride > 0; stride >>= 1) {
        if (elem < stride) {
            shared[elem] += shared[elem + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (elem == 0) {
        out[head * seq_len + pos] = shared[0] * norms[head * seq_len + pos] * scale[0];
    }
"""

_prerotate_query = None
_prerot_fused_qk = None


def prerotate_query(q: mx.array, signs: mx.array) -> mx.array:
    """Pre-rotate a decode-step query: signs * q → raw WHT butterfly.

    Args:
        q: (n_heads, dim)
        signs: (dim,) ±1 rotation signs (same convention as the encoder).

    Returns:
        (n_heads, dim) rotated query in the same space as the raw codebook.
    """
    global _prerotate_query
    if _prerotate_query is None:
        _prerotate_query = mx.fast.metal_kernel(
            name="tq_prerotate_query",
            input_names=["q_in", "signs", "dims"],
            output_names=["q_out"],
            source=PREROTATE_QUERY_KERNEL,
        )

    if q.ndim != 2:
        raise ValueError(f"prerotate_query expects (n_heads, dim), got {q.shape}")
    n_heads, dim = q.shape
    dims_arr = mx.array([dim], dtype=mx.uint32)
    outputs = _prerotate_query(
        inputs=[q.astype(mx.float32).reshape(n_heads * dim), signs, dims_arr],
        template=[("T", mx.float32)],
        grid=(n_heads * dim, 1, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_heads * dim,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_heads, dim)


def prerot_fused_qk_scores(
    q_rot: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    centroids: mx.array,
    dim: int,
    bits: int,
) -> mx.array:
    """Compute Q@K scores using a pre-rotated query.

    Args:
        q_rot: (n_heads, dim) output of prerotate_query.
        k_packed: (n_heads, seq_len, packed_dim) packed uint32 indices.
        k_norms: (n_heads, seq_len) per-position K vector norms.
        centroids: (n_levels,) Lloyd-Max centroids (same as encoder).
        dim: head dimension (must equal k_packed shape d, power of 2).
        bits: quantization bit width (1-4).

    Returns:
        (n_heads, seq_len) raw QK scores (attention scaling applied by caller).
    """
    global _prerot_fused_qk
    if _prerot_fused_qk is None:
        _prerot_fused_qk = mx.fast.metal_kernel(
            name="tq_prerot_fused_qk",
            input_names=["q_rot", "packed", "norms", "centroids", "scale", "dims"],
            output_names=["out"],
            source=PREROT_FUSED_QK_KERNEL,
        )

    n_heads, seq_len = k_norms.shape
    p_dim = k_packed.shape[-1]
    vpw = {1: 32, 2: 16, 3: 10, 4: 8}[bits]
    # Match the encoder/decoder scale convention in metal.py / kernels.py: the
    # raw WHT butterfly is carried un-normalized through encode, and decode
    # tacks on 1/sqrt(d). Here we pick up the *second* 1/sqrt(d) that a
    # paper-literal derivation would attach to the centroid side, so the scores
    # produced by this kernel stay consistent with packed_fused_qk_scores on
    # the same inputs. End-to-end, the outer attention code then multiplies
    # by 1/sqrt(d) one more time (attention scaling), matching the paper.
    scale = mx.array([1.0 / dim], dtype=mx.float32)
    dims_arr = mx.array([dim, seq_len, bits, vpw, p_dim], dtype=mx.uint32)

    outputs = _prerot_fused_qk(
        inputs=[
            q_rot.astype(mx.float32).reshape(n_heads * dim),
            k_packed.astype(mx.uint32).reshape(n_heads * seq_len * p_dim),
            k_norms.astype(mx.float32).reshape(n_heads * seq_len),
            centroids,
            scale,
            dims_arr,
        ],
        template=[("T", mx.float32)],
        grid=(seq_len * dim, n_heads, 1),
        threadgroup=(dim, 1, 1),
        output_shapes=[(n_heads * seq_len,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0].reshape(n_heads, seq_len)
