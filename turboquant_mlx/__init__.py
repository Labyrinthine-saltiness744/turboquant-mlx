from turboquant_mlx.cache import TurboQuantKVCache
from turboquant_mlx.v_only_cache import VOnlyTurboQuantCache
from turboquant_mlx.adaptive import make_adaptive_cache
from turboquant_mlx.patch import apply_patch, remove_patch
from turboquant_mlx.quantizer import PolarQuantizer
from turboquant_mlx.metal import fused_quantize, dequant_fp16
from turboquant_mlx.kernels import packed_dequantize, packed_fused_qk_scores
from turboquant_mlx.metal_kernels_v4 import (
    prerotate_query,
    prerot_fused_qk_scores,
    prerot_packed_dequantize,
)
from turboquant_mlx.sparse_v import sparse_v_matvec
from turboquant_mlx.flash_attention import flash_attention_turboquant
from turboquant_mlx.fused_attention import turboquant_attention
from turboquant_mlx.packing import pack_indices, unpack_indices, packed_dim
from turboquant_mlx.rotation import (
    walsh_hadamard_transform,
    randomized_hadamard_transform,
    inverse_randomized_hadamard,
    random_diagonal_sign,
)

__version__ = "0.3.0"
