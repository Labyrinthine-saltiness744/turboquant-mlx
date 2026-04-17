"""Tests for VOnlyTurboQuantCache and adaptive cache."""

import mlx.core as mx
import numpy as np
import pytest

from turboquant_mlx.cache import TurboQuantKVCache
from turboquant_mlx.v_only_cache import VOnlyTurboQuantCache
from turboquant_mlx.adaptive import make_adaptive_cache


class TestVOnlyCache:
    B, H, S, D = 1, 4, 32, 128

    def _fill(self, cache, S=None):
        S = S or self.S
        k = mx.random.normal(shape=(self.B, self.H, S, self.D))
        v = mx.random.normal(shape=(self.B, self.H, S, self.D))
        k_out, v_out = cache.update_and_fetch(k, v)
        mx.eval(k_out, v_out)
        return k, v, k_out, v_out

    def test_basic_roundtrip(self):
        cache = VOnlyTurboQuantCache(bits=3)
        k, v, k_out, v_out = self._fill(cache)
        # K should be preserved exactly (fp16 KVCache)
        assert k_out.shape == (self.B, self.H, self.S, self.D)
        assert v_out.shape == (self.B, self.H, self.S, self.D)
        # K cosine ~ 1.0 (fp16 roundtrip)
        k_cos = float(mx.sum(k * k_out) / (mx.linalg.norm(k) * mx.linalg.norm(k_out)))
        assert k_cos > 0.999, f"K roundtrip cosine {k_cos}"
        # V cosine > 0.95 (3-bit TQ lossy)
        v_cos = float(mx.sum(v * v_out) / (mx.linalg.norm(v) * mx.linalg.norm(v_out)))
        assert v_cos > 0.95, f"V roundtrip cosine {v_cos}"

    def test_offset_tracks(self):
        cache = VOnlyTurboQuantCache(bits=3)
        assert cache.offset == 0
        self._fill(cache, S=8)
        assert cache.offset == 8
        self._fill(cache, S=4)
        assert cache.offset == 12

    def test_empty(self):
        cache = VOnlyTurboQuantCache(bits=3)
        assert cache.empty()
        self._fill(cache, S=1)
        assert not cache.empty()

    def test_trim(self):
        cache = VOnlyTurboQuantCache(bits=3)
        self._fill(cache, S=16)
        assert cache.offset == 16
        cache.trim(4)
        assert cache.offset == 12

    def test_state_includes_v_data(self):
        cache = VOnlyTurboQuantCache(bits=3)
        self._fill(cache, S=8)
        state = cache.state
        # Should have K arrays (keys, values from KVCache) + V arrays (packed, norms)
        assert len(state) >= 4, f"state has {len(state)} arrays, expected >= 4"

    def test_meta_state_roundtrip(self):
        cache = VOnlyTurboQuantCache(bits=3)
        self._fill(cache, S=8)
        meta = cache.meta_state
        assert meta.startswith("VOnlyTQ,3,")

    def test_no_v_buffer_mode(self):
        cache = VOnlyTurboQuantCache(bits=3, no_v_buffer=True)
        k, v, k_out, v_out = self._fill(cache)
        assert v_out.shape == (self.B, self.H, self.S, self.D)
        v_cos = float(mx.sum(v * v_out) / (mx.linalg.norm(v) * mx.linalg.norm(v_out)))
        assert v_cos > 0.95


class TestAdaptiveCache:
    def test_basic_creation(self):
        caches = make_adaptive_cache(num_layers=12, bits=3, fp16_layers=2)
        assert len(caches) == 12
        # First 2 and last 2 should be KVCache
        from mlx_lm.models.cache import KVCache
        assert isinstance(caches[0], KVCache)
        assert isinstance(caches[1], KVCache)
        assert isinstance(caches[10], KVCache)
        assert isinstance(caches[11], KVCache)
        # Middle should be TurboQuantKVCache
        assert isinstance(caches[5], TurboQuantKVCache)

    def test_all_fp16_when_fp16_layers_large(self):
        from mlx_lm.models.cache import KVCache
        caches = make_adaptive_cache(num_layers=8, bits=3, fp16_layers=4)
        for c in caches:
            assert isinstance(c, KVCache)


class TestTQSerialization:
    def test_state_roundtrip(self):
        cache = TurboQuantKVCache(bits=3)
        k = mx.random.normal(shape=(1, 4, 16, 128))
        v = mx.random.normal(shape=(1, 4, 16, 128))
        cache.update_and_fetch(k, v)
        mx.eval(cache.k_packed)

        state = cache.state
        meta = cache.meta_state

        restored = TurboQuantKVCache.from_state(state, meta)
        assert restored.offset == cache.offset
        assert restored.quant_bits == cache.quant_bits

        # Packed data matches
        k_match = mx.array_equal(
            cache.k_packed[..., :cache.offset, :],
            restored.k_packed[..., :restored.offset, :],
        )
        assert k_match

    def test_from_state_can_extend(self):
        cache = TurboQuantKVCache(bits=3)
        k = mx.random.normal(shape=(1, 4, 8, 128))
        v = mx.random.normal(shape=(1, 4, 8, 128))
        cache.update_and_fetch(k, v)
        mx.eval(cache.k_packed)

        restored = TurboQuantKVCache.from_state(cache.state, cache.meta_state)
        # Should be able to add more tokens
        k2 = mx.random.normal(shape=(1, 4, 4, 128))
        v2 = mx.random.normal(shape=(1, 4, 4, 128))
        k_out, v_out = restored.update_and_fetch(k2, v2)
        mx.eval(k_out, v_out)
        assert restored.offset == 12
        assert k_out.shape == (1, 4, 12, 128)
