# Path: unplow/tests/test_subspace_coder.py
"""Unit tests for Phase 5 SVDSubspaceSteeringEngine and Order1BigramCacheCDF.

Covers multi-neuron SVD subspace vector computation, forward hook steering
cleanup, and Order-1 Markov bigram cache mixture arithmetic CDF monotonicity.
"""

import numpy as np
import torch
from unplow.models import DeterministicMockCausalLM, DeterministicMockTokenizer
from unplow.subspace_coder import (
    TOTAL_FREQ,
    Order1BigramCacheCDF,
    SVDSubspaceSteeringEngine,
    SubspaceSteeringRecord,
)


def test_svd_subspace_steering_and_hook_cleanup():
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    engine = SVDSubspaceSteeringEngine(model, tokenizer)

    vec, ratio = engine.compute_principal_subspace_vector(
        layer_idx=1, neuron_indices=[0, 1, 2]
    )
    assert vec.shape[0] == model.model.layers[1].mlp.down_proj.weight.shape[0]
    assert 0.0 < ratio <= 1.0

    rec = engine.steer_with_subspace(
        layer_idx=1,
        neuron_indices=[0, 1, 2],
        seed_prompt="def ",
        seq_len=10,
        steering_gain=4.0,
        steering_steps=4,
    )
    assert isinstance(rec, SubspaceSteeringRecord)
    assert rec.neuron_indices == [0, 1, 2]
    assert len(rec.extracted_text) >= 4
    assert len(model.model.layers[1].mlp.down_proj._forward_hooks) == 0


def test_order1_bigram_cache_cdf_transition_boost():
    vocab_size = 128
    cache = Order1BigramCacheCDF(
        vocab_size=vocab_size, unigram_weight=0.05, bigram_weight=0.20
    )
    logits = torch.zeros(vocab_size, dtype=torch.float32)

    # Record bigram transition 10 -> 25 three times, then land on token 10
    for _ in range(3):
        cache.update(10)
        cache.update(25)
    cache.update(10)

    cdf = cache.logits_to_bigram_mixed_cdf(logits)
    assert cdf[0] == 0
    assert cdf[-1] == TOTAL_FREQ
    assert np.all(np.diff(cdf) >= 1)

    span_25 = cdf[26] - cdf[25]
    span_other = cdf[31] - cdf[30]
    assert span_25 > span_other * 2


def test_order1_bigram_cache_cdf_reset_and_monotonicity():
    vocab_size = 64
    cache = Order1BigramCacheCDF(vocab_size=vocab_size, unigram_weight=0.08, bigram_weight=0.12)
    cache.update(5)
    cache.update(12)
    assert len(cache.history) == 2
    assert cache.prev_token == 12

    cache.reset()
    assert len(cache.history) == 0
    assert cache.prev_token is None
    assert np.sum(cache.unigram_counts) == 0
    assert len(cache.bigram_transitions) == 0

    logits = torch.randn(vocab_size, dtype=torch.float32)
    cdf = cache.logits_to_bigram_mixed_cdf(logits)
    assert cdf[0] == 0
    assert cdf[-1] == TOTAL_FREQ
    assert np.all(np.diff(cdf) >= 1)


def test_bigram_cache_lossless_container_roundtrip():
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    from unplow.paradigm2 import LLMArithmeticCodec

    codec = LLMArithmeticCodec(model=model, tokenizer=tokenizer)
    plaintext = "alpha beta gamma delta alpha beta gamma delta alpha beta"
    container_bytes, meta, stats = codec.compress_to_container(
        text=plaintext,
        enable_bigram_cache=True,
    )
    assert meta.bigram_cache is True
    assert meta.is_bigram_cache_enabled is True
    assert (meta.flags & 0x04) != 0

    reconstructed, dec_meta, is_valid = codec.decompress_from_container(container_bytes)
    assert is_valid is True
    assert reconstructed == plaintext

# end of file: unplow/tests/test_subspace_coder.py
