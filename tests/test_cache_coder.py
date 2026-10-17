# Path: unplow/tests/test_cache_coder.py
"""Unit tests for Phase 4 OnlinePrefixCacheCDF and Surprisal Profiler.

Covers online prefix-cache mixture monotonicity, bit packing of FLAG_CACHE_MIXTURE,
container roundtrip lossless verification with cache mixture, and surprisal savings.
"""

import numpy as np
import torch

from unplow.cache_coder import (
    FLAG_CACHE_MIXTURE,
    TOTAL_FREQ,
    OnlinePrefixCacheCDF,
    profile_sequence_surprisal,
)
from unplow.conditioning import pack_flags_with_domain, unpack_flags_with_domain
from unplow.models import DeterministicMockCausalLM, DeterministicMockTokenizer
from unplow.paradigm2 import LLMArithmeticCodec


def test_online_prefix_cache_cdf_monotonicity_and_boost():
    vocab_size = 256
    cache = OnlinePrefixCacheCDF(vocab_size=vocab_size, cache_weight=0.15, window_size=8)
    logits = torch.zeros(vocab_size, dtype=torch.float32)

    cdf_initial = cache.logits_to_mixed_integer_cdf(logits)
    assert cdf_initial[0] == 0
    assert cdf_initial[-1] == TOTAL_FREQ
    assert np.all(np.diff(cdf_initial) >= 1)

    target_tok = 42
    initial_span = cdf_initial[target_tok + 1] - cdf_initial[target_tok]

    # Repeat token 42 three times and verify its frequency mass strictly increases
    for _ in range(3):
        cache.update(target_tok)

    cdf_boosted = cache.logits_to_mixed_integer_cdf(logits)
    boosted_span = cdf_boosted[target_tok + 1] - cdf_boosted[target_tok]
    assert boosted_span > initial_span
    assert cdf_boosted[-1] == TOTAL_FREQ


def test_flag_cache_mixture_bit_packing():
    flags = pack_flags_with_domain(verify_hash=True, domain_id=2, cache_mixture=True)
    assert flags & FLAG_CACHE_MIXTURE == FLAG_CACHE_MIXTURE
    verify, dom_id = unpack_flags_with_domain(flags)
    assert verify is True
    assert dom_id == 2


def test_cache_mixture_lossless_container_roundtrip():
    codec = LLMArithmeticCodec(model_name="mock")
    text = "unplow_token unplow_token unplow_token lossless cache mixture verification."
    container_bytes = codec.compress_to_container(
        text, domain_id=1, use_cache_mixture=True
    )
    reconstructed, meta = codec.decompress_from_container(container_bytes)
    assert reconstructed == text
    assert meta.domain_id == 1
    assert meta.cache_mixture is True


def test_profile_sequence_surprisal_saves_bits_on_repetition():
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    repeated_text = "alpha beta alpha beta alpha beta alpha beta"

    report = profile_sequence_surprisal(
        model=model,
        tokenizer=tokenizer,
        plaintext=repeated_text,
        bos_token_id=tokenizer.bos_token_id,
        cache_weight=0.12,
    )
    assert report.total_tokens > 0
    assert report.ideal_bits_per_byte > 0.0
    assert report.cache_bits_saved > 0.0
    assert len(report.entries) == report.total_tokens

# end of file: unplow/tests/test_cache_coder.py
