# Path: unplow/tests/test_paradigm2.py
"""Unit tests for Paradigm 2: Lossless Autoregressive Arithmetic Codec."""

from pathlib import Path
import numpy as np
import pytest
import torch

from unplow.models import MockByteTokenizer, MockCausalLM
from unplow.paradigm2 import (
    CompressionStats,
    LLMArithmeticCodec,
    MAX_VAL,
    TOTAL_FREQ,
)


@pytest.fixture
def mock_codec():
    tok = MockByteTokenizer(vocab_size=260)
    model = MockCausalLM(vocab_size=260, hidden_dim=32)
    return LLMArithmeticCodec(model=model, tokenizer=tok, device="cpu")


def test_logits_to_integer_cdf():
    logits = torch.randn(260)
    cdf = LLMArithmeticCodec.logits_to_integer_cdf(logits)

    assert cdf[0] == 0
    assert cdf[-1] == TOTAL_FREQ
    assert len(cdf) == 261
    # Strictly monotonic
    diffs = np.diff(cdf)
    assert np.all(diffs >= 1)


def test_lossless_roundtrip_compression(mock_codec):
    original_text = "Language modeling is compression: Shannon, Solomonoff, and Delétang et al."
    payload, num_tokens, stats = mock_codec.compress(original_text)

    assert isinstance(payload, bytes)
    assert len(payload) > 0
    assert num_tokens > 0
    assert isinstance(stats, CompressionStats)

    reconstructed = mock_codec.decompress(payload, num_tokens)
    assert reconstructed == original_text, f"Mismatch: {reconstructed} != {original_text}"


def test_generative_manifold_decompression(mock_codec):
    seed = b"1234567890abcdef"
    unpacked = mock_codec.decompress_seed(seed, num_tokens=20)

    assert isinstance(unpacked, str)
    assert len(unpacked) > 0

    # Determinism: same seed yields same decompression
    unpacked_2 = mock_codec.decompress_seed(seed, num_tokens=20)
    assert unpacked == unpacked_2

# end of file: unplow/tests/test_paradigm2.py
