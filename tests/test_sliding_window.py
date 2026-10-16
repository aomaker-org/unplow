# Path: unplow/tests/test_sliding_window.py
"""Unit tests for sliding-window KV-cache horizon and container roundtrips."""

import pytest
import torch

from unplow.container import slice_kv_cache
from unplow.models import MockByteTokenizer, MockCausalLM
from unplow.paradigm2 import LLMArithmeticCodec


def test_slice_kv_cache_tuples():
    # Legacy tuple format: layers of (k, v), shape [batch, heads, seq_len, head_dim]
    k1 = torch.randn(1, 2, 64, 16)
    v1 = torch.randn(1, 2, 64, 16)
    k2 = torch.randn(1, 2, 64, 16)
    v2 = torch.randn(1, 2, 64, 16)
    past = ((k1, v1), (k2, v2))

    sliced = slice_kv_cache(past, keep_tokens=16)
    assert len(sliced) == 2
    assert sliced[0][0].shape == (1, 2, 16, 16)
    assert sliced[0][1].shape == (1, 2, 16, 16)


def test_compress_decompress_container_roundtrip():
    tok = MockByteTokenizer(vocab_size=260)
    model = MockCausalLM(vocab_size=260, hidden_dim=32)
    codec = LLMArithmeticCodec(model=model, tokenizer=tok, device="cpu")

    sample_doc = "Bounded memory KV-cache sliding window preserves exact bit-level arithmetic symmetry."
    container_bytes, meta, stats = codec.compress_to_container(sample_doc, max_ctx_len=16)

    assert len(container_bytes) > 18
    assert meta.num_tokens == len(tok.encode(sample_doc))
    assert meta.max_ctx_len == 16

    reconstructed, dec_meta, is_valid = codec.decompress_from_container(container_bytes)
    assert reconstructed == sample_doc
    assert is_valid is True

# end of file: unplow/tests/test_sliding_window.py
