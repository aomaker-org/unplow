# Path: unplow/tests/test_conditioning.py
"""Unit tests for zero-bit domain conditioning, nibble packing, and fast PyTorch CDF."""

import numpy as np
import pytest
import torch

from unplow.conditioning import (
    DOMAIN_PRIORS,
    logits_to_integer_cdf_fast,
    pack_flags_with_domain,
    prewarm_kv_cache_for_domain,
    resolve_domain_id,
    select_optimal_domain_mdl,
    unpack_flags_with_domain,
)
from unplow.container import FLAG_LOSSLESS_VERIFIED, pack_unp_container, unpack_unp_container
from unplow.models import MockByteTokenizer, MockCausalLM
from unplow.paradigm2 import LLMArithmeticCodec, TOTAL_FREQ


def test_flag_nibble_packing_roundtrip():
    for dom_id in range(16):
        for verify in (True, False):
            for bg in (True, False):
                packed = pack_flags_with_domain(verify_hash=verify, domain_id=dom_id, bigram_cache=bg)
                assert 0 <= packed <= 255
                un_verify, un_dom = unpack_flags_with_domain(packed)
                assert un_verify == verify
                assert un_dom == dom_id
                assert bool(packed & 0x04) == bg


def test_resolve_domain_id():
    assert resolve_domain_id(0) == 0
    assert resolve_domain_id("raw") == 0
    assert resolve_domain_id("prose_tech") == 1
    assert resolve_domain_id("python_code") == 2
    assert resolve_domain_id("c_firmware") == 3
    assert resolve_domain_id("json_telemetry") == 4
    with pytest.raises(ValueError):
        resolve_domain_id(16)
    with pytest.raises(ValueError):
        resolve_domain_id("non_existent_domain")


def test_logits_to_integer_cdf_fast():
    torch.manual_seed(42)
    vocab_size = 260
    logits = torch.randn(vocab_size)

    cdf = logits_to_integer_cdf_fast(logits)
    assert len(cdf) == vocab_size + 1
    assert cdf[0] == 0
    assert cdf[-1] == TOTAL_FREQ
    # Strictly monotonic
    diffs = np.diff(cdf)
    assert np.all(diffs >= 1)
    assert int(np.sum(diffs)) == TOTAL_FREQ


def test_prewarm_kv_cache():
    tok = MockByteTokenizer(vocab_size=260)
    model = MockCausalLM(vocab_size=260, hidden_dim=32)

    # Domain 0 (raw)
    curr_input, past_kv = prewarm_kv_cache_for_domain(
        model=model, tokenizer=tok, domain_id=0, bos_token_id=0, device="cpu"
    )
    assert curr_input.item() == 0
    assert past_kv is None

    # Domain 1 (prose_tech)
    curr_input, past_kv = prewarm_kv_cache_for_domain(
        model=model, tokenizer=tok, domain_id=1, bos_token_id=0, device="cpu"
    )
    assert curr_input is not None
    assert past_kv is not None


def test_select_optimal_domain_mdl():
    tok = MockByteTokenizer(vocab_size=260)
    model = MockCausalLM(vocab_size=260, hidden_dim=32)

    sample_text = "def test_function():\n    return 42\n"
    best_id, best_name, bpb = select_optimal_domain_mdl(
        model=model,
        tokenizer=tok,
        plaintext=sample_text,
        device="cpu",
        bos_token_id=0,
    )
    assert 0 <= best_id <= 15
    assert isinstance(best_name, str)
    assert isinstance(bpb, float)
    assert bpb > 0.0


def test_domain_conditioned_container_roundtrip():
    tok = MockByteTokenizer(vocab_size=260)
    model = MockCausalLM(vocab_size=260, hidden_dim=32)
    codec = LLMArithmeticCodec(model=model, tokenizer=tok, device="cpu")

    payload_text = "Lossless container test with 18-byte self-verifying header and python code."

    # Test explicit domain conditioning
    for dom in (0, 1, 2, "python_code", "auto"):
        container_bytes, meta, stats = codec.compress_to_container(
            text=payload_text,
            max_ctx_len=64,
            domain_id=dom,
        )
        assert len(container_bytes) >= 18
        assert meta.is_verified_lossless is True
        if dom == "auto":
            assert 0 <= meta.domain_id <= 15
        elif isinstance(dom, int):
            assert meta.domain_id == dom
        elif dom == "python_code":
            assert meta.domain_id == 2

        # Roundtrip decompression
        reconstructed, dec_meta, is_valid = codec.decompress_from_container(container_bytes)
        assert is_valid is True
        assert reconstructed == payload_text
        assert dec_meta.domain_id == meta.domain_id

# end of file: unplow/tests/test_conditioning.py
