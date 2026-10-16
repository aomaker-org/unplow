# Path: unplow/tests/test_models.py
"""Unit tests for unplow model loader and deterministic mock models."""

from pathlib import Path
import pytest
import torch

from unplow.models import (
    MockByteTokenizer,
    MockCausalLM,
    audit_system_models,
    resolve_model_path,
)


def test_mock_byte_tokenizer():
    tok = MockByteTokenizer(vocab_size=260)
    text = "unplow: LLM Decompression Engine"
    token_ids = tok.encode(text)

    assert len(token_ids) == len(text.encode("utf-8"))
    assert all(2 <= t < 260 for t in token_ids)

    reconstructed = tok.decode(token_ids)
    assert reconstructed == text


def test_mock_causal_lm():
    model = MockCausalLM(vocab_size=260, hidden_dim=32)
    input_ids = torch.tensor([[0, 10, 20, 30]])
    out = model(input_ids)

    assert out.logits.shape == (1, 4, 260)
    assert out.logits.dtype == torch.float32


def test_resolve_model_path():
    resolved = resolve_model_path("auto")
    assert isinstance(resolved, str)
    assert len(resolved) > 0

    custom = resolve_model_path("custom/test/path")
    assert "custom/test/path" in custom


def test_audit_system_models():
    models = audit_system_models()
    assert isinstance(models, list)
    # The local system has Qwen2.5-1.5B safetensors or GGUF or Ollama
    assert len(models) >= 1
    found_types = {m.get("type") for m in models}
    assert ("safetensors_snapshot" in found_types or "gguf" in found_types or "ollama" in found_types)

# end of file: unplow/tests/test_models.py
