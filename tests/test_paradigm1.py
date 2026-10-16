# Path: unplow/tests/test_paradigm1.py
"""Unit tests for Paradigm 1: Lossy Static Weight Decompression."""

from pathlib import Path
import pytest
import torch

from unplow.models import MockByteTokenizer, MockCausalLM
from unplow.paradigm1 import ExtractedRecord, WeightDecompressor


@pytest.fixture
def mock_decompressor():
    tok = MockByteTokenizer(vocab_size=260)
    model = MockCausalLM(vocab_size=260, hidden_dim=32)
    return WeightDecompressor(model=model, tokenizer=tok, device="cpu")


def test_compute_sequence_bits(mock_decompressor):
    bits = mock_decompressor.compute_sequence_bits("hello world")
    assert isinstance(bits, float)
    assert bits > 0.0

    # Empty string should return 0.0
    assert mock_decompressor.compute_sequence_bits("") == 0.0


def test_probe_weight_basins(mock_decompressor):
    records = mock_decompressor.probe_weight_basins(
        num_samples=2,
        seq_len=16,
        explore_steps=4,
        seed_prefixes=["def ", "class "],
    )

    assert len(records) == 2
    for r in records:
        assert isinstance(r, ExtractedRecord)
        assert len(r.text) > 0
        assert r.model_bits_per_byte > 0.0
        assert r.zlib_bits_per_byte > 0.0
        assert r.memorization_score >= 0.0
        assert r.casing_sensitivity >= 0.0

    # Ensure records are sorted descending by memorization_score
    assert records[0].memorization_score >= records[1].memorization_score


def test_save_records_jsonl(mock_decompressor, tmp_path: Path):
    records = mock_decompressor.probe_weight_basins(num_samples=1, seq_len=10)
    dest = tmp_path / "basins.jsonl"
    mock_decompressor.save_records_to_jsonl(records, dest)
    assert dest.exists()
    assert dest.stat().st_size > 0

# end of file: unplow/tests/test_paradigm1.py
