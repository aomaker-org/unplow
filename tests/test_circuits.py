# Path: unplow/tests/test_circuits.py
"""Unit tests for coupled SwiGLU circuits and anisotropic logit lens."""

import pytest
import torch

from unplow.circuits import AssociativeCircuitTrace, SwiGLUCircuitDecompressor
from unplow.models import MockByteTokenizer, MockCausalLM


@pytest.fixture
def mock_model_and_tokenizer():
    model = MockCausalLM(vocab_size=260, hidden_dim=32)
    tokenizer = MockByteTokenizer(vocab_size=260)
    return model, tokenizer


def test_is_trivial_numeral_or_punct():
    assert SwiGLUCircuitDecompressor._is_trivial_numeral_or_punct("8")
    assert SwiGLUCircuitDecompressor._is_trivial_numeral_or_punct("  6  ")
    assert SwiGLUCircuitDecompressor._is_trivial_numeral_or_punct("...")
    assert SwiGLUCircuitDecompressor._is_trivial_numeral_or_punct("a")  # single char
    assert not SwiGLUCircuitDecompressor._is_trivial_numeral_or_punct("operator")
    assert not SwiGLUCircuitDecompressor._is_trivial_numeral_or_punct(" loss")


def test_extract_coupled_circuits(mock_model_and_tokenizer):
    model, tokenizer = mock_model_and_tokenizer
    decompressor = SwiGLUCircuitDecompressor(model=model, tokenizer=tokenizer)

    circuits = decompressor.extract_coupled_circuits(
        layer_indices=[1, 2],
        candidates_per_layer=4,
        top_k_tokens=3,
        suppress_numerals=False,
    )

    assert len(circuits) > 0
    first = circuits[0]
    assert isinstance(first, AssociativeCircuitTrace)
    assert first.centered_l2_norm > 0.0
    assert isinstance(first.selectivity_margin, float)
    assert len(first.trigger_key_tokens) > 0
    assert len(first.target_value_tokens) > 0
    assert isinstance(first.synthesized_seed, str)

    # Check sorting by product of centered norm and selectivity margin
    for i in range(len(circuits) - 1):
        score_curr = circuits[i].centered_l2_norm * circuits[i].selectivity_margin
        score_next = circuits[i + 1].centered_l2_norm * circuits[i + 1].selectivity_margin
        assert score_curr >= score_next - 1e-5


def test_harvest_associative_seeds(mock_model_and_tokenizer):
    model, tokenizer = mock_model_and_tokenizer
    decompressor = SwiGLUCircuitDecompressor(model=model, tokenizer=tokenizer)

    seeds = decompressor.harvest_associative_seeds(
        max_seeds=5,
        layer_indices=[0, 1, 2],
        suppress_numerals=False,
    )

    assert isinstance(seeds, list)
    assert len(seeds) <= 5
    for s in seeds:
        assert isinstance(s, str)

# end of file: unplow/tests/test_circuits.py
