# Path: unplow/tests/test_mechanistic.py
"""Unit tests for mechanistic MLP weight inspection via Logit Lens."""

import pytest
import torch

from unplow.mechanistic import MechanisticWeightInspector, NeuronMemoryTrace
from unplow.models import MockByteTokenizer, MockCausalLM


@pytest.fixture
def mock_inspector():
    tok = MockByteTokenizer(vocab_size=260)
    model = MockCausalLM(vocab_size=260, hidden_dim=32)
    return MechanisticWeightInspector(model=model, tokenizer=tok)


def test_extract_mlp_memory_clusters(mock_inspector):
    traces = mock_inspector.extract_mlp_memory_clusters(
        layer_indices=[0, 1],
        top_neurons_per_layer=4,
        top_k_tokens=3,
    )

    assert len(traces) == 8
    for tr in traces:
        assert isinstance(tr, NeuronMemoryTrace)
        assert tr.layer_idx in (0, 1)
        assert tr.l2_norm >= 0.0
        assert len(tr.top_tokens) == 3
        assert len(tr.top_token_ids) == 3

    # Sorted descending by L2 norm
    assert traces[0].l2_norm >= traces[-1].l2_norm


def test_harvest_seed_prefixes(mock_inspector):
    seeds = mock_inspector.harvest_seed_prefixes(num_seeds=4)
    assert isinstance(seeds, list)
    for s in seeds:
        assert isinstance(s, str)
        assert len(s.strip()) >= 3

# end of file: unplow/tests/test_mechanistic.py
