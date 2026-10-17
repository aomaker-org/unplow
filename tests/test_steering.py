# Path: unplow/tests/test_steering.py
"""Unit tests for Phase 4 MechanisticSteeringEngine.

Covers RMSNorm-folded SwiGLU circuit extraction over mean-centered embeddings
and causal forward-hook activation-steered training basin decompression.
"""

from unplow.models import DeterministicMockCausalLM, DeterministicMockTokenizer
from unplow.steering import (
    FoldedCircuitTrace,
    MechanisticSteeringEngine,
    SteeredBasinRecord,
)


def test_rmsnorm_folded_circuit_extraction():
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    engine = MechanisticSteeringEngine(model, tokenizer)

    traces = engine.extract_rmsnorm_folded_circuits(
        layer_indices=[1, 2], candidates_per_layer=6, top_k_tokens=3
    )
    assert isinstance(traces, list)
    assert len(traces) > 0
    first = traces[0]
    assert isinstance(first, FoldedCircuitTrace)
    assert first.layer_idx in (1, 2)
    assert first.centered_l2_norm > 0.0
    assert len(first.folded_trigger_keys) > 0
    assert len(first.target_values) > 0


def test_decompress_neuron_basin_and_hook_cleanup():
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    engine = MechanisticSteeringEngine(model, tokenizer)

    target_down_proj = model.model.layers[1].mlp.down_proj
    initial_hooks = len(target_down_proj._forward_hooks)

    rec = engine.decompress_neuron_basin(
        layer_idx=1,
        neuron_idx=0,
        seed_prompt="def ",
        seq_len=12,
        steering_gain=3.5,
        steering_steps=6,
    )
    assert isinstance(rec, SteeredBasinRecord)
    assert rec.layer_idx == 1
    assert rec.neuron_idx == 0
    assert rec.steering_gain == 3.5
    assert len(rec.extracted_text) >= 4
    assert rec.model_bpb > 0.0
    assert rec.zlib_bpb > 0.0

    # Verify forward hook was cleanly removed in finally block
    assert len(target_down_proj._forward_hooks) == initial_hooks

# end of file: unplow/tests/test_steering.py
