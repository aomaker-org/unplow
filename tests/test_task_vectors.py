# Path: unplow/tests/test_task_vectors.py
"""Unit tests for Phase 6 CrossSubstrateTaskVectorEngine.

Tests GGUF-to-Safetensors task delta SVD extraction, Logit Lens projection,
and steered decompression without dual-model RAM overhead.
"""

import numpy as np
import torch
from unplow.gguf_inspector import GGML_TYPE_F32, build_synthetic_gguf_v3_bytes
from unplow.models import DeterministicMockCausalLM, DeterministicMockTokenizer
from unplow.task_vectors import (
    CrossSubstrateTaskVectorEngine,
    TaskVectorSpectrumReport,
    TaskVectorSteeredRecord,
)


def test_compute_layer_task_vector_and_steering(tmp_path):
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    engine = CrossSubstrateTaskVectorEngine(model, tokenizer)

    ref_w = model.model.layers[1].mlp.down_proj.weight.detach().cpu().numpy()
    perturbed_w = ref_w.copy()
    perturbed_w[:, 0] += 3.0  # Inject structured task delta into column 0

    gguf_bytes = build_synthetic_gguf_v3_bytes(
        "blk.1.ffn_down.weight", perturbed_w, ggml_type=GGML_TYPE_F32
    )
    gguf_file = tmp_path / "coder_delta.gguf"
    gguf_file.write_bytes(gguf_bytes)

    vec, report = engine.compute_layer_task_vector(
        gguf_file, layer_idx=1, top_k_tokens=4, top_k_neurons=3
    )
    assert isinstance(report, TaskVectorSpectrumReport)
    assert report.layer_idx == 1
    assert report.frobenius_delta_norm > 0.0
    assert report.top_divergent_neurons[0] == 0
    assert len(report.promoted_tokens) == 4
    assert len(report.suppressed_tokens) == 4
    assert vec.shape[0] == ref_w.shape[0]
    assert 0.0 < report.explained_variance_ratio <= 1.0

    rep_dict = report.to_dict()
    assert rep_dict["layer_idx"] == 1
    assert "top_singular_value" in rep_dict

    steered = engine.steer_with_gguf_task_vector(
        gguf_file,
        layer_idx=1,
        seed_prompt="def ",
        seq_len=10,
        steering_gain=4.0,
        steering_steps=4,
    )
    assert isinstance(steered, TaskVectorSteeredRecord)
    assert steered.layer_idx == 1
    assert len(steered.extracted_text) >= 4
    assert steered.steering_gain == 4.0
    assert len(model.model.layers[1].mlp.down_proj._forward_hooks) == 0

    steered_dict = steered.to_dict()
    assert steered_dict["layer_idx"] == 1
    assert "memorization_score" in steered_dict


def test_align_gguf_matrix_orientations():
    engine = CrossSubstrateTaskVectorEngine(
        DeterministicMockCausalLM(vocab_size=128), DeterministicMockTokenizer()
    )
    target = (32, 64)
    mat_exact = np.ones((32, 64), dtype=np.float32)
    t_exact = engine._align_gguf_matrix(mat_exact, target)
    assert tuple(t_exact.shape) == target

    mat_trans = np.ones((64, 32), dtype=np.float32)
    t_trans = engine._align_gguf_matrix(mat_trans, target)
    assert tuple(t_trans.shape) == target

    mat_flat = np.ones(2048, dtype=np.float32)
    t_flat = engine._align_gguf_matrix(mat_flat, target)
    assert tuple(t_flat.shape) == target

# end of file: unplow/tests/test_task_vectors.py
