# Path: unplow/tests/test_gguf_inspector.py
"""Unit tests for Phase 5 GGUFBinaryInspector and Vectorized Block Dequantizers.

Covers GGUF v3 header parsing, vectorized Q8_0/F32 dequantization, and
cross-substrate Task-Vector delta analysis against PyTorch matrices.
"""

import numpy as np
import pytest
import torch
from unplow.gguf_inspector import (
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGUFBinaryInspector,
    build_synthetic_gguf_v3_bytes,
)


def test_gguf_v3_header_and_q8_0_dequantization(tmp_path):
    rng = np.random.default_rng(42)
    orig_matrix = rng.standard_normal((4, 32), dtype=np.float32) * 1.5
    gguf_bytes = build_synthetic_gguf_v3_bytes(
        "blk.16.ffn_down.weight", orig_matrix, ggml_type=GGML_TYPE_Q8_0
    )

    gguf_file = tmp_path / "synthetic_q8_0.gguf"
    gguf_file.write_bytes(gguf_bytes)

    inspector = GGUFBinaryInspector(gguf_file)
    summary = inspector.parse_header()

    assert summary.version == 3
    assert summary.tensor_count == 1
    assert summary.metadata.get("general.architecture") == "qwen2"
    assert "blk.16.ffn_down.weight" in summary.tensors

    dequant = inspector.load_dequantized_tensor("blk.16.ffn_down.weight")
    assert dequant.shape == (4, 32)
    # Q8_0 quantization reconstruction error should be < 2% of dynamic range
    max_err = float(np.max(np.abs(dequant - orig_matrix)))
    assert max_err < 0.05


def test_gguf_reference_tensor_delta_comparison(tmp_path):
    orig_matrix = np.ones((2, 32), dtype=np.float32)
    gguf_bytes = build_synthetic_gguf_v3_bytes(
        "blk.0.ffn_down.weight", orig_matrix, ggml_type=GGML_TYPE_F32
    )
    gguf_file = tmp_path / "synthetic_f32.gguf"
    gguf_file.write_bytes(gguf_bytes)

    inspector = GGUFBinaryInspector(gguf_file)
    ref_tensor = torch.ones((2, 32), dtype=torch.float32)
    ref_tensor[:, 7] = 5.0  # Perturb column 7 to verify divergent column detection

    report = inspector.compare_with_reference_tensor(
        "blk.0.ffn_down.weight", ref_tensor, top_k_cols=3
    )
    assert report.ggml_type_name == "F32"
    assert report.shape == (2, 32)
    assert report.top_divergent_columns[0] == 7
    assert 0.0 < report.cosine_similarity < 1.0


def test_gguf_q4_0_dequantization_vectorized():
    # Test 32-element Q4_0 block dequantization
    # Scale = 1.0 (fp16), 16 bytes of nibbles
    scale_bytes = np.float16(1.0).tobytes()
    # Nibbles with offset 8: low=8 (0), high=9 (+1) => byte = 0x98
    nibble_bytes = bytes([0x98] * 16)
    block_bytes = scale_bytes + nibble_bytes
    dequant = GGUFBinaryInspector.dequantize_buffer(block_bytes, GGML_TYPE_Q4_0, 32)
    assert dequant.shape == (32,)
    # Low nibbles (indices 0..15) are 8 - 8 = 0.0
    assert np.allclose(dequant[:16], 0.0)
    # High nibbles (indices 16..31) are 9 - 8 = 1.0
    assert np.allclose(dequant[16:], 1.0)


def test_gguf_invalid_magic_rejected(tmp_path):
    bad_file = tmp_path / "bad.gguf"
    bad_file.write_bytes(b"BADMAGIC" + b"\x00" * 32)
    inspector = GGUFBinaryInspector(bad_file)
    with pytest.raises(ValueError, match="Invalid GGUF magic"):
        inspector.parse_header()

# end of file: unplow/tests/test_gguf_inspector.py
