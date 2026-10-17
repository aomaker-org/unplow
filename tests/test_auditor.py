# Path: unplow/tests/test_auditor.py
"""
Unit tests for ModelInternalAuditor and LayerAuditReport (Phase 7).
Validates layer-wise Logit Lens projections, FFN norms, and entropy profiles
using deterministic mock models and tokenizers.
"""

import pytest
import torch
import torch.nn as nn

from unplow.auditor import LayerAuditReport, ModelInternalAuditor
from unplow.models import MockByteTokenizer, MockCausalLM


def test_model_internal_auditor_layer_flow():
    """Verify layer-by-layer projection, norm tracking, and candidate tokens."""
    tokenizer = MockByteTokenizer(vocab_size=260)
    model = MockCausalLM(vocab_size=260, hidden_dim=32)
    auditor = ModelInternalAuditor(model, tokenizer)

    prompt = "def fibonacci(n):"
    reports = auditor.audit_sequence(prompt, top_k=5)

    # MockCausalLM has 4 transformer layers
    assert len(reports) == 4

    for idx, report in enumerate(reports):
        assert isinstance(report, LayerAuditReport)
        assert report.layer_idx == idx
        assert isinstance(report.top_predicted_token, str)
        assert isinstance(report.top_logit_value, float)
        assert report.residual_l2_norm > 0.0
        assert report.ffn_output_l2_norm > 0.0
        assert 0.0 <= report.top_probability <= 1.0
        assert report.entropy_bits >= 0.0
        assert len(report.top_k_candidates) == 5

        d = report.to_dict()
        assert d["layer_idx"] == idx
        assert "residual_l2_norm" in d
        assert "top_k_candidates" in d


def test_auditor_empty_prompt_and_format_table():
    """Verify empty prompt returns empty list and table formatter renders clean output."""
    tokenizer = MockByteTokenizer(vocab_size=260)
    model = MockCausalLM(vocab_size=260, hidden_dim=32)
    auditor = ModelInternalAuditor(model, tokenizer)

    # Empty prompt
    assert auditor.audit_sequence("") == []

    # Format table
    reports = auditor.audit_sequence("test prompt")
    table = auditor.format_audit_table(reports, prompt="test prompt")
    assert "=== Layer-Wise Internal Information Audit ===" in table
    assert "Prompt: 'test prompt'" in table
    assert "Layer" in table
    assert "Top Predicted" in table
    assert "Res Norm" in table


def test_auditor_missing_hidden_states_raises():
    """Verify ValueError is raised if model does not output hidden states."""
    tokenizer = MockByteTokenizer(vocab_size=260)

    class DummyNoHiddenLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Identity()
            self.lm_head = nn.Linear(16, 260)

        def forward(self, input_ids, output_hidden_states=False):
            class Out:
                logits = torch.zeros(1, 4, 260)
            return Out()

    dummy = DummyNoHiddenLM()
    auditor = ModelInternalAuditor(dummy, tokenizer)
    with pytest.raises(ValueError, match="did not return hidden_states"):
        auditor.audit_sequence("hello")


# end of file: unplow/tests/test_auditor.py
