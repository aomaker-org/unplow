# Path: unplow/tests/test_incremental_pipeline.py
"""Unit tests for the incremental multi-domain training pipeline.

Validates discrete epoch increments, loss/perplexity tracking, parameter
isolation for large backbones, and disk telemetry guardrails.
"""

import pytest
import torch
import torch.nn as nn

from unplow.incremental_pipeline import IncrementalTrainer
from unplow.models import load_causal_lm


def test_incremental_trainer_roundtrip(tmp_path):
    manifest_file = tmp_path / "manifest.toml"
    p_str = str(tmp_path).replace("\\", "/")
    manifest_content = f"""
[manifest]
version = "1.0.0"
target_free_percentage = 15.0
rclone_destination = "gdrive:transfer/test/"

[offload_targets.caches]
path = "{p_str}"
items = []

[offload_targets.checkpoints]
path = "{p_str}"
items = []

[offload_targets.archives]
path = "{p_str}"
pattern = "*.unp"
"""
    manifest_file.write_text(manifest_content.strip(), encoding="utf-8")

    tokenizer, model, _ = load_causal_lm("mock")
    trainer = IncrementalTrainer(
        model=model,
        tokenizer=tokenizer,
        min_free_pct=0.01,
        manifest_path=str(manifest_file),
    )

    corpus = ["def add(a, b): return a + b\n", "def mul(a, b): return a * b\n"]
    success = trainer.train_domain_incrementally(
        domain_name="python",
        corpus_texts=corpus,
        total_epochs=2,
        steps_per_epoch=2,
        dry_run_offload=True,
    )
    assert success is True


def test_incremental_trainer_space_breach_halting(tmp_path):
    manifest_file = tmp_path / "manifest.toml"
    p_str = str(tmp_path).replace("\\", "/")
    manifest_file.write_text(
        f"""
[manifest]
version = "1.0.0"
target_free_percentage = 99.99
rclone_destination = "gdrive:transfer/test/"

[offload_targets.caches]
path = "{p_str}"
items = []

[offload_targets.checkpoints]
path = "{p_str}"
items = []

[offload_targets.archives]
path = "{p_str}"
pattern = "*.unp"
""".strip(),
        encoding="utf-8",
    )

    tokenizer, model, _ = load_causal_lm("mock")
    # Setting min_free_pct to 99.99 forces a breach that dry_run cannot clear
    trainer = IncrementalTrainer(
        model=model,
        tokenizer=tokenizer,
        min_free_pct=99.99,
        manifest_path=str(manifest_file),
    )

    corpus = ["def add(a, b): return a + b\n"]
    success = trainer.train_domain_incrementally(
        domain_name="python",
        corpus_texts=corpus,
        total_epochs=1,
        steps_per_epoch=1,
        dry_run_offload=True,
    )
    # Reclaiming failed to satisfy 99.99% free space floor, so training halts
    assert success is False


def test_incremental_trainer_large_model_isolation():
    class DummyLargeModel(nn.Module):
        def __init__(self, vocab_size: int = 260):
            super().__init__()
            # 12 million parameters to trigger isolation guard
            self.linear1 = nn.Linear(3000, 4000)
            self.layers = nn.ModuleList([nn.Linear(10, 10)])
            self.vocab_size = vocab_size

        def forward(self, input_ids, **kwargs):
            seq_len = input_ids.shape[1]
            logits = torch.randn(1, seq_len, self.vocab_size, requires_grad=True)
            return type("Output", (), {"logits": logits, "loss": None})()

    model = DummyLargeModel()
    assert sum(p.numel() for p in model.parameters()) > 10_000_000

    tokenizer, _, _ = load_causal_lm("mock")
    trainer = IncrementalTrainer(model, tokenizer)
    # Calling train_domain_incrementally should isolate the final layer
    trainer.train_domain_incrementally(
        domain_name="gencode",
        corpus_texts=["code fragment"],
        total_epochs=1,
        steps_per_epoch=1,
    )

    assert model.linear1.weight.requires_grad is False
    assert model.layers[-1].weight.requires_grad is True


def test_incremental_trainer_checkpoint_persistence_and_status(tmp_path):
    manifest_file = tmp_path / "manifest.toml"
    p_str = str(tmp_path).replace("\\", "/")
    manifest_file.write_text(
        f"""
[manifest]
version = "1.0.0"
target_free_percentage = 15.0
rclone_destination = "gdrive:transfer/test/"

[offload_targets.caches]
path = "{p_str}"
items = []

[offload_targets.checkpoints]
path = "{p_str}"
items = []

[offload_targets.archives]
path = "{p_str}"
pattern = "*.unp"
""".strip(),
        encoding="utf-8",
    )

    tokenizer, model, _ = load_causal_lm("mock")
    cp_dir = tmp_path / "checkpoints"
    target_dir = tmp_path / "targets"
    trainer = IncrementalTrainer(
        model=model,
        tokenizer=tokenizer,
        min_free_pct=0.01,
        manifest_path=str(manifest_file),
        base_checkpoint_dir=str(cp_dir),
        base_target_dir=str(target_dir),
    )

    corpus = ["def test_func():\n    return 42\n"]
    success = trainer.train_domain_incrementally(
        domain_name="python",
        corpus_texts=corpus,
        total_epochs=1,
        steps_per_epoch=1,
    )
    assert success is True

    # Validate that adapter weights and config were saved
    expected_pt = cp_dir / "till-python-lora" / "adapter_model.pt"
    expected_cfg = cp_dir / "till-python-lora" / "adapter_config.json"
    assert expected_pt.exists()
    assert expected_cfg.exists()
    assert expected_pt.stat().st_size > 0

    # Validate that print_status_summary runs cleanly
    trainer.print_status_summary(iteration=1)
    assert "python" in trainer.domain_reports
    assert trainer.domain_reports["python"]["trainable_params"] > 0


# end of file: unplow/tests/test_incremental_pipeline.py

