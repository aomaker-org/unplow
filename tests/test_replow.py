# Path: unplow/tests/test_replow.py
"""
Unit tests for ReplowEngine (AST refactoring, syntax validation,
training dataset staging, distillation steps, and Bayesian CTS
container archiving of model work products).
"""

from pathlib import Path
import pytest

from unplow.models import DeterministicMockCausalLM, DeterministicMockTokenizer
from unplow.replow import ReplowArtifact, ReplowEngine


def test_replow_refactor_and_archive(tmp_path):
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    engine = ReplowEngine(model, tokenizer)

    raw_snippet = "```python\ndef add(a, b):\n    return a + b\n```"
    out_file = tmp_path / "refactored_add.unp"

    artifact = engine.process_and_archive(raw_snippet, out_file, domain_id=2)
    assert isinstance(artifact, ReplowArtifact)
    assert artifact.syntax_valid is True
    assert "def add" in artifact.refactored_text
    assert "```" not in artifact.refactored_text
    assert out_file.exists()
    assert artifact.compressed_bytes_len > 0
    assert artifact.bits_per_byte > 0.0

    d = artifact.to_dict()
    assert d["syntax_valid"] is True
    assert d["container_path"] == str(out_file)


def test_replow_refactor_fallback_and_invalid():
    # Fragment that has return outside function, recovered via stub
    fragment = "return x * 2"
    cleaned, valid = ReplowEngine.refactor_source_code(fragment)
    assert valid is True
    assert "__refactored_stub__" in cleaned

    # Completely broken syntax
    broken = "def (: invalid syntax ))"
    cleaned_b, valid_b = ReplowEngine.refactor_source_code(broken)
    assert valid_b is False


def test_replow_stage_dataset_and_train(tmp_path):
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    engine = ReplowEngine(model, tokenizer)

    samples = [
        "def foo():\n    return 42",
        "def bar(x):\n    return x + 1",
    ]
    ds_path = tmp_path / "train.jsonl"
    staged = engine.stage_training_dataset(samples, ds_path)
    assert staged.exists()
    lines = staged.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    # Run distillation training step
    loss = engine.train_step(samples, learning_rate=1e-3, steps=1)
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_replow_full_pipeline(tmp_path):
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    engine = ReplowEngine(model, tokenizer)

    raw_snippet = "def multiply(x, y):\n    return x * y"
    out_file = tmp_path / "multiply.unp"

    artifact, loss = engine.run_pipeline(
        raw_code=raw_snippet,
        output_unp_path=out_file,
        train_steps=1,
        domain_id=2,
    )
    assert artifact.syntax_valid is True
    assert out_file.exists()
    assert loss is not None
    assert loss >= 0.0


# end of file: unplow/tests/test_replow.py
