# Path: unplow/tests/test_cli.py
"""Unit tests for unplow command-line interface argument parsing and commands."""

import json
import subprocess
import sys
import pytest


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "unplow - Autonomous LLM Decompression Engine" in result.stdout
    assert "probe" in result.stdout
    assert "inspect" in result.stdout
    assert "compress" in result.stdout
    assert "decompress" in result.stdout
    assert "archive" in result.stdout
    assert "unarchive" in result.stdout
    assert "manifold" in result.stdout
    assert "benchmark" in result.stdout
    assert "models" in result.stdout
    assert "status" in result.stdout
    assert "daemon" in result.stdout


def test_cli_models():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "models"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Local Model Substrates & Coordinates" in result.stdout


def test_cli_status():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "status"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "unplow System Status" in result.stdout
    assert "Hostname" in result.stdout
    assert "Task Queue" in result.stdout


def test_cli_circuits_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "circuits", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--candidates" in result.stdout
    assert "--top-tokens" in result.stdout
    assert "--allow-numerals" in result.stdout
    assert "--harvest" in result.stdout


def test_cli_archive_domain_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "archive", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--domain" in result.stdout
    assert "--auto-domain" in result.stdout
    assert "--cache-mix" in result.stdout


def test_cli_circuits_folded_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "circuits", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--folded" in result.stdout


def test_cli_steer_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "steer", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--layer" in result.stdout
    assert "--neuron" in result.stdout
    assert "--gain" in result.stdout


def test_cli_profile_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "profile", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--cache-weight" in result.stdout


def test_cli_gguf_inspect_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "gguf-inspect", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "path" in result.stdout
    assert "--compare" in result.stdout
    assert "--tensor" in result.stdout


def test_cli_subspace_steer_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "subspace-steer", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--layer" in result.stdout
    assert "--neurons" in result.stdout
    assert "--gain" in result.stdout


def test_cli_archive_bigram_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "archive", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--bigram-mix" in result.stdout
    assert "--bayes-mix" in result.stdout


def test_cli_task_vector_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "task-vector", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--gguf" in result.stdout
    assert "--layer" in result.stdout
    assert "--steer" in result.stdout
    assert "--prompt" in result.stdout


def test_cli_audit_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "audit", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "prompt" in result.stdout
    assert "--top-k" in result.stdout
    assert "--json" in result.stdout


def test_cli_audit_execution_mock():
    result = subprocess.run(
        [
            sys.executable, "-m", "unplow.cli", "audit", "def test():",
            "-m", "mock", "--json"
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["layer_count"] == 4
    assert len(data["layers"]) == 4


def test_cli_replow_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "replow", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--raw" in result.stdout
    assert "--stage-dataset" in result.stdout
    assert "--train-steps" in result.stdout


def test_cli_replow_mock(tmp_path):
    out_unp = tmp_path / "mock_replow.unp"
    result = subprocess.run(
        [
            sys.executable, "-m", "unplow.cli", "replow",
            "--raw", "def compute(): return 100",
            "-o", str(out_unp),
            "-m", "mock",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["syntax_valid"] is True
    assert out_unp.exists()


def test_cli_eval_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "eval", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--artifact" in result.stdout
    assert "--test-suite" in result.stdout
    assert "--corpus" in result.stdout


def test_cli_eval_mock(tmp_path):
    code_file = tmp_path / "mock_solution.py"
    code_file.write_text("def is_even(n):\n    return n % 2 == 0\n", encoding="utf-8")

    test_file = tmp_path / "mock_tests.py"
    test_file.write_text(
        "assert is_even(4) is True\nassert is_even(3) is False\n",
        encoding="utf-8",
    )

    out_json = tmp_path / "eval_report.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "unplow.cli", "eval",
            "--artifact", str(code_file),
            "--test-suite", str(test_file),
            "-o", str(out_json),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["overall_status"] == "PASSED"
    assert data["functional_tests_passed"] == 2
    assert out_json.exists()


def test_cli_bundle_help():
    result = subprocess.run(
        [sys.executable, "-m", "unplow.cli", "bundle", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "source" in result.stdout
    assert "--output" in result.stdout


def test_cli_bundle_and_unbundle_mock(tmp_path):
    src_dir = tmp_path / "cli_bundle_src"
    src_dir.mkdir()
    (src_dir / "test.py").write_text("x = 42\n", encoding="utf-8")

    bundle_unp = tmp_path / "bundle.unp"
    res1 = subprocess.run(
        [
            sys.executable, "-m", "unplow.cli", "bundle",
            str(src_dir), "-o", str(bundle_unp),
            "-m", "mock", "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert res1.returncode == 0
    d1 = json.loads(res1.stdout)
    assert d1["total_files"] == 1
    assert bundle_unp.exists()

    restore_dir = tmp_path / "restored"
    res2 = subprocess.run(
        [
            sys.executable, "-m", "unplow.cli", "unbundle",
            str(bundle_unp), "-o", str(restore_dir),
            "-m", "mock", "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert res2.returncode == 0
    d2 = json.loads(res2.stdout)
    assert d2["total_files"] == 1
    assert (restore_dir / "test.py").read_text(encoding="utf-8") == "x = 42\n"


def test_cli_till_init(tmp_path):
    corpus_base = tmp_path / "till_corpus"
    result = subprocess.run(
        [
            sys.executable, "-m", "unplow.cli", "till-init",
            "--corpus-dir", str(corpus_base),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "telemetry" in data
    assert "python" in data["domains"]
    assert (corpus_base / "python").exists()


def test_cli_till_train(tmp_path):
    corpus_dir = tmp_path / "sample_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "sample.py").write_text("def ping(): return 'pong'\n", encoding="utf-8")
    out_unp = tmp_path / "till_sample.unp"

    result = subprocess.run(
        [
            sys.executable, "-m", "unplow.cli", "till-train",
            "--domain", "python",
            "--corpus", str(corpus_dir),
            "-o", str(out_unp),
            "-m", "mock",
            "--train-steps", "1",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["domain"] == "python"
    assert data["domain_id"] == 2
    assert out_unp.exists()


# end of file: unplow/tests/test_cli.py
