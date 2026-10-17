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


# end of file: unplow/tests/test_cli.py
