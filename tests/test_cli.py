# Path: unplow/tests/test_cli.py
"""Unit tests for unplow command-line interface argument parsing and commands."""

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

# end of file: unplow/tests/test_cli.py
