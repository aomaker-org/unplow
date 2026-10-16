# Path: unplow/tests/test_forensics.py
"""Unit tests for unplow forensics, cryptographic hashes, and telemetry."""

from pathlib import Path
import pytest

from unplow.forensics import (
    ForensicAuditLogger,
    compute_file_sha256,
    compute_str_sha256,
    get_telemetry_snapshot,
)


def test_sha256_computation(tmp_path: Path):
    text = "Lossless arithmetic decompression over latent manifolds"
    expected = "f050b4be93466ae7f0be9cbcfc6c42982d6da6bf802611756535dbe1ee0d0f01"
    digest = compute_str_sha256(text)
    assert len(digest) == 64

    test_file = tmp_path / "test.bin"
    test_file.write_bytes(text.encode("utf-8"))
    file_digest = compute_file_sha256(test_file)
    assert file_digest == digest


def test_telemetry_snapshot():
    telem = get_telemetry_snapshot()
    assert "hostname" in telem
    assert "pid" in telem
    assert "rss_mb" in telem
    assert isinstance(telem["rss_mb"], float)
    assert telem["rss_mb"] >= 0.0


def test_audit_logger(tmp_path: Path):
    logger = ForensicAuditLogger(base_dir=str(tmp_path / "logs"))
    payload = logger.log_event(
        phase="TEST_PHASE",
        task_id="task-12345",
        action="probe",
        details={"samples": 4},
    )
    assert payload["task_id"] == "task-12345"
    assert payload["phase"] == "TEST_PHASE"
    assert logger.journal_path.exists()
    assert logger.trace_log_path.exists()

    entries = logger.get_recent_entries(limit=10)
    assert len(entries) == 1
    assert entries[0]["task_id"] == "task-12345"

# end of file: unplow/tests/test_forensics.py
