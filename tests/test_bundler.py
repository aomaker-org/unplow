# Path: unplow/tests/test_bundler.py
"""Unit tests for WorkspaceBundler.

Tests multi-file directory tree bundling, Bayesian CTS compression,
and recursive unbundling integrity verification.
"""

from pathlib import Path
import pytest

from unplow.bundler import BundleManifestReport, WorkspaceBundler
from unplow.models import MockByteTokenizer, MockCausalLM


def test_workspace_bundler_roundtrip(tmp_path):
    src_dir = tmp_path / "project_src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("def main():\n    print('hello')\n", encoding="utf-8")
    sub = src_dir / "utils"
    sub.mkdir()
    (sub / "helper.py").write_text("def helper():\n    return 42\n", encoding="utf-8")

    unp_file = tmp_path / "workspace.unp"
    restore_dir = tmp_path / "project_restored"

    tokenizer = MockByteTokenizer(vocab_size=260)
    model = MockCausalLM(vocab_size=260, hidden_dim=32)
    bundler = WorkspaceBundler(model=model, tokenizer=tokenizer)

    # Bundle
    report = bundler.bundle_directory(src_dir, unp_file, domain_id=2)
    assert isinstance(report, BundleManifestReport)
    assert report.total_files == 2
    assert report.total_uncompressed_bytes > 0
    assert unp_file.exists()

    # Unbundle
    restore_report = bundler.unbundle_archive(unp_file, restore_dir)
    assert restore_report.total_files == 2
    assert (restore_dir / "main.py").read_text(encoding="utf-8") == (
        "def main():\n    print('hello')\n"
    )
    assert (restore_dir / "utils" / "helper.py").read_text(encoding="utf-8") == (
        "def helper():\n    return 42\n"
    )


def test_workspace_bundler_mock_codec(tmp_path):
    src_dir = tmp_path / "pkg_src"
    src_dir.mkdir()
    (src_dir / "app.py").write_text("print('mock bundler test')", encoding="utf-8")
    (src_dir / "config.json").write_text('{"mode": "test"}', encoding="utf-8")

    unp_file = tmp_path / "pkg.unp"
    restore_dir = tmp_path / "pkg_restored"

    bundler = WorkspaceBundler(model_name="mock")
    report = bundler.bundle_directory(src_dir, unp_file, domain_id=2)
    assert report.total_files == 2
    assert unp_file.exists()

    restore_report = bundler.unbundle_archive(unp_file, restore_dir)
    assert restore_report.total_files == 2
    assert (restore_dir / "app.py").read_text(encoding="utf-8") == "print('mock bundler test')"
    assert (restore_dir / "config.json").read_text(encoding="utf-8") == '{"mode": "test"}'


def test_workspace_bundler_report_table_and_dict(tmp_path):
    src_dir = tmp_path / "simple_src"
    src_dir.mkdir()
    (src_dir / "sample.txt").write_text("sample content", encoding="utf-8")

    unp_file = tmp_path / "simple.unp"
    bundler = WorkspaceBundler(model_name="mock")
    report = bundler.bundle_directory(src_dir, unp_file)

    table = report.format_manifest_table()
    assert "Workspace Bundler: Hierarchical Archive Manifest" in table
    assert "sample.txt" in table

    data = report.to_dict()
    assert data["total_files"] == 1
    assert len(data["files"]) == 1
    assert data["files"][0]["path"] == "sample.txt"


# end of file: unplow/tests/test_bundler.py
