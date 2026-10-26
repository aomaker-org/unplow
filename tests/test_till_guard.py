# Path: unplow/tests/test_till_guard.py
"""Unit tests for TillDiskGuardian, disk telemetry, and domain training staging.

Validates disk space monitoring, manifest parsing, multi-language corpus staging,
and domain-specific distillation pipelines.
"""

from pathlib import Path
import pytest

from unplow.till_guard import (
    TILL_DOMAINS,
    DiskTelemetry,
    TillDiskGuardian,
    TillTrainReport,
    init_till_staging,
    train_till_domain,
)


def test_disk_telemetry_properties():
    telem = DiskTelemetry(total_gb=100.0, free_gb=40.0, free_percent=40.0)
    assert telem.total_gb == 100.0
    assert telem.free_gb == 40.0
    assert telem.free_percent == 40.0
    d = telem.to_dict()
    assert d["total_gb"] == 100.0
    assert "DiskTelemetry" in str(telem)


def test_till_disk_guardian_check_disk():
    guardian = TillDiskGuardian(min_free_pct=15.0)
    telem = guardian.check_disk()
    assert isinstance(telem, DiskTelemetry)
    assert telem.total_gb > 0
    assert telem.free_gb > 0
    assert 0 <= telem.free_percent <= 100.0
    # In WSL environment, host disk telemetry should be captured
    if telem.is_wsl:
        assert telem.host_total_gb is not None
        assert telem.host_free_gb is not None
        assert telem.host_free_percent is not None
        assert "Windows Host" in str(telem)


def test_till_disk_guardian_manifest_and_candidates(tmp_path):
    manifest_file = tmp_path / "test_manifest.toml"
    p_str = str(tmp_path).replace("\\", "/")
    manifest_content = f"""
[manifest]
version = "1.0.0"
target_free_percentage = 15.0
rclone_destination = "gdrive:transfer/till-test/"

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
    (tmp_path / "dummy.unp").write_bytes(b"dummy")

    guardian = TillDiskGuardian(manifest_path=str(manifest_file))
    mf = guardian.load_manifest()
    assert mf["manifest"]["version"] == "1.0.0"

    candidates = guardian.get_offload_candidates()
    assert any("dummy.unp" in c for c in candidates)


def test_till_disk_guardian_enforce_space_satisfied():
    guardian = TillDiskGuardian(min_free_pct=0.01)
    res = guardian.enforce_space(target_dirs=[], dry_run=True)
    assert res is True


def test_till_disk_guardian_enforce_space_dry_run(tmp_path):
    # Test dry run execution when free space is forced to appear below threshold
    guardian = TillDiskGuardian(min_free_pct=99.99)
    target = tmp_path / "cache_dir"
    target.mkdir()
    (target / "data.bin").write_bytes(b"12345")

    # dry_run should simulate rclone move without raising errors
    res = guardian.enforce_space(target_dirs=[str(target)], dry_run=True)
    assert isinstance(res, bool)


def test_init_till_staging(tmp_path):
    corpus_base = tmp_path / "till_corpus"
    manifest_path = Path("workspace/till_manifest.toml")

    res = init_till_staging(
        manifest_path=str(manifest_path) if manifest_path.exists() else None,
        base_corpus_dir=str(corpus_base),
    )
    assert "telemetry" in res
    assert "domains" in res
    assert set(res["domains"].keys()) == {"python", "c", "cpp", "rust", "gencode"}

    for domain_name in TILL_DOMAINS:
        dom_dir = corpus_base / domain_name
        assert dom_dir.exists()
        assert any(dom_dir.iterdir())


def test_train_till_domain_roundtrip(tmp_path):
    corpus_dir = tmp_path / "python_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "math_utils.py").write_text(
        "def multiply(a: int, b: int) -> int:\n    return a * b\n",
        encoding="utf-8",
    )
    out_unp = tmp_path / "till_python.unp"

    report = train_till_domain(
        domain="python",
        corpus_dir=corpus_dir,
        output_unp_path=out_unp,
        model_name="mock",
        train_steps=1,
    )
    assert isinstance(report, TillTrainReport)
    assert report.domain == "python"
    assert report.domain_id == 2
    assert report.raw_bytes > 0
    assert report.compressed_bytes > 0
    assert out_unp.exists()
    assert report.compression_ratio > 0

    table = report.format_table()
    assert "till-python" in table
    assert "Domain ID: 2" in table

    d = report.to_dict()
    assert d["domain"] == "python"
    assert d["compressed_bytes"] == report.compressed_bytes


def test_train_till_domain_unknown_raises():
    with pytest.raises(ValueError, match="Unknown till domain 'cobol'"):
        train_till_domain(domain="cobol")


# end of file: unplow/tests/test_till_guard.py
