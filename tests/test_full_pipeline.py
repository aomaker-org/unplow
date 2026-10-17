# Path: unplow/tests/test_full_pipeline.py
"""Unit tests for the full multi-domain closed-loop training pipeline.

Validates end-to-end orchestration, disk telemetry gating, corpus stub staging,
and artifact archiving across specialized till-* domains.
"""

from pathlib import Path
from unplow.full_pipeline import execute_full_pipeline


def test_execute_full_pipeline_mock(tmp_path):
    manifest_file = tmp_path / "manifest.toml"
    p_str = str(tmp_path).replace("\\", "/")
    manifest_content = f"""
[manifest]
version = "1.0.0"
target_free_percentage = 15.0
rclone_destination = "gdrive:transfer/pipeline-test/"

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

    success = execute_full_pipeline(
        domains=["python"],
        manifest_path=str(manifest_file),
        min_free_pct=0.01,
        model_name="mock",
        dry_run_offload=True,
        train_steps=1,
    )
    assert success is True
    assert Path("workspace/targets/till_python.unp").exists()


# end of file: unplow/tests/test_full_pipeline.py
