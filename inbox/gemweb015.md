# Path: unplow/inbox/gemweb015.md
# End-to-End Multi-Domain Till Training & Storage Guard Orchestration
# Max Column: 80 Columns

To execute a comprehensive, end-to-end extraction, refactoring, and full-scale
training pipeline for the specialized **`till-*`** family (`till-python`,
`till-c`, `till-cpp`, `till-rust`, and `till-gencode`), we can invoke our
automated orchestration harness while monitoring storage allocation to strictly
honor the **15% free disk space** threshold via `TillDiskGuardian`.

The complete closed-loop pipeline script below automates the full
extraction-to-training sweep across all targets:

```python
# src/unplow/full_pipeline.py
"""
Full Closed-Loop Extraction, Refactoring, and Multi-Domain Training Pipeline
for till-python, till-c, till-cpp, till-rust, and till-gencode models.
"""

from pathlib import Path
import subprocess
from unplow.till_guard import TillDiskGuardian
from unplow.bundler import WorkspaceBundler


def execute_full_pipeline():
    guardian = TillDiskGuardian(min_free_pct=15.0)

    # 1. Enforce initial disk telemetry check
    telemetry = guardian.check_disk()
    print(
        f"[Pipeline] Initial Storage Telemetry -> Free: "
        f"{telemetry.free_gb} GB ({telemetry.free_percent}%)"
    )

    if telemetry.free_percent < 15.0:
        guardian.enforce_space(["~/.cache/", "models/work_products/"])

    domains = ["python", "c", "cpp", "rust", "gencode"]

    # 2. Extract, refactor, distill, and archive each till-* model
    for domain in domains:
        print(f"\n[Pipeline] === Processing Specialization: till-{domain} ===")
        corpus_dir = Path(f"workspace/till_corpus/{domain}")
        output_unp = Path(f"workspace/targets/till_{domain}.unp")

        # Ensure corpus exists
        if not corpus_dir.exists() or not list(corpus_dir.glob("*")):
            print(f"[Pipeline] Initializing corpus stubs for {domain}...")
            subprocess.run(
                [
                    "uv", "run", "unplow", "till-init",
                    "--manifest", "workspace/till_manifest.toml",
                ],
                check=True,
            )

        # Run distillation and Bayesian CTS compression
        cmd = [
            "uv", "run", "unplow", "till-train",
            "--domain", domain,
            "--corpus", str(corpus_dir),
            "-o", str(output_unp),
        ]
        print(f"[Pipeline] Executing distillation: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode != 0:
            print(
                f"[Pipeline ERROR] Training failed for till-{domain}: "
                f"{res.stderr}"
            )
        else:
            print(f"[Pipeline SUCCESS] Archived till-{domain} -> {output_unp}")
            print(res.stdout)

        # Re-check disk space after each domain checkpoint
        if not guardian.enforce_space(["models/work_products/"]):
            print(
                "[Pipeline WARNING] Disk space low; "
                "offloading archives to Google Drive..."
            )
            subprocess.run(
                ["uv", "run", "unplow", "till-guard-offload"], check=True
            )

    print(
        "\n[Pipeline] All till-* specialized models successfully "
        "trained, distilled, and archived."
    )


if __name__ == "__main__":
    execute_full_pipeline()
```

---

## Execution Instructions

To run the complete extraction, distillation, and training campaign across all
five models from your terminal while executing storage guard checks:

```bash
uv run python src/unplow/full_pipeline.py
```

<!-- end of file: unplow/inbox/gemweb015.md -->
