# Path: unplow/src/unplow/full_pipeline.py
"""Full Closed-Loop Extraction, Refactoring, and Multi-Domain Training Pipeline.

Orchestrates automated end-to-end extraction, distillation, and Bayesian CTS
compression across specialized domain models (till-python, till-c, till-cpp,
till-rust, till-gencode) while continuously enforcing storage quotas via
TillDiskGuardian.
"""

import argparse
from pathlib import Path
import subprocess
import sys
from typing import List, Optional

# Remove script directory from sys.path to avoid shadowing standard library modules (e.g. queue)
_script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _script_dir:
    sys.path.pop(0)
_repo_root = str(Path(__file__).resolve().parents[2])
_src_dir = str(Path(__file__).resolve().parents[1])
for _p in (_repo_root, _src_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from unplow.till_guard import TillDiskGuardian, TILL_DOMAINS


def execute_full_pipeline(
    domains: Optional[List[str]] = None,
    manifest_path: str = "workspace/till_manifest.toml",
    min_free_pct: float = 15.0,
    model_name: str = "auto",
    dry_run_offload: bool = False,
    train_steps: int = 1,
) -> bool:
    """Runs closed-loop extraction, distillation, and archiving across till-* domains."""
    guardian = TillDiskGuardian(manifest_path=manifest_path, min_free_pct=min_free_pct)

    # 1. Enforce initial disk telemetry check
    telemetry = guardian.check_disk()
    print(f"[Pipeline] Initial Storage Telemetry -> Free: {telemetry.free_gb} GB ({telemetry.free_percent}%)")

    if telemetry.free_percent < min_free_pct:
        guardian.enforce_space(["~/.cache/", "models/work_products/"], dry_run=dry_run_offload)

    target_domains = domains or list(TILL_DOMAINS.keys())
    all_success = True

    # 2. Extract, refactor, distill, and archive each till-* model
    for domain in target_domains:
        print(f"\n[Pipeline] === Processing Specialization: till-{domain} ===")
        corpus_dir = Path(f"workspace/till_corpus/{domain}")
        output_unp = Path(f"workspace/targets/till_{domain}.unp")

        # Ensure corpus exists
        if not corpus_dir.exists() or not list(corpus_dir.glob("*")):
            print(f"[Pipeline] Initializing corpus stubs for {domain}...")
            cmd_init = ["uv", "run", "unplow", "till-init", "--manifest", manifest_path]
            subprocess.run(cmd_init, check=True)

        # Run distillation and Bayesian CTS compression
        cmd = [
            "uv", "run", "unplow", "till-train",
            "--domain", domain,
            "--corpus", str(corpus_dir),
            "-o", str(output_unp),
            "-m", model_name,
            "--train-steps", str(train_steps),
        ]
        print(f"[Pipeline] Executing distillation: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)

        if res.returncode != 0:
            print(f"[Pipeline ERROR] Training failed for till-{domain}: {res.stderr}")
            all_success = False
        else:
            print(f"[Pipeline SUCCESS] Archived till-{domain} -> {output_unp}")
            if res.stdout:
                print(res.stdout)

        # Re-check disk space after each domain checkpoint
        if not guardian.enforce_space(["models/work_products/"], dry_run=dry_run_offload):
            print("[Pipeline WARNING] Disk space low; offloading archives to Google Drive...")
            offload_cmd = ["uv", "run", "unplow", "till-guard-offload"]
            if dry_run_offload:
                offload_cmd.append("--dry-run")
            subprocess.run(offload_cmd, check=True)

    if all_success:
        print("\n[Pipeline] All till-* specialized models successfully trained, distilled, and archived.")
    else:
        print("\n[Pipeline WARNING] One or more domains encountered errors during training.", file=sys.stderr)

    return all_success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full closed-loop extraction, refactoring, and multi-domain training pipeline",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["python", "c", "cpp", "rust", "gencode"],
        default=None,
        help="Subset of domains to process (default: all 5 domains)",
    )
    parser.add_argument(
        "--manifest",
        default="workspace/till_manifest.toml",
        help="Path to offload manifest (default: workspace/till_manifest.toml)",
    )
    parser.add_argument(
        "-m", "--model",
        default="auto",
        help="Model substrate identifier (default: auto)",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=1,
        help="Distillation training update steps (default: 1)",
    )
    parser.add_argument(
        "--dry-run-offload",
        action="store_true",
        help="Simulate rclone offload if free space drops below threshold",
    )
    args = parser.parse_args()

    success = execute_full_pipeline(
        domains=args.domains,
        manifest_path=args.manifest,
        model_name=args.model,
        dry_run_offload=args.dry_run_offload,
        train_steps=args.train_steps,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

# end of file: unplow/src/unplow/full_pipeline.py
