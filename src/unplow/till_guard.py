# Path: unplow/src/unplow/till_guard.py
"""Disk space guardian, rclone offloader, and till domain model staging framework.

Monitors storage utilization against the mandatory 15% free disk space threshold,
manages offload syncs via rclone, stages isolated multi-language training datasets,
and runs domain-specific distillation pipelines into Bayesian CTS .unp containers.
"""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tomllib
from typing import Any, Dict, List, Optional, Union

from .models import load_causal_lm
from .replow import ReplowEngine

logger = logging.getLogger("unplow.till_guard")

# Domain configurations for specialized till-* models
TILL_DOMAINS: Dict[str, Dict[str, Any]] = {
    "python": {
        "domain_id": 2,  # Python code prior
        "description": "Extracted Python AST stubs & standard library idioms",
        "corpus_dir": "workspace/till_corpus/python",
        "checkpoint_dir": "models/till-python-lora",
        "sample_filename": "stdlib_stubs.py",
        "seed_code": (
            "import os\nimport sys\nfrom typing import Any, List, Optional\n\n"
            "def sanitize_identifier(name: str) -> str:\n"
            "    \"\"\"Strip non-alphanumeric chars and normalize identifier.\"\"\"\n"
            "    return ''.join(c if c.isalnum() or c == '_' else '_' for c in name)\n\n"
            "class StderrLogger:\n"
            "    def __init__(self, prefix: str = '[LOG]'):\n"
            "        self.prefix = prefix\n\n"
            "    def log(self, msg: str) -> None:\n"
            "        sys.stderr.write(f'{self.prefix} {msg}\\n')\n"
        ),
    },
    "c": {
        "domain_id": 3,  # C / Low-level firmware prior
        "description": "Low-level system headers, POSIX syscalls & drivers",
        "corpus_dir": "workspace/till_corpus/c",
        "checkpoint_dir": "models/till-c-lora",
        "sample_filename": "kernel_hooks.c",
        "seed_code": (
            "#include <stdint.h>\n#include <stddef.h>\n\n"
            "typedef struct {\n"
            "    uint32_t magic;\n"
            "    uint32_t flags;\n"
            "    uint64_t offset;\n"
            "} unp_block_desc_t;\n\n"
            "int unp_verify_block(const unp_block_desc_t *desc) {\n"
            "    if (!desc) return -1;\n"
            "    return (desc->magic == 0x554E50) ? 0 : -2;\n"
            "}\n"
        ),
    },
    "cpp": {
        "domain_id": 1,  # Technical prose / C++
        "description": "Modern C++ AST, STL templates, AER diagnostics",
        "corpus_dir": "workspace/till_corpus/cpp",
        "checkpoint_dir": "models/till-cpp-lora",
        "sample_filename": "ring_buffer.hpp",
        "seed_code": (
            "#pragma once\n#include <vector>\n#include <cstddef>\n\n"
            "template <typename T, std::size_t Capacity>\n"
            "class StaticRingBuffer {\n"
            "    T buffer_[Capacity];\n"
            "    std::size_t head_ = 0;\n"
            "    std::size_t size_ = 0;\n"
            "public:\n"
            "    constexpr std::size_t capacity() const noexcept { return Capacity; }\n"
            "    constexpr std::size_t size() const noexcept { return size_; }\n"
            "};\n"
        ),
    },
    "rust": {
        "domain_id": 1,  # Technical prose / Rust
        "description": "Cargo workspace crates, safety wrappers & macros",
        "corpus_dir": "workspace/till_corpus/rust",
        "checkpoint_dir": "models/till-rust-lora",
        "sample_filename": "safe_alloc.rs",
        "seed_code": (
            "use std::alloc::{alloc, dealloc, Layout};\n\n"
            "pub struct BoundedBuffer {\n"
            "    ptr: *mut u8,\n"
            "    layout: Layout,\n"
            "}\n\n"
            "impl Drop for BoundedBuffer {\n"
            "    fn drop(&mut self) {\n"
            "        unsafe { dealloc(self.ptr, self.layout); }\n"
            "    }\n"
            "}\n"
        ),
    },
    "gencode": {
        "domain_id": 0,  # Raw BOS prior
        "description": "Multi-language translation & code synthesis pairs",
        "corpus_dir": "workspace/till_corpus/gencode",
        "checkpoint_dir": "models/till-gencode-lora",
        "sample_filename": "polyglot_spec.json",
        "seed_code": (
            "{\n"
            "  \"translation_pairs\": [\n"
            "    {\n"
            "      \"source\": \"def add(a, b): return a + b\",\n"
            "      \"c\": \"int add(int a, int b) { return a + b; }\",\n"
            "      \"rust\": \"fn add(a: i32, b: i32) -> i32 { a + b }\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
        ),
    },
}


@dataclass
class DiskTelemetry:
    total_gb: float
    free_gb: float
    free_percent: float
    host_total_gb: Optional[float] = None
    host_free_gb: Optional[float] = None
    host_free_percent: Optional[float] = None
    is_wsl: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "total_gb": self.total_gb,
            "free_gb": self.free_gb,
            "free_percent": self.free_percent,
            "is_wsl": self.is_wsl,
        }
        if self.host_total_gb is not None:
            d["host_total_gb"] = self.host_total_gb
            d["host_free_gb"] = self.host_free_gb
            d["host_free_percent"] = self.host_free_percent
        return d

    def __str__(self) -> str:
        s = (
            f"DiskTelemetry(total_gb={self.total_gb}, "
            f"free_gb={self.free_gb}, free_percent={self.free_percent}%)"
        )
        if self.is_wsl and self.host_free_percent is not None:
            s += (
                f" [Windows Host: total_gb={self.host_total_gb}, "
                f"free_gb={self.host_free_gb}, free_percent={self.host_free_percent}%]"
            )
        return s


class TillDiskGuardian:
    """Monitors disk utilization and offloads targets via rclone when below 15% free."""

    def __init__(
        self,
        min_free_pct: float = 15.0,
        gdrive_dest: str = "gdrive:transfer/till-20261017/",
        manifest_path: Optional[str] = None,
    ):
        self.min_free_pct = min_free_pct
        self.gdrive_dest = gdrive_dest
        self.manifest_path = manifest_path

    def check_disk(self, path: str = "/") -> DiskTelemetry:
        """Inspects disk usage for given path, taking into account Windows host space on WSL."""
        # On WSL, check working directory or path's actual mount rather than virtual root /
        probe_path = path
        if probe_path == "/" and os.path.exists("/home/fekerr/src/unplow"):
            probe_path = "/home/fekerr/src/unplow"

        usage = shutil.disk_usage(probe_path)
        total_gb = usage.total / (1024**3)
        free_gb = usage.free / (1024**3)
        free_pct = (usage.free / usage.total) * 100.0 if usage.total > 0 else 0.0

        # Detect WSL and probe Windows host drive (e.g. /mnt/c or WSL mount points)
        is_wsl = False
        host_total_gb: Optional[float] = None
        host_free_gb: Optional[float] = None
        host_free_pct: Optional[float] = None

        if Path("/proc/version").exists():
            try:
                with open("/proc/version", "r", encoding="utf-8") as f:
                    content = f.read().lower()
                    if "microsoft" in content or "wsl" in content:
                        is_wsl = True
            except Exception:
                pass

        if is_wsl:
            # Check primary Windows host mount point
            for win_mount in ["/mnt/c", "/mnt/d"]:
                if os.path.exists(win_mount):
                    try:
                        win_usage = shutil.disk_usage(win_mount)
                        host_total_gb = round(win_usage.total / (1024**3), 2)
                        host_free_gb = round(win_usage.free / (1024**3), 2)
                        host_free_pct = (
                            round((win_usage.free / win_usage.total) * 100.0, 2)
                            if win_usage.total > 0
                            else 0.0
                        )
                        break
                    except Exception:
                        pass

        # If on WSL, effective free percentage should reflect the bottleneck
        # (either the WSL VHD virtual drive or the underlying Windows host drive)
        effective_free_pct = free_pct
        if is_wsl and host_free_pct is not None:
            effective_free_pct = min(free_pct, host_free_pct)

        return DiskTelemetry(
            total_gb=round(total_gb, 2),
            free_gb=round(free_gb, 2),
            free_percent=round(effective_free_pct, 2),
            host_total_gb=host_total_gb,
            host_free_gb=host_free_gb,
            host_free_percent=host_free_pct,
            is_wsl=is_wsl,
        )

    def load_manifest(self, manifest_path: Optional[str] = None) -> Dict[str, Any]:
        """Loads and parses TOML manifest if available."""
        m_path = Path(manifest_path or self.manifest_path or "workspace/till_manifest.toml").expanduser()
        if not m_path.exists():
            return {}
        try:
            with open(m_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.warning("Failed to parse manifest at %s: %s", m_path, e)
            return {}

    def get_offload_candidates(self, manifest: Optional[Dict[str, Any]] = None) -> List[str]:
        """Collects candidate paths defined in the manifest offload targets."""
        mf = manifest if manifest is not None else self.load_manifest()
        candidates: List[str] = []
        offload_targets = mf.get("offload_targets", {})

        # Check caches
        caches = offload_targets.get("caches", {})
        base_cache = Path(caches.get("path", "~/.cache")).expanduser()
        for item in caches.get("items", []):
            p = base_cache / item.strip("/")
            if p.exists():
                candidates.append(str(p))

        # Check checkpoints
        checkpoints = offload_targets.get("checkpoints", {})
        base_cp = Path(checkpoints.get("path", "models")).expanduser()
        for item in checkpoints.get("items", []):
            p = base_cp / item.strip("/")
            if p.exists():
                candidates.append(str(p))

        # Check archives
        archives = offload_targets.get("archives", {})
        base_arc = Path(archives.get("path", "workspace/targets")).expanduser()
        pat = archives.get("pattern", "*.unp")
        if base_arc.exists():
            for p in base_arc.glob(pat):
                candidates.append(str(p))

        return candidates

    def enforce_space(
        self,
        target_dirs: Optional[List[str]] = None,
        dry_run: bool = False,
        verbose: bool = True,
    ) -> bool:
        telemetry = self.check_disk()
        if verbose:
            print(
                f"[DiskGuardian] Total: {telemetry.total_gb} GB | "
                f"Free: {telemetry.free_gb} GB ({telemetry.free_percent}%)"
            )

        if telemetry.free_percent >= self.min_free_pct:
            if verbose:
                print(f"[DiskGuardian] Disk space threshold satisfied (>= {self.min_free_pct}%).")
            return True

        if verbose:
            print(
                f"[DiskGuardian] WARNING: Free space {telemetry.free_percent}% is below "
                f"{self.min_free_pct}%. Initiating rclone offload..."
            )

        targets = target_dirs if target_dirs is not None else self.get_offload_candidates()

        for d in targets:
            p = Path(d).expanduser()
            if p.exists():
                remote_target = f"{self.gdrive_dest.rstrip('/')}/{p.name}/"
                cmd = ["rclone", "move", str(p), remote_target, "--transfers", "4", "--fast-list"]
                if verbose:
                    print(f"[DiskGuardian] Offloading {p} -> {remote_target}")
                if dry_run:
                    if verbose:
                        print(f"[DiskGuardian] [DRY RUN] Would execute: {' '.join(cmd)}")
                    continue
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode != 0:
                        if verbose:
                            print(f"[DiskGuardian] rclone error: {res.stderr.strip()}")
                    else:
                        if verbose:
                            print(f"[DiskGuardian] Successfully offloaded {p}")
                except Exception as ex:
                    if verbose:
                        print(f"[DiskGuardian] rclone invocation failure: {ex}")

        post_telemetry = self.check_disk()
        if verbose:
            print(
                f"[DiskGuardian] Post-offload Free: {post_telemetry.free_gb} GB "
                f"({post_telemetry.free_percent}%)"
            )
        return post_telemetry.free_percent >= self.min_free_pct


def init_till_staging(
    manifest_path: Optional[str] = "workspace/till_manifest.toml",
    base_corpus_dir: str = "workspace/till_corpus",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Initializes disk guardian, validates storage, and stages dataset directories."""
    guardian = TillDiskGuardian(manifest_path=manifest_path)
    telemetry = guardian.check_disk()
    guardian.enforce_space(verbose=verbose)

    base_dir = Path(base_corpus_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    staged_domains: Dict[str, Any] = {}
    for domain_name, config in TILL_DOMAINS.items():
        domain_dir = base_dir / domain_name
        domain_dir.mkdir(parents=True, exist_ok=True)

        sample_file = domain_dir / config["sample_filename"]
        if not sample_file.exists():
            sample_file.write_text(config["seed_code"], encoding="utf-8")

        cp_dir = Path(config["checkpoint_dir"]).resolve()
        cp_dir.mkdir(parents=True, exist_ok=True)

        staged_domains[domain_name] = {
            "domain_id": config["domain_id"],
            "corpus_dir": str(domain_dir),
            "checkpoint_dir": str(cp_dir),
            "sample_file": str(sample_file),
            "description": config["description"],
        }

    return {
        "telemetry": telemetry.to_dict(),
        "manifest_path": manifest_path,
        "base_corpus_dir": str(base_dir),
        "domains": staged_domains,
    }


@dataclass
class TillTrainReport:
    domain: str
    domain_id: int
    corpus_dir: str
    checkpoint_dir: str
    output_unp_path: str
    total_files: int
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    train_loss: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "domain_id": self.domain_id,
            "corpus_dir": self.corpus_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "output_unp_path": self.output_unp_path,
            "total_files": self.total_files,
            "raw_bytes": self.raw_bytes,
            "compressed_bytes": self.compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "train_loss": self.train_loss,
        }

    def format_table(self) -> str:
        loss_str = f"{self.train_loss:.4f}" if self.train_loss is not None else "N/A"
        lines = [
            "",
            "=" * 78,
            f"Till Model Distillation & Archiving: till-{self.domain}",
            "=" * 78,
            f"Domain                  : {self.domain} (Domain ID: {self.domain_id})",
            f"Corpus Directory        : {self.corpus_dir}",
            f"Checkpoint Directory    : {self.checkpoint_dir}",
            f"Output Archive          : {self.output_unp_path}",
            f"Total Corpus Files      : {self.total_files}",
            f"Raw Corpus Size         : {self.raw_bytes} bytes",
            f"Compressed Size (.unp)  : {self.compressed_bytes} bytes",
            f"Compression Ratio       : {self.compression_ratio:.3f}:1",
            f"Distillation Train Loss : {loss_str}",
            "=" * 78,
        ]
        return "\n".join(lines)


def train_till_domain(
    domain: str,
    corpus_dir: Optional[Union[str, Path]] = None,
    output_unp_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    train_steps: int = 1,
    learning_rate: float = 1e-4,
    verbose: bool = True,
) -> TillTrainReport:
    """Runs closed-loop training distillation and Bayesian CTS compression for a domain."""
    if domain not in TILL_DOMAINS:
        raise ValueError(
            f"Unknown till domain '{domain}'. Supported domains: {list(TILL_DOMAINS.keys())}"
        )

    cfg = TILL_DOMAINS[domain]
    c_dir = Path(corpus_dir or cfg["corpus_dir"]).resolve()
    out_unp = Path(output_unp_path or f"workspace/targets/till_{domain}.unp").resolve()
    cp_dir = Path(cfg["checkpoint_dir"]).resolve()

    c_dir.mkdir(parents=True, exist_ok=True)
    out_unp.parent.mkdir(parents=True, exist_ok=True)
    cp_dir.mkdir(parents=True, exist_ok=True)

    # Ensure disk threshold is enforced
    TillDiskGuardian().enforce_space(verbose=verbose)

    # Gather text samples from corpus
    files = sorted([p for p in c_dir.rglob("*") if p.is_file()])
    samples: List[str] = []
    total_raw = 0
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
            samples.append(content)
            total_raw += len(content.encode("utf-8"))
        except Exception:
            continue

    if not samples:
        # Seed default sample
        default_seed = cfg["seed_code"]
        samples.append(default_seed)
        total_raw = len(default_seed.encode("utf-8"))

    # Load model and tokenizer
    tokenizer, model, _ = load_causal_lm(model_identifier=model_name or "mock")

    engine = ReplowEngine(model=model, tokenizer=tokenizer)

    # Stage dataset
    dataset_path = cp_dir / "distillation_dataset.jsonl"
    engine.stage_training_dataset(samples, dataset_path)

    # Distillation training step
    train_loss: Optional[float] = None
    if train_steps > 0:
        train_loss = engine.train_step(samples, learning_rate=learning_rate, steps=train_steps)
        # Save adapter config
        adapter_cfg = {
            "base_model": model_name or "mock",
            "domain": domain,
            "train_steps": train_steps,
            "final_loss": train_loss,
        }
        (cp_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2), encoding="utf-8")

    # Combine samples and archive using Bayesian CTS
    combined_text = "\n\n".join(samples)
    artifact = engine.process_and_archive(
        raw_code=combined_text,
        output_path=out_unp,
        domain_id=cfg["domain_id"],
    )

    comp_bytes = artifact.compressed_bytes_len
    ratio = round(total_raw / max(comp_bytes, 1), 3)

    return TillTrainReport(
        domain=domain,
        domain_id=cfg["domain_id"],
        corpus_dir=str(c_dir),
        checkpoint_dir=str(cp_dir),
        output_unp_path=str(out_unp),
        total_files=max(len(files), 1),
        raw_bytes=total_raw,
        compressed_bytes=comp_bytes,
        compression_ratio=ratio,
        train_loss=train_loss,
    )


# end of file: unplow/src/unplow/till_guard.py
