# Path: unplow/inbox/gemweb012.md
# Disk Telemetry, Offload Manifest & Till Model Training Staging
# Max Column: 80 Columns

To initiate the creation and training of the **`till-python`**, **`till-c`**,
**`till-cpp`**, **`till-rust`**, and **`till-gencode`** specialized models
while strictly maintaining the **15% free disk space** threshold, we establish
an automated disk telemetry and offload manifest framework
(`workspace/till_manifest.toml`), coupled with our `replow` closed-loop
distillation pipeline.

---

## 1. Disk Space Telemetry & Offload Manifest (`workspace/till_manifest.toml`)

Before staging training corpora and model checkpoints, we audit current
storage and register offloadable cache directories and large archives for
`rclone` synchronization to `gdrive:transfer/till-20261017/`.

```toml
# workspace/till_manifest.toml
[manifest]
version = "1.0.0"
target_free_percentage = 15.0
rclone_destination = "gdrive:transfer/till-20261017/"

[offload_targets.caches]
path = "~/.cache/"
max_allowed_size_gb = 5.0
items = [
    "uv/",
    "pip/",
    "bazel/",
    "ccache/",
    "llama.cpp/"
]

[offload_targets.checkpoints]
path = "models/"
items = [
    "qwen2.5-1.5b-instruct/",
    "work_products/"
]

[offload_targets.archives]
path = "workspace/targets/"
pattern = "*.unp"
retention_days = 3
```

### Disk Enforcement & Offload Execution Script (`src/unplow/till_guard.py`)

```python
# src/unplow/till_guard.py
"""
Disk space guardian and rclone offloader ensuring >= 15% free disk space
by transferring aged checkpoints and caches to Google Drive.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional


@dataclass
class DiskTelemetry:
    total_gb: float
    free_gb: float
    free_percent: float


class TillDiskGuardian:
    """Monitors disk utilization and offloads targets via rclone."""

    def __init__(
        self,
        min_free_pct: float = 15.0,
        gdrive_dest: str = "gdrive:transfer/till-20261017/",
    ):
        self.min_free_pct = min_free_pct
        self.gdrive_dest = gdrive_dest

    def check_disk(self, path: str = "/") -> DiskTelemetry:
        usage = shutil.disk_usage(path)
        total_gb = usage.total / (1024**3)
        free_gb = usage.free / (1024**3)
        free_pct = (usage.free / usage.total) * 100.0
        return DiskTelemetry(
            round(total_gb, 2),
            round(free_gb, 2),
            round(free_pct, 2),
        )

    def enforce_space(self, target_dirs: List[str]) -> bool:
        telemetry = self.check_disk()
        print(
            f"[DiskGuardian] Total: {telemetry.total_gb} GB | "
            f"Free: {telemetry.free_gb} GB ({telemetry.free_percent}%)"
        )

        if telemetry.free_percent >= self.min_free_pct:
            print("[DiskGuardian] Disk space threshold satisfied (>= 15%).")
            return True

        print(
            f"[DiskGuardian] WARNING: Free space {telemetry.free_percent}% is "
            f"below {self.min_free_pct}%. Initiating rclone offload..."
        )

        for d in target_dirs:
            p = Path(d).expanduser()
            if p.exists():
                remote_target = f"{self.gdrive_dest}{p.name}/"
                cmd = [
                    "rclone", "move", str(p), remote_target,
                    "--transfers", "4", "--fast-list"
                ]
                print(f"[DiskGuardian] Offloading {p} -> {remote_target}")
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"[DiskGuardian] rclone error: {res.stderr}")
                else:
                    print(f"[DiskGuardian] Successfully offloaded {p}")

        post_telemetry = self.check_disk()
        print(
            f"[DiskGuardian] Post-offload Free: {post_telemetry.free_gb} GB "
            f"({post_telemetry.free_percent}%)"
        )
        return post_telemetry.free_percent >= self.min_free_pct

# end of file: unplow/src/unplow/till_guard.py
```

---

## 2. Training Pipeline Staging (`till-*` Models)

We initialize isolated dataset directories under `workspace/till_corpus/`
populated via our `replow` extraction engine from `Qwen2.5-Coder-1.5B`,
structured for targeted fine-tuning:

1. **`till-python`**:
   - Corpus: Extracted Python AST stubs & standard library idioms.
   - Checkpoint: `models/till-python-lora/`
   - Compression: `FLAG_BAYES_MIXER` (`0x08`) + Python Prior (`domain_id=2`).
2. **`till-c`**:
   - Corpus: Low-level system headers, POSIX syscalls & drivers.
   - Checkpoint: `models/till-c-lora/`
   - Compression: `FLAG_BAYES_MIXER` (`0x08`) + C Firmware (`domain_id=3`).
3. **`till-cpp`**:
   - Corpus: Modern C++ AST, STL templates, AER diagnostics.
   - Checkpoint: `models/till-cpp-lora/`
   - Compression: `FLAG_BAYES_MIXER` (`0x08`) + Technical Prose (`domain_id=1`).
4. **`till-rust`**:
   - Corpus: Cargo workspace crates, safety wrappers & macros.
   - Checkpoint: `models/till-rust-lora/`
   - Compression: `FLAG_BAYES_MIXER` (`0x08`) + Technical Prose (`domain_id=1`).
5. **`till-gencode`**:
   - Corpus: Multi-language translation & code synthesis pairs.
   - Checkpoint: `models/till-gencode-lora/`
   - Compression: `FLAG_BAYES_MIXER` (`0x08`) + Raw BOS Prior (`domain_id=0`).

---

## 3. Immediate Concrete Execution Steps

1. **Verify Disk Telemetry & Run Initial Disk Guard Check:**
```bash
uv run python -c \
  "from unplow.till_guard import TillDiskGuardian; \
print(TillDiskGuardian().check_disk())"
```

2. **Execute Offload & Initialize Training Staging for `till-*` Models:**
```bash
uv run unplow till-init --manifest workspace/till_manifest.toml
```

3. **Run Training Distillation & Bayesian Archiving:**
```bash
uv run unplow till-train \
  --domain python \
  --corpus workspace/till_corpus/python/ \
  -o workspace/targets/till_python.unp
```

<!-- end of file: unplow/inbox/gemweb012.md -->
