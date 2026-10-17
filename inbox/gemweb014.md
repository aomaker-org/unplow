# Path: unplow/inbox/gemweb014.md
# Phase 13 Operational Execution & Verification Checklist
# Max Column: 80 Columns

The successful merge of Phase 11 and Phase 12 (PR #11, version `0.12.0`,
112 unit tests passing cleanly) represents a major architectural milestone
for `unplow`.

The implementation of the **Adapter Parameter Isolation Guard** in `replow.py`
successfully resolves the CPU memory bottleneck: by freezing backbone layers
and isolating the final transformer block/adapter (~46.8M active parameters),
PyTorch AdamW state bloat was prevented, allowing stable local gradient
distillation on `Qwen2.5-1.5B-Instruct` without triggering OOM SIGKILL
exceptions.

Furthermore, all five specialized domain models (`till-python`, `till-c`,
`till-cpp`, `till-rust`, and `till-gencode`) were distilled and compressed into
high-density `.unp` containers using the 3-expert Bayesian Context-Tree
Switching mixer (`FLAG_BAYES_MIXER = 0x08`), achieving high compression ratios
(up to **13.344:1** on Python and **12.464:1** on C++) with verified integrity.

---

## Phase 13 Operational Execution & Verification Checklist

To transition the `till-*` specialized models into active validation and
deployment across your local development environment, execute the following
concrete verification and evaluation steps:

### 1. Functional Correctness Evaluation via Perplexity & Test Harness

Run the Phase 9 evaluation harness against each distilled and archived `till-*`
container to measure cross-entropy loss shifts ($\Delta \mathcal{L}_{\theta}$)
and verify functional test assertions:

```bash
uv run unplow eval \
  --artifact workspace/targets/till_python.unp \
  --test-suite tests/data/fib_tests.py

uv run unplow eval \
  --artifact workspace/targets/till_cpp.unp \
  --test-suite tests/data/fib_tests.py
```

### 2. Storage Guardian & Rclone Offload Verification

Verify that the `TillDiskGuardian` correctly reads
`workspace/till_manifest.toml`, audits partition telemetry (ensuring the
$\ge 15.0\%$ free space floor), and initiates secure transfers for aged cache
directories to Google Drive:

```bash
uv run python -c "
from unplow.till_guard import TillDiskGuardian
guardian = TillDiskGuardian()
print(guardian.check_disk())
guardian.enforce_space(['~/.cache/uv/', 'models/work_products/'])
"
```

### 3. Full Multi-Domain Build & Test Suite Synchronization

Execute a complete clean build and verify that all 112 unit tests pass across
the updated repository state:

```bash
make clean
make test
make sync
```

<!-- end of file: unplow/inbox/gemweb014.md -->
