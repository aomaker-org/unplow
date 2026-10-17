# Path: unplow/inbox/gemweb013.md
# Operational Command Reference: Till Model Training & Offloading
# Max Column: 80 Columns

With Phase 11 successfully completed and synchronized on `main` (PR #10, version
`0.11.0`, 112 passing unit tests), `unplow` has an active disk storage guardian
(`TillDiskGuardian`) maintaining a minimum of **15% free disk space** and
automating file offloads to Google Drive (`gdrive:transfer/till-20261017/`)
via `rclone`.

The automated corpora staging (`till-init`) and gradient distillation and
archiving pipeline (`till-train`) are fully operational across all five model
specializations: **`till-python`**, **`till-c`**, **`till-cpp`**,
**`till-rust`**, and **`till-gencode`**.

---

## Complete Operational Command Reference

### 1. Storage Telemetry & Enforcement Audit

To manually inspect live partition metrics or trigger the storage guardian's
offload routines based on `workspace/till_manifest.toml`:

```bash
uv run python -c \
  "from unplow.till_guard import TillDiskGuardian; \
print(TillDiskGuardian().check_disk())"
```

### 2. Initialize and Stage Domain Corpora

To populate or refresh domain-specific training stubs and datasets under
`workspace/till_corpus/`:

```bash
uv run unplow till-init --manifest workspace/till_manifest.toml
```

### 3. Execute Domain Distillation & Bayesian CTS Archiving (`till-*`)

To run local gradient distillation steps and compress the resulting
specialized adapters into high-density `.unp` containers using the 3-expert
Bayesian Context-Tree Switching mixer (`FLAG_BAYES_MIXER = 0x08`):

* **Python Specialization (`till-python`):**
```bash
uv run unplow till-train \
  --domain python \
  --corpus workspace/till_corpus/python/ \
  -o workspace/targets/till_python.unp
```

* **C System Specialization (`till-c`):**
```bash
uv run unplow till-train \
  --domain c \
  --corpus workspace/till_corpus/c/ \
  -o workspace/targets/till_c.unp
```

* **Modern C++ Specialization (`till-cpp`):**
```bash
uv run unplow till-train \
  --domain cpp \
  --corpus workspace/till_corpus/cpp/ \
  -o workspace/targets/till_cpp.unp
```

* **Rust Cargo Specialization (`till-rust`):**
```bash
uv run unplow till-train \
  --domain rust \
  --corpus workspace/till_corpus/rust/ \
  -o workspace/targets/till_rust.unp
```

* **Multi-Language Gencode Specialization (`till-gencode`):**
```bash
uv run unplow till-train \
  --domain gencode \
  --corpus workspace/till_corpus/gencode/ \
  -o workspace/targets/till_gencode.unp
```

<!-- end of file: unplow/inbox/gemweb013.md -->
