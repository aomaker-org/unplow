# Path: unplow/docs/scratch_model_training.md
# Scratch Model Training & Dynamic Data Harvesting Pipeline
# Max Column: 80 Columns

## 1. Overview & Architecture

This document describes the architectural enhancements introduced in the
`feature/agy-train-scratch-model` branch for training compact destination
models from scratch using mechanistic data harvested dynamically from
pre-trained LLM substrates (e.g. `Qwen/Qwen2.5-Coder-0.5B`).

Instead of fine-tuning or retraining isolated layers of heavy backbone
models, `unplow` instantiates a lightweight, standalone $N$-layer transformer
(4 or 6 layers, ~195M parameters) that shares tokenization and embedding
geometry with the substrate model.

```
+------------------------------------+
| Target LLM (e.g. Qwen2.5-Coder)   |
| (Mechanistic Parameter Extraction) |
+-----------------+------------------+
                  |
                  v
+-----------------+------------------+
| Raw Harvested Code Samples         |
| (Retained in till_corpus/)         |
+-----------------+------------------+
                  |
                  v
+-----------------+------------------+
| AST Complexity & Quality Filtering |
| (Manifest recorded in jsonl)       |
+-----------------+------------------+
                  |
                  v
+-----------------+------------------+
| Compact Scratch Transformer        |
| (4 or 6 layers, ~195M parameters)  |
+-----------------+------------------+
                  |
                  +--> Auto-resume from latest checkpoint
                  +--> Periodic side-by-side test evaluations
                  +--> Directive 17 heartbeat & storage telemetry
```

---

## 2. Core Components

### 2.1 Compact Scratch Model Instantiation
- Implemented in `src/unplow/models.py` via `create_scratch_model()`.
- Reads config geometry from the substrate model (`vocab_size`, `hidden_size`,
  `intermediate_size`, `num_attention_heads`, `num_key_value_heads`).
- Overrides `num_hidden_layers` to a compact depth (default 4, optional 6).
- Weight tensors are initialized with small pseudo-random Gaussian values
  ($\mathcal{N}(0, 0.02)$), with zero-initialized biases and unit-scaled
  RMS norms.

### 2.2 Dynamic Mechanistic Extraction & Harvesting
- Implemented in `src/unplow/incremental_pipeline.py`.
- Generates raw candidate code directly from parameter latent manifolds
  and retains them in `workspace/till_corpus/{domain}/raw_input_iterNNN.ext`.
- Analyzes code syntax, cyclomatic complexity, AST node count, token diversity,
  and composite quality score via `compute_code_metrics()` in `replow.py`.
- Manifest entries are recorded per iteration in
  `workspace/till_corpus/corpus_manifest.jsonl`.
- Staged training datasets are cached and re-used across iterations to save
  redundant model extraction compute.

### 2.3 Storage Guardian with WSL Host Telemetry
- Implemented in `src/unplow/till_guard.py`.
- Detects WSL virtual environments (`/proc/version`) and probes underlying
  Windows host drive partitions (`/mnt/c`, `/mnt/d`).
- Computes `effective_free_pct = min(wsl_vhd_free_pct, host_free_pct)`.
- If free storage drops below the configured floor (default 15.0%),
  training safely halts and triggers offload routines to prevent host
  drive exhaustion.

### 2.4 Auto-Resume & Periodic Evaluation
- Automatically searches `models/till-*-lora/adapter_model_rev*.pt` for the
  latest scratch model weights (>100MB) on launch and resumes training.
- Evaluates side-by-side prompt completions between substrate and scratch
  destination models every $N$ gradient steps (default: 100).
- Comparison logs are persisted to `logs/evaluations/step_comparisons.jsonl`.
- Emits Directive 17 heartbeats to `logs/heartbeat/unplow.json` at each epoch.

---

## 3. CLI Usage

To run the continuous multi-domain incremental training loop with scratch
model training:

```bash
uv run python src/unplow/incremental_pipeline.py \
  --model models/qwen2.5-coder-0.5b \
  --train-scratch \
  --scratch-layers 4 \
  --domains python c cpp rust gencode \
  --epochs 5 \
  --steps 10 \
  --step-eval-interval 100 \
  --loop
```

### Key Flags
- `--train-scratch`: Instantiate a new model initialized from scratch.
- `--scratch-layers {4,6}`: Number of transformer decoder layers (default: 4).
- `--resume` / `--no-resume`: Control auto-resuming from existing checkpoints.
- `--step-eval-interval N`: Run side-by-side test evaluations every $N$ steps.
- `--min-quality-score <float>`: Filter harvested code by quality threshold.
- `--min-complexity <int>`: Minimum cyclomatic complexity threshold.

<!-- end of file: unplow/docs/scratch_model_training.md -->
