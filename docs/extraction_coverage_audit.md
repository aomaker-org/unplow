# Path: unplow/docs/extraction_coverage_audit.md
# Input Model Dynamic Extraction Coverage & Telemetry Audit
# Max Column: 80 Columns

## 1. Executive Summary

This audit assesses the coverage, quality, and syntax validity of training
data dynamically harvested from the target substrate model
(`Qwen/Qwen2.5-Coder-0.5B`) and trained into the compact 4-layer destination
scratch model.

As of October 28, 2026, the extraction pipeline has completed **454 discrete
mechanistic extractions** spanning 5 core language domains. All generated code
samples have been filtered, AST-validated, and cataloged into
`workspace/till_corpus/corpus_manifest.jsonl`.

```
================================================================================
                    INPUT MODEL EXTRACTION AUDIT SUMMARY
================================================================================
Total Manifest Extractions : 454 samples
Extraction Model Substrate : Qwen/Qwen2.5-Coder-0.5B
Target Scratch Model Depth : 4 decoder layers (~195M parameters)
Refactored Syntax Validity : 100.0% (454 / 454)
Step Comparison Logs       : 187 comparative evaluation checkpoints
Evaluation Log Path        : logs/evaluations/step_comparisons.jsonl
Corpus Manifest Path       : workspace/till_corpus/corpus_manifest.jsonl
================================================================================
```

---

## 2. Domain Extraction Breakdown

| Domain | Extracted Samples | Unique Seeds | Raw Syntax Valid | Refactored Valid | Avg Tokens | Avg CC | Avg Quality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Python** | 115 | 5 | 13.0% | 100.0% | 15.7 | 1.04 | 0.96 |
| **C** | 91 | 5 | 100.0% | 100.0% | 25.0 | 1.67 | 0.93 |
| **C++** | 85 | 4 | 100.0% | 100.0% | 25.8 | 1.52 | 0.94 |
| **Rust** | 83 | 5 | 100.0% | 100.0% | 25.9 | 1.57 | 0.94 |
| **Gencode** | 80 | 1 | 100.0% | 100.0% | 23.4 | 1.05 | 0.91 |
| **Total** | **454** | **20** | **77.9%** | **100.0%** | **22.9** | **1.37** | **0.94** |

---

## 3. Extraction Seed Coverage

The extraction harness probes parameter basins across domain-specific seed
anchors defined in `DOMAIN_EXTRACTION_SEEDS`:

### 3.1 Python (115 samples)
- `def sanitize_identifier(name: str) -> str:` — 40 samples (34.8%)
- `class DataStreamProcessor:\n    def __init__(self):` — 28 samples (24.3%)
- `def calculate_crc32(payload: bytes) -> int:` — 19 samples (16.5%)
- `import sys\ndef emit_diagnostic(msg: str) -> None:` — 17 samples (14.8%)
- `def parse_token_stream(tokens: list) -> dict:` — 11 samples (9.6%)

### 3.2 C (91 samples)
- `#include <stdint.h>\nuint32_t compute_crc(...)` — 24 samples (26.4%)
- `#include <stddef.h>\nint verify_block_header(...)` — 21 samples (23.1%)
- `typedef struct { uint32_t id; uint64_t offset; } unp_chunk_t;` — 19 samples (20.9%)
- `static int decode_payload(...)` — 16 samples (17.6%)
- `#include <stdbool.h>\nbool is_aligned(...)` — 11 samples (12.1%)

### 3.3 C++ (85 samples)
- `template <typename T>` — 28 samples (32.9%)
- `#pragma once\ntemplate <typename T, std::size_t Cap> class FastRingBuffer` — 21 samples (24.7%)
- `#include <vector>\nnamespace unp { class ManifoldDecompressor` — 20 samples (23.5%)
- `class StaticMemoryArena { ... };` — 16 samples (18.8%)

### 3.4 Rust (83 samples)
- `pub fn verify_signature(data: &[u8], sig: &[u8]) -> bool` — 21 samples (25.3%)
- `pub struct BoundedBuffer<T>` — 19 samples (22.9%)
- `use std::alloc::{alloc, Layout};` — 17 samples (20.5%)
- `impl Drop for BoundedBuffer` — 16 samples (19.3%)
- `pub fn decode_varint(slice: &[u8]) -> Result<(u64, usize), ...>` — 10 samples (12.0%)

### 3.5 Gencode / Polyglot JSON (80 samples)
- Structured AST / Translation schemas: `{\n  "translation": ...` — 80 samples (100.0%)

---

## 4. Quality & Refactoring Dynamics

1. **AST & Syntax Normalization**:
   - Raw Python extractions directly from parameter manifolds frequently
     contain unclosed blocks or missing return types (13.0% raw validity).
   - The AST refactoring step rewrites incomplete fragments into syntactically
     sound, compilable routines before injection into the training pipeline
     (100.0% refactored validity).
2. **Complexity Metric Analysis**:
   - Systems code (C, C++, Rust) exhibited higher average cyclomatic
     complexity ($1.52 - 1.67$) and richer token density ($25 - 26$ tokens/block)
     than Python ($1.04$ CC) or Gencode schema JSON ($1.05$ CC).
3. **Artifact Retention**:
   - Staged inputs are permanently retained on disk:
     - 454 raw files (`workspace/till_corpus/{domain}/raw_input_iter*.ext`)
     - 474 refactored files (`workspace/till_corpus/{domain}/extracted_iter*.ext`)
     - 454 individual manifest records (`workspace/till_corpus/{domain}/input_manifest_iter*.json`)

---

## 5. Learning Evolution on Comparative Prompts

Progress evaluated every 100 steps on fixed test prompts logged in
`logs/evaluations/step_comparisons.jsonl` demonstrates steady convergence:
- **Steps 0–200**: Destination scratch model emits random character sequences
  and unaligned token bursts.
- **Steps 300–700**: Model adopts language-specific lexical tokens (`def`,
  `uint32_t`, `let mut`).
- **Steps 800–1300**: Generates syntactically coherent statements, variable
  assignments, structured loop bodies, and typed return statements
  approaching training loss $< 0.005$ and perplexity $\approx 1.01$.

<!-- end of file: unplow/docs/extraction_coverage_audit.md -->
