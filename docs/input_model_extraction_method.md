# Path: unplow/docs/input_model_extraction_method.md
# Input Model Mechanistic Extraction Method & Architecture
# Max Column: 80 Columns

## 1. Overview

This document details the mathematical and mechanistic extraction methodology
used by `unplow` to harvest training data directly from pre-trained neural
weights (such as `Qwen/Qwen2.5-Coder-0.5B`) to train compact destination
models.

The extraction pipeline treats the neural network as an indexed, compressed
probabilistic archive, bypassing synthetic prompt engineering in favor of
**Parameter Basin Probing** (Weight Decompression / Paradigm 1) coupled with
**AST-Guided Syntactic Refactoring** and quantitative complexity profiling.

```
+-------------------------------------------------------------------------+
|                  Target Model Parameter Tensor Space \theta             |
+------------------------------------+------------------------------------+
                                     |
                                     | Domain Seed Prefix Anchor
                                     v
+-------------------------------------------------------------------------+
|                      Weight Basin Probing Phase                         |
|   1. Temperature Exploration  : \tau_{explore} = 1.10 (Steps 1..6)      |
|   2. Temperature Annealing    : \tau_{lock}    = 0.30 (Steps 7..N)      |
|   3. Logit Filtering          : Top-K truncation (K = 60)               |
+------------------------------------+------------------------------------+
                                     |
                                     | Raw Harvested Code
                                     v
+-------------------------------------------------------------------------+
|                     AST Refactoring & Sanitization                      |
|   1. Code fence stripping & dedenting                                   |
|   2. Abstract Syntax Tree parsing (`ast.parse`)                         |
|   3. Snippet stub wrapping & compilability verification                 |
+------------------------------------+------------------------------------+
                                     |
                                     | Clean Code & Provenance
                                     v
+-------------------------------------------------------------------------+
|               Complexity, Quality & Manifest Indexing                   |
|   - Cyclomatic Complexity (CC)     - AST Node Density                   |
|   - Shannon Entropy & BPB          - Quality Score (\ge 0.80)           |
|   - Additive Storage in `workspace/till_corpus/{domain}/`               |
+-------------------------------------------------------------------------+
```

---

## 2. Extraction Phases

### Phase 1: Parameter Basin Exploration & Annealing

Probing is orchestrated by `WeightDecompressor.probe_weight_basins()` in
`src/unplow/paradigm1.py`. Rather than relying on uniform greedy decoding or
static sampling, extraction uses a two-stage temperature schedule:

1. **Exploration Window ($t \le 6$)**:
   - The sampling temperature is set to $\tau_{\text{explore}} = 1.10$.
   - Probes the high-entropy stochastic basin surrounding the domain seed
     prefix, forcing the model to diverge from superficial rote completions
     and navigate into memorized training subgraphs.
2. **Lock-In Annealing ($t > 6$)**:
   - The temperature drops abruptly to $\tau_{\text{lock}} = 0.30$.
   - Collapses the probability distribution into low-temperature greedy
     tracking, tracing the memorized parameter valley to reconstruct full
     routines with minimal hallucination.
3. **Top-K Truncation ($K = 60$)**:
   - At each step $t$, scaled logits $z_t / \tau$ are filtered to retain only
     the top 60 vocabulary tokens, truncating improbable tail tokens:

$$\mathcal{P}(x_t \mid x_{<t}) = \text{Softmax}\left(\text{Top}_K\left(\frac{z_t}{\tau}\right)\right)$$

---

## 3. Domain Seed Anchors

Extraction targets 5 specialized systems domains using deterministic seed
prefixes defined in `DOMAIN_EXTRACTION_SEEDS`:

1. **Python**:
   - Function signatures: `def sanitize_identifier(name: str) -> str:`
   - Object architectures: `class DataStreamProcessor:\n    def __init__(self):`
   - Numeric & diagnostic routines: `def calculate_crc32(payload: bytes) -> int:`
2. **C**:
   - Low-level hashing & checksums: `#include <stdint.h>\nuint32_t compute_crc(...)`
   - Header validations: `#include <stddef.h>\nint verify_block_header(...)`
   - Memory layout structs: `typedef struct { uint32_t id; uint64_t offset; } unp_chunk_t;`
3. **C++**:
   - Generic template containers: `template <typename T>`
   - RAII buffer designs: `#pragma once\ntemplate <typename T, std::size_t Cap> class FastRingBuffer`
   - Namespace utilities: `#include <vector>\nnamespace unp { class ManifoldDecompressor`
4. **Rust**:
   - Cryptographic verifications: `pub fn verify_signature(data: &[u8], sig: &[u8]) -> bool`
   - Memory allocation: `use std::alloc::{alloc, Layout};`
   - Lifetime & Drop implementations: `impl Drop for BoundedBuffer`
5. **Gencode**:
   - Structured JSON AST schemas and cross-language translation pairs:
     `{\n  "translation": ...`

---

## 4. AST Refactoring & Syntax Recovery

Raw text harvested directly from parameter manifolds can exhibit trailing
truncation or unclosed syntactic blocks (e.g. 13% raw Python validity).
The `ReplowEngine.refactor_source_code()` module normalizes all samples:

1. **Fencing & Formatting Cleanup**:
   - Strips extraneous Markdown markers (```) and dedents whitespace blocks.
2. **AST Parsing & Compilation Verification**:
   - Verifies code validity using Python's `ast.parse()` and `compile()`.
3. **Snippet Wrapping & Fallback**:
   - Incomplete statements are enclosed in syntactically complete routine stubs,
     ensuring that 100% of samples trained into the scratch model are valid.

---

## 5. Metadata Indexing & Curriculum Scoring

For every extraction, `compute_code_metrics()` calculates structural metrics
persisted in `workspace/till_corpus/{domain}/input_manifest_iter*.json` and
appended to `corpus_manifest.jsonl`:

- **Cyclomatic Complexity ($V(G)$)**: Decision point counting
  ($\text{if}, \text{for}, \text{while}, \text{try}, \text{and}, \text{or}$).
- **AST Node Count**: Total syntactic constructs present in the parsed tree.
- **Token Diversity Ratio**: $\frac{|\text{Unique Tokens}|}{|\text{Total Tokens}|}$.
- **Quality Score ($Q$)**:
  Composite metric weighting token diversity, syntax validity, and node density:

$$Q = 0.5 \cdot \text{syntax\_valid} + 0.3 \cdot \text{token\_diversity} + 0.2 \cdot \min\left(1.0, \frac{\text{ast\_nodes}}{20}\right)$$

---

## 6. Retention & Additive Storage

All extracted data is stored additively complying with **Directive 07**:
- `workspace/till_corpus/{domain}/raw_input_iterNNN.ext`: Raw output directly from the model.
- `workspace/till_corpus/{domain}/extracted_iterNNN.ext`: Sanitized and refactored code.
- `workspace/till_corpus/{domain}/input_manifest_iterNNN.json`: Metadata and provenance.
- `workspace/till_corpus/corpus_manifest.jsonl`: Unified sequential training log.

<!-- end of file: unplow/docs/input_model_extraction_method.md -->
