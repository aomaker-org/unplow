# Path: unplow/README.md
# unplow: Deep Dive into LLM Decompression & Latent Manifold Extraction
# Max Column: 80 Columns

Autonomous offline Large Language Model (LLM) decompression laboratory, latent
manifold extractor, and fixed-point arithmetic codec for edge, workstation, and
zero-cloud environments.

Repository: <https://github.com/aomaker-org/unplow>

---

## Theoretical Foundations

Treating a Large Language Model as compressed data sits at the intersection of
algorithmic information theory and autoregressive sequence modeling. Depending
on whether the neural weights themselves or an external bitstream serve as the
payload, this problem splits into two distinct mathematical paradigms:

1. **Lossy Static Weight Decompression (Carlini et al., 2021):**
   Treats static neural network parameters $\theta$ as a self-contained, lossy
   archive of the pre-training corpus $D$. Unpacks memorized sequences directly
   from weights with zero external bitstream via temperature-decayed basin
   probing and perplexity-to-`zlib` entropy calibration.
2. **Lossless Autoregressive Arithmetic Decompression (Delétang et al., 2023):**
   Treats the LLM as a frozen conditional probability measure
   $P_{\theta}(x_t \mid x_{<t})$ driving a 32-bit fixed-point integer
   Arithmetic Coder (Witten, Neal, & Cleary, 1987). Reconstructs exact text
   losslessly from compact bitstreams, or generatively decompresses arbitrary
   cryptographic seeds through the learned model manifold.

---

## Key Features

- **Paradigm 1: Static Weight Decompression**:
  - Temperature-annealed semantic basin probing ($\tau_{\text{high}} = 1.15 \to
    \tau_{\text{low}} = 0.25$).
  - Intrinsic syntactic entropy calibration: Carlini memorization score
    $\mathcal{M}(x) = H_{\text{zlib}}(x) / \mathcal{L}_{\theta}(x)$.
  - Casing perturbation sensitivity: $\mathcal{L}_{\theta}(\text{lower}(x)) /
    \mathcal{L}_{\theta}(x)$ detecting verbatim memorization of code identifiers
    and licenses.
- **Paradigm 2: Lossless Arithmetic Codec**:
  - 32-bit fixed-point integer interval partitioning ($[L, H]$ in
    $[0, 2^{32}-1]$) with Witten-Neal-Cleary underflow tracking.
  - Strictly monotonic integer CDF mapping ($M = 2^{24}$ mass, guaranteeing
    $\ge 1$ count per token to prevent zero-interval collapse).
  - Exact bit-for-bit lossless compression and decompression roundtrip.
  - Generative manifold decompression: maps arbitrary hashes (SHA-256) or seeds
    deterministically into coherent text across the LLM manifold.
- **Local Model Substrate & Air-Gap Mandate**:
  - Auto-resolves locally cached HuggingFace snapshots
    (`Qwen2.5-1.5B-Instruct`), local `.gguf` binaries, and Ollama endpoints.
  - Full compliance with Directive 18 (100% offline, zero cloud AI API burn).
- **Coupled SwiGLU Circuits & Anisotropic Logit Lens (Phase 3)**:
  - Direct static extraction of coupled trigger-emitter rules
    `(Trigger Key -> Target Value)` from SwiGLU gate/up/down projections.
  - Anisotropic mean-centering on late-layer value vectors.
- **Zero-Bit Domain Conditioning & MDL Auto-Router (Phase 3)**:
  - Packs 4-bit domain prior into `.unp` header flags upper nibble.
  - Pre-warms KV-cache to eliminate token-0 cold-start entropy.
  - Minimum Description Length auto-routing across prose, code, and telemetry.
- **RMSNorm-Folded Key Lenses & Activation Steering (Phase 4)**:
  - Folds `post_attention_layernorm.weight` into SwiGLU receptor keys over
    mean-centered $\tilde{W}_E$, removing multilingual rogue-cone artifacts.
  - Causal forward-hook steering on MLP `down_proj` layers, forcing the model
    to unpack memorized training basins from neutral prompts.
- **Online Prefix-Cache Mixture & Surprisal Profiler (Phase 4)**:
  - Dynamically blends static LLM softmax with deterministic prefix
    unigram histogram ($P_{\text{mix}} = (1-\lambda)P_{\theta} + \lambda
    P_{\text{cache}}$) via `.unp` Bit 1 (`FLAG_CACHE_MIXTURE`) with zero
    side-information bits.
  - Per-token Shannon surprisal attribution ledger quantifying empirical bits
    saved by continuous prefix caching.
- **GGUF v2/v3 Forensics & Vectorized Dequantizers (Phase 5)**:
  - Pure-Python `mmap` parser for GGUF headers, metadata, and tensor tables
    (Gerganov, 2023) operating 100% offline with zero external C++ libraries.
  - Vectorized NumPy block dequantizers for `F32`, `F16`, `Q8_0`, and `Q4_0`.
  - Cross-substrate Task-Vector divergence metrics (Ilharco et al., 2023)
    measuring cosine similarities and top divergent columns against PyTorch.
- **Multi-Neuron SVD Subspace Steering (Phase 5)**:
  - Synthesizes Rank-1 principal singular steering vector $\mathbf{u}_1$ across
    co-selective neuron ensembles (Elhage et al., 2022; Bricken et al., 2023),
    unlocking polysemantic basins without single-neuron saturation.
- **Order-1 Markov Bigram Cache Coding (Phase 5)**:
  - Blends static LLM softmax, Order-0 unigram frequencies, and Order-1 Markov
    bigram transition probabilities ($P_{\text{bi}}(x_t \mid x_{t-1})$) under
    `.unp` container header Bit 2 (`FLAG_BIGRAM_CACHE = 0x04`).
- **Cross-Substrate Task Vectors & Differential Steering (Phase 6)**:
  - Memory-maps GGUF models on disk to compute $\Delta W_{\text{down}}^{(l)} =
    W_{\text{GGUF}}^{(l)} - W_{\text{Safetensors}}^{(l)}$ without dual-model RAM
    overhead (Ilharco et al., 2023; Nasr et al., 2023).
  - Projects principal left singular vector $\mathbf{u}_{\Delta, 1}$ through
    model RMSNorm and `lm_head` (Logit Lens) to analyze vocabulary shifts.
  - Injects directional impulses into the residual stream via forward hooks,
    steering generation along cross-model specialization subspaces.
- **Online Bayesian Context-Tree Switching (CTS) Mixture Coding (Phase 6)**:
  - Adapts 3-expert posterior weights $\mathbf{w}_t = (w_{\text{LLM}},
    w_{\text{uni}}, w_{\text{bi}})$ dynamically using Vovk's aggregating
    algorithm with switching floor $\alpha=0.03$ (Willems et al., 1995;
    Vovk, 1990; Veness et al., 2012).
  - Assigned to container header Bit 3 (`FLAG_BAYES_MIXER = 0x08`), completing
    every bit in the 8-bit `.unp` header flag register with zero side bits.
- **Layer-Wise Internal Information Auditor (Phase 7)**:
  - Profiles intermediate Logit Lens projections, top token predictions, and
    entropy collapse across every Transformer layer.
  - Monitors residual stream $L_2$ vector norm dilation and FFN down-projection
    weight scaling across depth.
- **Closed-Loop Refactoring, Distillation & Archiving (Phase 8 - `replow`)**:
  - Mechanistic code harvesting and latent basin extraction via activation and
    task-vector steering.
  - AST-driven static sanitization, markdown fence removal, indentation
    normalization, and fallback function stub wrapping (`__refactored_stub__`).
  - Structured JSONL dataset staging and lightweight PyTorch next-token
    distillation fine-tuning gradient steps.
  - Direct end-to-end compression into `.unp` containers using the 3-expert
    Bayesian Context-Tree Switching (CTS) mixture arithmetic codec.
- **Automated Evaluation Harness & Perplexity Tracking (Phase 9 - `eval`)**:
  - Sandboxed execution (`FunctionalCorrectnessRunner`) isolating refactored
    code and unit test assertions in separate multiprocessing namespaces.
  - Per-token cross-entropy loss ($\mathcal{L}_{\theta}$) and perplexity
    ($\mathcal{P}$) shift tracking over validation corpora before and after
    local distillation.
  - Direct evaluation of `.unp` binary containers and `.py` source scripts with
    formatted ASCII diagnostic reports and structured JSON exports.
- **Hierarchical Workspace Bundler (Phase 10 - `bundler`)**:
  - Recursive packaging of multi-file directory structures (sources, configs,
    assets) into a single self-contained `.unp` container.
  - 3-expert Bayesian Context-Tree Switching arithmetic compression
    (`FLAG_BAYES_MIXER = 0x08`) maximizing cross-file pattern compression.
  - Strict per-file cryptographic SHA-256 integrity verification during lossless
    unbundling and workspace restoration.
- **Till Specialized Models & Storage Guardian (Phase 11 - `till`)**:
  - Continuous disk telemetry enforcement strictly maintaining >= 15% free
    partition space, dispatching configured `rclone` offload transfers.
  - Multi-language training corpus staging across five specialized domains:
    `till-python`, `till-c`, `till-cpp`, `till-rust`, and `till-gencode`.
  - Closed-loop PyTorch distillation fine-tuning, adapter configuration saving,
    and direct high-density archiving via Bayesian CTS
    (`FLAG_BAYES_MIXER = 0x08`).
- **Atomic File-Backed Queue & Observability**:
  - POSIX advisory file locking (`fcntl.flock`) over `records/queue/`.
  - Cryptographic audit trail logging SHA-256 hashes, compression stats, and
    RSS memory telemetry (`logs/forensics/audit_journal.jsonl`).
  - Real-time terminal activity viewer (`tools/unplow_viewer.sh`).

---

## Quickstart

### 1. Environment Setup (Mandatory `uv`)

```bash
cd /home/fekerr/src/unplow
uv venv
uv pip install -e ".[dev]"
```

### 2. Audit Available Local Models

```bash
uv run unplow models
```

### 3. Extract Coupled SwiGLU Circuits (Phase 3)

```bash
uv run unplow circuits --layer 16 --candidates 10 --seeds 4
```

### 4. Probe Static Weight Basins (Paradigm 1)

```bash
uv run unplow probe --samples 4 --seq-len 64
```

### 5. Extract RMSNorm-Folded Circuits & Steer Basins (Phase 4)

```bash
uv run unplow circuits --folded --layer 16 --candidates 10
uv run unplow steer --layer 16 --neuron 467 --gain 4.5 --prompt "/* "
```

### 6. Pack .unp Container with Domain Prior & Cache Mixture (Phase 4)

```bash
uv run unplow archive \
  --text "Lossless container test." --auto-domain --cache-mix -o test.unp
```

### 7. Profile Token Surprisal & Cache Mixture Savings (Phase 4)

```bash
uv run unplow profile \
  "unplow_token unplow_token unplow_token lossless cache mixture."
```

### 8. Unpack and Verify .unp Container

```bash
uv run unplow unarchive test.unp
```

### 9. Lossless Arithmetic Compression (Paradigm 2)

```bash
uv run unplow compress \
  "Information theory connects sequence modeling with data compression." \
  -o output.bin
```

### 10. Exact Lossless Decompression (Paradigm 2)

```bash
uv run unplow decompress output.bin --tokens 9
```

### 11. Generative Manifold Decompression from Seed

```bash
uv run unplow manifold --seed "unplow_latent_manifold" --tokens 30
```

### 12. Run Decompression Benchmark

```bash
uv run unplow benchmark --runs 3
```

### 13. Start Background Worker Daemon

```bash
uv run unplow daemon --interval 2.0
```

### 14. Inspect GGUF Binary & Compute Task Vector (Phase 5)

```bash
uv run unplow gguf-inspect /home/fekerr/ai_models/qwen2-1.5b.gguf --compare 16
```

### 15. Multi-Neuron SVD Subspace Steering (Phase 5)

```bash
uv run unplow subspace-steer --layer 16 --neurons 467,468,469 --gain 4.0
```

### 16. Pack Container with Order-1 Markov Bigram Cache (Phase 5)

```bash
uv run unplow archive \
  --text "Lossless bigram test." --bigram-mix -o test_bg.unp
```

### 17. Cross-Substrate Task-Vector Forensics & Steering (Phase 6)

```bash
uv run unplow task-vector \
  --gguf /home/fekerr/ai_models/qwen2.5-coder-1.5b.gguf \
  --layer 16 --steer --gain 2.5 --prompt "/* "
```

### 18. Pack Container with Online Bayesian CTS Dynamic Mixture (Phase 6)

```bash
uv run unplow archive \
  --file pyproject.toml --bayes-mix -o pyproject.unp
```

### 19. Layer-Wise Internal Information & Surprisal Audit (Phase 7)

```bash
uv run unplow audit "def fibonacci(n):"
```

### 20. Closed-Loop Refactoring & Archiving (Phase 8)

```bash
uv run unplow replow \
  --raw "```python\ndef fib(n):\n    return n\n```" \
  -o workspace/targets/fib.unp
```

### 21. Sandboxed Functional Correctness & Perplexity Evaluation (Phase 9)

```bash
uv run unplow eval \
  --artifact tests/data/fib_solution.py \
  --test-suite tests/data/fib_tests.py
```

### 22. Hierarchical Workspace Bundling & Restoration (Phase 10)

```bash
# Bundle directory into single .unp container
uv run unplow bundle src/ -o workspace/targets/src_bundle.unp

# Unbundle archive back with SHA-256 integrity verification
uv run unplow unbundle workspace/targets/src_bundle.unp -o workspace/restored/
```

### 23. Till Model Staging, Disk Telemetry & Distillation (Phase 11)

```bash
# Verify disk telemetry and 15% threshold
uv run python -c \
  "from unplow.till_guard import TillDiskGuardian; \
print(TillDiskGuardian().check_disk())"

# Stage domain training corpora
uv run unplow till-init --manifest workspace/till_manifest.toml

# Train specialized till-python model and archive with Bayesian CTS
uv run unplow till-train \
  --domain python \
  --corpus workspace/till_corpus/python/ \
  -o workspace/targets/till_python.unp
```

---

## Testing

Execute the test suite using pytest via `uv`:

```bash
uv run pytest
```

or via Makefile:

```bash
make test
```

---

## Repository Structure

```
unplow/
├── src/
│   └── unplow/              # Core LLM decompression package
│       ├── auditor.py       # Layer-wise internal info & logit lens audit
│       ├── bundler.py       # Hierarchical workspace bundler & unbundler
│       ├── cache_coder.py   # Online prefix-cache mixture & surprisal
│       ├── circuits.py      # Coupled SwiGLU circuits & logit lens
│       ├── cli.py           # Command-line interface
│       ├── conditioning.py  # Domain prior pre-warming & fast PyTorch CDF
│       ├── container.py     # 18-byte framed .unp binary container
│       ├── ctw_coder.py     # Online Bayesian CTS mixture arithmetic coding
│       ├── eval.py          # Sandboxed functional tests & perplexity tracker
│       ├── forensics.py     # Cryptographic audit ledger & telemetry
│       ├── gguf_inspector.py# GGUF v2/v3 parser & block dequantizer
│       ├── mechanistic.py   # Direct layer logit lens inspection
│       ├── models.py        # Local model locator & deterministic mocks
│       ├── paradigm1.py     # Lossy static weight decompressor
│       ├── paradigm2.py     # Lossless arithmetic codec & manifold engine
│       ├── queue.py         # Atomic POSIX file-locked queue
│       ├── replow.py        # Closed-loop AST refactor, distill & archive
│       ├── safety.py        # Air-gap enforcer & additive file safety
│       ├── steering.py      # RMSNorm-folded lens & activation steering
│       ├── subspace_coder.py# SVD subspace steering & Markov bigram cache
│       ├── task_vectors.py  # Cross-substrate task vectors & SVD steering
│       ├── till_guard.py    # Storage guardian & till domain model staging
│       └── worker.py        # Decompression worker daemon
├── models/
│   ├── configs/             # Model configurations
│   └── qwen2.5-1.5b-instruct# Symlink to local model snapshot
├── records/
│   └── queue/               # Atomic state partitions
│       ├── incoming/
│       ├── processing/
│       ├── done/
│       └── failed/
├── workspace/
│   ├── targets/             # Targets for compression/decompression
│   └── work_products/       # Additive work products & outputs
├── logs/
│   ├── forensics/           # Tamper-evident audit journals
│   └── heartbeat/           # Daemon worker telemetry
├── tools/
│   ├── run_with_logging.py  # Directive 02 universal execution wrapper
│   └── unplow_viewer.sh     # Real-time activity and artifact viewer
├── tests/                   # Complete pytest unit test suite
├── docs/                    # Architecture & decompression guides
├── inbox/                   # Ingested reference specifications
├── pyproject.toml           # Package configuration & entrypoints
├── Makefile                 # Standardized development targets
├── AI.md                    # Operational directives and kernel rules
├── AGENTS.md                # AI agent governance pointer
└── README.md                # Project documentation
```

---

## Academic & Historical Attribution

- **Claude Shannon (1948, 1951):** *A Mathematical Theory of Communication* &
  *Prediction and Entropy of Printed English*.
- **Ray Solomonoff (1964) & Marcus Hutter (2005):** Algorithmic probability,
  Kolmogorov complexity, and the Hutter Prize.
- **Jorma Rissanen (1976) & Ian Witten, Radford Neal, John Cleary (1987):**
  Arithmetic coding with fixed-point integer registers.
- **John Cleary & Ian Witten (1984):** *Data Compression Using Adaptive Coding
  and Partial String Matching (PPM)* (IEEE Trans. Commun.).
- **Frans Willems, Yuri Shtarkov, & Tjalling Tjalkens (1995):**
  *The Context-Tree Weighting Method: Basic Properties*
  (IEEE Trans. Inf. Theory).
- **Volodya Vovk (1990):** *Aggregating Strategies* (Proc. 3rd Annu. Workshop
  Comput. Learn. Theory).
- **Joel Veness, Kee Siong Ng, Marcus Hutter, William Uther, & David Silver
  (2012):** *Context Tree Switching* (IEEE Trans. Inf. Theory).
- **Roland Kuhn & Renato De Mori (1990):** *A Cache-Based Natural Language Model
  for Speech Recognition* (IEEE TPAMI).
- **Nicholas Carlini et al. (2021):** *Extracting Training Data from Large
  Language Models* (USENIX Security 2021).
- **Milad Nasr, Nicholas Carlini, et al. (2023):** *Scalable Extraction of
  Training Data from (Production) Language Models*.
- **Fabrice Bellard (2019-2021):** `NNCP` and `ts_zip` neural compressors.
- **Tim Dettmers et al. (2022):** *LLM.int8(): 8-bit Matrix Multiplication
  for Transformers at Scale* (NeurIPS 2022).
- **Kevin Meng et al. (2022) & Alex Turner et al. (2023):** Locating and
  editing facts (ROME) & activation addition / steering vectors.
- **Nelson Elhage et al. (2022) & Trenton Bricken et al. (2023):** *Toy Models
  of Superposition* & *Towards Monosemanticity* (Anthropic).
- **Gabriel Ilharco et al. (2023):** *Editing Models with Task Arithmetic*
  (ICLR 2023).
- **Georgi Gerganov (2023):** *The GGUF File Format Specification*
  (`ggml.ai`).
- **Guy Dar, Mor Geva, Ankit Gupta, & Jonathan Berant (2023):** *Analyzing
  Transformers in Embedding Space* (ACL 2023).
- **Edouard Grave, Armand Joulin, & Nicolas Usunier (2017):** *Improving Neural
  Language Models with a Continuous Cache* (ICLR 2017).
- **Grégoire Delétang et al. (ICLR 2024):** *Language Modeling Is Compression*
  (Google DeepMind).

<!-- end of file: unplow/README.md -->
