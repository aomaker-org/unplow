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

### 5. Pack Self-Verifying .unp Archive with MDL Auto-Domain (Phase 3)

```bash
uv run unplow archive \
  --text "Lossless container test." --auto-domain -o test.unp
```

### 6. Unpack and Verify .unp Container

```bash
uv run unplow unarchive test.unp
```

### 7. Lossless Arithmetic Compression (Paradigm 2)

```bash
uv run unplow compress \
  "Information theory connects sequence modeling with data compression." \
  -o output.bin
```

### 8. Exact Lossless Decompression (Paradigm 2)

```bash
uv run unplow decompress output.bin --tokens 9
```

### 9. Generative Manifold Decompression from Seed

```bash
uv run unplow manifold --seed "unplow_latent_manifold" --tokens 30
```

### 7. Run Decompression Benchmark

```bash
uv run unplow benchmark --runs 3
```

### 8. Start Background Worker Daemon

```bash
uv run unplow daemon --interval 2.0
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
│       ├── cli.py           # Command-line interface
│       ├── forensics.py     # Cryptographic audit ledger & telemetry
│       ├── models.py        # Local model locator & deterministic mocks
│       ├── paradigm1.py     # Lossy static weight decompressor
│       ├── paradigm2.py     # Lossless arithmetic codec & manifold engine
│       ├── queue.py         # Atomic POSIX file-locked queue
│       ├── safety.py        # Air-gap enforcer & additive file safety
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
- **Nicholas Carlini et al. (2021):** *Extracting Training Data from Large
  Language Models* (USENIX Security 2021).
- **Fabrice Bellard (2019-2021):** `NNCP` and `ts_zip` neural compressors.
- **Grégoire Delétang et al. (ICLR 2024):** *Language Modeling Is Compression*
  (Google DeepMind).

<!-- end of file: unplow/README.md -->
