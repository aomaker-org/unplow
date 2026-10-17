# Path: unplow/docs/architecture.md
# unplow Architectural Specification
# Max Column: 80 Columns

## 1. Overview & Purpose

`unplow` is a dedicated research laboratory and runtime engine designed for
treating Large Language Models as compressed data archives and implementing
dual mathematical decompression paradigms:

1. **Lossy Static Weight Decompression (Paradigm 1)**:
   Extracts memorized pre-training data directly from static parameters $\theta$
   using temperature-annealed parameter basin exploration and Carlini et al.
   entropy calibration without requiring any external bitstream.
   - Enhanced via **Mechanistic Logit Lens Inspection** (Geva et al., 2021):
     extracts high-salience vocabulary clusters directly from static MLP
     down-projection value vectors without forward passes.
2. **Lossless Autoregressive Arithmetic Decompression (Paradigm 2)**:
   Pairs the LLM's next-token predictive probability distribution with a
   32-bit fixed-point Arithmetic Coder to achieve lossless text compression/
   decompression and deterministic generative decompression of arbitrary seeds.
   - Enhanced via **Framed `.unp` Binary Container** with 18-byte headers and
     deterministic sliding KV-cache horizons.

---

## 2. Component Topology

```
+-------------------------------------------------------------------------+
|                          CLI Engine (unplow)                            |
| (probe, inspect, compress, decompress, archive, unarchive, manifold)    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|                   Decompression Worker Supervisor                       |
|           - Additive File Handler (Directive 07)                        |
|           - Forensic Audit Logger (Directive 02)                        |
|           - Process Heartbeat & Telemetry (Directive 17)                |
+------------------+----------------------------------+-------------------+
                   |                                  |
                   v                                  v
+------------------+---------------+  +---------------+-------------------+
|        File-Backed Queue         |  |         Decompression Engines     |
|  - records/queue/incoming/       |  |  [Paradigm 1] WeightDecompressor  |
|  - records/queue/processing/     |  |  - Temperature Basin Exploration  |
|  - records/queue/done/           |  |  - Carlini Memorization Scoring   |
|  - records/queue/failed/         |  |  - Mechanistic Logit Lens Prober  |
|  - POSIX fcntl.flock concurrency |  |  [Paradigm 2] LLMArithmeticCodec  |
+----------------------------------+  |  - 32-bit Integer Interval Coding |
                                      |  - Framed .unp Binary Container   |
                                      |  - Sliding-Window KV Truncation   |
                                      +-----------------+-----------------+
                                                        |
                                                        v
                                      +-----------------+-----------------+
                                      |        Model Substrate            |
                                      |  - Local Safetensors Snapshots    |
                                      |  - Deterministic Mock Evaluator   |
                                      |  - Ollama & GGUF Local Endpoints  |
                                      +-----------------------------------+
```

---

## 3. Framed `.unp` Binary Container Layout

Paradigm 2 packages arithmetic bitstreams into self-verifying `.unp` archives
comprising an 18-byte big-endian header followed by raw interval bits:

| Offset | Field | Type | Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0x00` | `magic` | ASCII | 6 B | Magic bytes `UNPLOW` |
| `0x06` | `version` | uint8 | 1 B | Archive format version (`0x01`) |
| `0x07` | `flags` | uint8 | 1 B | Bitfield: Bits 0..3 modes, 4..7 domain |
| `0x08` | `num_tokens`| uint32 | 4 B | Exact token count to decode |
| `0x0C` | `max_ctx_len`| uint16 | 2 B | KV-cache horizon stride |
| `0x0E` | `sha256_pre`| bytes | 4 B | First 32 bits of plaintext SHA-256 |
| `0x12` | `payload` | bytes | $N$ B | Raw 32-bit arithmetic bitstream |

---

## 4. Queue State Machine & Concurrency

The file-backed queue enforces atomic state transitions using POSIX advisory
file locking (`fcntl.flock(fd, LOCK_EX | LOCK_NB)`):

1. **`incoming/`**: Client or CLI enqueues task JSON payload atomically via
   dot-prefixed temporary file write and `os.replace`.
2. **`processing/`**: Worker acquires exclusive advisory lock, moves task file
   into `processing/`, records start timestamp and process PID.
3. **`done/`**: On successful execution, output result and metrics are stored,
   and file is moved atomically to `done/`.
4. **`failed/`**: On unhandled error, exception trace is recorded, and file is
   moved to `failed/`.

---

## 5. Air-Gap Safety & Forensic Audit Trail

- **Air-Gap Guardrails**: Directives 01 and 18 mandate 100% offline execution.
  Outbound network destinations are rejected at runtime; all operations are
  confined to localhost loopback interfaces and local model weights.
- **Cryptographic Audit Ledger**: Every task dispatch, prompt, compression
  payload, and extracted text is logged to
  `logs/forensics/audit_journal.jsonl` with ISO-8601 UTC microsecond timestamps,
  SHA-256 digests, and VmRSS memory metrics.

---

## 6. Coupled SwiGLU Circuits & Anisotropic Logit Lens

In Phase 3, `unplow` couples the input receptor geometry with output emission
projections directly from static weight matrices without forward passes:

1. **Anisotropic Mean Centering**: Late transformer layers exhibit
   representation collapse around a shared non-zero mean (Ethayarajh, 2019;
   Sun et al., 2024). Subtracting layer mean vector
   $\bar{\mathbf{v}} = \frac{1}{d}\sum \mathbf{v}_j$ from `down_proj` weights
   isolates distinct semantic circuits.
2. **Selectivity Margin Scoring**: Evaluates $S_i = z_{(1)} - z_{(50)}$ to
   filter out universal numeral amplifiers while retaining domain-specific
   concepts.
3. **SwiGLU Key Coupling**: Matches combined receptor keys
   $\mathbf{k}_i = \text{normalize}(\mathbf{w}_{\text{gate}, i} +
   \mathbf{w}_{\text{up}, i})$ against normalized token embeddings $W_E$,
   synthesizing full associative rules `(Trigger Context -> Target Value)`.

---

## 7. Zero-Bit Domain Conditioning & Vectorized PyTorch CDF

1. **Header Nibble Packing**: Bits 4..7 of the `.unp` header `flags` byte
   store a 4-bit domain ID (`0..15`). Bit 0 retains lossless verification.
2. **KV-Cache Pre-Warming**: Pre-warms `past_key_values` with a deterministic
   domain preamble (prose, Python, C firmware, JSON) prior to token 0, slashing
   cold-start entropy with zero extra header or bitstream bytes.
3. **MDL Auto-Routing**: Rissanen (1978) Minimum Description Length selector
   evaluates exact target cross-entropy across priors in a single batched pass.
4. **Fast PyTorch CDF**: Vectorized int64 cumulative frequency quantization
   eliminates per-token NumPy roundtrips on large vocabularies.

---

## 8. RMSNorm-Folded Key Lenses & Activation Steering

Phase 4 introduces mechanistic activation steering and continuous online cache
mixture arithmetic coding:

1. **RMSNorm Gain Folding**: Pre-MLP affine gains
   $\boldsymbol{\gamma}_{\text{post}}^{(l)}$ are folded into SwiGLU receptor
   vectors $\tilde{\mathbf{k}}_i = (\mathbf{w}_{\text{gate}, i} +
   \mathbf{w}_{\text{up}, i}) \odot \boldsymbol{\gamma}_{\text{post}}^{(l)}$
   and projected onto mean-centered embeddings $\tilde{W}_E = W_E - \bar{W}_E$
   (Dar et al., 2023), eliminating rogue-cone multilingual noise artifacts.
2. **Causal Forward-Hook Steering**: Attaches PyTorch forward hooks to target
   MLP `down_proj` layers, injecting directional impulses
   $\alpha \cdot \hat{\mathbf{v}}_i^{(l)}$ into the residual stream to
   causally force the LLM to decompress specific memorized training basins
   (Meng et al., 2022; Turner et al., 2023).
3. **Continuous Online Prefix-Cache Mixture**: Mixes static LLM probabilities
   with a lockstep empirical unigram frequency cache ($P_{\text{mix}} =
   (1-\lambda)P_{\theta} + \lambda P_{\text{cache}}$) via container bit-flag
   `0x02` (`FLAG_CACHE_MIXTURE`) without side-information bits (Kuhn & De Mori,
   1990; Grave et al., 2017).
4. **Per-Token Surprisal Profiler**: Quantifies exact Shannon bit costs
   $-\log_2 P(x_t \mid x_{<t})$ across sequence trajectories to attribute
   empirical entropy gains from continuous prefix caching.

---

## 9. GGUF Forensics, SVD Subspace Steering & Markov Bigram Cache

Phase 5 unlocks multi-substrate forensics and higher-order sequence modeling:

1. **Pure-Python GGUF v2/v3 Binary Inspector**: Parses GGUF container headers,
   metadata key-value tables, and tensor catalogs directly via `mmap` and
   `struct` (Gerganov, 2023) with zero external C++ dependencies.
2. **Vectorized Block Dequantizers**: Implements direct NumPy dequantization for
   `F32`, `F16`, `Q8_0` (fp16 scale + 32 int8 quants), and `Q4_0` (fp16 scale
   + 16 packed nibble bytes) block-quantized formats (Dettmers et al., 2022).
3. **Cross-Substrate Task-Vector Forensics**: Computes cosine similarities,
   relative Frobenius divergence, and identifies top divergent neuron columns
   between quantized GGUF weights and active PyTorch layers (Ilharco et al.,
   2023).
4. **Multi-Neuron SVD Subspace Steering**: Computes the first principal left
   singular vector $\mathbf{u}_1$ across co-selective neuron ensembles in MLP
   `down_proj` via SVD, steering generation along polysemantic subspaces
   without individual neuron saturation (Elhage et al., 2022; Bricken et al.,
   2023).
5. **Order-1 Markov Bigram Cache Mixture (`FLAG_BIGRAM_CACHE = 0x04`)**:
   Blends static LLM probabilities, Order-0 unigram frequencies, and dynamic
   Order-1 bigram transition probabilities $P_{\text{bi}}(x_t \mid x_{t-1})$
   (Cleary & Witten, 1984; Willems et al., 1995) to slash bitrates for
   recurrent multi-token identifiers in lockstep between encoder and decoder.

---

## 10. Cross-Substrate Task Vectors & Online Bayesian CTS Mixture Coding

Phase 6 completes the 8-bit `.unp` container flag register and bridges static
GGUF binaries with active Safetensors models:

1. **Cross-Substrate Task-Vector Forensics**: Bridges memory-mapped GGUFs and
   live Safetensors models via MLP weight delta:
   $$\Delta W_{\text{down}}^{(l)} = W_{\text{GGUF}}^{(l)} -
   W_{\text{Safetensors}}^{(l)}$$
   isolating domain specialization (e.g. `Coder - Instruct`) or pre-training
   priors (`Base - Instruct`) without dual-model RAM overhead (Ilharco et al.,
   2023; Nasr et al., 2023).
2. **Logit Lens Shift Projection**: Projects top left singular vector
   $\mathbf{u}_{\Delta, 1}^{(l)}$ through model RMSNorm and `lm_head` to
   identify vocabulary promotion and suppression directions.
3. **Differential SVD Steering**: Attaches forward hooks injecting
   $+\gamma \mathbf{u}_{\Delta, 1}^{(l)}$ into the residual stream to steer
   generation along inter-model specialization subspaces (Meng et al., 2022).
4. **Online Bayesian Context-Tree Switching (CTS) Mixture Coding
   (`FLAG_BAYES_MIXER = 0x08`)**: Dynamically updates 3-expert posterior weights
   $\mathbf{w}_t = (w_{t, \text{LLM}}, w_{t, \text{uni}}, w_{t, \text{bi}})$
   via Vovk's aggregating algorithm and Context-Tree Switching regularization
   floor $\alpha = 0.03$ (Willems et al., 1995; Vovk, 1990; Veness et al., 2012)
   with zero side-information bits.
5. **Completed `.unp` v1 Flag Specification**: Every bit in the 8-bit flag
   register is now assigned and fully backward-compatible:
   - `0x01`: Lossless Verified (Bit 0)
   - `0x02`: Prefix Unigram Cache (Bit 1)
   - `0x04`: Markov Bigram Cache (Bit 2)
   - `0x08`: Bayesian CTS Mixture (Bit 3)
   - `0xF0`: Domain Prior ID (Bits 4–7)

---

## 11. Layer-Wise Internal Information Auditor & Convergence Flow

Phase 7 introduces the unified internal representation auditor
(`src/unplow/auditor.py`) to diagnose how information accumulates and
compresses across network depth:

1. **Intermediate Logit Lens Unembedding**: Projects intermediate residual
   stream hidden states $\mathbf{h}_t^{(l)}$ at the final token position
   through the terminal layer norm and unembedding projection
   $\text{LogitLens}(\mathbf{h}_t^{(l)}) = W_U \text{Norm}(\mathbf{h}_t^{(l)})$
   (nostalgebraist, 2020; Geva et al., 2021).
2. **Layer-Wise Entropy & Probability Profiling**: Computes exact Shannon
   entropy $H(P^{(l)}) = -\sum_{v} P_v^{(l)} \log_2 P_v^{(l)}$ and top
   probability mass to measure the progressive collapse of predictive
   uncertainty across depth.
3. **Residual & FFN Norm Dynamics**: Tracks residual vector $L_2$ norm growth
   $\|\mathbf{h}_t^{(l)}\|_2$ and FFN downstream projection scale
   $\|W_{\text{down}}^{(l)}\|_2$ to isolate computational intensity across
   shallow, middle, and late layers.

---

## 12. Replow: Closed-Loop Refactoring, Distillation & High-Density Archiving

Phase 8 introduces the `replow` closed-loop pipeline (`src/unplow/replow.py`),
connecting extraction, syntax sanitization, dataset distillation staging, and
high-density `.unp` container archiving:

1. **Mechanistic Harvesting**: Extracts latent code basins from neural weights
   via SVD task-vector steering and causal forward-hook impulses.
2. **AST Static Analysis & Syntax Sanitization**: Strips markdown code fences,
   normalizes indentation, validates semantic compilability via `ast.parse` and
   `compile`, and provides automatic function stub wrappers for snippet
   fragments.
3. **Distillation Dataset Staging & Training Steps**: Formats refactored code
   into structured JSONL training datasets and executes lightweight PyTorch
   next-token cross-entropy gradient update steps across model parameters.
4. **Bayesian CTS Archiving**: Compresses refactored code directly into
   self-verifying `.unp` archives using the 3-expert Bayesian Context-Tree
   Switching mixture coder (`FLAG_BAYES_MIXER = 0x08`), maximizing compression
   density across structural code patterns.

---

## 13. Replow Evaluation Harness: Functional Correctness & Perplexity Tracking

Phase 9 introduces the evaluation harness (`src/unplow/eval.py`), combining
sandboxed execution against unit test assertions with per-token perplexity
and cross-entropy tracking before and after distillation:

1. **Sandboxed Functional Correctness (`FunctionalCorrectnessRunner`)**:
   Injects refactored Python code and test assertions into an isolated
   multiprocessing namespace, enforcing execution timeouts and capturing
   runtime exceptions or assertion failures.
2. **Perplexity & Surprisal Shift Tracking (`PerplexityTracker`)**:
   Measures cross-entropy loss ($\mathcal{L}_{\theta}$) and perplexity
   ($\mathcal{P} = 2^{\mathcal{L}_{\theta}}$) across validation corpora,
   tracking empirical compression and predictability shifts
   ($\Delta \mathcal{L}_{\theta}$) resulting from local gradient distillation.
3. **Container-Direct Evaluation**:
   Accepts both raw `.py` scripts and `.unp` binary archives directly,
   losslessly decompressing containers in memory before dispatching test
   assertions and corpus loss evaluation.

---

## 14. Workspace Bundler: Hierarchical Directory Tree Archiving

Phase 10 introduces the Workspace Bundler (`src/unplow/bundler.py`), enabling
recursive multi-file workspace directory tree packaging and extraction into
self-verifying, single `.unp` containers:

1. **Recursive Multi-File Packaging (`WorkspaceBundler.bundle_directory`)**:
   Traverses directory structures recursively, filtering non-essential build
   artifacts (`.git`, `__pycache__`, `node_modules`, `.venv`), computes per-file
   cryptographic SHA-256 hashes, and serializes a structured JSON manifest
   alongside concatenated byte streams.
2. **Bayesian CTS Mixture Compression**:
   Encodes multi-file container payloads through the 3-expert Bayesian
   Context-Tree Switching arithmetic codec (`FLAG_BAYES_MIXER = 0x08`),
   maximizing cross-file compression density and pattern reuse.
3. **Lossless Restoration & Integrity Verification (`unbundle_archive`)**:
   Decodes the `.unp` binary container, parses the JSON manifest, reconstructs
   the original hierarchical directory structure, and performs strict SHA-256
   checksum validation on every extracted file before completion.

---

## 15. Till Models: Disk Telemetry, Offload Manifest & Domain Staging

Phase 11 introduces the Disk Guardian and Till Staging pipeline
(`src/unplow/till_guard.py`), establishing automated disk telemetry, `rclone`
offload synchronization, and domain-specialized training pipelines:

1. **Storage Telemetry & 15% Free Space Guard (`TillDiskGuardian`)**:
   Tracks real-time disk partition capacity via `shutil.disk_usage`, validating
   that free disk space remains strictly at or above the 15% operational
   threshold. If disk exhaustion is imminent, dispatches configured offload
   transfers to Google Drive (`gdrive:transfer/till-20261017/`)
   via `rclone move`.
2. **Offload Manifest Framework (`workspace/till_manifest.toml`)**:
   Declaratively catalogs offloadable caches (`uv/`, `pip/`, `ccache/`), aged
   model checkpoints, and compressed `.unp` archive targets with configurable
   retention policies.
3. **Domain Corpus Staging (`init_till_staging`)**:
   Stages isolated training corpora under `workspace/till_corpus/{domain}/`
   across five specialization domains: `till-python` (AST stubs, `domain_id=2`),
   `till-c` (POSIX & driver headers, `domain_id=3`), `till-cpp` (STL templates,
   `domain_id=1`), `till-rust` (Cargo crates & macros, `domain_id=1`), and
   `till-gencode` (polyglot translation pairs, `domain_id=0`).
4. **Closed-Loop Distillation & Bayesian Archiving (`train_till_domain`)**:
   Executes domain-specific PyTorch gradient update steps, persists adapter
   metadata under `models/till-{domain}-lora/`, and compresses refactored code
   directly into self-verifying `.unp` archives with Bayesian CTS
   (`FLAG_BAYES_MIXER = 0x08`).

---

## 16. Neural Weight Visualizer & Embedded Provenance Metadata

Phase 19 and Phase 20 introduce the Checkpoint Visualizer
(`src/unplow/visualizer.py`), rendering raw neural weights, parameter update
deltas, and multi-projection attention mosaics as self-contained PNG heatmaps:

1. **Air-Gapped Pure-Python PNG Encoder (`encode_png_rgb`)**:
   Encodes raw RGB scanlines via `zlib` compression with zero external
   graphics libraries (no `PIL`/`Pillow`, no `matplotlib`), strictly adhering
   to Directive 18.
2. **Standard PNG Provenance Chunks (`tEXt`) & Bitmap Banners**:
   Embeds target checkpoint paths, comparison baselines, tensor names, and
   summary statistics directly into standard W3C PNG `tEXt` chunks. Draws
   a pure-Python 5x7 ASCII bitmap font header banner onto each rendered frame.
3. **Multi-Projection Attention Mosaics (`render_checkpoint_dashboard`)**:
   Produces 4-quadrant comparative heatmaps displaying raw query weights (TL),
   query update deltas (TR), key update deltas (BL), and output projection
   deltas (BR).

---

## 17. Headless Remote Web Studio & SSH Port-Forwarding

Phase 21 introduces the Web Visualizer Studio (`src/unplow/web.py`) with
remote SSH connectivity for headless server environments:

1. **Interactive Control Panel & Dynamic Telemetry**:
   Features multi-domain checkpoint selection, live parameter delta inspection,
   zoom/pan canvas viewing, and real-time disk quota telemetry.
2. **Headless Remote SSH Architecture**:
   Listens on `0.0.0.0:8765`, enabling secure SSH port-forwarding tunnels
   (`ssh -L 8765:localhost:8765 user@host`) or direct LAN access
   (`http://<lan-ip>:8765`) from remote workstations.

---

## 18. Checkpoint Evolution Timelapse Video

Phase 22 introduces chronological animated video synthesis across model
revisions (`render_checkpoint_timelapse` and `compile_images_to_apng`):

1. **RFC 7689 Animated PNG (APNG) Encoder (`encode_apng_rgb`)**:
   Implements a zero-dependency, pure-Python APNG encoder utilizing `acTL`,
   `fcTL`, `IDAT`, and `fdAT` binary chunk structures with monotonic frame
   sequence numbering. Supported natively across all modern browsers without
   `ffmpeg`.
2. **Multi-Mode Checkpoint Evolution Synthesis**:
   - `delta`: Cumulative weight drift relative to base revision (`rev001`).
   - `step`: Inter-iteration update shifts ($W_i - W_{i-1}$).
   - `raw`: Absolute weight intensity thermal evolution (Inferno palette).
   - `dashboard`: 4-quadrant multi-projection attention mosaic evolution.
3. **Interactive Video Studio Front End & CLI Integration**:
   Integrates video playback (play, pause, frame scrubbing, speed control)
   into the Web Studio and provides `unplow video` and `make video` targets.

## 19. Flicker-Free Video Synthesis & Double-Buffered Canvas Studio

Phase 23 eliminates frame blanking and inter-frame brightness jumps across
timelapse video generation and headless remote viewing:

1. **Architectural Cluster Filtering (`filter_homogeneous_checkpoints`)**:
   Automatically detects and clusters checkpoints by parameter shape and
   file size, excluding isolated mock checkpoints or disparate runs that
   previously induced abrupt frame blanking and visual distortion.
2. **Elimination of Zero-Delta Blank Frames**:
   In `delta` and `step` modes, sequences evolve directly across non-zero
   learned checkpoints, preventing an all-zeros blank frame at index 0.
3. **Global Dynamic Range Normalization (`smooth_scale`)**:
   Computes a global differential maximum scale across the entire sequence
   rather than normalizing each frame in isolation, ensuring consistent,
   flicker-free color contrast across revisions.
4. **Standalone APNG Frame Extractor (`extract_apng_frames`)**:
   RFC 7689 compliant decoder that parses `fcTL`, `IDAT`, and `fdAT` chunks
   to reconstruct standalone PNG byte streams in single-digit milliseconds.
5. **Double-Buffered HTML5 Canvas Video Player & Micro-Diff Inspection**:
   - Zero-flicker double-buffered `<canvas id="canvas-player">` playback.
   - Real-time frame scrubbing with zero decoding lag or blanking.
   - `⚡ A/B Blink`: Rapidly alternates between consecutive checkpoints.
   - `± Diff View`: Live pixel-difference rendering against previous frame.

<!-- end of file: unplow/docs/architecture.md -->
