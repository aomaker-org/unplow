# Path: unplow/docs/decompression_guide.md
# Deep Dive: Mathematical Paradigms of LLM Decompression
# Max Column: 80 Columns

## 1. Information Theoretic Foundations

Under Claude Shannon's source coding theorem (1948), an autoregressive language
model parameterizing conditional distributions $P_{\theta}(x_t \mid x_{<t})$
defines an optimal code length for any sequence:

$$
L(x_{1..T}) = \sum_{t=1}^{T} -\log_2 P_{\theta}(x_t \mid x_{<t})
\quad \text{bits}
$$

Minimizing cross-entropy loss during pre-training is mathematically identical to
minimizing the expected description length of the training corpus. When viewing
the resulting neural network as a compressed representation, two distinct
decompression paradigms emerge.

---

## 2. Paradigm 1: Lossy Static Weight Decompression (Carlini et al., 2021)

In Paradigm 1, the neural network weights $\theta$ themselves are treated as the
compressed archive. No external bitstream is provided. Decompressing the archive
means systematically extracting sequences memorized during pre-training.

### Two-Stage Basin Probing
Greedy decoding from `BOS` collapses into generic low-entropy loops. To sample
diverse parameter basins, `unplow` applies temperature-annealed exploration:
1. **Exploration Steps ($1 \le t \le k$):** High temperature ($\tau \approx
   1.15$) and top-$k$ filtering jump into diverse semantic subspaces.
2. **Lock-In Steps ($t > k$):** Low temperature ($\tau \approx 0.25$) forces the
   trajectory onto the highest-probability path in that basin.

### Carlini Memorization Scoring
A generated sequence $x$ is genuinely memorized (rather than syntactically
trivial repetition) if the LLM assigns it low cross-entropy $\mathcal{L}_\theta$
while a model-free syntactic compressor (`zlib`) finds it incompressible:

$$\mathcal{M}(x) = \frac{H_{\text{zlib}}(x)}{\mathcal{L}_{\theta}(x)}$$

Additionally, casing sensitivity $\mathcal{L}_\theta(\text{lower}(x)) /
\mathcal{L}_\theta(x)$ detects exact memorization of code symbols and licenses.

### Direct Mechanistic Inspection via Logit Lens (Geva et al., 2021)
Each column $\mathbf{v}_i^{(l)} = W_{\text{down}}^{(l)}[:, i]$ of an MLP down-
projection matrix acts as a value vector written directly into the residual
stream. Columns with the largest Euclidean norm $\Vert\mathbf{v}_i^{(l)}\Vert_2$
in upper-middle layers ($l \in [16, 26]$) encode localized key-value memories.
By projecting these value vectors through the final normalization and
unembedding matrix:

$$
\mathbf{z}_i^{(l)} = W_U \cdot
\text{RMSNorm}_{\text{final}}\left(\mathbf{v}_i^{(l)}\right)
$$

`unplow` extracts high-salience vocabulary clusters directly from static weights
without running forward attention passes, feeding them as high-yield seeds.

---

## 3. Paradigm 2: Lossless Arithmetic Codec (Delétang et al., 2023)

In Paradigm 2, the LLM provides dynamic conditional probability intervals for a
32-bit fixed-point Arithmetic Coder.

### Fixed-Point Integer Interval Partitioning
- Precision: 32 bits ($[0, 2^{32}-1]$).
- Total Frequency Mass: $M = 2^{24} = 16,777,216$.
- Quantized CDF: mapped so every token in vocabulary $\mathcal{V}$ receives
  frequency $C[v+1] - C[v] \ge 1$.
- At step $t$, with interval $[L, H]$ and span $S = H - L + 1$, the coder
  narrows the range to the sub-interval of the observed token and renormalizes.

### Bounded-Memory Sliding KV-Cache Window
For sequences exceeding context horizon $W$, `slice_kv_cache` truncates the
KV-cache to retain only the most recent $W/2$ tokens. Executed identically in
encoder and decoder, integer CDF trajectories remain bit-identical while capping
RAM to a deterministic constant.

### Self-Verifying `.unp` Binary Container
Framed with an 18-byte header (`UNPLOW`, version, flags, token count, window
stride, SHA-256 digest prefix), enabling self-contained archive distribution
with automatic cryptographic verification upon decompression.

### Domain-Adaptive Compression (Minimum Description Length)
Specialized code models ($\theta_{\text{coder}}$) assign lower cross-entropy to
AST syntax and indentation transitions. By matching the model prior to the
payload domain, arithmetic interval narrowing achieves higher compression
ratios ($<1.4$ bpb) without dictionary warmup penalties.

### Zero-Bit Domain Prior Pre-Warming (Phase 3)
In Phase 3, `.unp` binary archives utilize bits 4..7 of the 1-byte `flags` field
to encode a 4-bit Domain Prior ID (0..15) with zero bitstream overhead:
- **0**: Raw (unconditioned BOS fallback)
- **1**: `prose_tech` (technical specification on sequence modeling)
- **2**: `python_code` (Python 3 module header and standard imports)
- **3**: `c_firmware` (C/C++ firmware headers and low-level routines)
- **4**: `json_telemetry` (JSON telemetry schema preamble)

Both compressor and decompressor pre-warm the LLM KV-cache with the
deterministic domain preamble before token 0. The MDL auto-router
(`--auto-domain`) evaluates target Shannon cross-entropy across all
registered priors in a single batched pass and packs the optimal domain index
into the header.

### Mechanistic Weight Decompression via SwiGLU Circuits (Phase 3)
`unplow circuits` harvests multi-token associative rules directly from static
SwiGLU matrices:
1. Mean-centers `down_proj` columns:
   $\tilde{\mathbf{v}} = \mathbf{v} - \bar{\mathbf{v}}$ to suppress massive
   late-layer directional bias.
2. Filters by selectivity margin $z_{(1)} - z_{(50)}$ to remove universal
   numeral amplifiers.
3. Unembeds combined SwiGLU receptor keys
   ($\mathbf{w}_{\text{gate}} + \mathbf{w}_{\text{up}}$) against token
   embeddings $W_E$, discovering multi-token trigger phrases.

<!-- end of file: unplow/docs/decompression_guide.md -->
