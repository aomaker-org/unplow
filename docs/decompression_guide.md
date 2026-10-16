# Path: unplow/docs/decompression_guide.md
# Deep Dive: Mathematical Paradigms of LLM Decompression
# Max Column: 80 Columns

## 1. Information Theoretic Foundations

Under Claude Shannon's source coding theorem (1948), an autoregressive language
model parameterizing conditional distributions $P_{\theta}(x_t \mid x_{<t})$
defines an optimal code length for any sequence:

$$L(x_{1..T}) = \sum_{t=1}^{T} -\log_2 P_{\theta}(x_t \mid x_{<t}) \quad \text{bits}$$

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

### Generative Manifold Decompression
Because arithmetic coding establishes a measure-preserving bijection between
bitstreams and model probabilities, passing an arbitrary byte payload (such as a
SHA-256 hash or random seed) into the arithmetic decoder deterministically
projects those bits into coherent text according to the model's manifold.

---

## 4. Engineering Nuances & Floating-Point Determinism

- **FP32 vs FP16/BF16:** Floating-point matmul reductions across parallel CUDA
  threads are non-associative. For exact arithmetic decoding roundtrip, `unplow`
  runs logits evaluation in FP32 or deterministic quantized integer kernels.
- **Sliding Context Window:** For sequences exceeding model context length $W$,
  KV-caches are slid identically in encoder and decoder to maintain exact state
  synchronization.

<!-- end of file: unplow/docs/decompression_guide.md -->
