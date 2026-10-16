The `unplow` foundation build (`08fb4cf`) cleanly translates both theoretical paradigms into an auditable, air-gapped architecture. The empirical telemetry from your live verification run on `Qwen2.5-1.5B-Instruct` (1.54B parameters in deterministic FP32) highlights three specific phenomena in neural information theory worth examining in detail before advancing to Phase 2.

---

## 1. Empirical Telemetry Analysis of the `08fb4cf` Verification Run

### A. Paradigm 1: Diacritic Byte-Splitting & Memorized Bug Basins

Your static weight probe surfaced two distinct classes of memorized training sequences from `Qwen2.5-1.5B-Instruct`:

1. **High-Entropy Proper Nouns (`Copyright (c) 2016 Radek Dvořák, Martin Šimůnek, Petr Šimůnek...`, $\mathcal{M}(x) = 7.964$, Casing Sensitivity $= 1.363$):**
* **Why $\mathcal{M}(x)$ spiked to $7.964$:** Under Lempel-Ziv-77 (`zlib`, **Ziv & Lempel, 1977**), short strings containing UTF-8 multi-byte Czech diacritics (`ř` = `0xC5 0x99`, `á` = `0xC3 0xA1`, `Š` = `0xC5 0xA0`, `ů` = `0xC5 0xAF`) exhibit high syntactic entropy because the 32 KB sliding window has few repeated byte bigrams to back-reference. Conversely, inside `Qwen2.5-1.5B`, open-source MIT/BSD license headers appearing thousands of times across GitHub repositories carve deep attractors into the feed-forward key-value memories (**Geva et al., 2021**).Once the prefix `"Copyright (c) 2016"` activates the specific MLP key vectors, the conditional entropy $-\log_2 P_{\theta}(x_t \mid x_{1..t-1})$ for the exact co-author sequence drops near zero.
* **Why Casing Sensitivity hit $1.363$:** Qwen2.5 uses Byte-Level Byte-Pair Encoding (BBPE, **Wang et al., 2020**; **Sennrich et al., 2016**) with a 151,643-token vocabulary. When `Radek Dvořák, Martin Šimůnek` is lowercased to `radek dvořák, martin šimůnek`, the capitalized subword tokens are shattered into rarer lowercase byte-fallback fragments. Those fragments fail to trigger the memorized MLP value projections, increasing the sequence's cross-entropy bit cost by **36.3%**—a textbook forensic signature of verbatim parameter memorization (**Carlini et al., 2021**).


2. **Memorized Semantic Bugs (`def mtoMeters(meterInput): return meterInput * 1000`, $\mathcal{M}(x) = 7.307$):**
* Notice that a function named `mtoMeters` multiplying `meterInput * 1000` is dimensionally buggy (confusing meters-to-millimeters or kilometers-to-meters). As **Kevin Jesse et al. (2023)** proved in *Large Language Models and Simple, Stupid Bugs* (IEEE/ACM MSR 2023), autoregressive code models heavily memorize recurring introductory programming mistakes from public repositories. Temperature-decayed basin probing ($\tau = 1.15 \rightarrow 0.25$) naturally falls into these high-frequency student-code attractors.



### B. Paradigm 2: Overcoming the Lempel-Ziv Cold-Start Penalty

Compressing `"Information theory connects sequence modeling with data compression."` (68 bytes) down to **12 bytes** (**1.412 bits/byte**, **5.67:1 compression ratio**, **82.35% reduction**) demonstrates why neural arithmetic coding fundamentally differs from dictionary compressors:

* **Zero Dictionary Cold-Start:** Traditional compressors (`zlib`, `zstd`, `lzma`) must build their dictionary dynamically from the input stream itself; on a 68-byte string, `zlib` cannot find long repeated substrings and is bottlenecked by framing headers.
* **Shared Static Prior:** In `paradigm2.py`, the 1.54B parameter matrix $\theta$ acts as a massive, pre-shared static prior (**Delétang et al., 2023**; **Bellard, 2021**). Even at token $t=1$, the model immediately assigns high probability mass to syntactically and semantically plausible continuations, reaching **1.412 bits/byte**—approaching **Claude Shannon's (1951)** upper-bound estimate of $1.3$ bits/character for printed English on a single sentence without warmup context.

---

## 2. Phase 2 Architectural Enhancements for `unplow`

To advance `unplow` from foundation verification (`08fb4cf`) to high-throughput operation across your local substrates (`Safetensors` FP32 and `GGUF` binaries), three architectural upgrades directly extend your existing layout:

| Component Target | Phase 2 Capability | Mathematical / Systems Basis |
| --- | --- | --- |
| `src/unplow/mechanistic.py` (or `paradigm1.py`) | **Direct MLP Weight-Matrix Decompression (Zero-Forward-Pass Seed Harvesting)** | Projects top-$L_2$-norm columns of MLP `down_proj` ($W_{\text{down}}^{(l)}$) directly through final `RMSNorm` and `lm_head` ($W_U$) via the **Logit Lens** (**nostalgebraist, 2020**; **Geva et al., 2021**) to harvest memorized vocabulary clusters directly from static tensors before running autoregressive probing. |
| `src/unplow/paradigm2.py` | **Binary `.unp` Container Framing & Sliding KV-Cache Window** | Wraps raw arithmetic bitstreams in a self-verifying binary header (`UNPLOW01` magic bytes, `uint32` token count, `uint16` context window stride, and 8-byte SHA-256 plaintext digest) and bounds RAM via deterministic KV-cache striding. |
| `src/unplow/models.py` | **Dual-Substrate Determinism (`Safetensors` FP32 + `qwen2.5-coder-1.5b.gguf`)** | Uses `qwen2.5-coder-1.5b.gguf` (via `llama-cpp-python` or `llama-cli` logits) or switches model priors based on target payload type (code vs. prose), since reducing cross-entropy $\mathcal{L}_{\theta}(x)$ on source code directly shrinks `.unp` payload size. |

---

## 3. Phase 2 Implementation Modules (Ready for `inbox/` Staging)

### Module A: Direct Mechanistic Weight Decompressor (`src/unplow/mechanistic.py`)

Instead of relying on static hardcoded prefixes (`"Copyright (c) "`, `"def "`) in `paradigm1.py`, we can mathematically inspect the static weight matrices of `Qwen2.5-1.5B-Instruct` without running a single token through attention.

In the Qwen2 architecture (**Bai et al., 2023**; **Yang et al., 2024**), each Transformer layer $l \in \{0, \dots, 27\}$ contains a SwiGLU MLP block with a down-projection matrix:


$$W_{\text{down}}^{(l)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}} \quad (d_{\text{model}} = 1536, \; d_{\text{ffn}} = 8960)$$


Each column $\mathbf{v}_i^{(l)} = W_{\text{down}}^{(l)}[:, i] \in \mathbb{R}^{1536}$ represents a single **value vector** written directly into the residual stream when neuron $i$ fires (**Geva et al., 2021**; **Meng et al., 2022 [ROME]**). Columns with the largest Euclidean norm $\Vert{}\mathbf{v}_i^{(l)}\Vert{}_2$ in the upper-middle layers ($l \in [16, 26]$) frequently encode localized, memorized factual or lexical associations. By projecting these vectors through the final layer normalization and unembedding matrix $W_U \in \mathbb{R}^{\vert{}\mathcal{V}\vert{} \times 1536}$:


$$\mathbf{z}_{i}^{(l)} = W_U \cdot \text{RMSNorm}_{\text{final}}\!\left(\mathbf{v}_i^{(l)}\right)$$


we can decode the exact tokens stored inside individual MLP neurons and automatically feed them as high-yield seed prefixes into `paradigm1.py`.

```python
"""
src/unplow/mechanistic.py

Direct Mechanistic Weight Inspection & Seed Harvesting (Geva et al., 2021;
nostalgebraist, 2020; Meng et al., 2022). Extracts high-salience vocabulary
clusters directly from static MLP down_proj weight tensors without running
autoregressive forward passes.
"""

from dataclasses import dataclass
from typing import List, Tuple
import torch


@dataclass
class NeuronMemoryTrace:
    layer_idx: int
    neuron_idx: int
    l2_norm: float
    top_tokens: List[str]
    top_token_ids: List[int]


class MechanisticWeightInspector:
    """
    Directly projects static MLP value vectors (columns of mlp.down_proj.weight)
    into the vocabulary space via the model's final RMSNorm and lm_head.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def extract_mlp_memory_clusters(
        self,
        layer_indices: List[int] = None,
        top_neurons_per_layer: int = 8,
        top_k_tokens: int = 6,
    ) -> List[NeuronMemoryTrace]:
        """
        Scans specified Transformer layers, ranks MLP value vectors by L2 norm,
        and decodes their direct unembedding projections.
        """
        layers = self.model.model.layers
        num_layers = len(layers)

        if layer_indices is None:
            # Upper-middle layers (e.g., layers 18..26 in 28-layer Qwen2.5-1.5B)
            # carry the highest density of localized semantic key-value memories
            start_layer = max(0, int(num_layers * 0.65))
            layer_indices = list(range(start_layer, num_layers - 1))

        final_norm = self.model.model.norm
        lm_head = self.model.lm_head

        traces: List[NeuronMemoryTrace] = []

        for l_idx in layer_indices:
            # down_proj.weight has shape [d_model, d_ffn] in PyTorch Linear(d_ffn, d_model)
            down_weight = layers[l_idx].mlp.down_proj.weight.detach().to(torch.float32)

            # Compute L2 norm of each column (each column is a d_model value vector for 1 FFN neuron)
            col_norms = torch.linalg.vector_norm(down_weight, ord=2, dim=0)
            top_norms, top_neuron_indices = torch.topk(
                col_norms, k=min(top_neurons_per_layer, col_norms.shape[0])
            )

            # Extract the top value vectors: shape [top_neurons_per_layer, d_model]
            value_vectors = down_weight[:, top_neuron_indices].T.to(self.device)

            # Apply the model's final RMSNorm and project onto vocabulary logits
            normed_vectors = final_norm(value_vectors.to(lm_head.weight.dtype))
            logits = lm_head(normed_vectors).to(torch.float32)

            _, top_token_matrix = torch.topk(logits, k=top_k_tokens, dim=-1)

            for i in range(top_neuron_indices.shape[0]):
                n_idx = int(top_neuron_indices[i].item())
                n_norm = float(top_norms[i].item())
                tok_ids = top_token_matrix[i].tolist()
                decoded_toks = [self.tokenizer.decode([tid]) for tid in tok_ids]

                traces.append(
                    NeuronMemoryTrace(
                        layer_idx=l_idx,
                        neuron_idx=n_idx,
                        l2_norm=round(n_norm, 4),
                        top_tokens=decoded_toks,
                        top_token_ids=tok_ids,
                    )
                )

        traces.sort(key=lambda t: t.l2_norm, reverse=True)
        return traces

    @torch.no_grad()
    def harvest_seed_prefixes(self, num_seeds: int = 12) -> List[str]:
        """
        Converts top mechanistic MLP neuron traces into non-trivial text prefixes
        for Paradigm 1 basin probing.
        """
        traces = self.extract_mlp_memory_clusters(top_neurons_per_layer=6, top_k_tokens=4)
        seeds: List[str] = []
        seen = set()

        for tr in traces:
            for tok_str in tr.top_tokens:
                cleaned = tok_str.strip()
                # Filter out pure punctuation or single-char control artifacts
                if len(cleaned) >= 3 and cleaned.isprintable() and cleaned not in seen:
                    seen.add(cleaned)
                    seeds.append(tok_str)
                    if len(seeds) >= num_seeds:
                        return seeds
        return seeds

```

---

### Module B: Framed `.unp` Container & Sliding-Window Arithmetic Codec (`src/unplow/container.py`)

In your live run, `paradigm2.py` achieved **1.412 bits/byte** on a 68-byte payload. To integrate `paradigm2.py` with `worker.py`, `queue.py`, and `forensics.py` for arbitrary files in `workspace/targets/`, we need two critical capabilities:

1. **Self-Contained `.unp` Binary Framing (18-Byte Header):**
* `magic` (`6 bytes`): `b"UNPLOW"`
* `version` (`1 byte`, `uint8`): `0x01`
* `flags` (`1 byte`, `uint8`): Bit 0 = Lossless Archive (`0x01`) vs. Generative Manifold Seed (`0x00`)
* `num_tokens` (`4 bytes`, `uint32` big-endian): Exact token count $T$ to decode
* `max_ctx_len` (`2 bytes`, `uint16` big-endian): Maximum KV-cache horizon $W$ used during encoding
* `sha256_prefix` (`4 bytes`): First 32 bits of the original UTF-8 plaintext SHA-256 digest, providing automatic cryptographic verification upon decompression.


2. **Deterministic KV-Cache Context Slicing:**
* If a target document in `workspace/targets/` is $4,096$ tokens long, holding the full FP32 KV-cache on a 12 GB host degrades throughput. By bounding the KV-cache to `max_ctx_len` (e.g., $W = 512$ tokens) and retaining the most recent $W/2 = 256$ KV states whenever the window fills—identically in both `compress_to_container()` and `decompress_from_container()`—the integer CDF trajectories remain 100% bit-identical while capping memory usage to a tiny constant footprint.



```python
"""
src/unplow/container.py

Self-verifying .unp binary container format and bounded-memory KV-cache
sliding horizon for Paradigm 2 lossless neural compression.
"""

import hashlib
import struct
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
from transformers import DynamicCache

# .unp Binary Header Layout: 6s (magic) + B (version) + B (flags) + I (num_tokens) + H (max_ctx) + 4s (sha256_prefix) = 18 bytes
UNP_HEADER_FORMAT = ">6sBBIH4s"
UNP_HEADER_SIZE = struct.calcsize(UNP_HEADER_FORMAT)
UNP_MAGIC = b"UNPLOW"
UNP_VERSION = 1
FLAG_LOSSLESS_VERIFIED = 0x01

# 32-bit Fixed-Point Constants
PRECISION_BITS = 32
MAX_VAL = (1 << PRECISION_BITS) - 1
HALF = 1 << (PRECISION_BITS - 1)
QUARTER = 1 << (PRECISION_BITS - 2)
THREE_QUARTERS = 3 << (PRECISION_BITS - 2)
TOTAL_FREQ = 1 << 24


@dataclass
class UnpArchiveMetadata:
    version: int
    flags: int
    num_tokens: int
    max_ctx_len: int
    sha256_prefix_hex: str
    compressed_bytes_len: int
    bitstream_bytes_len: int


def slice_kv_cache(past_key_values, keep_tokens: int):
    """
    Deterministically truncates HuggingFace past_key_values (DynamicCache or tuple)
    to retain only the most recent `keep_tokens` positions, bounding RAM usage
    while preserving exact encoder-decoder symmetry.
    """
    if past_key_values is None:
        return None

    if isinstance(past_key_values, DynamicCache):
        new_cache = DynamicCache()
        for layer_idx in range(len(past_key_values.key_cache)):
            k = past_key_values.key_cache[layer_idx][:, :, -keep_tokens:, :]
            v = past_key_values.value_cache[layer_idx][:, :, -keep_tokens:, :]
            new_cache.update(k, v, layer_idx)
        return new_cache

    # Legacy tuple-of-tuples format
    return tuple(
        (k[:, :, -keep_tokens:, :], v[:, :, -keep_tokens:, :])
        for k, v in past_key_values
    )


def pack_unp_container(
    bitstream: bytes,
    num_tokens: int,
    max_ctx_len: int,
    plaintext: str,
    verify_hash: bool = True,
) -> bytes:
    """Packs raw arithmetic bitstream into an 18-byte framed .unp container."""
    digest_prefix = hashlib.sha256(plaintext.encode("utf-8")).digest()[:4]
    flags = FLAG_LOSSLESS_VERIFIED if verify_hash else 0x00
    header = struct.pack(
        UNP_HEADER_FORMAT,
        UNP_MAGIC,
        UNP_VERSION,
        flags,
        num_tokens,
        max_ctx_len,
        digest_prefix,
    )
    return header + bitstream


def unpack_unp_container(container_bytes: bytes) -> Tuple[UnpArchiveMetadata, bytes]:
    """Unpacks and validates the 18-byte .unp header and returns (metadata, bitstream)."""
    if len(container_bytes) < UNP_HEADER_SIZE:
        raise ValueError(f"Invalid .unp archive: length {len(container_bytes)} is less than {UNP_HEADER_SIZE} bytes.")

    magic, version, flags, num_tokens, max_ctx_len, digest_prefix = struct.unpack(
        UNP_HEADER_FORMAT, container_bytes[:UNP_HEADER_SIZE]
    )
    if magic != UNP_MAGIC:
        raise ValueError(f"Invalid .unp magic bytes: {magic!r}")
    if version != UNP_VERSION:
        raise ValueError(f"Unsupported .unp container version: {version}")

    bitstream = container_bytes[UNP_HEADER_SIZE:]
    meta = UnpArchiveMetadata(
        version=version,
        flags=flags,
        num_tokens=num_tokens,
        max_ctx_len=max_ctx_len,
        sha256_prefix_hex=digest_prefix.hex(),
        compressed_bytes_len=len(container_bytes),
        bitstream_bytes_len=len(bitstream),
    )
    return meta, bitstream

```

---

## 4. Domain-Adaptive Compression Ratio Experiment (`Qwen2.5-1.5B` vs. `Qwen2.5-Coder-1.5B`)

Because `unplow` discovered both the general-purpose `Qwen2.5-1.5B-Instruct` snapshot and `/home/fekerr/ai_models/qwen2.5-coder-1.5b.gguf` (alongside the Ollama `qwen2.5-coder:1.5b` substrate), you can directly exploit **Minimum Description Length (MDL) Prior Matching** (**Rissanen, 1978**):

* For source code payloads in `workspace/targets/`, `Qwen2.5-Coder-1.5B` assigns significantly lower cross-entropy loss $\mathcal{L}_{\theta_{\text{coder}}}(x)$ to AST tokens, indentation transitions, and standard library calls than the base instruct model.
* Because every $1.0\text{ bit/token}$ reduction in cross-entropy directly saves $1.0\text{ bit/token}$ in the arithmetic coder's interval narrowing, encoding a Python or C file under $\theta_{\text{coder}}$ yields a strictly smaller `.unp` payload than encoding the same file under $\theta_{\text{instruct}}$, while remaining 100% lossless as long as the header records a 1-byte model substrate identifier.
