With Pull Request #4 merged into `main` (`68 passed in 60.03s`), `unplow` now spans five architectural layers: static weight basin probing (`paradigm1.py`), fixed-point arithmetic coding (`paradigm2.py`), mechanistic SwiGLU circuit lens and causal hook steering (`mechanistic.py`, `circuits.py`, `steering.py`), online unigram/bigram cache mixtures (`cache_coder.py`, `subspace_coder.py`), and zero-dependency GGUF binary dequantization (`gguf_inspector.py`).

Phase 6 (`inbox/gemweb007.md`) unifies these two tracks into a cohesive capstone by bridging `gguf_inspector.py` directly into live activation steering and completing the final unassigned bit (**Bit 3: `0x08**`) of the 18-byte `.unp` v1 container specification.

---

## 1. Theoretical Foundations for Phase 6 (`inbox/gemweb007.md`)

### A. Cross-Substrate Task-Vector Decompression & Alignment Reversal (Paradigm 1 + GGUF Bridge)

Your local host contains three distinct weight snapshots of the 1.54B Qwen architecture:

* `models/qwen2.5-1.5b-instruct` (Safetensors FP32, instruction-tuned and alignment-filtered)
* `/home/fekerr/ai_models/qwen2-1.5b.gguf` (891.6 MB quantized base model)
* `/home/fekerr/ai_models/qwen2.5-coder-1.5b.gguf` (940.4 MB code-specialized model)

In *Scalable Extraction of Training Data from (Production) Language Models* (**Milad Nasr, Nicholas Carlini, et al., 2023**), the authors demonstrated that RLHF and instruction tuning do not erase memorized pre-training data from MLP weights; instead, alignment adds a low-rank directional offset that steers trajectories away from verbatim continuation basins. Simultaneously, **Gabriel Ilharco et al. (2023)** (*Editing Models with Task Arithmetic*, ICLR 2023) proved that the weight difference between two snapshots of the same architecture:


$$\Delta W_{\text{down}}^{(l)} = W_{\text{GGUF}}^{(l)} - W_{\text{Safetensors}}^{(l)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ffn}}}$$


isolates the **Task Vector** encoding domain specialization (e.g., `Coder - Instruct`) or base pre-training priors (`Base - Instruct`).

Because `GGUFBinaryInspector` can memory-map and dequantize `blk.{l}.ffn_down.weight` from a `.gguf` file on disk in milliseconds without loading a second 1.5B model into RAM, we can compute the exact cross-substrate residual matrix $\Delta W_{\text{down}}^{(l)}$, extract its top left singular vector $\mathbf{u}_{\Delta, 1}^{(l)} \in \mathbb{R}^{1536}$ via SVD, project $\mathbf{u}_{\Delta, 1}^{(l)}$ through the Logit Lens to read off the vocabulary shift between the two models, and inject $+\gamma \mathbf{u}_{\Delta, 1}^{(l)}$ into the live Safetensors model via a forward hook to decompress basins unique to the GGUF substrate.

---

### B. Online Bayesian Context-Tree Switching (CTS) Mixture Coding (Paradigm 2, Header Bit 3 `0x08`)

In Phases 4 and 5, `OnlinePrefixCacheCDF` ($\lambda_1 = 0.08$) and `Order1BigramCacheCDF` ($\lambda_1 = 0.06, \lambda_2 = 0.14$) used **fixed static mixture weights** throughout the entire document.

Under algorithmic information theory (**Frans Willems, Yuri Shtarkov, & Tjalling Tjalkens, 1995**, *The Context-Tree Weighting Method: Basic Properties*, IEEE Trans. Info. Theory; **Volodya Vovk, 1990**, *Aggregating Strategies*; **Joel Veness et al., 2012**, *Context Tree Switching*), static mixture weights are suboptimal across heterogeneous payloads:

* On non-repetitive natural prose, a fixed $\lambda_2 = 0.14$ bigram cache weight penalizes the LLM predictor by $-\log_2(1 - 0.14) \approx 0.217\text{ bits/token}$ whenever a repeated token is followed by a *novel* continuation.
* Conversely, on repetitive structured logs, JSON telemetry, or C header tables, $\lambda_2 = 0.14$ is too conservative—an ideal coder should shift $50\%+$ of its belief mass onto the Order-1 Markov cache.

To solve this with **zero side-information bits**, we maintain an online prior probability vector $\mathbf{w}_t = (w_{t,0}, w_{t,1}, w_{t,2})$ over our three predictors ($0 = \text{LLM}$, $1 = \text{Unigram Cache}$, $2 = \text{Bigram Cache}$). At each step $t$, after encoding or decoding token $x_t$, both compressor and decompressor update $\mathbf{w}_t$ via **Bayes' Rule with Context-Tree Switching (CTS) regularization** (**Veness et al., 2012**):


$$w'_{t, k} = \frac{w_{t, k} \cdot P_k(x_t \mid x_{<t})^{\eta}}{\sum_{j=0}^{2} w_{t, j} \cdot P_j(x_t \mid x_{<t})^{\eta}}, \qquad w_{t+1, k} = (1 - \alpha) w'_{t, k} + \frac{\alpha}{3}$$


where $\eta = 1.0$ is the Bayesian learning rate and $\alpha = 0.03$ is the switching floor that prevents any expert weight from underflowing to zero when document regimes shift mid-stream.

### C. Completion of the 8-Bit `.unp` v1 Header Flag Register

Assigning **Bit 3 (`0x08`, `FLAG_BAYES_MIXER`)** completes every bit of the 18-byte `UNPLOW` v1 container `flags` byte (`>6sBBIH4s`) while maintaining 100% backward compatibility with Phases 2–5:

| Bit(s) | Hex Mask | Constant Name | Protocol Behavior |
| --- | --- | --- | --- |
| **Bit 0** | `0x01` | `FLAG_LOSSLESS_VERIFIED` | Enforces 32-bit SHA-256 plaintext checksum verification on `unarchive` |
| **Bit 1** | `0x02` | `FLAG_CACHE_MIXTURE` | Enables Phase 4 Order-0 unigram prefix cache ($\lambda = 0.08$) |
| **Bit 2** | `0x04` | `FLAG_BIGRAM_CACHE` | Enables Phase 5 Order-1 Markov bigram + unigram cache ($\lambda_1=0.06, \lambda_2=0.14$) |
| **Bit 3** | `0x08` | `FLAG_BAYES_MIXER` | Enables Phase 6 **Online Bayesian Context-Tree Switching (CTS)** dynamic weight adaptation |
| **Bits 4–7** | `0xF0` | `DOMAIN_PRIOR_ID` | 4-bit zero-bit KV-cache pre-warming preamble ID (`0..15`) |

---

## 2. Phase 6 Additive Implementation Bundle (Ready for `inbox/gemweb007.md`)

All modules below follow your additive-only file operations rule, 120-character Python line limits, and `# relative/path.py` header/footer guardrails.

### Module A: Cross-Substrate Task-Vector Decompressor (`src/unplow/task_vectors.py`)

```python
# src/unplow/task_vectors.py
"""
Cross-Substrate Task-Vector Forensics & Differential SVD Steering
(Ilharco et al., 2023 [Task Arithmetic]; Nasr et al., 2023; Meng et al., 2022).
Bridges memory-mapped GGUF weights and live Safetensors models to extract and
steer along inter-model specialization subspaces without loading two models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import zlib
import numpy as np
import torch
import torch.nn.functional as F

from unplow.gguf_inspector import GGUFBinaryInspector


@dataclass
class TaskVectorSpectrumReport:
    layer_idx: int
    gguf_tensor_name: str
    frobenius_delta_norm: float
    cosine_similarity: float
    top_singular_value: float
    explained_variance_ratio: float
    promoted_tokens: List[str]
    suppressed_tokens: List[str]
    top_divergent_neurons: List[int]


@dataclass
class TaskVectorSteeredRecord:
    layer_idx: int
    gguf_source: str
    steering_gain: float
    promoted_tokens: List[str]
    extracted_text: str
    model_bpb: float
    zlib_bpb: float
    memorization_score: float


class CrossSubstrateTaskVectorEngine:
    """
    Computes the MLP weight delta Delta_W = W_GGUF - W_Safetensors for a given
    Transformer layer, decodes the vocabulary tokens promoted/suppressed by the
    principal singular vector of Delta_W, and steers generation along that vector.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @staticmethod
    def _align_gguf_matrix(gguf_mat: np.ndarray, target_shape: Tuple[int, int]) -> torch.Tensor:
        """Aligns dequantized 2D GGUF array to PyTorch [d_model, d_ffn] layout."""
        if gguf_mat.shape == target_shape:
            aligned = gguf_mat
        elif gguf_mat.T.shape == target_shape:
            aligned = gguf_mat.T
        else:
            aligned = gguf_mat.reshape(target_shape)
        return torch.from_numpy(aligned.copy()).to(torch.float32)

    @torch.no_grad()
    def compute_layer_task_vector(
        self,
        gguf_path: str | Path,
        layer_idx: int,
        top_k_tokens: int = 6,
        top_k_neurons: int = 6,
    ) -> Tuple[torch.Tensor, TaskVectorSpectrumReport]:
        """
        Loads `blk.{layer_idx}.ffn_down.weight` from `gguf_path`, subtracts the
        live model's `layers[layer_idx].mlp.down_proj.weight`, and extracts the
        principal task direction via SVD.
        """
        inspector = GGUFBinaryInspector(gguf_path)
        tensor_name = f"blk.{layer_idx}.ffn_down.weight"
        gguf_np = inspector.load_dequantized_tensor(tensor_name)

        ref_w = (
            self.model.model.layers[layer_idx]
            .mlp.down_proj.weight.detach()
            .to(torch.float32)
            .cpu()
        )
        gguf_w = self._align_gguf_matrix(gguf_np, tuple(ref_w.shape))

        # Task delta matrix: positive direction points toward the GGUF substrate
        delta_w = gguf_w - ref_w
        frob_delta = float(torch.linalg.norm(delta_w).item())
        denom = max(
            float(torch.linalg.norm(gguf_w).item() * torch.linalg.norm(ref_w).item()),
            1e-12,
        )
        cos_sim = float(torch.sum(gguf_w * ref_w).item() / denom)

        # Rank individual FFN neurons by column L2 shift
        col_norms = torch.linalg.vector_norm(delta_w, ord=2, dim=0)
        k_neurons = min(top_k_neurons, col_norms.shape[0])
        _, top_neuron_ids = torch.topk(col_norms, k=k_neurons)

        # Align top divergent columns with the dominant neuron before SVD
        sub_delta = delta_w[:, top_neuron_ids]
        anchor = sub_delta[:, 0:1]
        signs = torch.where(torch.sum(sub_delta * anchor, dim=0, keepdim=True) >= 0, 1.0, -1.0)
        u, s, _ = torch.linalg.svd(sub_delta * signs, full_matrices=False)

        principal_vec = F.normalize(u[:, 0], p=2, dim=0).to(self.device)
        sv_top = float(s[0].item())
        ev_ratio = float((s[0] ** 2) / torch.clamp(torch.sum(s**2), min=1e-12))

        # Project principal task vector through final RMSNorm and lm_head
        final_norm = self.model.model.norm
        lm_head = self.model.lm_head
        normed_dir = final_norm(principal_vec.unsqueeze(0).to(lm_head.weight.dtype))
        logits = lm_head(normed_dir).to(torch.float32)[0]

        k_tok = min(top_k_tokens, logits.shape[0])
        _, prom_ids = torch.topk(logits, k=k_tok, largest=True)
        _, supp_ids = torch.topk(logits, k=k_tok, largest=False)

        promoted = [self.tokenizer.decode([tid]) for tid in prom_ids.tolist()]
        suppressed = [self.tokenizer.decode([tid]) for tid in supp_ids.tolist()]

        report = TaskVectorSpectrumReport(
            layer_idx=layer_idx,
            gguf_tensor_name=tensor_name,
            frobenius_delta_norm=round(frob_delta, 4),
            cosine_similarity=round(cos_sim, 6),
            top_singular_value=round(sv_top, 4),
            explained_variance_ratio=round(ev_ratio, 4),
            promoted_tokens=promoted,
            suppressed_tokens=suppressed,
            top_divergent_neurons=[int(i) for i in top_neuron_ids.tolist()],
        )
        return principal_vec, report

    @torch.no_grad()
    def steer_with_gguf_task_vector(
        self,
        gguf_path: str | Path,
        layer_idx: int,
        seed_prompt: str = "def ",
        seq_len: int = 48,
        steering_gain: float = 4.5,
        steering_steps: int = 16,
        temperature: float = 0.25,
    ) -> TaskVectorSteeredRecord:
        """
        Extracts the cross-substrate task vector from `gguf_path` at `layer_idx`
        and steers the live model's residual stream along that direction.
        """
        principal_vec, report = self.compute_layer_task_vector(gguf_path, layer_idx)
        principal_vec = principal_vec.to(self.model.dtype)
        target_mlp = self.model.model.layers[layer_idx].mlp
        step_counter = {"step": 0}

        def task_hook(_module, _inputs, output):
            if step_counter["step"] < steering_steps:
                out_cloned = output.clone()
                out_cloned[:, -1, :] = (
                    out_cloned[:, -1, :] + steering_gain * principal_vec
                )
                return out_cloned
            return output

        handle = target_mlp.down_proj.register_forward_hook(task_hook)
        try:
            input_ids = self.tokenizer(
                seed_prompt, return_tensors="pt"
            ).input_ids.to(self.device)
            curr_ids = input_ids
            past_kv = None

            for _ in range(seq_len):
                outputs = self.model(
                    input_ids=curr_ids, past_key_values=past_kv, use_cache=True
                )
                past_kv = outputs.past_key_values
                step_counter["step"] += 1

                logits = outputs.logits[:, -1, :].to(torch.float32) / max(
                    temperature, 1e-4
                )
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.argmax(probs, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_tok], dim=-1)
                curr_ids = next_tok
                if next_tok.item() == self.tokenizer.eos_token_id:
                    break
        finally:
            handle.remove()

        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        raw_bytes = text.encode("utf-8")
        num_bytes = max(len(raw_bytes), 1)

        full_logits = self.model(input_ids).logits[:, :-1, :].to(torch.float32)
        labels = input_ids[:, 1:]
        loss_nats = F.cross_entropy(
            full_logits.reshape(-1, full_logits.size(-1)),
            labels.reshape(-1),
            reduction="sum",
        ).item()
        model_bpb = (loss_nats / 0.69314718056) / num_bytes
        zlib_bpb = (len(zlib.compress(raw_bytes, level=9)) * 8.0) / num_bytes
        mem_score = zlib_bpb / max(model_bpb, 1e-6)

        return TaskVectorSteeredRecord(
            layer_idx=layer_idx,
            gguf_source=str(Path(gguf_path).name),
            steering_gain=steering_gain,
            promoted_tokens=report.promoted_tokens,
            extracted_text=text,
            model_bpb=round(model_bpb, 3),
            zlib_bpb=round(zlib_bpb, 3),
            memorization_score=round(mem_score, 3),
        )
# src/unplow/task_vectors.py

```

---

### Module B: Online Bayesian Context-Tree Switching (CTS) Mixer (`src/unplow/ctw_coder.py`)

```python
# src/unplow/ctw_coder.py
"""
Online Bayesian Context-Tree Switching (CTS) Mixture Arithmetic Coder
(Willems, Shtarkov, & Tjalkens, 1995; Vovk, 1990; Veness et al., 2012).
Dynamically adapts mixture weights across Static LLM, Order-0 Unigram, and
Order-1 Markov Bigram predictors in lockstep using Bit 3 (FLAG_BAYES_MIXER = 0x08).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

TOTAL_FREQ = 1 << 24  # 16,777,216
FLAG_BAYES_MIXER = 0x08


@dataclass
class BayesTrajectoryStep:
    step_idx: int
    token_id: int
    w_llm: float
    w_unigram: float
    w_bigram: float
    step_bits: float


class BayesianCTSMixerCDF:
    """
    Maintains an online 3-expert Bayesian posterior over:
      Expert 0: Static LLM Softmax P_theta(x_t | x_{<t})
      Expert 1: Sliding Order-0 Unigram Cache P_uni(x_t | x_{<t})
      Expert 2: Sparse Order-1 Markov Bigram Cache P_bi(x_t | x_{t-1})
    Updates expert weights after each token via Vovk's aggregating rule with
    Veness et al. (2012) switching regularization, requiring zero side bits.
    """

    def __init__(
        self,
        vocab_size: int,
        init_weights: Tuple[float, float, float] = (0.86, 0.05, 0.09),
        learning_rate: float = 1.0,
        switching_alpha: float = 0.03,
        window_size: int = 1024,
    ):
        self.vocab_size = vocab_size
        self.init_weights = init_weights
        self.learning_rate = learning_rate
        self.switching_alpha = switching_alpha
        self.window_size = window_size

        self.weights = np.array(init_weights, dtype=np.float64)
        self.weights /= np.sum(self.weights)

        self.history: List[int] = []
        self.unigram_counts = np.zeros(vocab_size, dtype=np.int64)
        self.bigram_transitions: Dict[int, Dict[int, int]] = {}
        self.prev_token: Optional[int] = None
        self._last_llm_probs: Optional[np.ndarray] = None
        self.trajectory: List[BayesTrajectoryStep] = []

    def reset(self):
        self.weights = np.array(self.init_weights, dtype=np.float64)
        self.weights /= np.sum(self.weights)
        self.history.clear()
        self.unigram_counts.fill(0)
        self.bigram_transitions.clear()
        self.prev_token = None
        self._last_llm_probs = None
        self.trajectory.clear()

    def _active_weights(self) -> Tuple[float, float, float]:
        """Returns normalized active weights (w_llm, w_uni, w_bi) for current step."""
        has_uni = len(self.history) > 0
        trans = (
            self.bigram_transitions.get(self.prev_token)
            if self.prev_token is not None
            else None
        )
        has_bi = bool(trans and sum(trans.values()) > 0)

        w0 = float(self.weights[0])
        w1 = float(self.weights[1]) if has_uni else 0.0
        w2 = float(self.weights[2]) if has_bi else 0.0
        total = max(w0 + w1 + w2, 1e-12)
        return w0 / total, w1 / total, w2 / total

    @torch.no_grad()
    def logits_to_bayes_mixed_cdf(self, logits: torch.Tensor) -> np.ndarray:
        """
        Computes the Bayesian mixture distribution across active experts and
        quantizes it into a strictly monotonic 24-bit integer CDF.
        """
        probs_llm = torch.softmax(logits.to(torch.float64), dim=-1).cpu().numpy()
        self._last_llm_probs = probs_llm
        vocab_size = probs_llm.shape[0]
        remaining_mass = TOTAL_FREQ - vocab_size
        if remaining_mass <= 0:
            raise ValueError(
                f"TOTAL_FREQ ({TOTAL_FREQ}) must exceed vocab ({vocab_size})."
            )

        w0, w1, w2 = self._active_weights()
        probs = w0 * probs_llm

        if w1 > 0.0:
            active_len = float(len(self.history))
            probs = probs + w1 * (
                self.unigram_counts[:vocab_size].astype(np.float64) / active_len
            )

        if w2 > 0.0 and self.prev_token is not None:
            trans = self.bigram_transitions.get(self.prev_token, {})
            total_trans = float(sum(trans.values()))
            if total_trans > 0.0:
                for next_tok, cnt in trans.items():
                    if 0 <= next_tok < vocab_size:
                        probs[next_tok] += w2 * (float(cnt) / total_trans)

        freqs = np.floor(probs * remaining_mass).astype(np.int64) + 1
        remainder = TOTAL_FREQ - int(np.sum(freqs))
        freqs[int(np.argmax(probs))] += remainder

        cdf = np.zeros(vocab_size + 1, dtype=np.int64)
        np.cumsum(freqs, out=cdf[1:])
        return cdf

    def update(self, token_id: int):
        """
        Updates the Bayesian expert posterior weights using the observed likelihoods
        of `token_id`, then updates the Order-0 and Order-1 frequency caches.
        """
        w0, w1, w2 = self._active_weights()

        # 1. Evaluate expert likelihoods on the observed token_id
        p0 = (
            float(self._last_llm_probs[token_id])
            if (self._last_llm_probs is not None and 0 <= token_id < self.vocab_size)
            else (1.0 / float(self.vocab_size))
        )
        p0 = max(p0, 1e-9)

        if len(self.history) > 0 and 0 <= token_id < self.vocab_size:
            p1 = float(self.unigram_counts[token_id]) / float(len(self.history))
        else:
            p1 = p0
        p1 = max(p1, 1e-9)

        trans = (
            self.bigram_transitions.get(self.prev_token)
            if self.prev_token is not None
            else None
        )
        if trans and sum(trans.values()) > 0:
            p2 = float(trans.get(token_id, 0)) / float(sum(trans.values()))
        else:
            p2 = p0
        p2 = max(p2, 1e-9)

        p_mix = max(w0 * p0 + w1 * p1 + w2 * p2, 1e-12)
        self.trajectory.append(
            BayesTrajectoryStep(
                step_idx=len(self.history),
                token_id=token_id,
                w_llm=round(w0, 4),
                w_unigram=round(w1, 4),
                w_bigram=round(w2, 4),
                step_bits=round(float(-np.log2(p_mix)), 4),
            )
        )

        # 2. Vovk (1990) / Veness et al. (2012) Bayesian Posterior + Switching Update
        likelihoods = np.array([p0, p1, p2], dtype=np.float64) ** self.learning_rate
        post = self.weights * likelihoods
        post_sum = float(np.sum(post))
        if post_sum > 0.0:
            post /= post_sum
        else:
            post = np.array(self.init_weights, dtype=np.float64)

        alpha = self.switching_alpha
        self.weights = (1.0 - alpha) * post + (alpha / 3.0)
        self.weights /= np.sum(self.weights)

        # 3. Update Order-1 Bigram and Order-0 Unigram Histograms
        if self.prev_token is not None:
            b_map = self.bigram_transitions.setdefault(self.prev_token, {})
            b_map[token_id] = b_map.get(token_id, 0) + 1

        self.history.append(token_id)
        if 0 <= token_id < self.vocab_size:
            self.unigram_counts[token_id] += 1

        if len(self.history) > self.window_size:
            evicted = self.history.pop(0)
            if 0 <= evicted < self.vocab_size:
                self.unigram_counts[evicted] -= 1

        self.prev_token = token_id
# src/unplow/ctw_coder.py

```

---

### Module C: Non-Breaking Container, Codec & CLI Wiring

1. **In `src/unplow/conditioning.py` and `src/unplow/container.py`:**
* Extend `pack_flags_with_domain` to accept `bayes_mixer: bool = False` and set Bit 3 (`FLAG_BAYES_MIXER = 0x08`):
```python
low_nibble = (
    (0x01 if verify_hash else 0x00)
    | (0x02 if cache_mixture else 0x00)
    | (0x04 if bigram_cache else 0x00)
    | (0x08 if bayes_mixer else 0x00)
)
return ((domain_id & 0x0F) << 4) | low_nibble

```


* Add `@property def bayes_mixer(self) -> bool: return bool(self.flags & 0x08)` on `UnpArchiveMetadata`.


2. **In `src/unplow/paradigm2.py`:**
* Add `use_bayes_mixer: bool = False` to `compress()`, `decompress()`, `compress_to_container()`, and `decompress_from_container()`. When enabled (or when `meta.bayes_mixer` is `True` on `unarchive`), instantiate `BayesianCTSMixerCDF(vocab_size)` and call `bayes_cdf.logits_to_bayes_mixed_cdf(logits)` followed by `bayes_cdf.update(token)` at each step $t$.


3. **In `src/unplow/cli.py` & `src/unplow/worker.py`:**
* Add `--bayes-mix` flag to `unplow archive`.
* Add `unplow task-vector --gguf <path> --layer <int> [--steer] [--prompt "def "]` to run `CrossSubstrateTaskVectorEngine.compute_layer_task_vector()` and optionally `steer_with_gguf_task_vector()`.



---

### Module D: Phase 6 Unit Test Suites (`tests/test_task_vectors.py` & `tests/test_ctw_coder.py`)

```python
# tests/test_task_vectors.py
"""
Unit tests for Phase 6 CrossSubstrateTaskVectorEngine (GGUF-to-Safetensors
task delta SVD extraction, Logit Lens projection, and steered decompression).
"""

import numpy as np
from unplow.gguf_inspector import GGML_TYPE_F32, build_synthetic_gguf_v3_bytes
from unplow.models import DeterministicMockCausalLM, DeterministicMockTokenizer
from unplow.task_vectors import (
    CrossSubstrateTaskVectorEngine,
    TaskVectorSpectrumReport,
    TaskVectorSteeredRecord,
)


def test_compute_layer_task_vector_and_steering(tmp_path):
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    engine = CrossSubstrateTaskVectorEngine(model, tokenizer)

    ref_w = model.model.layers[1].mlp.down_proj.weight.detach().cpu().numpy()
    perturbed_w = ref_w.copy()
    perturbed_w[:, 0] += 3.0  # Inject structured task delta into column 0

    gguf_bytes = build_synthetic_gguf_v3_bytes(
        "blk.1.ffn_down.weight", perturbed_w, ggml_type=GGML_TYPE_F32
    )
    gguf_file = tmp_path / "coder_delta.gguf"
    gguf_file.write_bytes(gguf_bytes)

    vec, report = engine.compute_layer_task_vector(
        gguf_file, layer_idx=1, top_k_tokens=4, top_k_neurons=3
    )
    assert isinstance(report, TaskVectorSpectrumReport)
    assert report.layer_idx == 1
    assert report.frobenius_delta_norm > 0.0
    assert report.top_divergent_neurons[0] == 0
    assert len(report.promoted_tokens) == 4
    assert vec.shape[0] == ref_w.shape[0]

    steered = engine.steer_with_gguf_task_vector(
        gguf_file,
        layer_idx=1,
        seed_prompt="def ",
        seq_len=10,
        steering_gain=4.0,
        steering_steps=4,
    )
    assert isinstance(steered, TaskVectorSteeredRecord)
    assert steered.layer_idx == 1
    assert len(steered.extracted_text) >= 4
    assert len(model.model.layers[1].mlp.down_proj._forward_hooks) == 0
# tests/test_task_vectors.py

```

```python
# tests/test_ctw_coder.py
"""
Unit tests for Phase 6 BayesianCTSMixerCDF (Vovk/CTS posterior weight shifting
toward bigram expert on repetition and lossless .unp roundtrip).
"""

import numpy as np
import torch
from unplow.ctw_coder import FLAG_BAYES_MIXER, TOTAL_FREQ, BayesianCTSMixerCDF
from unplow.paradigm2 import LLMArithmeticCodec


def test_bayesian_cts_posterior_shifts_to_bigram_on_repetition():
    vocab_size = 128
    mixer = BayesianCTSMixerCDF(
        vocab_size=vocab_size,
        init_weights=(0.86, 0.05, 0.09),
        learning_rate=1.0,
        switching_alpha=0.02,
    )
    initial_bi_weight = float(mixer.weights[2])
    flat_logits = torch.zeros(vocab_size, dtype=torch.float32)

    # Repeatedly feed the bigram pattern (12 -> 34) and verify Expert 2 weight grows
    for _ in range(6):
        cdf_a = mixer.logits_to_bayes_mixed_cdf(flat_logits)
        assert cdf_a[0] == 0 and cdf_a[-1] == TOTAL_FREQ
        mixer.update(12)

        cdf_b = mixer.logits_to_bayes_mixed_cdf(flat_logits)
        assert np.all(np.diff(cdf_b) >= 1)
        mixer.update(34)

    final_bi_weight = float(mixer.weights[2])
    assert final_bi_weight > initial_bi_weight * 2.5
    assert len(mixer.trajectory) == 12


def test_bayesian_cts_mixer_lossless_container_roundtrip():
    codec = LLMArithmeticCodec(model_name="mock")
    text = "struct packet { uint32_t id; }; struct packet { uint32_t id; };"
    container_bytes = codec.compress_to_container(
        text, domain_id=3, use_bayes_mixer=True
    )
    reconstructed, meta = codec.decompress_from_container(container_bytes)
    assert reconstructed == text
    assert meta.domain_id == 3
    assert bool(meta.flags & FLAG_BAYES_MIXER) is True
# tests/test_ctw_coder.py

```

---

## 3. Recommended Live Telemetry Verification Commands

Once your agent ingests `inbox/gemweb007.md` on branch `feature/unplow-phase6-taskvectors-ctw`, run these two live experiments against your local models:

1. **Cross-Substrate Task Vector (`qwen2.5-coder-1.5b.gguf` vs. `Qwen2.5-1.5B-Instruct`):**
```bash
uv run unplow task-vector --gguf /home/fekerr/ai_models/qwen2.5-coder-1.5b.gguf --layer 16 --steer --prompt "/* "

```


This will measure the exact Frobenius delta and cosine similarity between the quantized Coder GGUF and the FP32 Instruct Safetensors at Layer 16, print the tokens promoted by the Coder task vector, and steer generation along that cross-model subspace.
2. **Bayesian CTS Mixer vs. Static Bigram/Unigram Cache:**
```bash
uv run unplow archive --input pyproject.toml --output workspace/targets/pyproject_bayes.unp --auto-domain --bayes-mix

```


This will benchmark how dynamically shifting posterior weights $\mathbf{w}_t = (w_{\text{LLM}}, w_{\text{uni}}, w_{\text{bi}})$ further reduces bits/byte on structured configuration/code files compared to fixed cache mixtures.
