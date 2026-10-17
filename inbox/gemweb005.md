### 1. Triage of the `agy` `Read` Tool Schema Error (`missing property 'toolSummary'`)

The error thrown by your local agent harness on `Read(~/src/unplow/inbox/gemweb004.md)` stems from two strict validation rules in the `agy` tool-call JSON schema:

1. **Mandatory `toolSummary` Parameter:** The `agy` tool dispatcher enforces a JSON Schema where every tool invocation (`Read`, `Edit`, `Write`, `Bash`) requires `"toolSummary": "<brief description>"` in the arguments payload. When the local model emits `{"file_path": "~/src/unplow/inbox/gemweb004.md"}` without `"toolSummary"`, the schema validator aborts before touching the filesystem.
2. **Tilde (`~`) Expansion in Tool Paths:** Passing a literal `~` in `file_path` can bypass workspace-relative path canonicalization. Prompting the agent with the repo-relative path (`inbox/gemweb004.md`) or absolute path (`/home/fekerr/src/unplow/inbox/gemweb004.md`) and explicitly instructing it:
> *"Read `inbox/gemweb004.md` (include `toolSummary` in all tool calls) and implement Phase 4 on branch `feature/unplow-phase4-steering-cache`."*
> resolves the schema rejection immediately.



---

### 2. Complete Phase 4 Specification & Test Suite (`inbox/gemweb004.md`)

To make your `inbox/gemweb004.md` ingestion completely self-contained in a single turn, below is the full **Phase 4 Implementation Bundle**—incorporating the additive modules (`src/unplow/steering.py` and `src/unplow/cache_coder.py`), the non-breaking `.unp` v1 flag-bit `0x02` (`FLAG_CACHE_MIXTURE`) extensions, CLI/worker subcommands (`steer`, `profile`, `archive --cache-mix`), and the full unit test suite (`tests/test_steering.py` and `tests/test_cache_coder.py`). All files strictly follow the 120-character Python line limit, additive-only operations, and `# relative/path.py` header/footer guardrails.

#### A. Additive Module 1: `src/unplow/steering.py`

Implements **RMSNorm-Folded SwiGLU Circuit Projection** (**Dar et al., 2023**, *Analyzing Transformers in Embedding Space*; **Elhage et al., 2021**) to eliminate the non-ASCII rogue-cone tokens (`' viên'`, `' nhiều'`) seen in Phase 3, alongside **Causal Activation Steering** (**Meng et al., 2022 [ROME]**; **Turner et al., 2023**; **Zou et al., 2023**) via PyTorch `register_forward_hook` on `mlp.down_proj`.

```python
# src/unplow/steering.py
"""
RMSNorm-Folded SwiGLU Circuit Lens and Activation-Steered Weight Decompression
(Dar et al., 2023; Meng et al., 2022 [ROME]; Turner et al., 2023; Zou et al., 2023).
"""

from dataclasses import dataclass
from typing import List
import zlib
import torch
import torch.nn.functional as F


@dataclass
class FoldedCircuitTrace:
    layer_idx: int
    neuron_idx: int
    centered_l2_norm: float
    selectivity_margin: float
    folded_trigger_keys: List[str]
    target_values: List[str]
    coherence_score: float


@dataclass
class SteeredBasinRecord:
    layer_idx: int
    neuron_idx: int
    steering_gain: float
    extracted_text: str
    model_bpb: float
    zlib_bpb: float
    memorization_score: float


class MechanisticSteeringEngine:
    """
    1. Refines SwiGLU receptor key projection by folding post_attention_layernorm
       gains and mean-centering the input embedding basis (Dar et al., 2023).
    2. Performs causal activation steering via PyTorch forward hooks on MLP
       down_proj layers to decompress neuron-specific training data basins.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @staticmethod
    def _is_clean_ascii_identifier(token_str: str) -> bool:
        """Filters out non-ASCII rogue-cone subwords and trivial single chars."""
        s = token_str.strip()
        if len(s) < 2 or s.isdigit():
            return False
        if not s.isascii():
            return False
        return any(ch.isalpha() for ch in s)

    @torch.no_grad()
    def extract_rmsnorm_folded_circuits(
        self,
        layer_indices: List[int] = None,
        candidates_per_layer: int = 20,
        top_k_tokens: int = 5,
    ) -> List[FoldedCircuitTrace]:
        """
        Projects SwiGLU receptor keys through post_attention_layernorm.weight
        onto a mean-centered input embedding matrix to eliminate multilingual
        anisotropy artifacts.
        """
        layers = self.model.model.layers
        num_layers = len(layers)
        if layer_indices is None:
            start_l = max(0, int(num_layers * 0.50))
            end_l = max(start_l + 1, int(num_layers * 0.88))
            layer_indices = list(range(start_l, end_l))

        final_norm = self.model.model.norm
        lm_head = self.model.lm_head

        # Mean-center the input embedding matrix W_E (Dar et al., 2023)
        raw_embed = self.model.model.embed_tokens.weight.detach().to(torch.float32)
        centered_embed = raw_embed - raw_embed.mean(dim=0, keepdim=True)
        embed_normed = F.normalize(centered_embed, p=2, dim=-1).to(self.device)

        traces: List[FoldedCircuitTrace] = []

        for l_idx in layer_indices:
            layer = layers[l_idx]
            mlp = layer.mlp
            post_norm_gamma = (
                layer.post_attention_layernorm.weight.detach()
                .to(torch.float32)
                .to(self.device)
            )

            down_w = mlp.down_proj.weight.detach().to(torch.float32)
            centered_down = down_w - down_w.mean(dim=1, keepdim=True)
            col_norms = torch.linalg.vector_norm(centered_down, ord=2, dim=0)

            k_cand = min(candidates_per_layer, col_norms.shape[0])
            top_norms, top_indices = torch.topk(col_norms, k=k_cand)

            cand_values = centered_down[:, top_indices].T.to(self.device)
            normed_vals = final_norm(cand_values.to(lm_head.weight.dtype))
            val_logits = lm_head(normed_vals).to(torch.float32)

            vocab_len = val_logits.shape[-1]
            k_margin = min(50, vocab_len)
            top_m_vals, top_m_ids = torch.topk(val_logits, k=k_margin, dim=-1)
            margins = top_m_vals[:, 0] - top_m_vals[:, -1]

            # Fold post_attention_layernorm affine gain into SwiGLU receptor keys
            gate_vecs = (
                mlp.gate_proj.weight.detach()[top_indices, :]
                .to(torch.float32)
                .to(self.device)
            )
            up_vecs = (
                mlp.up_proj.weight.detach()[top_indices, :]
                .to(torch.float32)
                .to(self.device)
            )
            folded_receptors = (gate_vecs + up_vecs) * post_norm_gamma.unsqueeze(0)
            folded_keys = F.normalize(folded_receptors, p=2, dim=-1)

            # Cosine similarity between folded receptor keys and mean-centered W_E
            key_scores = torch.matmul(folded_keys, embed_normed.T)
            k_search = min(top_k_tokens * 4, vocab_len)
            _, top_key_ids = torch.topk(key_scores, k=k_search, dim=-1)

            for i in range(top_indices.shape[0]):
                n_idx = int(top_indices[i].item())
                c_norm = float(top_norms[i].item())
                margin = float(margins[i].item())

                v_ids = top_m_ids[i, :top_k_tokens].tolist()
                v_toks = [self.tokenizer.decode([tid]) for tid in v_ids]
                if not any(self._is_clean_ascii_identifier(t) for t in v_toks[:3]):
                    continue

                raw_k_ids = top_key_ids[i].tolist()
                raw_k_toks = [self.tokenizer.decode([tid]) for tid in raw_k_ids]
                clean_k_toks = [
                    t for t in raw_k_toks if self._is_clean_ascii_identifier(t)
                ][:top_k_tokens]
                if not clean_k_toks:
                    clean_k_toks = raw_k_toks[:top_k_tokens]

                coherence = round(c_norm * margin, 4)
                traces.append(
                    FoldedCircuitTrace(
                        layer_idx=l_idx,
                        neuron_idx=n_idx,
                        centered_l2_norm=round(c_norm, 4),
                        selectivity_margin=round(margin, 4),
                        folded_trigger_keys=clean_k_toks,
                        target_values=v_toks,
                        coherence_score=coherence,
                    )
                )

        traces.sort(key=lambda tr: tr.coherence_score, reverse=True)
        return traces

    @torch.no_grad()
    def decompress_neuron_basin(
        self,
        layer_idx: int,
        neuron_idx: int,
        seed_prompt: str = "/* ",
        seq_len: int = 48,
        steering_gain: float = 4.5,
        steering_steps: int = 16,
        temperature: float = 0.25,
    ) -> SteeredBasinRecord:
        """
        Clamps/steers the residual stream along `down_proj[:, neuron_idx]` at
        `layer_idx` during the first `steering_steps` forward passes, forcing
        the model to unpack the specific training basin linked to that neuron.
        """
        target_mlp = self.model.model.layers[layer_idx].mlp
        value_vec = (
            target_mlp.down_proj.weight.detach()[:, neuron_idx]
            .to(self.model.dtype)
            .to(self.device)
        )
        unit_value_vec = F.normalize(value_vec.to(torch.float32), p=2, dim=0).to(
            self.model.dtype
        )

        step_counter = {"step": 0}

        def steering_hook(_module, _inputs, output):
            if step_counter["step"] < steering_steps:
                out_cloned = output.clone()
                out_cloned[:, -1, :] = (
                    out_cloned[:, -1, :] + steering_gain * unit_value_vec
                )
                return out_cloned
            return output

        handle = target_mlp.down_proj.register_forward_hook(steering_hook)
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

        return SteeredBasinRecord(
            layer_idx=layer_idx,
            neuron_idx=neuron_idx,
            steering_gain=steering_gain,
            extracted_text=text,
            model_bpb=round(model_bpb, 3),
            zlib_bpb=round(zlib_bpb, 3),
            memorization_score=round(mem_score, 3),
        )
# src/unplow/steering.py

```

---

#### B. Additive Module 2: `src/unplow/cache_coder.py`

Implements **Online Prefix-Cache Mixture Arithmetic Coding** (**Kuhn & De Mori, 1990**; **Grave, Joulin, & Usunier, 2017**; **Willems, Shtarkov, & Tjalkens, 1995**) and the **Per-Token Shannon Surprisal Attribution Ledger**.

```python
# src/unplow/cache_coder.py
"""
Online Prefix-Cache Mixture CDF (Kuhn & De Mori, 1990; Grave et al., 2017)
and Per-Token Shannon Surprisal Attribution Ledger for Paradigm 2.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
import torch

TOTAL_FREQ = 1 << 24  # 16,777,216
FLAG_CACHE_MIXTURE = 0x02


@dataclass
class TokenSurprisalEntry:
    step_idx: int
    token_id: int
    token_str: str
    prob_llm: float
    prob_mixed: float
    ideal_shannon_bits: float
    cumulative_bits: float


@dataclass
class SurprisalProfileReport:
    total_tokens: int
    total_bytes: int
    ideal_total_bits: float
    ideal_bits_per_byte: float
    cache_bits_saved: float
    entries: List[TokenSurprisalEntry]


class OnlinePrefixCacheCDF:
    """
    Maintains a deterministic online unigram frequency cache over already
    encoded/decoded prefix tokens x_{1..t-1} in lockstep between encoder
    and decoder, boosting probabilities of locally repeated tokens inside
    the 24-bit integer CDF without adding side-information bits.
    """

    def __init__(
        self, vocab_size: int, cache_weight: float = 0.08, window_size: int = 1024
    ):
        self.vocab_size = vocab_size
        self.cache_weight = cache_weight
        self.window_size = window_size
        self.history: List[int] = []
        self.counts = np.zeros(vocab_size, dtype=np.int64)

    def reset(self):
        self.history.clear()
        self.counts.fill(0)

    def update(self, token_id: int):
        """Records `token_id` into the sliding prefix histogram after step t."""
        self.history.append(token_id)
        if 0 <= token_id < self.vocab_size:
            self.counts[token_id] += 1
        if len(self.history) > self.window_size:
            evicted = self.history.pop(0)
            if 0 <= evicted < self.vocab_size:
                self.counts[evicted] -= 1

    @torch.no_grad()
    def logits_to_mixed_integer_cdf(self, logits: torch.Tensor) -> np.ndarray:
        """
        Combines LLM softmax probabilities with the online prefix histogram
        to produce a strictly monotonic integer CDF with total mass 2^24.
        """
        probs_llm = torch.softmax(logits.to(torch.float64), dim=-1).cpu().numpy()
        vocab_size = probs_llm.shape[0]
        remaining_mass = TOTAL_FREQ - vocab_size
        if remaining_mass <= 0:
            raise ValueError(
                f"TOTAL_FREQ ({TOTAL_FREQ}) must exceed vocab ({vocab_size})."
            )

        active_len = len(self.history)
        if active_len > 0 and self.cache_weight > 0.0:
            probs_cache = self.counts[:vocab_size].astype(np.float64) / float(
                active_len
            )
            probs = (
                1.0 - self.cache_weight
            ) * probs_llm + self.cache_weight * probs_cache
        else:
            probs = probs_llm

        freqs = np.floor(probs * remaining_mass).astype(np.int64) + 1
        remainder = TOTAL_FREQ - int(np.sum(freqs))
        freqs[int(np.argmax(probs))] += remainder

        cdf = np.zeros(vocab_size + 1, dtype=np.int64)
        np.cumsum(freqs, out=cdf[1:])
        return cdf


@torch.no_grad()
def profile_sequence_surprisal(
    model,
    tokenizer,
    plaintext: str,
    bos_token_id: int,
    cache_weight: float = 0.08,
) -> SurprisalProfileReport:
    """
    Computes an exact per-token Shannon surprisal ledger (-log2 P(x_t | x_{<t}))
    comparing static LLM probabilities against the online cache mixture.
    """
    device = next(model.parameters()).device
    token_ids = tokenizer.encode(plaintext, add_special_tokens=False)
    num_bytes = max(len(plaintext.encode("utf-8")), 1)

    if not token_ids:
        return SurprisalProfileReport(0, num_bytes, 0.0, 0.0, 0.0, [])

    seq = [bos_token_id] + token_ids
    input_tensor = torch.tensor([seq[:-1]], device=device)
    logits_all = model(input_tensor).logits[0].to(torch.float64)
    probs_all = torch.softmax(logits_all, dim=-1).cpu().numpy()

    vocab_size = probs_all.shape[-1]
    counts = np.zeros(vocab_size, dtype=np.int64)

    entries: List[TokenSurprisalEntry] = []
    cum_bits = 0.0
    llm_only_bits = 0.0

    for step_idx, tok_id in enumerate(token_ids):
        p_llm = float(max(probs_all[step_idx, tok_id], 1e-12))
        if step_idx > 0 and cache_weight > 0.0:
            p_cache = float(counts[tok_id]) / float(step_idx)
            p_mix = (1.0 - cache_weight) * p_llm + cache_weight * p_cache
        else:
            p_mix = p_llm

        bits_llm = float(-np.log2(p_llm))
        bits_mix = float(-np.log2(max(p_mix, 1e-12)))
        llm_only_bits += bits_llm
        cum_bits += bits_mix

        counts[tok_id] += 1
        entries.append(
            TokenSurprisalEntry(
                step_idx=step_idx,
                token_id=tok_id,
                token_str=tokenizer.decode([tok_id]),
                prob_llm=round(p_llm, 6),
                prob_mixed=round(p_mix, 6),
                ideal_shannon_bits=round(bits_mix, 3),
                cumulative_bits=round(cum_bits, 3),
            )
        )

    return SurprisalProfileReport(
        total_tokens=len(token_ids),
        total_bytes=num_bytes,
        ideal_total_bits=round(cum_bits, 3),
        ideal_bits_per_byte=round(cum_bits / num_bytes, 3),
        cache_bits_saved=round(llm_only_bits - cum_bits, 3),
        entries=entries,
    )
# src/unplow/cache_coder.py

```

---

#### C. Non-Breaking Header Flag & CLI Integration Wiring

1. **In `src/unplow/conditioning.py` and `src/unplow/container.py`:**
* Extend `pack_flags_with_domain(verify_hash: bool, domain_id: int = 0, cache_mixture: bool = False) -> int` so Bit 1 (`0x02`) encodes `cache_mixture`:
```python
low_nibble = (0x01 if verify_hash else 0x00) | (0x02 if cache_mixture else 0x00)
return ((domain_id & 0x0F) << 4) | low_nibble

```


* Add a `@property def cache_mixture(self) -> bool: return bool(self.flags & 0x02)` property on `UnpArchiveMetadata` in `src/unplow/container.py`.


2. **In `src/unplow/paradigm2.py`:**
* Add `use_cache_mixture: bool = False` to `compress()`, `decompress()`, `compress_to_container()`, and `decompress_from_container()`. When enabled (or when `meta.cache_mixture` is `True` on `unarchive`), instantiate `OnlinePrefixCacheCDF(vocab_size)` and call `cache_cdf.logits_to_mixed_integer_cdf(logits)` followed by `cache_cdf.update(token)` at each step $t$.


3. **In `src/unplow/cli.py` & `src/unplow/worker.py`:**
* Add `--cache-mix` boolean flag to `unplow archive`.
* Add `unplow steer --layer <int> --neuron <int> [--gain 4.5] [--prompt "/* "]` to invoke `MechanisticSteeringEngine.decompress_neuron_basin()` and `unplow circuits --folded` to invoke `extract_rmsnorm_folded_circuits()`.
* Add `unplow profile --text <str>` (or `--input <path>`) to run `profile_sequence_surprisal()` and print the per-token Shannon surprisal table and `cache_bits_saved`.



---

#### D. Unit Test Suite 1: `tests/test_steering.py`

```python
# tests/test_steering.py
"""
Unit tests for Phase 4 MechanisticSteeringEngine (RMSNorm-folded circuit
extraction and causal activation-steered weight decompression).
"""

import torch
from unplow.models import DeterministicMockCausalLM, DeterministicMockTokenizer
from unplow.steering import (
    FoldedCircuitTrace,
    MechanisticSteeringEngine,
    SteeredBasinRecord,
)


def test_rmsnorm_folded_circuit_extraction():
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    engine = MechanisticSteeringEngine(model, tokenizer)

    traces = engine.extract_rmsnorm_folded_circuits(
        layer_indices=[1, 2], candidates_per_layer=6, top_k_tokens=3
    )
    assert isinstance(traces, list)
    assert len(traces) > 0
    first = traces[0]
    assert isinstance(first, FoldedCircuitTrace)
    assert first.layer_idx in (1, 2)
    assert first.centered_l2_norm > 0.0
    assert len(first.folded_trigger_keys) > 0
    assert len(first.target_values) > 0


def test_decompress_neuron_basin_and_hook_cleanup():
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    engine = MechanisticSteeringEngine(model, tokenizer)

    target_down_proj = model.model.layers[1].mlp.down_proj
    initial_hooks = len(target_down_proj._forward_hooks)

    rec = engine.decompress_neuron_basin(
        layer_idx=1,
        neuron_idx=0,
        seed_prompt="def ",
        seq_len=12,
        steering_gain=3.5,
        steering_steps=6,
    )
    assert isinstance(rec, SteeredBasinRecord)
    assert rec.layer_idx == 1
    assert rec.neuron_idx == 0
    assert rec.steering_gain == 3.5
    assert len(rec.extracted_text) >= 4
    assert rec.model_bpb > 0.0
    assert rec.zlib_bpb > 0.0

    # Verify forward hook was cleanly removed in finally block
    assert len(target_down_proj._forward_hooks) == initial_hooks
# tests/test_steering.py

```

---

#### E. Unit Test Suite 2: `tests/test_cache_coder.py`

```python
# tests/test_cache_coder.py
"""
Unit tests for Phase 4 OnlinePrefixCacheCDF, per-token Shannon surprisal
profiling, and .unp container FLAG_CACHE_MIXTURE roundtrip verification.
"""

import numpy as np
import torch
from unplow.cache_coder import (
    FLAG_CACHE_MIXTURE,
     TOTAL_FREQ,
    OnlinePrefixCacheCDF,
    profile_sequence_surprisal,
)
from unplow.conditioning import pack_flags_with_domain, unpack_flags_with_domain
from unplow.models import DeterministicMockCausalLM, DeterministicMockTokenizer
from unplow.paradigm2 import LLMArithmeticCodec


def test_online_prefix_cache_cdf_monotonicity_and_boost():
    vocab_size = 256
    cache = OnlinePrefixCacheCDF(vocab_size=vocab_size, cache_weight=0.15, window_size=8)
    logits = torch.zeros(vocab_size, dtype=torch.float32)

    cdf_initial = cache.logits_to_mixed_integer_cdf(logits)
    assert cdf_initial[0] == 0
    assert cdf_initial[-1] == TOTAL_FREQ
    assert np.all(np.diff(cdf_initial) >= 1)

    target_tok = 42
    initial_span = cdf_initial[target_tok + 1] - cdf_initial[target_tok]

    # Repeat token 42 three times and verify its frequency mass strictly increases
    for _ in range(3):
        cache.update(target_tok)

    cdf_boosted = cache.logits_to_mixed_integer_cdf(logits)
    boosted_span = cdf_boosted[target_tok + 1] - cdf_boosted[target_tok]
    assert boosted_span > initial_span
    assert cdf_boosted[-1] == TOTAL_FREQ


def test_flag_cache_mixture_bit_packing():
    flags = pack_flags_with_domain(verify_hash=True, domain_id=2, cache_mixture=True)
    assert flags & FLAG_CACHE_MIXTURE == FLAG_CACHE_MIXTURE
    verify, dom_id = unpack_flags_with_domain(flags)
    assert verify is True
    assert dom_id == 2


def test_cache_mixture_lossless_container_roundtrip():
    codec = LLMArithmeticCodec(model_name="mock")
    text = "unplow_token unplow_token unplow_token lossless cache mixture verification."
    container_bytes = codec.compress_to_container(
        text, domain_id=1, use_cache_mixture=True
    )
    reconstructed, meta = codec.decompress_from_container(container_bytes)
    assert reconstructed == text
    assert meta.domain_id == 1
    assert meta.cache_mixture is True


def test_profile_sequence_surprisal_saves_bits_on_repetition():
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    repeated_text = "alpha beta alpha beta alpha beta alpha beta"

    report = profile_sequence_surprisal(
        model=model,
        tokenizer=tokenizer,
        plaintext=repeated_text,
        bos_token_id=tokenizer.bos_token_id,
        cache_weight=0.12,
    )
    assert report.total_tokens > 0
    assert report.ideal_bits_per_byte > 0.0
    assert report.cache_bits_saved > 0.0
    assert len(report.entries) == report.total_tokens
# tests/test_cache_coder.py

```
