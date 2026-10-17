# Path: unplow/src/unplow/subspace_coder.py
"""Multi-Neuron SVD Subspace Steering & Order-1 Markov Bigram Cache Coding.

Implements:
1. Multi-Neuron SVD Subspace Steering (Elhage et al., 2022; Bricken et al., 2023)
   extracting the principal left singular vector u_1 across co-selective MLP
   down_proj columns and steering generation without single-neuron saturation.
2. Order-1 Markov Bigram + Unigram Cache Mixture Arithmetic Coding
   (Cleary & Witten, 1984 [PPM]; Willems, Shtarkov, & Tjalkens, 1995 [CTW])
   blending static LLM softmax, unigram cache, and dynamic bigram transitions
   under .unp header flag 0x04.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple
import zlib
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

TOTAL_FREQ = 1 << 24  # 16,777,216
FLAG_BIGRAM_CACHE = 0x04


@dataclass
class SubspaceSteeringRecord:
    """Record of an SVD subspace steering extraction and basin evaluation."""

    layer_idx: int
    neuron_indices: List[int]
    singular_value_ratio: float
    steering_gain: float
    extracted_text: str
    model_bpb: float
    zlib_bpb: float
    memorization_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SVDSubspaceSteeringEngine:
    """Synthesizes Rank-1 principal steering directions via SVD across neuron ensembles."""

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def compute_principal_subspace_vector(
        self, layer_idx: int, neuron_indices: List[int]
    ) -> Tuple[torch.Tensor, float]:
        """Extracts the top left singular vector u_1 across down_proj[:, neuron_indices].

        Returns (unit_principal_vector, explained_variance_ratio).
        """
        down_w = (
            self.model.model.layers[layer_idx]
            .mlp.down_proj.weight.detach()
            .to(torch.float32)
        )
        submatrix = down_w[:, neuron_indices]  # shape: [d_model, k]

        # Align column signs with the anchor neuron (neuron_indices[0]) before SVD
        anchor = submatrix[:, 0:1]
        dots = torch.sum(submatrix * anchor, dim=0, keepdim=True)
        signs = torch.where(dots >= 0, 1.0, -1.0)
        aligned = submatrix * signs

        u, s, _ = torch.linalg.svd(aligned, full_matrices=False)
        principal_vec = F.normalize(u[:, 0], p=2, dim=0)
        energy_ratio = float((s[0] ** 2) / torch.clamp(torch.sum(s**2), min=1e-12))
        return principal_vec.to(self.device), round(energy_ratio, 4)

    @torch.no_grad()
    def steer_with_subspace(
        self,
        layer_idx: int,
        neuron_indices: List[int],
        seed_prompt: str = "/* ",
        seq_len: int = 48,
        steering_gain: float = 5.0,
        steering_steps: int = 16,
        temperature: float = 0.25,
    ) -> SubspaceSteeringRecord:
        """Steers generation along the principal SVD direction of neuron_indices at layer_idx."""
        principal_vec, sv_ratio = self.compute_principal_subspace_vector(
            layer_idx, neuron_indices
        )
        target_dtype = getattr(self.model, "dtype", torch.float32)
        if target_dtype in (torch.int32, torch.int64):
            target_dtype = torch.float32
        principal_vec = principal_vec.to(target_dtype)
        target_mlp = self.model.model.layers[layer_idx].mlp
        step_counter = {"step": 0}

        def subspace_hook(_module, _inputs, output):
            if step_counter["step"] < steering_steps:
                out_cloned = output.clone()
                out_cloned[:, -1, :] = (
                    out_cloned[:, -1, :] + steering_gain * principal_vec
                )
                return out_cloned
            return output

        handle = target_mlp.down_proj.register_forward_hook(subspace_hook)
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

        if input_ids.shape[1] > 1:
            full_logits = self.model(input_ids).logits[:, :-1, :].to(torch.float32)
            labels = input_ids[:, 1:]
            loss_nats = F.cross_entropy(
                full_logits.reshape(-1, full_logits.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            ).item()
            model_bpb = (loss_nats / 0.69314718056) / num_bytes
        else:
            model_bpb = 1.0

        zlib_bpb = (len(zlib.compress(raw_bytes, level=9)) * 8.0) / num_bytes
        mem_score = zlib_bpb / max(model_bpb, 1e-6)

        return SubspaceSteeringRecord(
            layer_idx=layer_idx,
            neuron_indices=list(neuron_indices),
            singular_value_ratio=sv_ratio,
            steering_gain=steering_gain,
            extracted_text=text,
            model_bpb=round(model_bpb, 3),
            zlib_bpb=round(zlib_bpb, 3),
            memorization_score=round(mem_score, 3),
        )


class Order1BigramCacheCDF:
    """Blends LLM Softmax + Order-0 Unigram Cache + Order-1 Markov Bigram Cache.

    Activated via .unp container header bit 0x04.
    """

    def __init__(
        self,
        vocab_size: int,
        unigram_weight: float = 0.06,
        bigram_weight: float = 0.14,
        window_size: int = 1024,
    ):
        self.vocab_size = vocab_size
        self.unigram_weight = unigram_weight
        self.bigram_weight = bigram_weight
        self.window_size = window_size
        self.history: List[int] = []
        self.unigram_counts = np.zeros(vocab_size, dtype=np.int64)
        # Sparse transition map: prev_token -> {next_token: count}
        self.bigram_transitions: Dict[int, Dict[int, int]] = {}
        self.prev_token: Optional[int] = None

    def reset(self):
        """Clears prefix history, unigram counts, and bigram transition map."""
        self.history.clear()
        self.unigram_counts.fill(0)
        self.bigram_transitions.clear()
        self.prev_token = None

    def _ensure_vocab_size(self, needed_size: int):
        if needed_size > self.vocab_size:
            new_counts = np.zeros(needed_size, dtype=np.int64)
            new_counts[: self.vocab_size] = self.unigram_counts
            self.unigram_counts = new_counts
            self.vocab_size = needed_size

    def update(self, token_id: int):
        """Updates both Order-0 unigram and Order-1 bigram counts after step t."""
        if token_id >= self.vocab_size:
            self._ensure_vocab_size(token_id + 1)

        if self.prev_token is not None:
            trans = self.bigram_transitions.setdefault(self.prev_token, {})
            trans[token_id] = trans.get(token_id, 0) + 1

        self.history.append(token_id)
        if 0 <= token_id < self.vocab_size:
            self.unigram_counts[token_id] += 1

        if len(self.history) > self.window_size:
            evicted = self.history.pop(0)
            if 0 <= evicted < self.vocab_size:
                self.unigram_counts[evicted] -= 1

        self.prev_token = token_id

    @torch.no_grad()
    def logits_to_bigram_mixed_cdf(self, logits: torch.Tensor) -> np.ndarray:
        """Blends P_theta, P_unigram, and P_bigram into a strictly monotonic 24-bit integer CDF."""
        probs_llm = torch.softmax(logits.to(torch.float64), dim=-1).cpu().numpy()
        vocab_size = probs_llm.shape[0]
        if vocab_size > self.vocab_size:
            self._ensure_vocab_size(vocab_size)

        remaining_mass = TOTAL_FREQ - vocab_size
        if remaining_mass <= 0:
            raise ValueError(
                f"TOTAL_FREQ ({TOTAL_FREQ}) must exceed vocab ({vocab_size})."
            )

        active_len = len(self.history)
        w_uni = self.unigram_weight if active_len > 0 else 0.0

        # Check if we have observed transitions out of self.prev_token
        trans = (
            self.bigram_transitions.get(self.prev_token)
            if self.prev_token is not None
            else None
        )
        w_bi = self.bigram_weight if (trans and sum(trans.values()) > 0) else 0.0
        w_llm = 1.0 - w_uni - w_bi

        probs = w_llm * probs_llm
        if w_uni > 0.0:
            probs = probs + w_uni * (
                self.unigram_counts[:vocab_size].astype(np.float64) / float(active_len)
            )
        if w_bi > 0.0 and trans is not None:
            total_trans = float(sum(trans.values()))
            for next_tok, cnt in trans.items():
                if 0 <= next_tok < vocab_size:
                    probs[next_tok] += w_bi * (float(cnt) / total_trans)

        freqs = np.floor(probs * remaining_mass).astype(np.int64) + 1
        remainder = TOTAL_FREQ - int(np.sum(freqs))
        freqs[int(np.argmax(probs))] += remainder

        cdf = np.zeros(vocab_size + 1, dtype=np.int64)
        np.cumsum(freqs, out=cdf[1:])
        return cdf

# end of file: unplow/src/unplow/subspace_coder.py
