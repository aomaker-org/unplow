# Path: unplow/src/unplow/cache_coder.py
"""Online Prefix-Cache Mixture CDF & Per-Token Surprisal Profiler.

Implements:
1. Online prefix-cache mixture arithmetic coding (Kuhn & De Mori, 1990; Grave et al., 2017)
   dynamically mixing static LLM softmax probabilities with a local empirical n-gram/unigram
   frequency histogram in lockstep between encoder and decoder without side-information bits.
2. Per-token Shannon surprisal profiler (-log2 P(x_t | x_{<t})) measuring exact bit costs
   and quantifying entropy reductions delivered by continuous caching.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List
import numpy as np
import torch

logger = logging.getLogger("unplow.cache_coder")

TOTAL_FREQ = 1 << 24  # 16,777,216
FLAG_CACHE_MIXTURE = 0x02


@dataclass
class TokenSurprisalEntry:
    """Individual token surprisal attribution entry."""

    step_idx: int
    token_id: int
    token_str: str
    prob_llm: float
    prob_mixed: float
    ideal_shannon_bits: float
    cumulative_bits: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SurprisalProfileReport:
    """Aggregated sequence surprisal and cache bit savings report."""

    total_tokens: int
    total_bytes: int
    ideal_total_bits: float
    ideal_bits_per_byte: float
    cache_bits_saved: float
    entries: List[TokenSurprisalEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "total_bytes": self.total_bytes,
            "ideal_total_bits": self.ideal_total_bits,
            "ideal_bits_per_byte": self.ideal_bits_per_byte,
            "cache_bits_saved": self.cache_bits_saved,
            "entries": [e.to_dict() for e in self.entries],
        }


class OnlinePrefixCacheCDF:
    """Maintains a deterministic online unigram frequency cache over already

    encoded/decoded prefix tokens in lockstep between encoder and decoder,
    boosting probabilities of locally repeated tokens inside the 24-bit integer CDF.
    """

    def __init__(self, vocab_size: int, cache_weight: float = 0.08, window_size: int = 1024):
        self.vocab_size = vocab_size
        self.cache_weight = cache_weight
        self.window_size = window_size
        self.history: List[int] = []
        self.counts = np.zeros(vocab_size, dtype=np.int64)

    def reset(self) -> None:
        """Reset prefix history and frequency histogram."""
        self.history.clear()
        self.counts.fill(0)

    def update(self, token_id: int) -> None:
        """Records `token_id` into the sliding prefix histogram after step t."""
        self.history.append(token_id)
        if token_id >= len(self.counts):
            new_size = max(token_id + 1, len(self.counts) * 2)
            new_counts = np.zeros(new_size, dtype=np.int64)
            new_counts[:len(self.counts)] = self.counts
            self.counts = new_counts
        if 0 <= token_id < len(self.counts):
            self.counts[token_id] += 1
        if len(self.history) > self.window_size:
            evicted = self.history.pop(0)
            if 0 <= evicted < len(self.counts):
                self.counts[evicted] -= 1

    @torch.no_grad()
    def logits_to_mixed_integer_cdf(self, logits: torch.Tensor) -> np.ndarray:
        """Combines LLM softmax probabilities with the online prefix histogram

        to produce a strictly monotonic integer CDF with total mass 2^24.
        """
        probs_llm = torch.softmax(logits.to(torch.float64), dim=-1).cpu().numpy()
        vocab_size = probs_llm.shape[0]
        remaining_mass = TOTAL_FREQ - vocab_size
        if remaining_mass <= 0:
            raise ValueError(f"TOTAL_FREQ ({TOTAL_FREQ}) must exceed vocab_size ({vocab_size})")

        active_len = len(self.history)
        if active_len > 0 and self.cache_weight > 0.0:
            probs_cache = np.zeros(vocab_size, dtype=np.float64)
            valid_len = min(vocab_size, len(self.counts))
            probs_cache[:valid_len] = self.counts[:valid_len].astype(np.float64) / float(active_len)
            probs = (1.0 - self.cache_weight) * probs_llm + self.cache_weight * probs_cache
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
    model: Any,
    tokenizer: Any,
    plaintext: str,
    bos_token_id: int,
    cache_weight: float = 0.08,
) -> SurprisalProfileReport:
    """Computes an exact per-token Shannon surprisal ledger (-log2 P(x_t | x_{<t}))

    comparing static LLM probabilities against the online cache mixture.
    """
    device = next(model.parameters()).device
    token_ids = tokenizer.encode(plaintext, add_special_tokens=False)
    num_bytes = max(len(plaintext.encode("utf-8")), 1)

    if not token_ids:
        return SurprisalProfileReport(
            total_tokens=0,
            total_bytes=num_bytes,
            ideal_total_bits=0.0,
            ideal_bits_per_byte=0.0,
            cache_bits_saved=0.0,
            entries=[],
        )

    seq = [bos_token_id] + token_ids
    input_tensor = torch.tensor([seq[:-1]], device=device)
    outputs = model(input_tensor)
    logits_all = outputs.logits[0].to(torch.float64) if hasattr(outputs, "logits") else outputs[0].to(torch.float64)
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
        tok_str = tokenizer.decode([tok_id]) if hasattr(tokenizer, "decode") else str(tok_id)
        entries.append(
            TokenSurprisalEntry(
                step_idx=step_idx,
                token_id=tok_id,
                token_str=tok_str,
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

# end of file: unplow/src/unplow/cache_coder.py
