# Path: unplow/src/unplow/ctw_coder.py
"""Online Bayesian Context-Tree Switching (CTS) Mixture Arithmetic Coder.

(Willems, Shtarkov, & Tjalkens, 1995; Vovk, 1990; Veness et al., 2012).
Dynamically adapts mixture weights across Static LLM, Order-0 Unigram, and
Order-1 Markov Bigram predictors in lockstep using Bit 3 (FLAG_BAYES_MIXER = 0x08).
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

TOTAL_FREQ = 1 << 24  # 16,777,216
FLAG_BAYES_MIXER = 0x08


@dataclass
class BayesTrajectoryStep:
    """Snapshot of Bayesian expert mixture weights and Shannon surprisal at step t."""

    step_idx: int
    token_id: int
    w_llm: float
    w_unigram: float
    w_bigram: float
    step_bits: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BayesianCTSMixerCDF:
    """Maintains an online 3-expert Bayesian posterior over LLM, Unigram, and Bigram caches.

    Expert 0: Static LLM Softmax P_theta(x_t | x_<t)
    Expert 1: Sliding Order-0 Unigram Cache P_uni(x_t | x_<t)
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

    def reset(self) -> None:
        """Resets dynamic expert weights, histograms, and transition cache."""
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
        """Computes Bayesian mixture distribution and quantizes into a strictly monotonic 24-bit CDF."""
        if logits.ndim > 1:
            logits = logits.squeeze()

        probs_llm = torch.softmax(logits.to(torch.float64), dim=-1).cpu().numpy()
        self._last_llm_probs = probs_llm
        vocab_size = probs_llm.shape[0]

        if vocab_size > len(self.unigram_counts):
            new_counts = np.zeros(vocab_size, dtype=np.int64)
            new_counts[: len(self.unigram_counts)] = self.unigram_counts
            self.unigram_counts = new_counts
            self.vocab_size = vocab_size

        remaining_mass = TOTAL_FREQ - vocab_size
        if remaining_mass <= 0:
            raise ValueError(
                f"TOTAL_FREQ ({TOTAL_FREQ}) must exceed vocab ({vocab_size})."
            )

        w0, w1, w2 = self._active_weights()
        probs = w0 * probs_llm

        if w1 > 0.0:
            active_len = max(float(len(self.history)), 1.0)
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

    def update(self, token_id: int) -> None:
        """Updates Bayesian expert posterior weights via Vovk/CTS rule, then updates caches."""
        w0, w1, w2 = self._active_weights()

        # 1. Evaluate expert likelihoods on observed token_id
        p0 = (
            float(self._last_llm_probs[token_id])
            if (self._last_llm_probs is not None and 0 <= token_id < self.vocab_size)
            else (1.0 / float(max(self.vocab_size, 1)))
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

# end of file: unplow/src/unplow/ctw_coder.py
