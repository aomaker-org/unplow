# Path: unplow/src/unplow/paradigm1.py
"""Paradigm 1: Lossy Static Weight Decompression (Carlini et al., 2021).

Treats static neural network parameters theta as a self-contained, lossy archive
of the pre-training corpus, extracting memorized sequences directly from weights
with zero external bitstream via temperature-decayed basin exploration and
Carlini et al. zlib-to-perplexity entropy calibration.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import zlib

import torch
import torch.nn.functional as F

from .models import load_causal_lm, resolve_model_path

logger = logging.getLogger("unplow.paradigm1")


@dataclass
class ExtractedRecord:
    """Record representing a sequence extracted from static LLM weights."""

    text: str
    model_bits_per_byte: float
    zlib_bits_per_byte: float
    memorization_score: float
    casing_sensitivity: float
    token_count: int = 0
    raw_bytes_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WeightDecompressor:
    """Unpacks memorized training sequences directly from static LLM weights."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None and tokenizer is not None:
            self.tokenizer = tokenizer
            self.model = model
            self.model_path = "custom"
        else:
            self.tokenizer, self.model, self.model_path = load_causal_lm(
                model_identifier=model_name,
                device=self.device,
            )

        self.bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if self.bos_token_id is None:
            self.bos_token_id = getattr(self.tokenizer, "eos_token_id", 0)

    @torch.no_grad()
    def compute_sequence_bits(self, text: str) -> float:
        """Compute exact Shannon information content (in bits) assigned to text.

        Calculates: sum_{t} -log_2 P_theta(x_t | x_{<t}).
        """
        if not text:
            return 0.0

        if hasattr(self.tokenizer, "__call__") and callable(getattr(self.tokenizer, "__call__")):
            try:
                enc = self.tokenizer(text, return_tensors="pt")
                input_ids = enc.input_ids.to(self.device)
            except TypeError:
                input_ids = torch.tensor([self.tokenizer.encode(text)], device=self.device)
        else:
            input_ids = torch.tensor([self.tokenizer.encode(text)], device=self.device)

        if input_ids.shape[1] < 2:
            return float("inf")

        outputs = self.model(input_ids)
        shift_logits = outputs.logits[:, :-1, :].contiguous().to(torch.float32)
        shift_labels = input_ids[:, 1:].contiguous()

        loss_nats = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        # Convert cross-entropy from nats to base-2 bits: bits = nats / ln(2)
        return float(loss_nats.item() / torch.log(torch.tensor(2.0)).item())

    @torch.no_grad()
    def probe_weight_basins(
        self,
        num_samples: int = 16,
        seq_len: int = 128,
        explore_steps: int = 12,
        temp_explore: float = 1.15,
        temp_lock: float = 0.25,
        top_k: int = 60,
        seed_prefixes: Optional[List[str]] = None,
        use_mechanistic_seeds: bool = False,
    ) -> List[ExtractedRecord]:
        """Probe LLM parameter space via temperature-decayed exploration.

        Steps into diverse semantic basins during `explore_steps` and then
        anneals into low-temperature greedy extraction to follow memorized basins.
        """
        if not seed_prefixes:
            if use_mechanistic_seeds:
                from .mechanistic import MechanisticWeightInspector
                inspector = MechanisticWeightInspector(model=self.model, tokenizer=self.tokenizer)
                harvested = inspector.harvest_seed_prefixes(num_seeds=max(num_samples, 4))
                seed_prefixes = harvested if harvested else [
                    "Copyright (c) ",
                    "def ",
                    "The following is a verbatim ",
                    "According to the official ",
                ]
            else:
                seed_prefixes = [
                    "Copyright (c) ",
                    "def ",
                    "The following is a verbatim ",
                    "According to the official ",
                ]

        records: List[ExtractedRecord] = []

        for i in range(num_samples):
            prefix = seed_prefixes[i % len(seed_prefixes)]
            if hasattr(self.tokenizer, "__call__") and callable(getattr(self.tokenizer, "__call__")):
                try:
                    input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)
                except TypeError:
                    input_ids = torch.tensor([self.tokenizer.encode(prefix)], device=self.device)
            else:
                input_ids = torch.tensor([self.tokenizer.encode(prefix)], device=self.device)

            past_key_values = None
            curr_ids = input_ids

            for step in range(seq_len):
                outputs = self.model(
                    input_ids=curr_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :].to(torch.float32)

                temp = temp_explore if step < explore_steps else temp_lock
                scaled_logits = logits / max(temp, 1e-4)

                if top_k > 0:
                    k_val = min(top_k, scaled_logits.size(-1))
                    v, _ = torch.topk(scaled_logits, k_val)
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                curr_ids = next_token

                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_id is not None and next_token.item() == eos_id:
                    break

            decoded_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            raw_bytes = decoded_text.encode("utf-8")
            num_bytes = max(len(raw_bytes), 1)

            # 1. Model compressed description length (bits per byte)
            model_total_bits = self.compute_sequence_bits(decoded_text)
            model_bpb = model_total_bits / num_bytes

            # 2. Intrinsic syntactic entropy via zlib (bits per byte)
            zlib_total_bits = len(zlib.compress(raw_bytes, level=9)) * 8.0
            zlib_bpb = zlib_total_bits / num_bytes

            # 3. Lowercase perturbation ratio (detects verbatim casing memorization)
            lower_bits = self.compute_sequence_bits(decoded_text.lower())
            casing_ratio = lower_bits / max(model_total_bits, 1e-6)

            # Carlini et al. memorization score: high zlib entropy + low model perplexity
            memorization_score = zlib_bpb / max(model_bpb, 1e-6)

            records.append(
                ExtractedRecord(
                    text=decoded_text,
                    model_bits_per_byte=round(model_bpb, 3),
                    zlib_bits_per_byte=round(zlib_bpb, 3),
                    memorization_score=round(memorization_score, 3),
                    casing_sensitivity=round(casing_ratio, 3),
                    token_count=int(input_ids.shape[1]),
                    raw_bytes_count=num_bytes,
                )
            )

        # Sort descending by memorization score
        records.sort(key=lambda r: r.memorization_score, reverse=True)
        return records

    @staticmethod
    def save_records_to_jsonl(records: List[ExtractedRecord], destination: Path) -> Path:
        """Save extracted records to JSONL file."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")
        return destination

# end of file: unplow/src/unplow/paradigm1.py
