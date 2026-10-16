import zlib
from dataclasses import dataclass
from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ExtractedRecord:
    text: str
    model_bits_per_byte: float
    zlib_bits_per_byte: float
    memorization_score: float
    casing_sensitivity: float


class WeightDecompressor:
    """
    Unpacks memorized training sequences directly from static LLM weights
    using temperature-decayed basin probing and Carlini et al. (2021)
    zlib-to-perplexity entropy calibration.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def compute_sequence_bits(self, text: str) -> float:
        """
        Computes the exact Shannon information content (in bits) assigned to `text`
        by the LLM: sum_{t} -log_2 P_theta(x_t | x_{<t}).
        """
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = enc.input_ids
        if input_ids.shape[1] < 2:
            return float("inf")

        outputs = self.model(input_ids)
        # Shift logits and targets for causal next-token evaluation
        shift_logits = outputs.logits[:, :-1, :].contiguous().to(torch.float32)
        shift_labels = input_ids[:, 1:].contiguous()

        # Cross-entropy in nats, converted to base-2 bits via 1 / ln(2)
        loss_nats = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
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
        seed_prefixes: List[str] = None,
    ) -> List[ExtractedRecord]:
        """
        Probes the LLM weight space by dispersing into diverse semantic basins
        during `explore_steps` and then annealing into low-temperature extraction.
        """
        if not seed_prefixes:
            # Diverse structural anchors to jump into distinct parameter subspaces
            seed_prefixes = [
                "Copyright (c) ",
                "def ",
                "The following is a verbatim ",
                "According to the official ",
            ]

        records: List[ExtractedRecord] = []

        for i in range(num_samples):
            prefix = seed_prefixes[i % len(seed_prefixes)]
            input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)
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

                # High temperature early to jump basins; low temperature later to lock onto memorized paths
                temp = temp_explore if step < explore_steps else temp_lock
                scaled_logits = logits / temp

                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                curr_ids = next_token

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
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
            casing_ratio = (lower_bits / max(model_total_bits, 1e-6))

            # Carlini et al. memorization score: high zlib entropy + low model perplexity
            memorization_score = zlib_bpb / max(model_bpb, 1e-6)

            records.append(
                ExtractedRecord(
                    text=decoded_text,
                    model_bits_per_byte=round(model_bpb, 3),
                    zlib_bits_per_byte=round(zlib_bpb, 3),
                    memorization_score=round(memorization_score, 3),
                    casing_sensitivity=round(casing_ratio, 3),
                )
            )

        # Sort descending by memorization score (highest ratio of intrinsic complexity to model surprise)
        records.sort(key=lambda r: r.memorization_score, reverse=True)
        return records


if __name__ == "__main__":
    # Can use "gpt2" for quick CPU testing or "Qwen/Qwen2.5-1.5B"
    extractor = WeightDecompressor(model_name="gpt2")
    extracted = extractor.probe_weight_basins(num_samples=4, seq_len=64)
    for idx, rec in enumerate(extracted, 1):
        print(f"\n--- Decompressed Weight Basin #{idx} ---")
        print(f"Memorization Score (Zlib/LLM): {rec.memorization_score} "
              f"(LLM: {rec.model_bits_per_byte} bpb | Zlib: {rec.zlib_bits_per_byte} bpb | "
              f"Case Ratio: {rec.casing_sensitivity})")
        print(rec.text)
