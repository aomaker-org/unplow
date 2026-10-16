# Path: unplow/src/unplow/circuits.py
"""Coupled SwiGLU Key-to-Value Circuit Decompressor & Anisotropy-Corrected Logit Lens.

Implements static circuit decompression based on:
1. SwiGLU feedforward receptor-emitter geometry (Shazeer, 2020).
2. Transformer MLP key-value associative memory circuits (Geva et al., 2021).
3. Anisotropic mean centering to correct late-layer representation collapse (Ethayarajh, 2019; Sun et al., 2024).
4. Latent multilingual detokenization & cross-lingual realization filtering (Wendler et al., 2024).

Directly harvests multi-token associative seeds (Trigger Context -> Memorized Target)
without performing forward passes.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional
import torch

from .models import load_causal_lm

logger = logging.getLogger("unplow.circuits")


@dataclass
class AssociativeCircuitTrace:
    """Represents a coupled SwiGLU receptor-emitter associative circuit."""

    layer_idx: int
    neuron_idx: int
    centered_l2_norm: float
    selectivity_margin: float
    trigger_key_tokens: List[str]
    target_value_tokens: List[str]
    synthesized_seed: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SwiGLUCircuitDecompressor:
    """Extracts static (Key Trigger -> Value Output) associative circuits from

    SwiGLU MLP layers by coupling gate_proj/up_proj receptor vectors with
    mean-centered down_proj value vectors.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.model_path = "custom"
        else:
            self.tokenizer, self.model, self.model_path = load_causal_lm(
                model_identifier=model_name,
                device=self.device,
                dtype=torch.float32,
            )

    @staticmethod
    def _is_trivial_numeral_or_punct(token_str: str) -> bool:
        """Filters out pure digits, single characters, and whitespace/punctuation."""
        stripped = token_str.strip()
        if len(stripped) < 2:
            return True
        if stripped.isdigit():
            return True
        # Filter pure ASCII/Unicode numeric tokens that dominate Layer 26 norms
        if all(not ch.isalpha() for ch in stripped):
            return True
        return False

    @torch.no_grad()
    def extract_coupled_circuits(
        self,
        layer_indices: Optional[List[int]] = None,
        candidates_per_layer: int = 24,
        top_k_tokens: int = 5,
        suppress_numerals: bool = True,
    ) -> List[AssociativeCircuitTrace]:
        """Scans MLP layers using Anisotropic Mean Centering on down_proj and

        couples top selective value vectors with their gate_proj + up_proj trigger keys.
        """
        model_body = getattr(self.model, "model", self.model)
        layers = getattr(model_body, "layers", [])
        num_layers = len(layers)

        if num_layers == 0:
            return []

        if layer_indices is None:
            # Middle-to-upper layers (50% to 88% depth) capture semantic/code circuits
            # before the final layer detokenization/numeral amplifiers
            start_l = max(0, int(num_layers * 0.50))
            end_l = max(start_l + 1, int(num_layers * 0.88))
            layer_indices = list(range(start_l, end_l))

        final_norm = getattr(model_body, "norm", None)
        lm_head = getattr(self.model, "lm_head", None)

        if hasattr(self.model, "get_input_embeddings") and self.model.get_input_embeddings() is not None:
            embed_weight = self.model.get_input_embeddings().weight.detach().to(torch.float32)
        elif hasattr(model_body, "embed_tokens"):
            embed_weight = model_body.embed_tokens.weight.detach().to(torch.float32)
        else:
            embed_weight = getattr(self.model, "embed").weight.detach().to(torch.float32)

        # Normalize input embeddings for cosine/dot-product receptor matching
        embed_normed = torch.nn.functional.normalize(embed_weight, p=2, dim=-1)

        circuits: List[AssociativeCircuitTrace] = []

        for l_idx in layer_indices:
            if l_idx < 0 or l_idx >= num_layers:
                continue
            mlp = layers[l_idx].mlp
            down_w = mlp.down_proj.weight.detach().to(torch.float32)

            # 1. Anisotropic Mean Centering (Ethayarajh, 2019)
            mean_vec = down_w.mean(dim=1, keepdim=True)
            centered_down = down_w - mean_vec

            col_norms = torch.linalg.vector_norm(centered_down, ord=2, dim=0)
            k_candidates = min(candidates_per_layer, col_norms.shape[0])
            top_norms, top_indices = torch.topk(col_norms, k=k_candidates)

            # 2. Project candidate value vectors through final RMSNorm and lm_head
            cand_values = centered_down[:, top_indices].T.to(self.device)
            if final_norm is not None and lm_head is not None:
                normed_vals = final_norm(cand_values.to(lm_head.weight.dtype))
                val_logits = lm_head(normed_vals).to(torch.float32)
            elif lm_head is not None:
                val_logits = lm_head(cand_values).to(torch.float32)
            else:
                continue

            # Compute Selectivity Margin: (Top-1 logit) - (50th-rank logit)
            k_margin = min(50, val_logits.shape[-1])
            top_val_vals, top_val_ids = torch.topk(val_logits, k=k_margin, dim=-1)
            margins = top_val_vals[:, 0] - top_val_vals[:, -1]

            # 3. Extract Coupled SwiGLU Receptor Keys (gate_proj + up_proj rows)
            has_gate = hasattr(mlp, "gate_proj") and mlp.gate_proj is not None
            has_up = hasattr(mlp, "up_proj") and mlp.up_proj is not None

            if has_gate and has_up:
                gate_vecs = mlp.gate_proj.weight.detach()[top_indices, :].to(torch.float32)
                up_vecs = mlp.up_proj.weight.detach()[top_indices, :].to(torch.float32)
                coupled_keys = torch.nn.functional.normalize(
                    gate_vecs + up_vecs, p=2, dim=-1
                ).to(self.device)
                key_scores = torch.matmul(coupled_keys, embed_normed.to(self.device).T)
                k_key = min(top_k_tokens, key_scores.shape[-1])
                _, top_key_ids = torch.topk(key_scores, k=k_key, dim=-1)
            else:
                top_key_ids = None

            for i in range(top_indices.shape[0]):
                n_idx = int(top_indices[i].item())
                c_norm = float(top_norms[i].item())
                margin = float(margins[i].item())

                k_val = min(top_k_tokens, top_val_ids.shape[-1])
                v_ids = top_val_ids[i, :k_val].tolist()
                v_toks = [self.tokenizer.decode([tid]) for tid in v_ids]

                if suppress_numerals and all(
                    self._is_trivial_numeral_or_punct(t) for t in v_toks[:min(3, len(v_toks))]
                ):
                    continue

                if top_key_ids is not None:
                    k_ids = top_key_ids[i].tolist()
                    k_toks = [self.tokenizer.decode([tid]) for tid in k_ids]
                else:
                    k_toks = ["<unk>"]

                # Synthesize a 2-part associative seed from best printable trigger + value
                best_k = next(
                    (t for t in k_toks if not self._is_trivial_numeral_or_punct(t)),
                    k_toks[0] if k_toks else "",
                )
                best_v = next(
                    (t for t in v_toks if not self._is_trivial_numeral_or_punct(t)),
                    v_toks[0] if v_toks else "",
                )
                seed_phrase = f"{best_k}{best_v}"

                circuits.append(
                    AssociativeCircuitTrace(
                        layer_idx=l_idx,
                        neuron_idx=n_idx,
                        centered_l2_norm=round(c_norm, 4),
                        selectivity_margin=round(margin, 4),
                        trigger_key_tokens=k_toks,
                        target_value_tokens=v_toks,
                        synthesized_seed=seed_phrase,
                    )
                )

        # Rank by joint product of centered norm and logit selectivity margin
        circuits.sort(
            key=lambda c: c.centered_l2_norm * c.selectivity_margin, reverse=True
        )
        return circuits

    def harvest_associative_seeds(
        self,
        max_seeds: int = 16,
        layer_indices: Optional[List[int]] = None,
        suppress_numerals: bool = True,
    ) -> List[str]:
        """Harvest clean associative seed phrases suitable for generative probing."""
        circuits = self.extract_coupled_circuits(
            layer_indices=layer_indices,
            suppress_numerals=suppress_numerals,
        )
        seeds: List[str] = []
        seen = set()
        for c in circuits:
            s = c.synthesized_seed.strip()
            if s and len(s) >= 3 and s not in seen:
                seen.add(s)
                seeds.append(s)
                if len(seeds) >= max_seeds:
                    break
        return seeds

# end of file: unplow/src/unplow/circuits.py
