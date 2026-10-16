# Path: unplow/src/unplow/mechanistic.py
"""Direct Mechanistic Weight Inspection & Seed Harvesting.

Mathematical attribution:
- Mor Geva et al. (2021): Transformer Feed-Forward Layers Are Key-Value Memories.
- nostalgebraist (2020): Interpreting GPT: the Logit Lens.
- Kevin Meng et al. (2022): Locating and Editing Factual Associations in GPT (ROME).

Directly projects static MLP value vectors (columns of mlp.down_proj.weight)
into vocabulary space via the model's final RMSNorm and lm_head (Logit Lens)
to harvest memorized semantic clusters without running forward attention passes.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional
import torch

from .models import load_causal_lm

logger = logging.getLogger("unplow.mechanistic")


@dataclass
class NeuronMemoryTrace:
    """Record representing a static MLP neuron's direct projection into vocabulary."""

    layer_idx: int
    neuron_idx: int
    l2_norm: float
    top_tokens: List[str]
    top_token_ids: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MechanisticWeightInspector:
    """Projects static MLP down_proj value vectors into vocabulary logits."""

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer, self.model, _ = load_causal_lm(
                model_identifier=model_name,
                device=device or "cpu",
            )

        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def extract_mlp_memory_clusters(
        self,
        layer_indices: Optional[List[int]] = None,
        top_neurons_per_layer: int = 8,
        top_k_tokens: int = 6,
    ) -> List[NeuronMemoryTrace]:
        """Scans specified Transformer layers and decodes top-L2 value vectors via Logit Lens."""
        # Detect layers hierarchy (Qwen / LLaMA / Mock models)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
            final_norm = getattr(self.model.model, "norm", None)
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
            final_norm = getattr(self.model, "norm", None)
        else:
            # Fallback for simple mock models
            layers = getattr(self.model, "mock_layers", [])
            final_norm = getattr(self.model, "mock_norm", None)

        num_layers = len(layers)
        if num_layers == 0:
            logger.warning("No discrete transformer layers detected for mechanistic inspection.")
            return []

        if layer_indices is None:
            # Upper-middle layers carry highest density of localized key-value memories
            start_layer = max(0, int(num_layers * 0.60))
            layer_indices = list(range(start_layer, num_layers))

        lm_head = getattr(self.model, "lm_head", getattr(self.model, "head", None))
        if lm_head is None:
            logger.warning("No lm_head found on model.")
            return []

        traces: List[NeuronMemoryTrace] = []

        for l_idx in layer_indices:
            if l_idx >= num_layers:
                continue
            layer = layers[l_idx]

            # Extract down_proj weight
            down_weight = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                down_weight = layer.mlp.down_proj.weight.detach().to(torch.float32)
            elif hasattr(layer, "down_proj"):
                down_weight = layer.down_proj.weight.detach().to(torch.float32)
            elif hasattr(layer, "weight"):
                down_weight = layer.weight.detach().to(torch.float32)

            if down_weight is None:
                continue

            # Compute L2 norm of each column (value vector for 1 neuron)
            # In Linear(in_features=d_ffn, out_features=d_model), weight has shape [d_model, d_ffn]
            col_norms = torch.linalg.vector_norm(down_weight, ord=2, dim=0)
            k_neurons = min(top_neurons_per_layer, col_norms.shape[0])
            top_norms, top_neuron_indices = torch.topk(col_norms, k=k_neurons)

            # Shape: [k_neurons, d_model]
            value_vectors = down_weight[:, top_neuron_indices].T.to(self.device)

            # Project through final RMSNorm and lm_head
            norm_dtype = next(lm_head.parameters()).dtype
            if final_norm is not None:
                normed_vectors = final_norm(value_vectors.to(norm_dtype))
            else:
                normed_vectors = value_vectors.to(norm_dtype)

            logits = lm_head(normed_vectors).to(torch.float32)
            _, top_token_matrix = torch.topk(logits, k=top_k_tokens, dim=-1)

            for i in range(top_neuron_indices.shape[0]):
                n_idx = int(top_neuron_indices[i].item())
                n_norm = float(top_norms[i].item())
                tok_ids = top_token_matrix[i].tolist()
                decoded_toks = [
                    self.tokenizer.decode([tid], skip_special_tokens=True)
                    for tid in tok_ids
                ]

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
        """Harvest top non-trivial tokens decoded from MLP neurons to seed Paradigm 1 probing."""
        traces = self.extract_mlp_memory_clusters(top_neurons_per_layer=8, top_k_tokens=6)
        seeds: List[str] = []
        seen = set()

        for tr in traces:
            for tok_str in tr.top_tokens:
                cleaned = tok_str.strip()
                if len(cleaned) >= 3 and cleaned.isprintable() and cleaned.lower() not in seen:
                    seen.add(cleaned.lower())
                    seeds.append(tok_str)
                    if len(seeds) >= num_seeds:
                        return seeds

        return seeds

# end of file: unplow/src/unplow/mechanistic.py
