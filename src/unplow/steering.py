# Path: unplow/src/unplow/steering.py
"""RMSNorm-Folded SwiGLU Circuit Lens & Activation-Steered Weight Decompression.

Implements:
1. RMSNorm-folded receptor key projection over mean-centered input embedding space
   (Dar et al., 2023; Belrose et al., 2023) to eliminate rogue-cone multilingual noise.
2. Causal activation steering via PyTorch forward hooks on target MLP down_proj
   layers during generation (Meng et al., 2022 [ROME]; Turner et al., 2023; Zou et al., 2023)
   to decompress training basins directly tied to individual neurons.
"""

from dataclasses import asdict, dataclass
import logging
from typing import Any, Dict, List, Optional
import zlib
import torch
import torch.nn.functional as F

from .models import load_causal_lm

logger = logging.getLogger("unplow.steering")


@dataclass
class FoldedCircuitTrace:
    """Trace of an RMSNorm-folded SwiGLU associative circuit."""

    layer_idx: int
    neuron_idx: int
    centered_l2_norm: float
    selectivity_margin: float
    folded_trigger_keys: List[str]
    target_values: List[str]
    coherence_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SteeredBasinRecord:
    """Record of an activation-steered training basin decompression."""

    layer_idx: int
    neuron_idx: int
    steering_gain: float
    extracted_text: str
    model_bpb: float
    zlib_bpb: float
    memorization_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MechanisticSteeringEngine:
    """Refines SwiGLU circuit key projection with RMSNorm gain folding and executes

    causal activation steering on target MLP neurons to decompress specific basins.
    """

    def __init__(
        self,
        model_or_name: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[str] = None,
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
    ):
        if model is None and model_or_name is not None and not isinstance(model_or_name, str):
            model = model_or_name
        if model_name is None and isinstance(model_or_name, str):
            model_name = model_or_name

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            try:
                self.device = device or (
                    next(model.parameters()).device.type
                    if hasattr(model, "parameters") and list(model.parameters())
                    else self.device
                )
            except Exception:
                pass
            self.model_path = "custom"
        else:
            self.tokenizer, self.model, self.model_path = load_causal_lm(
                model_identifier=model_name,
                device=self.device,
                dtype=torch.float32,
            )

    @staticmethod
    def _is_clean_ascii_identifier(token_str: str) -> bool:
        """Filters out non-ASCII rogue-cone subwords and trivial single characters."""
        s = token_str.strip()
        if len(s) < 2 or s.isdigit():
            return False
        if not s.isascii():
            return False
        return any(ch.isalpha() for ch in s)

    @torch.no_grad()
    def extract_rmsnorm_folded_circuits(
        self,
        layer_indices: Optional[List[int]] = None,
        candidates_per_layer: int = 20,
        top_k_tokens: int = 5,
    ) -> List[FoldedCircuitTrace]:
        """Projects SwiGLU receptor keys through post_attention_layernorm gain onto

        mean-centered W_E to eliminate rogue-cone multilingual noise.
        """
        model_body = getattr(self.model, "model", self.model)
        layers = getattr(model_body, "layers", [])
        num_layers = len(layers)
        if num_layers == 0:
            return []

        if layer_indices is None:
            start_l = max(0, int(num_layers * 0.50))
            end_l = max(start_l + 1, int(num_layers * 0.88))
            layer_indices = list(range(start_l, end_l))

        final_norm = getattr(model_body, "norm", None)
        lm_head = getattr(self.model, "lm_head", None)

        if hasattr(self.model, "get_input_embeddings") and self.model.get_input_embeddings() is not None:
            raw_embed = self.model.get_input_embeddings().weight.detach().to(torch.float32)
        elif hasattr(model_body, "embed_tokens"):
            raw_embed = model_body.embed_tokens.weight.detach().to(torch.float32)
        else:
            raw_embed = getattr(self.model, "embed").weight.detach().to(torch.float32)

        # Mean-center input embedding matrix W_E (Dar et al., 2023)
        centered_embed = raw_embed - raw_embed.mean(dim=0, keepdim=True)
        embed_normed = F.normalize(centered_embed, p=2, dim=-1).to(self.device)

        traces: List[FoldedCircuitTrace] = []

        for l_idx in layer_indices:
            if l_idx < 0 or l_idx >= num_layers:
                continue
            layer = layers[l_idx]
            mlp = layer.mlp

            post_norm = getattr(layer, "post_attention_layernorm", None)
            if post_norm is not None and hasattr(post_norm, "weight") and post_norm.weight is not None:
                post_norm_gamma = post_norm.weight.detach().to(torch.float32).to(self.device)
            else:
                post_norm_gamma = None

            down_w = mlp.down_proj.weight.detach().to(torch.float32)
            centered_down = down_w - down_w.mean(dim=1, keepdim=True)
            col_norms = torch.linalg.vector_norm(centered_down, ord=2, dim=0)

            k_cand = min(candidates_per_layer, col_norms.shape[0])
            top_norms, top_indices = torch.topk(col_norms, k=k_cand)

            cand_values = centered_down[:, top_indices].T.to(self.device)
            if final_norm is not None and lm_head is not None:
                normed_vals = final_norm(cand_values.to(lm_head.weight.dtype))
                val_logits = lm_head(normed_vals).to(torch.float32)
            elif lm_head is not None:
                val_logits = lm_head(cand_values).to(torch.float32)
            else:
                continue

            k_margin = min(50, val_logits.shape[-1])
            top_val_logits, top_val_ids = torch.topk(val_logits, k=k_margin, dim=-1)
            margins = top_val_logits[:, 0] - top_val_logits[:, -1]

            has_gate = hasattr(mlp, "gate_proj") and mlp.gate_proj is not None
            has_up = hasattr(mlp, "up_proj") and mlp.up_proj is not None

            if has_gate and has_up:
                gate_vecs = mlp.gate_proj.weight.detach()[top_indices, :].to(torch.float32).to(self.device)
                up_vecs = mlp.up_proj.weight.detach()[top_indices, :].to(torch.float32).to(self.device)
                if post_norm_gamma is not None:
                    folded_receptors = (gate_vecs + up_vecs) * post_norm_gamma.unsqueeze(0)
                else:
                    folded_receptors = gate_vecs + up_vecs
                folded_keys = F.normalize(folded_receptors, p=2, dim=-1)

                key_scores = torch.matmul(folded_keys, embed_normed.T)
                k_probe = min(top_k_tokens * 4, key_scores.shape[-1])
                _, top_key_ids = torch.topk(key_scores, k=k_probe, dim=-1)
            else:
                top_key_ids = None

            for i in range(top_indices.shape[0]):
                n_idx = int(top_indices[i].item())
                c_norm = float(top_norms[i].item())
                margin = float(margins[i].item())

                k_val = min(top_k_tokens, top_val_ids.shape[-1])
                v_ids = top_val_ids[i, :k_val].tolist()
                v_toks = [self.tokenizer.decode([tid]) for tid in v_ids]
                if not any(self._is_clean_ascii_identifier(t) for t in v_toks[:3]):
                    continue

                if top_key_ids is not None:
                    raw_k_ids = top_key_ids[i].tolist()
                    raw_k_toks = [self.tokenizer.decode([tid]) for tid in raw_k_ids]
                    clean_k_toks = [
                        t for t in raw_k_toks if self._is_clean_ascii_identifier(t)
                    ][:top_k_tokens]
                    if not clean_k_toks:
                        clean_k_toks = raw_k_toks[:top_k_tokens]
                else:
                    clean_k_toks = ["<unk>"]

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
        """Clamps/steers residual stream along down_proj[:, neuron_idx] at layer_idx

        during early forward passes, forcing model to decompress the training basin.
        """
        model_body = getattr(self.model, "model", self.model)
        target_mlp = model_body.layers[layer_idx].mlp

        value_vec = (
            target_mlp.down_proj.weight.detach()[:, neuron_idx]
            .to(torch.float32)
            .to(self.device)
        )
        unit_value_vec = F.normalize(value_vec, p=2, dim=0)

        step_counter = {"step": 0}

        def steering_hook(_module, _inputs, output):
            if step_counter["step"] < steering_steps:
                out_cloned = output.clone()
                # Inject directional impulse into final token position
                vec_dtype = unit_value_vec.to(output.dtype)
                out_cloned[:, -1, :] = out_cloned[:, -1, :] + steering_gain * vec_dtype
                return out_cloned
            return output

        handle = target_mlp.down_proj.register_forward_hook(steering_hook)
        try:
            if hasattr(self.tokenizer, "__call__") and callable(self.tokenizer):
                tok_out = self.tokenizer(seed_prompt, return_tensors="pt")
                if hasattr(tok_out, "input_ids"):
                    input_ids = tok_out.input_ids.to(self.device)
                else:
                    input_ids = tok_out.to(self.device)
            else:
                input_ids = torch.tensor([self.tokenizer.encode(seed_prompt)], device=self.device)

            curr_ids = input_ids
            past_kv = None
            eos_id = getattr(self.tokenizer, "eos_token_id", None)

            for _ in range(seq_len):
                outputs = self.model(
                    input_ids=curr_ids, past_key_values=past_kv, use_cache=True
                )
                past_kv = outputs.past_key_values
                step_counter["step"] += 1

                logits = outputs.logits[:, -1, :].to(torch.float32) / max(temperature, 1e-4)
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.argmax(probs, dim=-1, keepdim=True)

                input_ids = torch.cat([input_ids, next_tok], dim=-1)
                curr_ids = next_tok
                if eos_id is not None and next_tok.item() == eos_id:
                    break
        finally:
            handle.remove()

        text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        raw_bytes = text.encode("utf-8")
        num_bytes = max(len(raw_bytes), 1)

        # Evaluate unsteered cross-entropy vs zlib syntactic entropy
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

# end of file: unplow/src/unplow/steering.py
