# Path: unplow/src/unplow/auditor.py
"""
Unified Model Internal Auditor: Traces Logit Lens projections, FFN norm shifts,
and intermediate layer representations across Transformer depth.
"""

from dataclasses import dataclass, field
import math
from typing import Any, List, Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class LayerAuditReport:
    """Diagnostic profile for a single Transformer layer."""

    layer_idx: int
    top_predicted_token: str
    top_logit_value: float
    residual_l2_norm: float
    ffn_output_l2_norm: float
    top_probability: float = 0.0
    entropy_bits: float = 0.0
    top_k_candidates: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layer_idx": self.layer_idx,
            "top_predicted_token": self.top_predicted_token,
            "top_logit_value": self.top_logit_value,
            "residual_l2_norm": self.residual_l2_norm,
            "ffn_output_l2_norm": self.ffn_output_l2_norm,
            "top_probability": self.top_probability,
            "entropy_bits": self.entropy_bits,
            "top_k_candidates": self.top_k_candidates,
        }


class ModelInternalAuditor:
    """
    Performs a full pass over all Transformer layers using intermediate
    unembedding projections (Logit Lens) to map how text representations
    and predictive confidence evolve across network depth.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    @torch.no_grad()
    def audit_sequence(
        self,
        prompt: str,
        top_k: int = 5,
    ) -> List[LayerAuditReport]:
        """
        Runs a forward pass with output_hidden_states=True and projects each
        layer's final residual token hidden state through the model unembedding
        head (Logit Lens).
        """
        tok_out = self.tokenizer(prompt, return_tensors="pt")
        input_ids = (
            tok_out.input_ids.to(self.device)
            if hasattr(tok_out, "input_ids")
            else tok_out["input_ids"].to(self.device)
        )
        if input_ids.numel() == 0:
            return []

        outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(
                "Model forward pass did not return hidden_states. Ensure output_hidden_states=True is supported."
            )

        # Locate final layer norm and unembedding head
        final_norm = getattr(self.model.model, "norm", getattr(self.model, "norm", None))
        lm_head = getattr(self.model, "lm_head", getattr(self.model, "head", None))
        if lm_head is None:
            raise ValueError("Model does not expose an accessible lm_head or head module.")

        reports: List[LayerAuditReport] = []
        layers = getattr(self.model.model, "layers", getattr(self.model, "layers", []))

        # hidden_states[0] is token embedding; hidden_states[1..N] are transformer layer outputs
        for idx, h_state in enumerate(hidden_states[1:], start=1):
            token_hidden = h_state[:, -1, :]
            head_dtype = getattr(lm_head, "weight", None)
            target_dtype = head_dtype.dtype if head_dtype is not None else token_hidden.dtype

            if final_norm is not None:
                normed = final_norm(token_hidden.to(target_dtype))
            else:
                normed = token_hidden.to(target_dtype)

            logits = lm_head(normed)[0]
            probs = F.softmax(logits, dim=-1)

            top_id = int(torch.argmax(logits).item())
            top_val = float(logits[top_id].item())
            top_prob = float(probs[top_id].item())
            tok_str = self.tokenizer.decode([top_id])

            # Shannon entropy: - sum(p * log2(p))
            clamped_probs = torch.clamp(probs, min=1e-12)
            entropy = float(-torch.sum(clamped_probs * torch.log2(clamped_probs)).item())

            res_norm = float(torch.linalg.vector_norm(token_hidden).item())

            # Top-k candidate tokens
            top_k_indices = torch.topk(logits, k=min(top_k, logits.shape[-1])).indices.tolist()
            top_k_pairs = []
            for tid in top_k_indices:
                t_str = self.tokenizer.decode([tid])
                t_val = float(logits[tid].item())
                top_k_pairs.append((t_str, round(t_val, 3)))

            # Approximate FFN weight scale if layer exists
            ffn_norm = 0.0
            layer_idx = idx - 1
            if layer_idx < len(layers):
                layer_mod = layers[layer_idx]
                mlp = getattr(layer_mod, "mlp", None)
                if mlp is not None:
                    down_proj = getattr(mlp, "down_proj", None)
                    if down_proj is not None and hasattr(down_proj, "weight"):
                        ffn_norm = float(torch.linalg.vector_norm(down_proj.weight).item())

            reports.append(
                LayerAuditReport(
                    layer_idx=layer_idx,
                    top_predicted_token=tok_str,
                    top_logit_value=round(top_val, 3),
                    residual_l2_norm=round(res_norm, 3),
                    ffn_output_l2_norm=round(ffn_norm, 3),
                    top_probability=round(top_prob, 4),
                    entropy_bits=round(entropy, 3),
                    top_k_candidates=top_k_pairs,
                )
            )

        return reports

    def format_audit_table(self, reports: List[LayerAuditReport], prompt: str) -> str:
        """Renders an ASCII diagnostic table of the layer audit report."""
        hdr = (
            f"{'Layer':<6} | {'Top Predicted':<16} | {'Logit':<7} | {'Prob':<7} | "
            f"{'Entropy':<7} | {'Res Norm':<9} | {'FFN Norm':<9}"
        )
        sep = "-" * len(hdr)
        lines = [
            "=== Layer-Wise Internal Information Audit ===",
            f"Prompt: {repr(prompt)}",
            hdr,
            sep,
        ]
        for r in reports:
            clean_tok = repr(r.top_predicted_token)
            if len(clean_tok) > 16:
                clean_tok = clean_tok[:13] + "..."
            row = (
                f"{r.layer_idx:<6} | {clean_tok:<16} | {r.top_logit_value:<7.3f} | "
                f"{r.top_probability:<7.4f} | {r.entropy_bits:<7.3f} | "
                f"{r.residual_l2_norm:<9.3f} | {r.ffn_output_l2_norm:<9.3f}"
            )
            lines.append(row)
        return "\n".join(lines)


# end of file: unplow/src/unplow/auditor.py
