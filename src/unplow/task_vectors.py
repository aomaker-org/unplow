# Path: unplow/src/unplow/task_vectors.py
"""Cross-Substrate Task-Vector Forensics & Differential SVD Steering.

(Ilharco et al., 2023 [Task Arithmetic]; Nasr et al., 2023; Meng et al., 2022).
Bridges memory-mapped GGUF weights and live Safetensors models to extract and
steer along inter-model specialization subspaces without loading two models.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import zlib
import numpy as np
import torch
import torch.nn.functional as F

from unplow.gguf_inspector import GGUFBinaryInspector


@dataclass
class TaskVectorSpectrumReport:
    """Spectral decomposition and token projections of a layer task vector."""

    layer_idx: int
    gguf_tensor_name: str
    frobenius_delta_norm: float
    cosine_similarity: float
    top_singular_value: float
    explained_variance_ratio: float
    promoted_tokens: List[str]
    suppressed_tokens: List[str]
    top_divergent_neurons: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskVectorSteeredRecord:
    """Diagnostics and text extracted by steering along a task vector."""

    layer_idx: int
    gguf_source: str
    steering_gain: float
    promoted_tokens: List[str]
    extracted_text: str
    model_bpb: float
    zlib_bpb: float
    memorization_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrossSubstrateTaskVectorEngine:
    """Computes the MLP weight delta Delta_W = W_GGUF - W_Safetensors for a layer.

    Decodes vocabulary tokens promoted/suppressed by the principal singular vector
    of Delta_W, and steers generation along that vector without dual-model RAM overhead.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @staticmethod
    def _align_gguf_matrix(gguf_mat: np.ndarray, target_shape: Tuple[int, int]) -> torch.Tensor:
        """Aligns dequantized 2D GGUF array to PyTorch [d_model, d_ffn] layout."""
        if gguf_mat.shape == target_shape:
            aligned = gguf_mat
        elif len(gguf_mat.shape) == 2 and gguf_mat.T.shape == target_shape:
            aligned = gguf_mat.T
        else:
            aligned = gguf_mat.reshape(target_shape)
        return torch.from_numpy(aligned.copy()).to(torch.float32)

    @torch.no_grad()
    def compute_layer_task_vector(
        self,
        gguf_path: str | Path,
        layer_idx: int,
        top_k_tokens: int = 6,
        top_k_neurons: int = 6,
    ) -> Tuple[torch.Tensor, TaskVectorSpectrumReport]:
        """Extracts the principal task direction Delta_W = W_GGUF - W_ref via SVD."""
        inspector = GGUFBinaryInspector(gguf_path)
        summary = inspector.parse_header()
        tensor_name = f"blk.{layer_idx}.ffn_down.weight"
        if tensor_name not in summary.tensors:
            matching = [k for k in summary.tensors if f"blk.{layer_idx}." in k and "down" in k]
            if matching:
                tensor_name = matching[0]

        gguf_np = inspector.load_dequantized_tensor(tensor_name)

        ref_w = (
            self.model.model.layers[layer_idx]
            .mlp.down_proj.weight.detach()
            .to(torch.float32)
            .cpu()
        )
        gguf_w = self._align_gguf_matrix(gguf_np, tuple(ref_w.shape))

        # Task delta matrix: positive direction points toward the GGUF substrate
        delta_w = gguf_w - ref_w
        frob_delta = float(torch.linalg.norm(delta_w).item())
        denom = max(
            float(torch.linalg.norm(gguf_w).item() * torch.linalg.norm(ref_w).item()),
            1e-12,
        )
        cos_sim = float(torch.sum(gguf_w * ref_w).item() / denom)

        # Rank individual FFN neurons by column L2 shift
        col_norms = torch.linalg.vector_norm(delta_w, ord=2, dim=0)
        k_neurons = min(top_k_neurons, col_norms.shape[0])
        _, top_neuron_ids = torch.topk(col_norms, k=k_neurons)

        # Align top divergent columns with the dominant neuron before SVD
        sub_delta = delta_w[:, top_neuron_ids]
        anchor = sub_delta[:, 0:1]
        signs = torch.where(torch.sum(sub_delta * anchor, dim=0, keepdim=True) >= 0, 1.0, -1.0)
        u, s, _ = torch.linalg.svd(sub_delta * signs, full_matrices=False)

        principal_vec = F.normalize(u[:, 0], p=2, dim=0).to(self.device)
        sv_top = float(s[0].item())
        ev_ratio = float((s[0] ** 2) / torch.clamp(torch.sum(s**2), min=1e-12))

        # Project principal task vector through final RMSNorm and lm_head
        final_norm = self.model.model.norm
        lm_head = getattr(self.model, "lm_head", getattr(self.model, "head", None))
        head_dtype = next(lm_head.parameters()).dtype
        normed_dir = final_norm(principal_vec.unsqueeze(0).to(head_dtype))
        logits = lm_head(normed_dir).to(torch.float32)[0]

        k_tok = min(top_k_tokens, logits.shape[0])
        _, prom_ids = torch.topk(logits, k=k_tok, largest=True)
        _, supp_ids = torch.topk(logits, k=k_tok, largest=False)

        promoted = [self.tokenizer.decode([tid]) for tid in prom_ids.tolist()]
        suppressed = [self.tokenizer.decode([tid]) for tid in supp_ids.tolist()]

        report = TaskVectorSpectrumReport(
            layer_idx=layer_idx,
            gguf_tensor_name=tensor_name,
            frobenius_delta_norm=round(frob_delta, 4),
            cosine_similarity=round(cos_sim, 6),
            top_singular_value=round(sv_top, 4),
            explained_variance_ratio=round(ev_ratio, 4),
            promoted_tokens=promoted,
            suppressed_tokens=suppressed,
            top_divergent_neurons=[int(i) for i in top_neuron_ids.tolist()],
        )
        return principal_vec, report

    @torch.no_grad()
    def steer_with_gguf_task_vector(
        self,
        gguf_path: str | Path,
        layer_idx: int,
        seed_prompt: str = "def ",
        seq_len: int = 48,
        steering_gain: float = 4.5,
        steering_steps: int = 16,
        temperature: float = 0.25,
    ) -> TaskVectorSteeredRecord:
        """Extracts the cross-substrate task vector and steers the live model residual stream."""
        principal_vec, report = self.compute_layer_task_vector(gguf_path, layer_idx)
        model_dtype = getattr(self.model, "dtype", next(self.model.parameters()).dtype)
        principal_vec = principal_vec.to(model_dtype)
        target_mlp = self.model.model.layers[layer_idx].mlp
        step_counter = {"step": 0}

        def task_hook(_module, _inputs, output):
            if step_counter["step"] < steering_steps:
                out_cloned = output.clone()
                out_cloned[:, -1, :] = out_cloned[:, -1, :] + steering_gain * principal_vec
                return out_cloned
            return output

        handle = target_mlp.down_proj.register_forward_hook(task_hook)
        try:
            tok_out = self.tokenizer(seed_prompt, return_tensors="pt")
            if hasattr(tok_out, "input_ids"):
                input_ids = tok_out.input_ids.to(self.device)
            else:
                input_ids = tok_out["input_ids"].to(self.device)
            curr_ids = input_ids
            past_kv = None

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
                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_id is not None and next_tok.item() == eos_id:
                    break
        finally:
            handle.remove()

        text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
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

        return TaskVectorSteeredRecord(
            layer_idx=layer_idx,
            gguf_source=str(Path(gguf_path).name),
            steering_gain=steering_gain,
            promoted_tokens=report.promoted_tokens,
            extracted_text=text,
            model_bpb=round(model_bpb, 3),
            zlib_bpb=round(zlib_bpb, 3),
            memorization_score=round(mem_score, 3),
        )

# end of file: unplow/src/unplow/task_vectors.py
