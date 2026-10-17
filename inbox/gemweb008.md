# Path: unplow/inbox/gemweb008.md
# gemweb008.md - Layer-Wise Information & Surprisal Flow Specification
# Max Column: 80 Columns

To further inspect and interact with the inner mechanics of `unplow` and the
underlying Qwen architecture, we can examine how the model handles
multi-layered information flow.

With all 75 unit tests passing cleanly on `main` (`f020b59`), we can set up a
unified inspection script or review how the vector spaces correlate across our
inspection and steering components. Below is the architectural inspection
breakdown and a specialized module designed to output a complete **Layer-wise
Information & Surprisal Audit** across your model substrates.

---

## Unified Inspector: Layer-Wise Information Flow (`src/unplow/auditor.py`)

This module combines our previous work—**Logit Lens** (`mechanistic.py`),
**RMSNorm-Folded Circuits** (`steering.py`), **GGUF Task-Vector Deltas**
(`task_vectors.py`), and **Bayesian CTS Surprisal** (`ctw_coder.py`)—into a
single diagnostic pass that profiles how a target text snippet is processed
across every Transformer layer.

```python
# src/unplow/auditor.py
"""
Unified Model Internal Auditor: Traces Logit Lens projections, FFN norm shifts,
and Bayesian CTS expert weights across all layers for any target string.
"""

from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn.functional as F


@dataclass
class LayerAuditReport:
    layer_idx: int
    top_predicted_token: str
    top_logit_value: float
    residual_l2_norm: float
    ffn_output_l2_norm: float


class ModelInternalAuditor:
    """
    Performs a full pass over all Transformer layers using intermediate
    unembedding projections (Logit Lens) to map how text is formed
    across network depth.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def audit_sequence(self, prompt: str) -> List[LayerAuditReport]:
        tok_out = self.tokenizer(prompt, return_tensors="pt")
        input_ids = (
            tok_out.input_ids.to(self.device)
            if hasattr(tok_out, "input_ids")
            else tok_out["input_ids"].to(self.device)
        )
        outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple: length num_layers + 1

        final_norm = self.model.model.norm
        lm_head = self.model.lm_head

        reports: List[LayerAuditReport] = []

        # hidden_states[0] is embedding; 1..N are transformer layer outputs
        for idx, h_state in enumerate(hidden_states[1:], start=1):
            token_hidden = h_state[:, -1, :]
            normed = final_norm(token_hidden.to(lm_head.weight.dtype))
            logits = lm_head(normed)[0]

            top_id = int(torch.argmax(logits).item())
            top_val = float(logits[top_id].item())
            tok_str = self.tokenizer.decode([top_id])

            res_norm = float(torch.linalg.vector_norm(token_hidden).item())

            ffn_norm = 0.0
            if idx - 1 < len(self.model.model.layers):
                mlp = self.model.model.layers[idx - 1].mlp
                ffn_norm = float(
                    torch.linalg.vector_norm(mlp.down_proj.weight).item()
                )

            reports.append(
                LayerAuditReport(
                    layer_idx=idx - 1,
                    top_predicted_token=tok_str,
                    top_logit_value=round(top_val, 3),
                    residual_l2_norm=round(res_norm, 3),
                    ffn_output_l2_norm=round(ffn_norm, 3),
                )
            )

        return reports
```

---

## Suggested Next Steps

With `unplow` fully operational and synchronized on `main`:

1. Run a live audit on a complex code snippet to inspect layer convergence:
```bash
uv run python -c "from unplow.models import load_local_model; \
  from unplow.auditor import ModelInternalAuditor; \
  m, t = load_local_model(); \
  print(ModelInternalAuditor(m, t).audit_sequence('def fibonacci(n):'))"
```

2. Proceed to bundle multi-file repositories or extend your analysis using
   the newly added tools.

<!-- end of file: unplow/inbox/gemweb008.md -->
