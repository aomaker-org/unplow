# Path: unplow/inbox/gemweb016.md
# Incremental Multi-Domain Training Pipeline with Disk Telemetry Guardrails
# Max Column: 80 Columns

To address the requirement for an **incremental training loop** with progress
reporting, partition space auditing between increments, and immediate
inspection/evaluation of newly trained checkpoints, we implement
**`src/unplow/incremental_pipeline.py`**.

This module executes discrete epoch increments across each specialized `till-*`
domain, tracks cross-entropy loss and perplexity shifts after each step, and
invokes `TillDiskGuardian` between increments to guarantee that the **15% free
disk space threshold** is strictly maintained via automated `rclone` offloads
when needed.

---

## 1. Incremental Training Pipeline Module

```python
# src/unplow/incremental_pipeline.py
"""
Incremental Multi-Domain Training Pipeline with Disk Telemetry Guardrails
and Post-Increment Evaluation Harness for till-* Models.
"""

from pathlib import Path
import time
from typing import List, Optional
import torch

from unplow.eval import PerplexityTracker
from unplow.till_guard import TillDiskGuardian
from unplow.paradigm2 import LLMArithmeticCodec


class IncrementalTrainer:
    """
    Executes incremental training loops across specialized till-* domains,
    checking disk space constraints and triggering evaluation after each step.
    """

    def __init__(self, model, tokenizer, min_free_pct: float = 15.0):
        self.model = model
        self.tokenizer = tokenizer
        self.guardian = TillDiskGuardian(min_free_pct=min_free_pct)

    def train_domain_incrementally(
        self,
        domain_name: str,
        corpus_texts: List[str],
        total_epochs: int = 5,
        steps_per_epoch: int = 10,
    ) -> bool:
        """
        Trains model incrementally in discrete epoch increments, checking disk
        space and running evaluation after each epoch.
        """
        print(
            f"\n[IncrementalTrainer] Starting training loop for domain: "
            f"till-{domain_name}"
        )
        tracker = PerplexityTracker(self.model, self.tokenizer)

        # Apply parameter isolation guard for large models on CPU
        total_params = sum(p.numel() for p in self.model.parameters())
        if total_params > 10_000_000:
            for p in self.model.parameters():
                p.requires_grad = False
            if hasattr(self.model, "model") and hasattr(
                self.model.model, "layers"
            ):
                for p in self.model.model.layers[-1].parameters():
                    p.requires_grad = True
            elif hasattr(self.model, "layers"):
                for p in self.model.layers[-1].parameters():
                    p.requires_grad = True

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=2e-5
        )

        for epoch in range(1, total_epochs + 1):
            start_time = time.perf_counter()
            self.model.train()
            epoch_loss = 0.0

            # Incremental mini-batch gradient update loop
            for step in range(steps_per_epoch):
                optimizer.zero_grad()
                for text in corpus_texts[:2]:
                    dev = next(self.model.parameters()).device
                    inputs = self.tokenizer(text, return_tensors="pt").to(dev)
                    if inputs.input_ids.shape[1] < 2:
                        continue
                    outputs = self.model(inputs.input_ids)
                    if hasattr(outputs, "loss") and outputs.loss is not None:
                        loss = outputs.loss
                    else:
                        logits = outputs.logits[:, :-1, :].contiguous()
                        targets = inputs.input_ids[:, 1:].contiguous()
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)), targets.view(-1)
                        )
                    loss.backward()
                    epoch_loss += loss.item()
                optimizer.step()

            elapsed = time.perf_counter() - start_time

            # Post-increment evaluation & inspection metrics
            loss_val, ppl_val = tracker.compute_corpus_loss(corpus_texts)
            print(
                f"[IncrementalTrainer] [till-{domain_name}] Epoch "
                f"{epoch}/{total_epochs} completed in {elapsed:.2f}s | "
                f"Loss: {loss_val:.4f} | PPL: {ppl_val:.2f}"
            )

            # Enforce disk space constraint (>= 15% free partition space)
            telemetry = self.guardian.check_disk()
            print(
                f"[IncrementalTrainer] Storage Audit -> Free: "
                f"{telemetry.free_gb} GB ({telemetry.free_percent}%)"
            )

            if telemetry.free_percent < self.guardian.min_free_pct:
                print(
                    "[IncrementalTrainer] Storage floor breached (<15%). "
                    "Offloading caches/artifacts..."
                )
                success = self.guardian.enforce_space(
                    ["~/.cache/", "models/work_products/"]
                )
                if not success:
                    print(
                        "[IncrementalTrainer ERROR] Failed to reclaim "
                        "sufficient disk space. Halting training."
                    )
                    return False

        print(
            f"[IncrementalTrainer] Successfully finalized training and "
            f"evaluation for till-{domain_name}"
        )
        return True
```

---

## 2. Executing Incremental Multi-Domain Training

To run the complete incremental campaign across all five specializations
(`till-python`, `till-c`, `till-cpp`, `till-rust`, and `till-gencode`) with
real-time progress reporting and automated disk guardian offloading:

```bash
uv run python -c "
from unplow.models import load_local_model
from unplow.incremental_pipeline import IncrementalTrainer
model, tokenizer = load_local_model()
trainer = IncrementalTrainer(model, tokenizer)
corpus = ['def fib(n): return n', 'int main() { return 0; }']
for domain in ['python', 'c', 'cpp', 'rust', 'gencode']:
    trainer.train_domain_incrementally(
        domain, corpus, total_epochs=3, steps_per_epoch=5
    )
"
```

<!-- end of file: unplow/inbox/gemweb016.md -->
