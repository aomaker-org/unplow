# Path: unplow/inbox/gemweb010.md
# gemweb010.md - Replow Evaluation Harness Specification
# Max Column: 80 Columns

To evaluate the effectiveness of the `replow` closed-loop pipeline, we introduce
**`src/unplow/eval.py`**—an automated evaluation harness that combines
**functional correctness testing** (sandboxed execution against unit test
assertions) with **per-token perplexity and cross-entropy tracking**
($\Delta \mathcal{L}_{\theta}$) before and after local gradient distillation.

---

## 1. Architectural Design of the Evaluation Harness

The harness performs two distinct evaluation passes on any refactored artifact
produced by `replow`:

1. **Functional Correctness Sandbox (`FunctionalCorrectnessRunner`):**
   Dynamically injects the refactored code and associated unit tests into an
   isolated Python namespace using `exec()`, enforcing execution timeout
   guards and capturing assertion failures or runtime exceptions.
2. **Perplexity & Surprisal Shift Tracker (`PerplexityTracker`):**
   Measures the exact cross-entropy loss ($\mathcal{L}_{\theta}$) and
   perplexity ($\mathcal{P} = 2^{\mathcal{L}_{\theta}}$) of the model over the
   evaluation corpus before and after training steps, quantifying how much more
   "compressible" and predictable the code has become to the weights $\theta$.

---

## 2. Implementation: `src/unplow/eval.py`

```python
# src/unplow/eval.py
"""
Replow Evaluation Harness: Functional correctness testing via sandboxed
execution and pre/post-training perplexity shift measurement (Delta L_theta).
"""

from dataclasses import dataclass, field
import multiprocessing
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FunctionalTestResult:
    test_name: str
    passed: bool
    execution_time_sec: float
    error_message: Optional[str] = None


@dataclass
class PerplexityReport:
    pre_training_loss: float
    pre_training_perplexity: float
    post_training_loss: float
    post_training_perplexity: float
    delta_loss: float
    perplexity_reduction_pct: float


@dataclass
class ReplowEvaluationSummary:
    artifact_name: str
    functional_tests_passed: int
    functional_tests_total: int
    test_results: List[FunctionalTestResult]
    perplexity_report: PerplexityReport
    overall_status: str


class FunctionalCorrectnessRunner:
    """
    Executes refactored Python code against test assertions in an isolated
    process namespace with a strict execution timeout.
    """

    @staticmethod
    def _run_test_worker(
        code_str: str, test_code_str: str, queue: multiprocessing.Queue
    ):
        local_scope: Dict[str, Any] = {}
        try:
            # Execute refactored code definition
            exec(code_str, local_scope, local_scope)
            # Execute unit test assertions against local scope
            exec(test_code_str, local_scope, local_scope)
            queue.put((True, None))
        except Exception as e:
            queue.put((False, str(e)))

    def run_tests(
        self,
        code_str: str,
        test_cases: List[Tuple[str, str]],
        timeout_sec: float = 3.0,
    ) -> List[FunctionalTestResult]:
        """
        Runs a suite of (test_name, test_assertion_code) pairs against
        `code_str`.
        """
        results: List[FunctionalTestResult] = []

        for test_name, test_assertion in test_cases:
            queue: multiprocessing.Queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=self._run_test_worker,
                args=(code_str, test_assertion, queue),
            )
            start_t = time.perf_counter()
            p.start()
            p.join(timeout=timeout_sec)

            elapsed = time.perf_counter() - start_t

            if p.is_alive():
                p.terminate()
                p.join()
                results.append(
                    FunctionalTestResult(
                        test_name=test_name,
                        passed=False,
                        execution_time_sec=round(elapsed, 4),
                        error_message="Execution Timeout Exceeded",
                    )
                )
                continue

            if not queue.empty():
                success, err_msg = queue.get()
                results.append(
                    FunctionalTestResult(
                        test_name=test_name,
                        passed=success,
                        execution_time_sec=round(elapsed, 4),
                        error_message=err_msg,
                    )
                )
            else:
                results.append(
                    FunctionalTestResult(
                        test_name=test_name,
                        passed=False,
                        execution_time_sec=round(elapsed, 4),
                        error_message=(
                            "Worker process terminated without output"
                        ),
                    )
                )

        return results


class PerplexityTracker:
    """
    Measures cross-entropy loss and perplexity over a validation corpus
    before and after model distillation/training steps.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def compute_corpus_loss(
        self, corpus_texts: List[str]
    ) -> Tuple[float, float]:
        """Computes average cross-entropy loss (nats) and perplexity."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for text in corpus_texts:
            input_ids = self.tokenizer(
                text, return_tensors="pt"
            ).input_ids.to(self.device)
            if input_ids.shape[1] < 2:
                continue
            outputs = self.model(input_ids)
            logits = outputs.logits[:, :-1, :].contiguous().to(torch.float32)
            labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += labels.numel()

        if total_tokens == 0:
            return 0.0, 1.0

        mean_loss = total_loss / total_tokens
        perplexity = float(np.exp(mean_loss))
        return round(mean_loss, 4), round(perplexity, 4)

    def evaluate_shift(
        self, corpus_texts: List[str], train_callback=None, epochs: int = 1
    ) -> PerplexityReport:
        """
        Measures baseline perplexity, executes an optional training callback
        (distillation step), and measures post-training perplexity shift.
        """
        pre_loss, pre_ppl = self.compute_corpus_loss(corpus_texts)

        if train_callback is not None:
            self.model.train()
            for _ in range(epochs):
                train_callback(self.model, self.tokenizer, corpus_texts)

        post_loss, post_ppl = self.compute_corpus_loss(corpus_texts)
        delta_loss = round(post_loss - pre_loss, 4)
        ppl_reduction = round(
            ((pre_ppl - post_ppl) / max(pre_ppl, 1e-6)) * 100.0, 2
        )

        return PerplexityReport(
            pre_training_loss=pre_loss,
            pre_training_perplexity=pre_ppl,
            post_training_loss=post_loss,
            post_training_perplexity=post_ppl,
            delta_loss=delta_loss,
            perplexity_reduction_pct=ppl_reduction,
        )
# src/unplow/eval.py
```

---

## 3. Unit Test Suite: `tests/test_eval.py`

```python
# tests/test_eval.py
"""
Unit tests for FunctionalCorrectnessRunner and PerplexityTracker.
"""

from unplow.eval import FunctionalCorrectnessRunner, PerplexityTracker
from unplow.models import DeterministicMockCausalLM, DeterministicMockTokenizer


def test_functional_correctness_runner():
    runner = FunctionalCorrectnessRunner()
    code = "def add(a, b):\n    return a + b\n"
    tests = [
        ("test_positive", "assert add(2, 3) == 5"),
        ("test_negative", "assert add(-1, 1) == 0"),
        ("test_failure", "assert add(1, 1) == 3"),  # Expected fail
    ]
    results = runner.run_tests(code, tests)
    assert len(results) == 3
    assert results[0].passed is True
    assert results[1].passed is True
    assert results[2].passed is False


def test_perplexity_tracker():
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    tracker = PerplexityTracker(model, tokenizer)

    corpus = ["def fib(n): return n", "class Pipeline: pass"]
    report = tracker.evaluate_shift(corpus, train_callback=None)

    assert report.pre_training_loss > 0.0
    assert report.pre_training_perplexity > 0.0
    assert report.delta_loss == 0.0
# tests/test_eval.py
```

---

## 4. CLI Wiring & Execution

1. **Add `unplow eval` Subcommand in `src/unplow/cli.py`:**
```python
# In cli.py parser setup:
eval_parser = subparsers.add_parser(
    "eval", help="Run functional tests and perplexity evaluation."
)
eval_parser.add_argument(
    "--artifact", required=True, help="Path to refactored .unp or .py file."
)
eval_parser.add_argument(
    "--test-suite", required=True,
    help="Python file containing test assertions.",
)
```

2. **Execute via Makefile:**
```bash
uv run unplow eval \
  --artifact workspace/targets/fib_replow.py \
  --test-suite tests/data/fib_tests.py
```

<!-- end of file: unplow/inbox/gemweb010.md -->
