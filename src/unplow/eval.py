# Path: unplow/src/unplow/eval.py
"""Replow Evaluation Harness: Functional correctness sandbox and perplexity shift tracker.

Combines sandboxed execution against unit test assertions with per-token
perplexity and cross-entropy tracking (Delta L_theta) before and after
local gradient distillation.
"""

import ast
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .paradigm2 import LLMArithmeticCodec


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
    perplexity_report: Optional[PerplexityReport] = None
    overall_status: str = "PASSED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation summary to a structured serializable dictionary."""
        return {
            "artifact_name": self.artifact_name,
            "overall_status": self.overall_status,
            "functional_tests_passed": self.functional_tests_passed,
            "functional_tests_total": self.functional_tests_total,
            "pass_rate_pct": (
                round(
                    (self.functional_tests_passed / max(self.functional_tests_total, 1))
                    * 100.0,
                    2,
                )
            ),
            "test_results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "execution_time_sec": r.execution_time_sec,
                    "error_message": r.error_message,
                }
                for r in self.test_results
            ],
            "perplexity_report": (
                {
                    "pre_training_loss": self.perplexity_report.pre_training_loss,
                    "pre_training_perplexity": self.perplexity_report.pre_training_perplexity,
                    "post_training_loss": self.perplexity_report.post_training_loss,
                    "post_training_perplexity": self.perplexity_report.post_training_perplexity,
                    "delta_loss": self.perplexity_report.delta_loss,
                    "perplexity_reduction_pct": self.perplexity_report.perplexity_reduction_pct,
                }
                if self.perplexity_report
                else None
            ),
        }


class FunctionalCorrectnessRunner:
    """Executes refactored Python code against test assertions in an isolated process namespace."""

    @staticmethod
    def _run_test_worker(
        code_str: str, test_code_str: str, queue: Any
    ) -> None:
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
        """Runs a suite of (test_name, test_assertion_code) pairs against code_str."""
        results: List[FunctionalTestResult] = []
        ctx = multiprocessing.get_context()

        for test_name, test_assertion in test_cases:
            queue = ctx.Queue()
            p = ctx.Process(
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
                try:
                    success, err_msg = queue.get_nowait()
                    results.append(
                        FunctionalTestResult(
                            test_name=test_name,
                            passed=success,
                            execution_time_sec=round(elapsed, 4),
                            error_message=err_msg,
                        )
                    )
                except Exception as e:
                    results.append(
                        FunctionalTestResult(
                            test_name=test_name,
                            passed=False,
                            execution_time_sec=round(elapsed, 4),
                            error_message=f"Queue read error: {e}",
                        )
                    )
            else:
                results.append(
                    FunctionalTestResult(
                        test_name=test_name,
                        passed=False,
                        execution_time_sec=round(elapsed, 4),
                        error_message="Worker process terminated without output",
                    )
                )

        return results

    @staticmethod
    def load_test_cases_from_file(suite_path: str) -> List[Tuple[str, str]]:
        """Extract test cases from a Python file or assertion script."""
        p = Path(suite_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Test suite file not found: {suite_path}")

        content = p.read_text(encoding="utf-8")
        test_cases: List[Tuple[str, str]] = []

        # Check for explicit TEST_CASES variable in module
        try:
            scope: Dict[str, Any] = {}
            exec(compile(content, str(p), "exec"), scope, scope)
            if "TEST_CASES" in scope and isinstance(scope["TEST_CASES"], list):
                return scope["TEST_CASES"]
            if "test_cases" in scope and isinstance(scope["test_cases"], list):
                return scope["test_cases"]
        except Exception:
            pass

        # Try AST parsing for test_* functions
        try:
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    fn_code = ast.unparse(node)
                    invocation = f"{fn_code}\n{node.name}()"
                    test_cases.append((node.name, invocation))
        except Exception:
            pass

        if test_cases:
            return test_cases

        # Fallback: extract assert lines
        for idx, line in enumerate(content.splitlines(), 1):
            s = line.strip()
            if s.startswith("assert ") or s.startswith("assert("):
                test_cases.append((f"assert_line_{idx}", s))

        if not test_cases:
            test_cases.append((p.stem, content))

        return test_cases


class PerplexityTracker:
    """Measures cross-entropy loss and perplexity over a validation corpus before and after training."""

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def compute_corpus_loss(self, corpus_texts: List[str]) -> Tuple[float, float]:
        """Computes average cross-entropy loss (nats) and perplexity over corpus."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for text in corpus_texts:
            encoded = self.tokenizer(text, return_tensors="pt")
            input_ids = encoded.input_ids.to(self.device)
            if input_ids.shape[1] < 2:
                continue
            outputs = self.model(input_ids)
            logits = outputs.logits[:, :-1, :].contiguous().to(torch.float32)
            labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += labels.numel()

        if total_tokens == 0:
            return 0.0, 1.0

        mean_loss = total_loss / total_tokens
        perplexity = float(np.exp(mean_loss))
        return round(mean_loss, 4), round(perplexity, 4)

    def evaluate_shift(
        self,
        corpus_texts: List[str],
        train_callback: Optional[Any] = None,
        epochs: int = 1,
    ) -> PerplexityReport:
        """Measures baseline perplexity, executes optional distillation step, and measures post-training shift."""
        pre_loss, pre_ppl = self.compute_corpus_loss(corpus_texts)

        if train_callback is not None:
            self.model.train()
            for _ in range(epochs):
                train_callback(self.model, self.tokenizer, corpus_texts)

        post_loss, post_ppl = self.compute_corpus_loss(corpus_texts)
        delta_loss = round(post_loss - pre_loss, 4)
        ppl_reduction = round(((pre_ppl - post_ppl) / max(pre_ppl, 1e-6)) * 100.0, 2)

        return PerplexityReport(
            pre_training_loss=pre_loss,
            pre_training_perplexity=pre_ppl,
            post_training_loss=post_loss,
            post_training_perplexity=post_ppl,
            delta_loss=delta_loss,
            perplexity_reduction_pct=ppl_reduction,
        )


def evaluate_artifact(
    artifact_path: str,
    test_suite_path: Optional[str] = None,
    test_cases: Optional[List[Tuple[str, str]]] = None,
    corpus_texts: Optional[List[str]] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    timeout_sec: float = 3.0,
) -> ReplowEvaluationSummary:
    """Evaluate a refactored code artifact (.py or .unp) for functional correctness and perplexity shift."""
    p = Path(artifact_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    # Determine code string from .unp or .py
    if p.suffix.lower() == ".unp":
        container_bytes = p.read_bytes()
        codec = LLMArithmeticCodec(model=model, tokenizer=tokenizer)
        code_str, _, _ = codec.decompress_from_container(container_bytes)
    else:
        code_str = p.read_text(encoding="utf-8")

    # Resolve test cases
    runner = FunctionalCorrectnessRunner()
    if test_suite_path and not test_cases:
        test_cases = runner.load_test_cases_from_file(test_suite_path)

    test_results: List[FunctionalTestResult] = []
    if test_cases:
        test_results = runner.run_tests(code_str, test_cases, timeout_sec=timeout_sec)

    passed_count = sum(1 for r in test_results if r.passed)
    total_count = len(test_results)
    overall_status = "PASSED" if passed_count == total_count else "FAILED"

    # Perplexity tracking
    ppl_report: Optional[PerplexityReport] = None
    if corpus_texts and model is not None and tokenizer is not None:
        tracker = PerplexityTracker(model, tokenizer)
        ppl_report = tracker.evaluate_shift(corpus_texts)

    return ReplowEvaluationSummary(
        artifact_name=p.name,
        functional_tests_passed=passed_count,
        functional_tests_total=total_count,
        test_results=test_results,
        perplexity_report=ppl_report,
        overall_status=overall_status,
    )


def format_summary_table(summary: ReplowEvaluationSummary) -> str:
    """Format evaluation summary as an ASCII diagnostic report table."""
    lines = [
        "",
        "=" * 78,
        "Replow Evaluation Harness: Functional Correctness & Perplexity Report",
        "=" * 78,
        f"Artifact Name           : {summary.artifact_name}",
        f"Overall Status          : {summary.overall_status}",
        f"Functional Tests Passed : {summary.functional_tests_passed} / {summary.functional_tests_total} "
        f"({(summary.functional_tests_passed / max(summary.functional_tests_total, 1)) * 100.0:.1f}%)",
    ]
    if summary.test_results:
        lines.append("-" * 78)
        lines.append(f"{'Test Name':<32} | {'Status':<8} | {'Time (s)':<10} | {'Details'}")
        lines.append("-" * 78)
        for r in summary.test_results:
            st = "PASS" if r.passed else "FAIL"
            err = r.error_message or ""
            if len(err) > 22:
                err = err[:19] + "..."
            lines.append(f"{r.test_name:<32} | {st:<8} | {r.execution_time_sec:<10.4f} | {err}")
    if summary.perplexity_report:
        p = summary.perplexity_report
        lines.append("-" * 78)
        lines.append(
            f"Pre-Training Loss       : {p.pre_training_loss:.4f} nats "
            f"(PPL: {p.pre_training_perplexity:.2f})"
        )
        lines.append(
            f"Post-Training Loss      : {p.post_training_loss:.4f} nats "
            f"(PPL: {p.post_training_perplexity:.2f})"
        )
        lines.append(f"Delta Loss (nats)       : {p.delta_loss:+.4f}")
        lines.append(f"Perplexity Reduction    : {p.perplexity_reduction_pct:+.2f}%")
    lines.append("=" * 78)
    return "\n".join(lines)


# end of file: unplow/src/unplow/eval.py
