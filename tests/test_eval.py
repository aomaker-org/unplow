# Path: unplow/tests/test_eval.py
"""Unit tests for FunctionalCorrectnessRunner and PerplexityTracker."""

import tempfile
from pathlib import Path

from unplow.eval import (
    FunctionalCorrectnessRunner,
    PerplexityTracker,
    evaluate_artifact,
    format_summary_table,
)
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


def test_functional_correctness_timeout():
    runner = FunctionalCorrectnessRunner()
    code = "def slow():\n    import time\n    time.sleep(2)\n    return 42\n"
    tests = [
        ("test_slow_timeout", "assert slow() == 42"),
    ]
    results = runner.run_tests(code, tests, timeout_sec=0.2)
    assert len(results) == 1
    assert results[0].passed is False
    assert "Timeout" in (results[0].error_message or "")


def test_load_test_cases_from_file_and_eval():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        code_file = tmp_p / "solution.py"
        code_file.write_text("def mul(x, y):\n    return x * y\n", encoding="utf-8")

        test_file = tmp_p / "tests.py"
        test_file.write_text(
            "def test_mul_pos():\n    assert mul(3, 4) == 12\n\n"
            "def test_mul_zero():\n    assert mul(5, 0) == 0\n",
            encoding="utf-8",
        )

        cases = FunctionalCorrectnessRunner.load_test_cases_from_file(str(test_file))
        assert len(cases) == 2
        assert cases[0][0] == "test_mul_pos"
        assert cases[1][0] == "test_mul_zero"

        summary = evaluate_artifact(
            artifact_path=str(code_file),
            test_suite_path=str(test_file),
        )
        assert summary.functional_tests_passed == 2
        assert summary.functional_tests_total == 2
        assert summary.overall_status == "PASSED"

        data = summary.to_dict()
        assert data["overall_status"] == "PASSED"
        assert data["pass_rate_pct"] == 100.0

        table = format_summary_table(summary)
        assert "Replow Evaluation Harness" in table
        assert "PASS" in table


# end of file: unplow/tests/test_eval.py
