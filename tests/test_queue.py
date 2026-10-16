# Path: unplow/tests/test_queue.py
"""Unit tests for unplow atomic file-backed task queue."""

from pathlib import Path
import pytest

from unplow.queue import FileTaskQueue, Task, TaskStatus


def test_enqueue_and_claim(tmp_path: Path):
    queue = FileTaskQueue(base_dir=str(tmp_path / "queue"))
    task = queue.enqueue(action="probe", payload="", extra={"samples": 2})

    assert task.task_id.startswith("task-")
    assert task.status == TaskStatus.INCOMING.value

    counts = queue.get_counts()
    assert counts["incoming"] == 1
    assert counts["processing"] == 0

    claimed = queue.claim_next_task()
    assert claimed is not None
    claimed_task, proc_path = claimed

    assert claimed_task.task_id == task.task_id
    assert claimed_task.status == TaskStatus.PROCESSING.value
    assert proc_path.exists()

    counts_after = queue.get_counts()
    assert counts_after["incoming"] == 0
    assert counts_after["processing"] == 1


def test_complete_and_fail_task(tmp_path: Path):
    queue = FileTaskQueue(base_dir=str(tmp_path / "queue"))
    task1 = queue.enqueue(action="compress", payload="hello world")
    task2 = queue.enqueue(action="decompress", payload="invalid")

    # Claim & complete task 1
    claimed1 = queue.claim_next_task()
    assert claimed1 is not None
    t1, p1 = claimed1
    queue.complete_task(t1, p1, result="OK", model_used="test_model")

    # Claim & fail task 2
    claimed2 = queue.claim_next_task()
    assert claimed2 is not None
    t2, p2 = claimed2
    queue.fail_task(t2, p2, error_msg="Syntax error")

    counts = queue.get_counts()
    assert counts["incoming"] == 0
    assert counts["processing"] == 0
    assert counts["done"] == 1
    assert counts["failed"] == 1

# end of file: unplow/tests/test_queue.py
