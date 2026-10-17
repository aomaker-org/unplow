# Path: unplow/tests/test_worker.py
"""Unit tests for unplow background decompression worker daemon."""

import base64
from pathlib import Path
import pytest

from unplow.models import MockByteTokenizer, MockCausalLM
from unplow.paradigm1 import WeightDecompressor
from unplow.paradigm2 import LLMArithmeticCodec
from unplow.queue import FileTaskQueue
from unplow.worker import DecompressionWorker


@pytest.fixture
def mock_worker(tmp_path: Path):
    queue = FileTaskQueue(base_dir=str(tmp_path / "queue"))
    tok = MockByteTokenizer(vocab_size=260)
    model = MockCausalLM(vocab_size=260, hidden_dim=32)

    p1 = WeightDecompressor(model=model, tokenizer=tok, device="cpu")
    p2 = LLMArithmeticCodec(model=model, tokenizer=tok, device="cpu")

    worker = DecompressionWorker(
        queue=queue,
        model_name="mock",
        heartbeat_path=str(tmp_path / "logs" / "heartbeat" / "unplow.json"),
        forensics_dir=str(tmp_path / "logs" / "forensics"),
        work_products_dir=str(tmp_path / "workspace" / "work_products"),
    )
    worker._p1_decompressor = p1
    worker._p2_codec = p2
    return worker


def test_worker_probe_task(mock_worker):
    task = mock_worker.queue.enqueue(
        action="probe",
        extra={"num_samples": 2, "seq_len": 12},
    )
    claimed = mock_worker.queue.claim_next_task()
    assert claimed is not None
    t, proc_path = claimed

    mock_worker.process_task(t, proc_path)
    assert mock_worker.processed_count == 1

    counts = mock_worker.queue.get_counts()
    assert counts["done"] == 1
    assert counts["failed"] == 0

    done_file = mock_worker.queue.done_dir / f"{task.task_id}.json"
    assert done_file.exists()


def test_worker_compress_and_decompress(mock_worker):
    text = "Lossless test string for queue processing"
    comp_task = mock_worker.queue.enqueue(action="compress", payload=text)

    claimed = mock_worker.queue.claim_next_task()
    assert claimed is not None
    t, p = claimed
    mock_worker.process_task(t, p)

    # Check work products
    bin_file = mock_worker.work_products_dir / f"{comp_task.task_id}.bin"
    assert bin_file.exists()
    payload_bytes = bin_file.read_bytes()
    b64_payload = base64.b64encode(payload_bytes).decode("ascii")

    # Now enqueue decompress task
    n_tokens = len(mock_worker.get_p2_codec().tokenizer.encode(text, add_special_tokens=False))
    decomp_task = mock_worker.queue.enqueue(
        action="decompress",
        payload=b64_payload,
        extra={"num_tokens": n_tokens},
    )

    claimed2 = mock_worker.queue.claim_next_task()
    assert claimed2 is not None
    t2, p2 = claimed2
    mock_worker.process_task(t2, p2)

    recon_file = mock_worker.work_products_dir / f"{decomp_task.task_id}_reconstructed.txt"
    assert recon_file.exists()
    assert recon_file.read_text(encoding="utf-8") == text


def test_worker_heartbeat(mock_worker):
    mock_worker.emit_heartbeat(status="IDLE")
    assert mock_worker.heartbeat_path.exists()
    content = mock_worker.heartbeat_path.read_text(encoding="utf-8")
    assert '"status": "IDLE"' in content


def test_worker_archive_and_unarchive_with_domain(mock_worker):
    text = "Domain-conditioned queue test payload."
    arch_task = mock_worker.queue.enqueue(
        action="archive",
        payload=text,
        extra={"domain_id": 2, "max_ctx_len": 64},
    )
    claimed = mock_worker.queue.claim_next_task()
    assert claimed is not None
    t, p = claimed
    mock_worker.process_task(t, p)

    unp_file = mock_worker.work_products_dir / f"{arch_task.task_id}.unp"
    assert unp_file.exists()
    container_bytes = unp_file.read_bytes()
    b64_container = base64.b64encode(container_bytes).decode("ascii")

    # Unarchive
    unarch_task = mock_worker.queue.enqueue(
        action="unarchive",
        payload=b64_container,
    )
    claimed2 = mock_worker.queue.claim_next_task()
    assert claimed2 is not None
    t2, p2 = claimed2
    mock_worker.process_task(t2, p2)

    assert mock_worker.queue.get_counts()["done"] == 2


def test_worker_circuits_task(mock_worker):
    task = mock_worker.queue.enqueue(
        action="circuits",
        extra={"candidates": 2, "top_tokens": 2, "layers": [1]},
    )
    claimed = mock_worker.queue.claim_next_task()
    assert claimed is not None
    t, p = claimed
    mock_worker.process_task(t, p)

    circuits_file = mock_worker.work_products_dir / f"{task.task_id}_circuits.json"
    assert circuits_file.exists()


def test_worker_steer_task(mock_worker):
    task = mock_worker.queue.enqueue(
        action="steer",
        extra={"layer": 1, "neuron": 0, "gain": 2.0, "seq_len": 8, "steps": 4},
    )
    claimed = mock_worker.queue.claim_next_task()
    assert claimed is not None
    t, p = claimed
    mock_worker.process_task(t, p)

    steer_file = mock_worker.work_products_dir / f"{task.task_id}_steered.json"
    assert steer_file.exists()


def test_worker_profile_task(mock_worker):
    task = mock_worker.queue.enqueue(
        action="profile",
        payload="token repeat token repeat",
        extra={"cache_weight": 0.1},
    )
    claimed = mock_worker.queue.claim_next_task()
    assert claimed is not None
    t, p = claimed
    mock_worker.process_task(t, p)

    surprisal_file = mock_worker.work_products_dir / f"{task.task_id}_surprisal.json"
    assert surprisal_file.exists()

# end of file: unplow/tests/test_worker.py
