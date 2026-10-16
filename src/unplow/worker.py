# Path: unplow/src/unplow/worker.py
"""Background decompression worker daemon and queue supervisor for unplow.

Dispatches queued tasks to Paradigm 1 (static weight basin probing) or
Paradigm 2 (arithmetic compression/decompression/manifold), maintains
Directive 17 telemetry heartbeats, and persists tamper-evident forensic records.
"""

import base64
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

from .forensics import (
    ForensicAuditLogger,
    compute_str_sha256,
    get_telemetry_snapshot,
)
from .models import resolve_model_path
from .paradigm1 import WeightDecompressor
from .paradigm2 import LLMArithmeticCodec
from .queue import FileTaskQueue, Task, TaskStatus
from .safety import AdditiveFileHandler

logger = logging.getLogger("unplow.worker")


class DecompressionWorker:
    """Worker daemon executing decompression jobs from the atomic file queue."""

    def __init__(
        self,
        queue: Optional[FileTaskQueue] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        heartbeat_path: str = "logs/heartbeat/unplow.json",
        forensics_dir: str = "logs/forensics",
        work_products_dir: str = "workspace/work_products",
    ):
        self.queue = queue or FileTaskQueue()
        self.model_name = model_name
        self.device = device or ("cuda" if False else "cpu")
        self.heartbeat_path = Path(heartbeat_path).resolve()
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

        self.forensics = ForensicAuditLogger(base_dir=forensics_dir)
        self.work_products_dir = Path(work_products_dir).resolve()
        self.work_products_dir.mkdir(parents=True, exist_ok=True)
        self.file_handler = AdditiveFileHandler(base_dir=self.work_products_dir.parent.parent)

        self.processed_count = 0
        self.failed_count = 0
        self.running = False

        # Lazy initialized codecs
        self._p1_decompressor: Optional[WeightDecompressor] = None
        self._p2_codec: Optional[LLMArithmeticCodec] = None

    def get_p1_decompressor(self) -> WeightDecompressor:
        if self._p1_decompressor is None:
            logger.info("Initializing WeightDecompressor for worker...")
            self._p1_decompressor = WeightDecompressor(model_name=self.model_name, device=self.device)
        return self._p1_decompressor

    def get_p2_codec(self) -> LLMArithmeticCodec:
        if self._p2_codec is None:
            logger.info("Initializing LLMArithmeticCodec for worker...")
            self._p2_codec = LLMArithmeticCodec(model_name=self.model_name, device=self.device)
        return self._p2_codec

    def emit_heartbeat(self, status: str = "IDLE") -> None:
        """Record process telemetry heartbeat complying with Directive 17."""
        telemetry = get_telemetry_snapshot()
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "pid": telemetry["pid"],
            "hostname": telemetry["hostname"],
            "rss_mb": telemetry["rss_mb"],
            "model_path": resolve_model_path(self.model_name),
            "processed_tasks": self.processed_count,
            "failed_tasks": self.failed_count,
            "queue_counts": self.queue.get_counts(),
        }
        tmp_hb = self.heartbeat_path.parent / f".{self.heartbeat_path.name}.tmp"
        with open(tmp_hb, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_hb, self.heartbeat_path)

    def process_task(self, task: Task, proc_path: Path) -> None:
        """Execute a claimed task according to its specified decompression paradigm."""
        self.emit_heartbeat(status="PROCESSING")
        self.forensics.log_event(
            phase="WORKER_DISPATCH",
            task_id=task.task_id,
            action=task.action,
            details={"extra": task.extra},
        )

        try:
            action = task.action.lower()
            if action in ("probe", "paradigm1", "p1"):
                num_samples = task.extra.get("num_samples", 4)
                seq_len = task.extra.get("seq_len", 64)
                prefixes = task.extra.get("seed_prefixes")
                decompressor = self.get_p1_decompressor()
                records = decompressor.probe_weight_basins(
                    num_samples=num_samples,
                    seq_len=seq_len,
                    seed_prefixes=prefixes,
                )

                output_dict = {
                    "paradigm": 1,
                    "model": decompressor.model_path,
                    "num_extracted": len(records),
                    "records": [r.to_dict() for r in records],
                }
                result_json = json.dumps(output_dict, indent=2)
                art_path = self.work_products_dir / f"{task.task_id}_weight_basins.json"
                self.file_handler.write_additive_text(art_path, result_json)
                self.queue.complete_task(task, proc_path, result=result_json, model_used=decompressor.model_path)
                self.processed_count += 1

            elif action in ("compress", "p2_compress"):
                text = task.payload or task.extra.get("text", "")
                codec = self.get_p2_codec()
                payload_bytes, n_tokens, stats = codec.compress(text)
                b64_payload = base64.b64encode(payload_bytes).decode("ascii")

                output_dict = {
                    "paradigm": 2,
                    "action": "compress",
                    "model": codec.model_path,
                    "b64_payload": b64_payload,
                    "num_tokens": n_tokens,
                    "stats": stats.to_dict(),
                }
                result_json = json.dumps(output_dict, indent=2)
                bin_path = self.work_products_dir / f"{task.task_id}.bin"
                self.file_handler.write_additive_bytes(bin_path, payload_bytes)
                self.queue.complete_task(task, proc_path, result=result_json, model_used=codec.model_path)
                self.processed_count += 1

            elif action in ("decompress", "p2_decompress"):
                codec = self.get_p2_codec()
                b64_payload = task.payload or task.extra.get("b64_payload", "")
                num_tokens = task.extra.get("num_tokens", 0)
                raw_payload = base64.b64decode(b64_payload.encode("ascii"))
                reconstructed = codec.decompress(raw_payload, num_tokens=num_tokens)

                output_dict = {
                    "paradigm": 2,
                    "action": "decompress",
                    "model": codec.model_path,
                    "reconstructed_text": reconstructed,
                    "reconstructed_sha256": compute_str_sha256(reconstructed),
                }
                result_json = json.dumps(output_dict, indent=2)
                art_path = self.work_products_dir / f"{task.task_id}_reconstructed.txt"
                self.file_handler.write_additive_text(art_path, reconstructed)
                self.queue.complete_task(task, proc_path, result=result_json, model_used=codec.model_path)
                self.processed_count += 1

            elif action in ("manifold", "p2_manifold"):
                seed = task.payload or task.extra.get("seed", "unplow")
                num_tokens = task.extra.get("num_tokens", 32)
                codec = self.get_p2_codec()
                unpacked = codec.decompress_seed(seed, num_tokens=num_tokens)

                output_dict = {
                    "paradigm": 2,
                    "action": "manifold",
                    "seed": seed,
                    "unpacked_text": unpacked,
                    "model": codec.model_path,
                }
                result_json = json.dumps(output_dict, indent=2)
                art_path = self.work_products_dir / f"{task.task_id}_manifold.txt"
                self.file_handler.write_additive_text(art_path, unpacked)
            elif action in ("archive", "pack"):
                text = task.payload or task.extra.get("text", "")
                max_ctx = task.extra.get("max_ctx_len", 512)
                domain_id = task.extra.get("domain_id", 0)
                codec = self.get_p2_codec()
                container_bytes, meta, stats = codec.compress_to_container(
                    text, max_ctx_len=max_ctx, verify_hash=True, domain_id=domain_id
                )
                b64_container = base64.b64encode(container_bytes).decode("ascii")

                output_dict = {
                    "paradigm": 2,
                    "action": "archive",
                    "model": codec.model_path,
                    "b64_container": b64_container,
                    "metadata": meta.to_dict(),
                    "stats": stats.to_dict(),
                }
                result_json = json.dumps(output_dict, indent=2)
                unp_path = self.work_products_dir / f"{task.task_id}.unp"
                self.file_handler.write_additive_bytes(unp_path, container_bytes)
                self.queue.complete_task(task, proc_path, result=result_json, model_used=codec.model_path)
                self.processed_count += 1

            elif action in ("circuits", "circuit", "swiglu"):
                from .circuits import SwiGLUCircuitDecompressor
                decompressor = SwiGLUCircuitDecompressor(model_name=self.model_name)
                candidates = task.extra.get("candidates", 24)
                top_tokens = task.extra.get("top_tokens", 5)
                layers = task.extra.get("layers", None)
                suppress_num = task.extra.get("suppress_numerals", True)
                circuits = decompressor.extract_coupled_circuits(
                    layer_indices=layers,
                    candidates_per_layer=candidates,
                    top_k_tokens=top_tokens,
                    suppress_numerals=suppress_num,
                )
                seeds = decompressor.harvest_associative_seeds(
                    max_seeds=task.extra.get("seeds", 12),
                    layer_indices=layers,
                    suppress_numerals=suppress_num,
                )
                output_dict = {
                    "action": "circuits",
                    "model": decompressor.model_path,
                    "num_circuits": len(circuits),
                    "traces": [c.to_dict() for c in circuits],
                    "harvested_seeds": seeds,
                }
                result_json = json.dumps(output_dict, indent=2)
                art_path = self.work_products_dir / f"{task.task_id}_circuits.json"
                self.file_handler.write_additive_text(art_path, result_json)
                self.queue.complete_task(task, proc_path, result=result_json, model_used=decompressor.model_path)
                self.processed_count += 1

            elif action in ("unarchive", "unpack"):
                codec = self.get_p2_codec()
                b64_container = task.payload or task.extra.get("b64_container", "")
                container_bytes = base64.b64decode(b64_container.encode("ascii"))
                reconstructed, meta, is_valid = codec.decompress_from_container(container_bytes)

                output_dict = {
                    "paradigm": 2,
                    "action": "unarchive",
                    "model": codec.model_path,
                    "reconstructed_text": reconstructed,
                    "metadata": meta.to_dict(),
                    "checksum_valid": is_valid,
                }
                result_json = json.dumps(output_dict, indent=2)
                art_path = self.work_products_dir / f"{task.task_id}_unpacked.txt"
                self.file_handler.write_additive_text(art_path, reconstructed)
                self.queue.complete_task(task, proc_path, result=result_json, model_used=codec.model_path)
                self.processed_count += 1

            else:
                raise ValueError(f"Unknown decompression action '{task.action}'")

            self.forensics.log_event(
                phase="WORKER_COMPLETE",
                task_id=task.task_id,
                action=task.action,
                details={"status": "DONE"},
            )

        except Exception as e:
            logger.exception("Error processing task %s: %s", task.task_id, e)
            self.failed_count += 1
            self.queue.fail_task(task, proc_path, error_msg=str(e))
            self.forensics.log_event(
                phase="WORKER_ERROR",
                task_id=task.task_id,
                action=task.action,
                details={"error": str(e)},
                level="ERROR",
            )
        finally:
            self.emit_heartbeat(status="IDLE")

    def run_loop(self, interval: float = 2.0, max_iterations: Optional[int] = None) -> None:
        """Continuous polling loop processing queued decompression tasks."""
        self.running = True
        logger.info("DecompressionWorker loop started (interval=%.1fs)...", interval)
        iterations = 0

        try:
            while self.running:
                self.emit_heartbeat(status="IDLE")
                claimed = self.queue.claim_next_task()
                if claimed:
                    task, proc_path = claimed
                    self.process_task(task, proc_path)
                else:
                    time.sleep(interval)

                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
        except KeyboardInterrupt:
            logger.info("Worker stopped by user.")
        finally:
            self.running = False
            self.emit_heartbeat(status="STOPPED")

# end of file: unplow/src/unplow/worker.py
