# Path: unplow/src/unplow/queue.py
"""File-backed asynchronous task queue for unplow.

Implements an atomic directory-based state machine with POSIX advisory file
locking (fcntl.flock) across:
  - incoming/   : New unassigned decompression tasks
  - processing/ : Tasks currently being executed by an unplow worker
  - done/       : Successfully completed decompression runs with records
  - failed/     : Tasks that encountered execution errors
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import fcntl
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger("unplow.queue")


class TaskStatus(str, Enum):
    INCOMING = "incoming"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    """Atomic task definition for unplow queue processing."""

    task_id: str
    action: str
    payload: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = TaskStatus.INCOMING.value
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(**data)


class FileTaskQueue:
    """Manages atomic file-based task scheduling with fcntl locking."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            self.base_dir = Path("records/queue").resolve()

        self.incoming_dir = self.base_dir / "incoming"
        self.processing_dir = self.base_dir / "processing"
        self.done_dir = self.base_dir / "done"
        self.failed_dir = self.base_dir / "failed"

        for d in [self.incoming_dir, self.processing_dir, self.done_dir, self.failed_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        action: str,
        payload: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Enqueue a new task into incoming/."""
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        task = Task(
            task_id=task_id,
            action=action,
            payload=payload,
            extra=extra or {},
        )
        task_file = self.incoming_dir / f"{task_id}.json"
        tmp_file = self.incoming_dir / f".{task_id}.json.tmp"

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2)
        os.replace(tmp_file, task_file)

        logger.info("Enqueued task %s (%s) -> %s", task_id, action, task_file)
        return task

    def claim_next_task(self) -> Optional[Tuple[Task, Path]]:
        """Atomically claim the oldest task in incoming/ by moving to processing/."""
        incoming_files = sorted(self.incoming_dir.glob("task-*.json"))
        for candidate in incoming_files:
            try:
                with open(candidate, "r+", encoding="utf-8") as f:
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except (BlockingIOError, OSError):
                        continue

                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        fcntl.flock(f, fcntl.LOCK_UN)
                        continue

                    task = Task.from_dict(data)
                    task.status = TaskStatus.PROCESSING.value
                    task.started_at = datetime.now(timezone.utc).isoformat()

                    dest_path = self.processing_dir / candidate.name
                    tmp_dest = self.processing_dir / f".{candidate.name}.tmp"

                    with open(tmp_dest, "w", encoding="utf-8") as df:
                        json.dump(task.to_dict(), df, indent=2)
                    os.replace(tmp_dest, dest_path)

                    fcntl.flock(f, fcntl.LOCK_UN)
                    candidate.unlink(missing_ok=True)
                    logger.info("Claimed task %s -> %s", task.task_id, dest_path)
                    return task, dest_path
            except Exception as e:
                logger.debug("Failed claiming %s: %s", candidate.name, e)
                continue

        return None

    def complete_task(
        self,
        task: Task,
        processing_path: Path,
        result: str,
        model_used: Optional[str] = None,
    ) -> Path:
        """Move task from processing/ to done/ with output result."""
        task.status = TaskStatus.DONE.value
        task.result = result
        task.model_used = model_used
        task.completed_at = datetime.now(timezone.utc).isoformat()

        dest_path = self.done_dir / f"{task.task_id}.json"
        tmp_dest = self.done_dir / f".{task.task_id}.json.tmp"

        with open(tmp_dest, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2)
        os.replace(tmp_dest, dest_path)
        processing_path.unlink(missing_ok=True)

        logger.info("Completed task %s -> %s", task.task_id, dest_path)
        return dest_path

    def fail_task(
        self,
        task: Task,
        processing_path: Path,
        error_msg: str,
    ) -> Path:
        """Move task from processing/ to failed/ with error details."""
        task.status = TaskStatus.FAILED.value
        task.error = error_msg
        task.completed_at = datetime.now(timezone.utc).isoformat()

        dest_path = self.failed_dir / f"{task.task_id}.json"
        tmp_dest = self.failed_dir / f".{task.task_id}.json.tmp"

        with open(tmp_dest, "w", encoding="utf-8") as f:
            json.dump(task.to_dict(), f, indent=2)
        os.replace(tmp_dest, dest_path)
        processing_path.unlink(missing_ok=True)

        logger.error("Failed task %s: %s -> %s", task.task_id, error_msg, dest_path)
        return dest_path

    def get_counts(self) -> Dict[str, int]:
        """Return counts of tasks across states."""
        return {
            "incoming": len(list(self.incoming_dir.glob("task-*.json"))),
            "processing": len(list(self.processing_dir.glob("task-*.json"))),
            "done": len(list(self.done_dir.glob("task-*.json"))),
            "failed": len(list(self.failed_dir.glob("task-*.json"))),
        }

# end of file: unplow/src/unplow/queue.py
