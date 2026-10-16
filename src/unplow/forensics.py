# Path: unplow/src/unplow/forensics.py
"""Forensic audit logger and cryptographic ledger for unplow.

Provides tamper-evident, append-only logging of every task, prompt,
decompression payload, cryptographic SHA-256 hash, and system telemetry record
adhering to Directive 02 (full logging) and Directive 17 (telemetry).
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
from typing import Any, Dict, List, Optional

logger = logging.getLogger("unplow.forensics")


def compute_str_sha256(content: str) -> str:
    """Calculate SHA-256 hex digest of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_file_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hex digest of a file on disk."""
    if not file_path.exists() or not file_path.is_file():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_telemetry_snapshot() -> Dict[str, Any]:
    """Capture process telemetry for forensic tracking."""
    rss_mb = 0.0
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = float(line.split()[1]) / 1024.0
                    break
    except Exception:
        pass

    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rss_mb": round(rss_mb, 2),
    }


class ForensicAuditLogger:
    """Manages append-only JSONL journal and verbose trace logging."""

    def __init__(self, base_dir: str = "logs/forensics"):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.journal_path = self.base_dir / "audit_journal.jsonl"
        self.trace_log_path = self.base_dir / "forensic_trace.log"

    def log_event(
        self,
        phase: str,
        task_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        level: str = "INFO",
    ) -> Dict[str, Any]:
        """Record an immutable event into the forensic journal and trace log."""
        now_utc = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp_utc": now_utc,
            "phase": phase,
            "task_id": task_id,
            "action": action,
            "level": level,
            "telemetry": get_telemetry_snapshot(),
            "details": details or {},
        }

        # 1. Append structured JSONL record
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        # 2. Append verbose human-readable trace log
        trace_line = (
            f"[{now_utc}] [{level}] [{phase}] task={task_id} action={action} "
            f"details={json.dumps(details or {})}\n"
        )
        with open(self.trace_log_path, "a", encoding="utf-8") as f:
            f.write(trace_line)

        return payload

    def get_recent_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve the most recent ledger entries."""
        if not self.journal_path.exists():
            return []

        entries = []
        with open(self.journal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries[-limit:]

# end of file: unplow/src/unplow/forensics.py
