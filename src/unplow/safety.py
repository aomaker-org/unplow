# Path: unplow/src/unplow/safety.py
"""Safety, air-gap enforcer, and additive file/Git history preservation guardrails.

Enforces:
1. Strict local-only network air-gap (zero external I/O or cloud API leakage).
2. Scoped additive file operations (untracked worker artifacts use _rev001;
   canonical Git-tracked source is updated in place via feature branches).
3. Non-destructive Git history management (blocks force-push, reset --hard, etc).
"""

from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from typing import List, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger("unplow.safety")


class AirGapViolationError(RuntimeError):
    """Raised when non-localhost or external network destinations are targeted."""
    pass


class GitDestructiveOperationError(RuntimeError):
    """Raised when a destructive Git command or option is attempted."""
    pass


def validate_local_only_url(url: str) -> None:
    """Verify that an endpoint URL strictly targets localhost/loopback."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    allowed_hosts: Set[str] = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

    if hostname not in allowed_hosts:
        raise AirGapViolationError(
            f"Air-gap mandate violation: unplow is restricted to local-only "
            f"resources. Target host '{hostname}' is not a local loopback interface."
        )


CANONICAL_SOURCE_DIRS: tuple[str, ...] = (
    "src",
    "tests",
    "docs",
    "tools",
)
CANONICAL_ROOT_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "Makefile",
    "AI.md",
    "AGENTS.md",
    "README.md",
    "config_env",
    "config_env_models",
    ".gitignore",
    ".editorconfig",
    ".python-version",
)


def validate_additive_artifact_path(
    target_path: str | Path,
    repo_root: str | Path | None = None,
) -> Path:
    """Ensure AdditiveFileHandler (_rev001) is not used on Git-tracked source.

    Per Directive 07, canonical repository source files under src/, tests/,
    docs/, tools/, and root config files must retain canonical filenames and
    rely on atomic Git commits for history rather than _revNNN duplicates.
    """
    resolved_target = Path(target_path).resolve()
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )

    try:
        rel_path = resolved_target.relative_to(root)
    except ValueError:
        # Target is outside the repository root (e.g., /tmp or pytest tmp_path)
        return resolved_target

    parts = rel_path.parts
    if not parts:
        return resolved_target

    if parts[0] in CANONICAL_SOURCE_DIRS or (
        len(parts) == 1 and parts[0] in CANONICAL_ROOT_FILES
    ):
        raise ValueError(
            f"Directive 07 violation: '{rel_path}' is inside a canonical "
            f"Git-tracked path ({parts[0]}). Update canonical source files "
            f"in place on a feature/* branch; reserve AdditiveFileHandler "
            f"(_rev001) for untracked worker artifacts and drafts."
        )

    return resolved_target


class AdditiveFileHandler:
    """Manages strictly additive file persistence with _revNNN revisions."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = (base_dir or Path(".")).resolve()

    def get_next_revision_path(self, target_path: Path) -> Path:
        """Find the next available _revNNN path if target already exists."""
        target_path = validate_additive_artifact_path(target_path, self.base_dir)

        if not target_path.exists():
            return target_path

        stem = target_path.stem
        suffix = target_path.suffix
        parent = target_path.parent

        rev = 1
        while True:
            candidate = parent / f"{stem}_rev{rev:03d}{suffix}"
            if not candidate.exists():
                return candidate
            rev += 1

    def write_additive_text(
        self,
        target_path: Path,
        content: str,
        encoding: str = "utf-8",
    ) -> Path:
        """Write content safely without overwriting historical artifacts."""
        safe_path = self.get_next_revision_path(target_path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding=encoding)
        logger.info("Additive write persisted to %s", safe_path)
        return safe_path

    def write_additive_bytes(
        self,
        target_path: Path,
        content: bytes,
    ) -> Path:
        """Write binary payload safely without overwriting historical artifacts."""
        safe_path = self.get_next_revision_path(target_path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_bytes(content)
        logger.info("Additive binary write persisted to %s", safe_path)
        return safe_path


class GitSafetyGuard:
    """Guards against destructive Git operations adhering to Directive 07."""

    PROHIBITED_COMMANDS: Set[str] = {
        "reset",
        "clean",
        "rebase",
    }
    PROHIBITED_FLAGS: Set[str] = {
        "--hard",
        "--force",
        "-f",
        "-D",
    }

    @classmethod
    def audit_git_command(cls, cmd_args: List[str]) -> None:
        """Raise GitDestructiveOperationError if command violates Directive 07."""
        if not cmd_args or cmd_args[0] != "git":
            return

        for arg in cmd_args[1:]:
            if arg in cls.PROHIBITED_COMMANDS:
                raise GitDestructiveOperationError(
                    f"Prohibited git command '{arg}' detected. Directive 07 "
                    f"mandates strictly additive Git history."
                )
            if arg in cls.PROHIBITED_FLAGS:
                raise GitDestructiveOperationError(
                    f"Prohibited destructive git flag '{arg}' detected. "
                    f"Directive 07 forbids destructive history manipulation."
                )

# end of file: unplow/src/unplow/safety.py
