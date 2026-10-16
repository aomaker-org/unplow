# Path: unplow/tests/test_safety.py
"""Unit tests for unplow safety, air-gap enforcer, and additive file guards."""

from pathlib import Path
import pytest

from unplow.safety import (
    AdditiveFileHandler,
    AirGapViolationError,
    GitDestructiveOperationError,
    GitSafetyGuard,
    validate_additive_artifact_path,
    validate_local_only_url,
)


def test_air_gap_validation():
    # Valid local URLs
    validate_local_only_url("http://127.0.0.1:11434")
    validate_local_only_url("http://localhost:8000/v1")
    validate_local_only_url("http://0.0.0.0:5000")

    # Prohibited external URLs
    with pytest.raises(AirGapViolationError):
        validate_local_only_url("https://api.openai.com/v1")

    with pytest.raises(AirGapViolationError):
        validate_local_only_url("https://api.anthropic.com")

    with pytest.raises(AirGapViolationError):
        validate_local_only_url("https://generativelanguage.googleapis.com")


def test_additive_file_handler_revisions(tmp_path: Path):
    handler = AdditiveFileHandler(base_dir=tmp_path)
    base_file = tmp_path / "artifacts" / "decompressed.txt"

    p1 = handler.write_additive_text(base_file, "Revision 0 content")
    assert p1 == base_file
    assert p1.read_text(encoding="utf-8") == "Revision 0 content"

    p2 = handler.write_additive_text(base_file, "Revision 1 content")
    assert p2 == tmp_path / "artifacts" / "decompressed_rev001.txt"
    assert p2.read_text(encoding="utf-8") == "Revision 1 content"

    p3 = handler.write_additive_text(base_file, "Revision 2 content")
    assert p3 == tmp_path / "artifacts" / "decompressed_rev002.txt"
    assert p3.read_text(encoding="utf-8") == "Revision 2 content"


def test_canonical_path_protection(tmp_path: Path):
    # Ensure attempting additive revisions on canonical git files raises ValueError
    with pytest.raises(ValueError, match="Directive 07 violation"):
        validate_additive_artifact_path("src/unplow/paradigm1.py", repo_root=Path(__file__).parents[1])

    with pytest.raises(ValueError, match="Directive 07 violation"):
        validate_additive_artifact_path("pyproject.toml", repo_root=Path(__file__).parents[1])


def test_git_safety_guard():
    # Safe commands
    GitSafetyGuard.audit_git_command(["git", "status"])
    GitSafetyGuard.audit_git_command(["git", "commit", "-m", "chore"])
    GitSafetyGuard.audit_git_command(["git", "checkout", "-b", "feature/test"])

    # Prohibited destructive commands
    with pytest.raises(GitDestructiveOperationError):
        GitSafetyGuard.audit_git_command(["git", "reset", "--hard", "HEAD~1"])

    with pytest.raises(GitDestructiveOperationError):
        GitSafetyGuard.audit_git_command(["git", "clean", "-fd"])

    with pytest.raises(GitDestructiveOperationError):
        GitSafetyGuard.audit_git_command(["git", "push", "--force", "origin", "main"])

# end of file: unplow/tests/test_safety.py
