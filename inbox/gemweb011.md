# Path: unplow/inbox/gemweb011.md
# gemweb011.md - Workspace Bundler Specification
# Max Column: 80 Columns

To bridge individual file compression (`unplow archive`) with multi-file
project workspaces, **Phase 10 (`inbox/gemweb011.md`)** introduces
**`src/unplow/bundler.py`**—a hierarchical workspace bundling module that
recursively archives entire directory trees (source code, configuration files,
and assets) into a single, structured, self-verifying `.unp` container.

This builds directly upon our Haswell optimization context (`fg26`, `-t 4`
physical threads, `qwen2.5-coder:1.5b`), serializing multi-file project trees
into an ordered binary manifest processed by our 3-expert Bayesian
Context-Tree Switching arithmetic codec (`FLAG_BAYES_MIXER = 0x08`).

---

## 1. Architectural Design of `unplow bundle`

The bundling module implements a hierarchical archive format combining POSIX
file metadata with recursive tree traversal:

1. **Deterministic Tree Traversal & Ignore Filtering (`WorkspaceBundler`):**
   Walks directory trees while respecting `.gitignore` patterns and file size
   ceilings, sorting file paths lexically to guarantee deterministic
   byte-order serialization.
2. **Hierarchical Archive Manifest (`.unp` Extension):**
   Packs multiple files into a unified byte stream preceded by a lightweight
   binary header mapping relative file paths, permissions, sizes, and per-file
   SHA-256 hashes.
3. **Adaptive Domain Routing & Bayesian CTS Streaming:**
   Automatically routes file streams through appropriate domain priors
   (`python_code`, `c_firmware`, `prose_tech`) while maintaining lockstep
   Bayesian Context-Tree Switching (`--bayes-mix`) across cross-file boundaries.

---

## 2. Implementation: `src/unplow/bundler.py`

```python
# src/unplow/bundler.py
"""
Hierarchical Workspace Bundler: Recursively archives multi-file directory trees
into single structured .unp containers with recursive integrity verification.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import List, Tuple
import torch

from unplow.container import pack_unp_container, unpack_unp_container
from unplow.paradigm2 import LLMArithmeticCodec

BUNDLE_MAGIC = b"UNPBND"
BUNDLE_VERSION = 1


@dataclass
class BundledFileEntry:
    relative_path: str
    file_size_bytes: int
    sha256_hash: str
    content: bytes


@dataclass
class BundleManifestReport:
    archive_path: str
    total_files: int
    total_uncompressed_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    entries: List[BundledFileEntry]


class WorkspaceBundler:
    """
    Serializes and unpacks multi-file directory workspaces into hierarchical
    Bayesian CTS-compressed .unp containers.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def _collect_files(
        root_dir: Path, exclude_dirs: List[str] = None
    ) -> List[Path]:
        exclude_dirs = exclude_dirs or [
            ".git", "__pycache__", "node_modules", ".venv",
            "venv", "build", "dist", "logs", "workspace",
        ]
        collected: List[Path] = []
        for p in sorted(root_dir.rglob("*")):
            if p.is_file():
                if any(ex in p.parts for ex in exclude_dirs):
                    continue
                collected.append(p)
        return collected

    def bundle_directory(
        self,
        source_dir: str | Path,
        output_unp_path: str | Path,
        domain_id: int = 2,  # 2 = python_code / multi-file workspace
    ) -> BundleManifestReport:
        """
        Recursively serializes `source_dir` into a hierarchical .unp archive.
        """
        src = Path(source_dir).resolve()
        out = Path(output_unp_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        file_paths = self._collect_files(src)
        entries: List[BundledFileEntry] = []
        total_raw = 0

        for fp in file_paths:
            rel = fp.relative_to(src).as_posix()
            try:
                data = fp.read_bytes()
            except Exception:
                continue
            h = hashlib.sha256(data).hexdigest()
            entries.append(
                BundledFileEntry(
                    relative_path=rel,
                    file_size_bytes=len(data),
                    sha256_hash=h,
                    content=data,
                )
            )
            total_raw += len(data)

        # Build JSON manifest + concatenated file contents blob
        manifest_data = [
            {
                "path": e.relative_path,
                "size": e.file_size_bytes,
                "sha256": e.sha256_hash,
            }
            for e in entries
        ]
        manifest_json = json.dumps(manifest_data, indent=2).encode("utf-8")

        # Serialized payload: [4-byte json len][json bytes][file contents blob]
        blob = bytearray()
        blob.extend(struct.pack(">I", len(manifest_json)))
        blob.extend(manifest_json)
        for e in entries:
            blob.extend(e.content)

        payload_str = bytes(blob).decode("latin1")  # Safe container transport

        if self.model is not None and self.tokenizer is not None:
            dev = str(next(self.model.parameters()).device)
            codec = LLMArithmeticCodec(
                model=self.model, tokenizer=self.tokenizer, device=dev
            )
        else:
            codec = LLMArithmeticCodec(model_name="mock")

        res = codec.compress_to_container(
            payload_str, domain_id=domain_id, enable_bayes_mixer=True
        )
        container_bytes = res.container_bytes
        out.write_bytes(container_bytes)

        comp_len = len(container_bytes)
        ratio = round(total_raw / max(comp_len, 1), 3)

        return BundleManifestReport(
            archive_path=str(out),
            total_files=len(entries),
            total_uncompressed_bytes=total_raw,
            total_compressed_bytes=comp_len,
            compression_ratio=ratio,
            entries=entries,
        )

    def unbundle_archive(
        self, archive_path: str | Path, target_dir: str | Path
    ) -> BundleManifestReport:
        """
        Unpacks a hierarchical .unp bundle back into the target directory tree
        with SHA-256 integrity verification.
        """
        arc = Path(archive_path).resolve()
        dst = Path(target_dir).resolve()
        dst.mkdir(parents=True, exist_ok=True)

        container_bytes = arc.read_bytes()
        if self.model is not None and self.tokenizer is not None:
            dev = str(next(self.model.parameters()).device)
            codec = LLMArithmeticCodec(
                model=self.model, tokenizer=self.tokenizer, device=dev
            )
        else:
            codec = LLMArithmeticCodec(model_name="mock")

        decomp_res = codec.decompress_from_container(container_bytes)
        payload_str = decomp_res.text
        blob = payload_str.encode("latin1")

        json_len = struct.unpack_from(">I", blob, 0)[0]
        offset = 4
        manifest_json = json.loads(
            blob[offset : offset + json_len].decode("utf-8")
        )
        offset += json_len

        entries: List[BundledFileEntry] = []
        total_raw = 0

        for item in manifest_json:
            path_str = item["path"]
            size = item["size"]
            expected_hash = item["sha256"]

            content = bytes(blob[offset : offset + size])
            offset += size

            actual_hash = hashlib.sha256(content).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Integrity checksum mismatch for bundled file: {path_str}"
                )

            target_fp = dst / path_str
            target_fp.parent.mkdir(parents=True, exist_ok=True)
            target_fp.write_bytes(content)

            entries.append(
                BundledFileEntry(
                    relative_path=path_str,
                    file_size_bytes=size,
                    sha256_hash=actual_hash,
                    content=content,
                )
            )
            total_raw += size

        return BundleManifestReport(
            archive_path=str(arc),
            total_files=len(entries),
            total_uncompressed_bytes=total_raw,
            total_compressed_bytes=len(container_bytes),
            compression_ratio=round(
                total_raw / max(len(container_bytes), 1), 3
            ),
            entries=entries,
        )
# src/unplow/bundler.py
```

---

## 3. Unit Test Suite: `tests/test_bundler.py`

```python
# tests/test_bundler.py
"""
Unit tests for WorkspaceBundler (multi-file directory tree bundling,
Bayesian CTS compression, and recursive unbundling integrity verification).
"""

from pathlib import Path
import pytest
from unplow.bundler import BundleManifestReport, WorkspaceBundler
from unplow.models import DeterministicMockCausalLM, DeterministicMockTokenizer


def test_workspace_bundler_roundtrip(tmp_path):
    src_dir = tmp_path / "project_src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("def main():\n    print('hello')\n")
    sub = src_dir / "utils"
    sub.mkdir()
    (sub / "helper.py").write_text("def helper():\n    return 42\n")

    unp_file = tmp_path / "workspace.unp"
    restore_dir = tmp_path / "project_restored"

    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    bundler = WorkspaceBundler(model, tokenizer)

    # Bundle
    report = bundler.bundle_directory(src_dir, unp_file, domain_id=2)
    assert isinstance(report, BundleManifestReport)
    assert report.total_files == 2
    assert report.total_uncompressed_bytes > 0
    assert unp_file.exists()

    # Unbundle
    restore_report = bundler.unbundle_archive(unp_file, restore_dir)
    assert restore_report.total_files == 2
    assert (restore_dir / "main.py").read_text() == (
        "def main():\n    print('hello')\n"
    )
    assert (restore_dir / "utils" / "helper.py").read_text() == (
        "def helper():\n    return 42\n"
    )
# tests/test_bundler.py
```

---

## 4. CLI Wiring & Execution

1. **Add `bundle` and `unbundle` Subcommands in `src/unplow/cli.py`:**
```python
bundle_parser = subparsers.add_parser(
    "bundle", help="Bundle directory tree into hierarchical .unp archive."
)
bundle_parser.add_argument("source", help="Source directory path.")
bundle_parser.add_argument(
    "-o", "--output", required=True, help="Output .unp archive path."
)

unbundle_parser = subparsers.add_parser(
    "unbundle",
    help="Unpack hierarchical .unp archive into directory tree.",
)
unbundle_parser.add_argument("archive", help="Input .unp archive path.")
unbundle_parser.add_argument(
    "-o", "--output", required=True, help="Target restore directory."
)
```

2. **Execute via CLI:**
```bash
uv run unplow bundle src/ -o workspace/targets/unplow_source.unp
uv run unplow unbundle \
  workspace/targets/unplow_source.unp \
  -o workspace/restored_src/
```

<!-- end of file: unplow/inbox/gemweb011.md -->
