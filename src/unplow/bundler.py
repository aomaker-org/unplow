# Path: unplow/src/unplow/bundler.py
"""Hierarchical Workspace Bundler: Recursively archives multi-file directory trees.

Packages directory structures into single structured .unp containers using
3-expert Bayesian Context-Tree Switching arithmetic compression with per-file
cryptographic integrity verification.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Union

import torch

from .paradigm2 import LLMArithmeticCodec

BUNDLE_MAGIC = b"UNPBND"
BUNDLE_VERSION = 1


@dataclass
class BundledFileEntry:
    relative_path: str
    file_size_bytes: int
    sha256_hash: str
    content: bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.relative_path,
            "size": self.file_size_bytes,
            "sha256": self.sha256_hash,
        }


@dataclass
class BundleManifestReport:
    archive_path: str
    total_files: int
    total_uncompressed_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    entries: List[BundledFileEntry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archive_path": self.archive_path,
            "total_files": self.total_files,
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
            "total_compressed_bytes": self.total_compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "savings_percent": round(
                max(
                    0.0,
                    (1.0 - self.total_compressed_bytes / max(self.total_uncompressed_bytes, 1))
                    * 100.0,
                ),
                2,
            ),
            "files": [e.to_dict() for e in self.entries],
        }

    def format_manifest_table(self) -> str:
        """Format bundle manifest report as an ASCII diagnostic table."""
        lines = [
            "",
            "=" * 78,
            "Workspace Bundler: Hierarchical Archive Manifest",
            "=" * 78,
            f"Archive Path            : {self.archive_path}",
            f"Total Files             : {self.total_files}",
            f"Uncompressed Size       : {self.total_uncompressed_bytes} bytes",
            f"Compressed Container    : {self.total_compressed_bytes} bytes",
            f"Compression Ratio       : {self.compression_ratio:.3f}:1",
            "-" * 78,
            f"{'Relative File Path':<44} | {'Size (B)':<10} | {'SHA-256 (Prefix)':<16}",
            "-" * 78,
        ]
        for e in self.entries:
            display_path = e.relative_path
            if len(display_path) > 44:
                display_path = "..." + display_path[-41:]
            lines.append(
                f"{display_path:<44} | {e.file_size_bytes:<10} | {e.sha256_hash[:16]}"
            )
        lines.append("=" * 78)
        return "\n".join(lines)


class WorkspaceBundler:
    """Serializes and unpacks multi-file directory workspaces into hierarchical .unp containers."""

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self._codec: Optional[LLMArithmeticCodec] = None

    def get_codec(self) -> LLMArithmeticCodec:
        """Instantiate or reuse synchronized LLMArithmeticCodec."""
        if self._codec is None:
            if self.model is not None and self.tokenizer is not None:
                dev = self.device or str(next(self.model.parameters()).device)
                self._codec = LLMArithmeticCodec(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=dev,
                )
            else:
                mod_id = self.model_name or "mock"
                self._codec = LLMArithmeticCodec(model_name=mod_id, device=self.device)
        return self._codec

    @staticmethod
    def _collect_files(
        root_dir: Path, exclude_dirs: Optional[List[str]] = None
    ) -> List[Path]:
        """Collect and sort files lexically, filtering common build and cache directories."""
        excludes = exclude_dirs or [
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "build",
            "dist",
            "logs",
            "workspace",
            ".pytest_cache",
        ]
        collected: List[Path] = []
        for p in sorted(root_dir.rglob("*")):
            if p.is_file():
                if any(ex in p.parts for ex in excludes):
                    continue
                collected.append(p)
        return collected

    def bundle_directory(
        self,
        source_dir: Union[str, Path],
        output_unp_path: Union[str, Path],
        domain_id: int = 2,  # 2 = python_code / multi-file workspace
        max_ctx_len: int = 512,
    ) -> BundleManifestReport:
        """Recursively serialize source_dir into a hierarchical Bayesian CTS .unp archive."""
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

        payload_str = bytes(blob).decode("latin1")  # Safe lossless byte bijection

        codec = self.get_codec()
        res = codec.compress_to_container(
            payload_str,
            domain_id=domain_id,
            enable_bayes_mixer=True,
            max_ctx_len=max_ctx_len,
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
        self,
        archive_path: Union[str, Path],
        target_dir: Union[str, Path],
    ) -> BundleManifestReport:
        """Unpack a hierarchical .unp bundle back into target_dir with SHA-256 verification."""
        arc = Path(archive_path).resolve()
        dst = Path(target_dir).resolve()
        dst.mkdir(parents=True, exist_ok=True)

        container_bytes = arc.read_bytes()
        codec = self.get_codec()
        decomp_res = codec.decompress_from_container(container_bytes)
        payload_str = decomp_res.text
        blob = payload_str.encode("latin1")

        json_len = struct.unpack_from(">I", blob, 0)[0]
        offset = 4
        manifest_json = json.loads(blob[offset : offset + json_len].decode("utf-8"))
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
                    f"Integrity checksum mismatch for bundled file: {path_str} "
                    f"(expected {expected_hash}, got {actual_hash})"
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
            compression_ratio=round(total_raw / max(len(container_bytes), 1), 3),
            entries=entries,
        )


# end of file: unplow/src/unplow/bundler.py
