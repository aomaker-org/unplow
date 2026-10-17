# Path: unplow/src/unplow/container.py
"""Self-verifying .unp binary container format and bounded-memory KV-cache sliding horizon.

Specifies:
1. 18-Byte Framed Binary Header:
   - 6 bytes  : Magic ASCII signature b"UNPLOW"
   - 1 byte   : Version (uint8 = 0x01)
   - 1 byte   : Flags (uint8, 0x01 = Lossless Verified, 0x00 = Manifold Seed)
   - 4 bytes  : Token count (uint32 big-endian)
   - 2 bytes  : Max context length stride (uint16 big-endian)
   - 4 bytes  : SHA-256 digest prefix (first 32 bits of plaintext hash)
2. Bounded KV-cache horizon management preserving exact bit-level arithmetic determinism.
"""

from dataclasses import asdict, dataclass
import hashlib
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from transformers import DynamicCache
except ImportError:
    DynamicCache = None

# .unp Header Layout: 6s (magic) + B (ver) + B (flags) + I (tokens) + H (max_ctx) + 4s (sha_prefix) = 18 bytes
UNP_HEADER_FORMAT = ">6sBBIH4s"
UNP_HEADER_SIZE = struct.calcsize(UNP_HEADER_FORMAT)
UNP_MAGIC = b"UNPLOW"
UNP_VERSION = 1
FLAG_LOSSLESS_VERIFIED = 0x01
FLAG_CACHE_MIXTURE = 0x02
FLAG_MANIFOLD_SEED = 0x00


@dataclass
class UnpArchiveMetadata:
    """Header metadata for an unplow (.unp) binary archive."""

    version: int
    flags: int
    num_tokens: int
    max_ctx_len: int
    sha256_prefix_hex: str
    compressed_bytes_len: int
    bitstream_bytes_len: int

    @property
    def is_verified_lossless(self) -> bool:
        return bool(self.flags & FLAG_LOSSLESS_VERIFIED)

    @property
    def cache_mixture(self) -> bool:
        """Bit 1: True if online prefix-cache mixture CDF is enabled."""
        return bool(self.flags & FLAG_CACHE_MIXTURE)

    @property
    def is_cache_mixture_enabled(self) -> bool:
        return self.cache_mixture

    @property
    def domain_id(self) -> int:
        """4-bit domain conditioning prior ID (0..15) stored in upper nibble."""
        return (self.flags >> 4) & 0x0F

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["domain_id"] = self.domain_id
        d["cache_mixture"] = self.cache_mixture
        d["is_cache_mixture_enabled"] = self.is_cache_mixture_enabled
        return d


def slice_kv_cache(past_key_values: Any, keep_tokens: int) -> Any:
    """Deterministically truncates past_key_values to retain the most recent keep_tokens.

    Preserves exact symmetry between encoder and decoder while bounding memory.
    """
    if past_key_values is None or keep_tokens <= 0:
        return past_key_values

    # Check if DynamicCache
    if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
        new_cache = DynamicCache()
        # In newer transformers, key_cache / value_cache are lists of tensors
        key_cache = getattr(past_key_values, "key_cache", getattr(past_key_values, "layers", []))
        value_cache = getattr(past_key_values, "value_cache", [])
        if key_cache and value_cache and len(key_cache) == len(value_cache):
            for l_idx in range(len(key_cache)):
                k = key_cache[l_idx][:, :, -keep_tokens:, :]
                v = value_cache[l_idx][:, :, -keep_tokens:, :]
                new_cache.update(k, v, l_idx)
            return new_cache

    # Tuple of tuples format
    if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
        if isinstance(past_key_values[0], tuple) and len(past_key_values[0]) >= 2:
            return tuple(
                (k[:, :, -keep_tokens:, :], v[:, :, -keep_tokens:, :])
                for k, v in past_key_values
            )

    return past_key_values


def pack_unp_container(
    bitstream: bytes,
    num_tokens: int,
    max_ctx_len: int = 512,
    plaintext: Optional[str] = None,
    verify_hash: bool = True,
    domain_id: int = 0,
    enable_cache_mixture: bool = False,
    use_cache_mixture: Optional[bool] = None,
) -> bytes:
    """Pack raw arithmetic bitstream into an 18-byte framed .unp container."""
    if use_cache_mixture is not None:
        enable_cache_mixture = use_cache_mixture

    if plaintext is not None and verify_hash:
        digest_prefix = hashlib.sha256(plaintext.encode("utf-8")).digest()[:4]
        low_nibble = FLAG_LOSSLESS_VERIFIED
    else:
        digest_prefix = b"\x00\x00\x00\x00"
        low_nibble = FLAG_MANIFOLD_SEED

    if enable_cache_mixture:
        low_nibble |= FLAG_CACHE_MIXTURE

    flags = ((domain_id & 0x0F) << 4) | (low_nibble & 0x0F)

    header = struct.pack(
        UNP_HEADER_FORMAT,
        UNP_MAGIC,
        UNP_VERSION,
        flags,
        num_tokens,
        max_ctx_len,
        digest_prefix,
    )
    return header + bitstream


def unpack_unp_container(container_bytes: bytes) -> Tuple[UnpArchiveMetadata, bytes]:
    """Unpack and validate the 18-byte .unp header, returning metadata and bitstream."""
    if len(container_bytes) < UNP_HEADER_SIZE:
        raise ValueError(
            f"Invalid .unp archive: size {len(container_bytes)} smaller than header size ({UNP_HEADER_SIZE} bytes)."
        )

    magic, version, flags, num_tokens, max_ctx_len, digest_prefix = struct.unpack(
        UNP_HEADER_FORMAT, container_bytes[:UNP_HEADER_SIZE]
    )

    if magic != UNP_MAGIC:
        raise ValueError(f"Invalid .unp container: magic bytes {magic!r} do not match {UNP_MAGIC!r}")

    if version != UNP_VERSION:
        raise ValueError(f"Unsupported .unp version {version}; expected {UNP_VERSION}")

    bitstream = container_bytes[UNP_HEADER_SIZE:]
    meta = UnpArchiveMetadata(
        version=version,
        flags=flags,
        num_tokens=num_tokens,
        max_ctx_len=max_ctx_len,
        sha256_prefix_hex=digest_prefix.hex(),
        compressed_bytes_len=len(container_bytes),
        bitstream_bytes_len=len(bitstream),
    )
    return meta, bitstream


def verify_unp_checksum(meta: UnpArchiveMetadata, decompressed_text: str) -> bool:
    """Verify that decompressed plaintext matches the 32-bit SHA-256 header prefix."""
    if not meta.is_verified_lossless:
        return True
    actual_prefix = hashlib.sha256(decompressed_text.encode("utf-8")).digest()[:4].hex()
    return actual_prefix == meta.sha256_prefix_hex

# end of file: unplow/src/unplow/container.py
