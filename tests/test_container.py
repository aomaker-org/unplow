# Path: unplow/tests/test_container.py
"""Unit tests for .unp binary container packaging and header verification."""

import pytest

from unplow.container import (
    FLAG_LOSSLESS_VERIFIED,
    FLAG_MANIFOLD_SEED,
    UNP_HEADER_SIZE,
    UNP_MAGIC,
    UNP_VERSION,
    UnpArchiveMetadata,
    pack_unp_container,
    unpack_unp_container,
    verify_unp_checksum,
)


def test_pack_and_unpack_container():
    raw_bitstream = b"\xde\xad\xbe\xef\xca\xfe"
    plaintext = "Mechanistic decompression and arithmetic coding."
    num_tokens = 8
    max_ctx_len = 512

    container = pack_unp_container(
        bitstream=raw_bitstream,
        num_tokens=num_tokens,
        max_ctx_len=max_ctx_len,
        plaintext=plaintext,
        verify_hash=True,
    )

    assert len(container) == UNP_HEADER_SIZE + len(raw_bitstream)
    assert container.startswith(UNP_MAGIC)

    meta, unpacked_bitstream = unpack_unp_container(container)
    assert isinstance(meta, UnpArchiveMetadata)
    assert meta.version == UNP_VERSION
    assert meta.num_tokens == num_tokens
    assert meta.max_ctx_len == max_ctx_len
    assert meta.is_verified_lossless is True
    assert unpacked_bitstream == raw_bitstream

    # Checksum verification
    assert verify_unp_checksum(meta, plaintext) is True
    assert verify_unp_checksum(meta, "Corrupted text") is False


def test_invalid_header_rejection():
    # Header too small
    with pytest.raises(ValueError, match="less than|smaller than"):
        unpack_unp_container(b"SHORT")

    # Corrupted magic bytes
    bad_magic = b"BADMAG\x01\x01\x00\x00\x00\x08\x02\x00\x00\x00\x00\x00"
    with pytest.raises(ValueError, match="magic bytes"):
        unpack_unp_container(bad_magic)

    # Unsupported version
    bad_version = b"UNPLOW\x99\x01\x00\x00\x00\x08\x02\x00\x00\x00\x00\x00"
    with pytest.raises(ValueError, match="version"):
        unpack_unp_container(bad_version)

# end of file: unplow/tests/test_container.py
