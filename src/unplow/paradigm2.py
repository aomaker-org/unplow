# Path: unplow/src/unplow/paradigm2.py
"""Paradigm 2: Lossless Autoregressive Arithmetic Compression & Decompression Engine.

Pairs an autoregressive causal language model with a 32-bit fixed-point integer
arithmetic coder (Delétang et al., 2023; Bellard, 2021; Witten, Neal, & Cleary, 1987).
Provides:
1. Exact, bit-for-bit lossless text compression and decompression.
2. Generative decompression: maps arbitrary byte bitstreams or seeds
   deterministically into coherent text across the learned model manifold.
"""

from dataclasses import asdict, dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .container import (
    UnpArchiveMetadata,
    pack_unp_container,
    slice_kv_cache,
    unpack_unp_container,
    verify_unp_checksum,
)
from .models import load_causal_lm

logger = logging.getLogger("unplow.paradigm2")

# 32-bit Fixed-Point Arithmetic Coding Constants (Witten, Neal, & Cleary, 1987)
PRECISION_BITS = 32
MAX_VAL = (1 << PRECISION_BITS) - 1          # 0xFFFFFFFF
HALF = 1 << (PRECISION_BITS - 1)             # 0x80000000
QUARTER = 1 << (PRECISION_BITS - 2)          # 0x40000000
THREE_QUARTERS = 3 << (PRECISION_BITS - 2)   # 0xC0000000
TOTAL_FREQ = 1 << 24                         # 16,777,216 (supports vocabularies > 150k)


@dataclass
class CompressionStats:
    """Metrics and profiling for an arithmetic compression execution."""

    original_bytes: int
    compressed_bytes: int
    num_tokens: int
    bits_per_byte: float
    compression_ratio: float
    savings_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMArithmeticCodec:
    """Lossless compressor, decompressor, and latent manifold decompressor."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None and tokenizer is not None:
            self.tokenizer = tokenizer
            self.model = model
            self.model_path = "custom"
        else:
            self.tokenizer, self.model, self.model_path = load_causal_lm(
                model_identifier=model_name,
                device=self.device,
                dtype=torch.float32,  # Deterministic logit evaluation
            )

        self.bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if self.bos_token_id is None:
            self.bos_token_id = getattr(self.tokenizer, "eos_token_id", 0)

    @staticmethod
    def logits_to_integer_cdf(logits: torch.Tensor) -> np.ndarray:
        """Convert raw 1D token logits into a strictly monotonic integer CDF.

        Ensures C[0] = 0, C[V] = TOTAL_FREQ, and C[i+1] - C[i] >= 1 for all i.
        """
        probs = torch.softmax(logits.to(torch.float64), dim=-1).cpu().numpy()
        vocab_size = probs.shape[0]
        remaining_mass = TOTAL_FREQ - vocab_size
        if remaining_mass <= 0:
            raise ValueError(
                f"TOTAL_FREQ ({TOTAL_FREQ}) must exceed model vocab_size ({vocab_size})."
            )

        # Every token receives at least 1 unit of frequency mass
        freqs = np.floor(probs * remaining_mass).astype(np.int64) + 1
        remainder = TOTAL_FREQ - int(np.sum(freqs))
        freqs[np.argmax(probs)] += remainder

        cdf = np.zeros(vocab_size + 1, dtype=np.int64)
        np.cumsum(freqs, out=cdf[1:])
        return cdf

    @torch.no_grad()
    def compress(
        self,
        text: str,
        max_ctx_len: Optional[int] = None,
    ) -> Tuple[bytes, int, CompressionStats]:
        """Compress text into an arithmetic bitstream using the LLM distribution.

        Returns (compressed_bytes, num_tokens, stats).
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(token_ids)
        raw_bytes = text.encode("utf-8")
        num_raw_bytes = max(len(raw_bytes), 1)

        low = 0
        high = MAX_VAL
        pending_bits = 0
        output_bits: List[int] = []

        def emit_bit(bit: int):
            nonlocal pending_bits
            output_bits.append(bit)
            while pending_bits > 0:
                output_bits.append(1 - bit)
                pending_bits -= 1

        curr_input = torch.tensor([[self.bos_token_id]], device=self.device)
        past_key_values = None

        for step, token in enumerate(token_ids):
            if max_ctx_len and max_ctx_len >= 4 and step > 0 and step % (max_ctx_len // 2) == 0:
                past_key_values = slice_kv_cache(past_key_values, keep_tokens=max_ctx_len // 2)

            outputs = self.model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            cdf = self.logits_to_integer_cdf(outputs.logits[0, -1, :])

            span = high - low + 1
            c_low = int(cdf[token])
            c_high = int(cdf[token + 1])

            high = low + (span * c_high) // TOTAL_FREQ - 1
            low = low + (span * c_low) // TOTAL_FREQ

            while True:
                if high < HALF:
                    emit_bit(0)
                elif low >= HALF:
                    emit_bit(1)
                    low -= HALF
                    high -= HALF
                elif low >= QUARTER and high < THREE_QUARTERS:
                    pending_bits += 1
                    low -= QUARTER
                    high -= QUARTER
                else:
                    break
                low = (low << 1) & MAX_VAL
                high = ((high << 1) | 1) & MAX_VAL

            curr_input = torch.tensor([[token]], device=self.device)

        # Flush final interval state
        pending_bits += 1
        if low < QUARTER:
            emit_bit(0)
        else:
            emit_bit(1)

        # Pack bits into bytes
        while len(output_bits) % 8 != 0:
            output_bits.append(0)
        byte_array = bytearray()
        for i in range(0, len(output_bits), 8):
            byte_val = 0
            for b in output_bits[i : i + 8]:
                byte_val = (byte_val << 1) | b
            byte_array.append(byte_val)

        payload = bytes(byte_array)
        comp_len = len(payload)
        stats = CompressionStats(
            original_bytes=num_raw_bytes,
            compressed_bytes=comp_len,
            num_tokens=num_tokens,
            bits_per_byte=round((comp_len * 8.0) / num_raw_bytes, 3),
            compression_ratio=round(num_raw_bytes / max(comp_len, 1), 3),
            savings_percent=round(max(0.0, (1.0 - comp_len / num_raw_bytes) * 100.0), 2),
        )

        return payload, num_tokens, stats

    @torch.no_grad()
    def decompress(
        self,
        compressed_bytes: bytes,
        num_tokens: int,
        max_ctx_len: Optional[int] = None,
    ) -> str:
        """Decompress an arithmetic bitstream back to exact original text.

        Also operates as a deterministic generative manifold decompressor when
        supplied with arbitrary seed bytes.
        """
        bits: List[int] = []
        for byte_val in compressed_bytes:
            for shift in range(7, -1, -1):
                bits.append((byte_val >> shift) & 1)

        bit_idx = 0

        def read_bit() -> int:
            nonlocal bit_idx
            if bit_idx < len(bits):
                b = bits[bit_idx]
                bit_idx += 1
                return b
            return 0  # Zero-padding beyond payload EOF

        low = 0
        high = MAX_VAL
        value = 0
        for _ in range(PRECISION_BITS):
            value = (value << 1) | read_bit()

        decoded_tokens: List[int] = []
        curr_input = torch.tensor([[self.bos_token_id]], device=self.device)
        past_key_values = None

        for step in range(num_tokens):
            if max_ctx_len and max_ctx_len >= 4 and step > 0 and step % (max_ctx_len // 2) == 0:
                past_key_values = slice_kv_cache(past_key_values, keep_tokens=max_ctx_len // 2)

            outputs = self.model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            cdf = self.logits_to_integer_cdf(outputs.logits[0, -1, :])

            span = high - low + 1
            scaled_target = ((value - low + 1) * TOTAL_FREQ - 1) // span
            token = int(np.searchsorted(cdf, scaled_target, side="right") - 1)
            decoded_tokens.append(token)

            c_low = int(cdf[token])
            c_high = int(cdf[token + 1])
            high = low + (span * c_high) // TOTAL_FREQ - 1
            low = low + (span * c_low) // TOTAL_FREQ

            while True:
                if high < HALF:
                    pass
                elif low >= HALF:
                    value -= HALF
                    low -= HALF
                    high -= HALF
                elif low >= QUARTER and high < THREE_QUARTERS:
                    value -= QUARTER
                    low -= QUARTER
                    high -= QUARTER
                else:
                    break
                low = (low << 1) & MAX_VAL
                high = ((high << 1) | 1) & MAX_VAL
                value = ((value << 1) | read_bit()) & MAX_VAL

            curr_input = torch.tensor([[token]], device=self.device)

        return self.tokenizer.decode(decoded_tokens, clean_up_tokenization_spaces=False)

    def compress_to_container(
        self,
        text: str,
        max_ctx_len: int = 512,
        verify_hash: bool = True,
    ) -> Tuple[bytes, UnpArchiveMetadata, CompressionStats]:
        """Compress text directly into a self-verifying .unp binary archive."""
        payload, num_tokens, stats = self.compress(text, max_ctx_len=max_ctx_len)
        container_bytes = pack_unp_container(
            bitstream=payload,
            num_tokens=num_tokens,
            max_ctx_len=max_ctx_len,
            plaintext=text,
            verify_hash=verify_hash,
        )
        meta, _ = unpack_unp_container(container_bytes)
        return container_bytes, meta, stats

    def decompress_from_container(
        self,
        container_bytes: bytes,
    ) -> Tuple[str, UnpArchiveMetadata, bool]:
        """Decompress a .unp binary archive and verify cryptographic integrity."""
        meta, bitstream = unpack_unp_container(container_bytes)
        text = self.decompress(
            bitstream,
            num_tokens=meta.num_tokens,
            max_ctx_len=meta.max_ctx_len if meta.max_ctx_len > 0 else None,
        )
        is_valid = verify_unp_checksum(meta, text)
        return text, meta, is_valid

    def decompress_seed(
        self,
        seed: Union[bytes, str],
        num_tokens: int = 32,
    ) -> str:
        """Generatively decompress an arbitrary seed or digest through the LLM manifold.

        Takes arbitrary bytes or hashes and projects them into coherent natural
        language tokens proportional to the learned pre-training distribution.
        """
        if isinstance(seed, str):
            seed_bytes = hashlib.sha256(seed.encode("utf-8")).digest()
        else:
            seed_bytes = seed

        return self.decompress(seed_bytes, num_tokens=num_tokens)

# end of file: unplow/src/unplow/paradigm2.py
