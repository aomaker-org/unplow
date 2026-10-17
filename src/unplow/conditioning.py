# Path: unplow/src/unplow/conditioning.py
"""Zero-Bit Domain Prior Pre-Warming & Fast Vectorized PyTorch Integer CDF.

Provides:
1. Deterministic Domain Conditioning Priors packed into the upper nibble of the
   existing 18-byte .unp binary container `flags` field (zero bitstream overhead).
2. KV-cache pre-warming to eliminate token-0 cold-start entropy on technical formats.
3. Minimum Description Length (MDL) auto-router evaluating the optimal domain prior (Rissanen, 1978).
4. Pure PyTorch int64 cumulative frequency quantization (logits_to_integer_cdf_fast).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

TOTAL_FREQ = 1 << 24  # 16,777,216

# Deterministic Domain Conditioning Priors (4-bit ID: 0..15 stored in flags >> 4)
DOMAIN_PRIORS: Dict[int, Tuple[str, str]] = {
    0: ("raw", ""),  # Unconditioned BOS (v1 default backward compatibility)
    1: (
        "prose_tech",
        "Technical specification on information theory, sequence modeling, and lossless data compression:\n",
    ),
    2: (
        "python_code",
        '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Core module implementation."""\nimport os\nimport sys\n\n',
    ),
    3: (
        "c_firmware",
        "/* SPDX-License-Identifier: MIT\n * Low-level system firmware and hardware validation routines.\n */\n"
        "#include <stdint.h>\n#include <stddef.h>\n\n",
    ),
    4: (
        "json_telemetry",
        '{"schema_version": 1, "status": "verified", "telemetry": ',
    ),
}

DOMAIN_NAME_TO_ID: Dict[str, int] = {
    name: dom_id for dom_id, (name, _) in DOMAIN_PRIORS.items()
}


def resolve_domain_id(identifier: Union[int, str]) -> int:
    """Resolves an integer or string name into a valid domain ID (0..15)."""
    if isinstance(identifier, int):
        if not (0 <= identifier <= 15):
            raise ValueError(f"Domain ID must be in 0..15, got {identifier}")
        return identifier
    identifier_str = str(identifier).strip().lower()
    if identifier_str.isdigit():
        dom_id = int(identifier_str)
        if not (0 <= dom_id <= 15):
            raise ValueError(f"Domain ID must be in 0..15, got {dom_id}")
        return dom_id
    if identifier_str in DOMAIN_NAME_TO_ID:
        return DOMAIN_NAME_TO_ID[identifier_str]
    raise ValueError(f"Unknown domain identifier: {identifier!r}. Available: {list(DOMAIN_NAME_TO_ID.keys())}")


def pack_flags_with_domain(
    verify_hash: bool,
    domain_id: int = 0,
    cache_mixture: bool = False,
    bigram_cache: bool = False,
) -> int:
    """Packs domain_id (bits 4..7), bigram_cache (bit 2), cache_mixture (bit 1), verify_hash (bit 0)."""
    if not (0 <= domain_id <= 15):
        raise ValueError(f"domain_id must be in 0..15, got {domain_id}")
    low_nibble = (
        (0x01 if verify_hash else 0x00)
        | (0x02 if cache_mixture else 0x00)
        | (0x04 if bigram_cache else 0x00)
    )
    return ((domain_id & 0x0F) << 4) | low_nibble


def unpack_flags_with_domain(flags: int) -> Tuple[bool, int]:
    """Extracts (verify_hash, domain_id) from the uint8 flags byte."""
    verify_hash = bool(flags & 0x01)
    domain_id = (flags >> 4) & 0x0F
    return verify_hash, domain_id


@torch.no_grad()
def prewarm_kv_cache_for_domain(
    model: Any,
    tokenizer: Any,
    domain_id: int,
    bos_token_id: int,
    device: str,
) -> Tuple[torch.Tensor, Optional[Any]]:
    """Pre-warms the LLM KV-cache using the deterministic preamble for `domain_id`.

    Returns (curr_input_tensor, past_key_values) ready for token t=0.
    """
    _, preamble = DOMAIN_PRIORS.get(domain_id, DOMAIN_PRIORS[0])
    if not preamble:
        return torch.tensor([[bos_token_id]], device=device), None

    preamble_ids = tokenizer.encode(preamble, add_special_tokens=False)
    full_prefix = [bos_token_id] + preamble_ids

    # Feed prefix tokens through the model to build past_key_values in a single batched pass
    if len(full_prefix) > 1:
        prefix_tensor = torch.tensor([full_prefix[:-1]], device=device)
        outputs = model(input_ids=prefix_tensor, use_cache=True)
        past_kv = outputs.past_key_values
    else:
        past_kv = None

    # The final token of the preamble serves as curr_input for step t=0
    curr_input = torch.tensor([[full_prefix[-1]]], device=device)
    return curr_input, past_kv


@torch.no_grad()
def logits_to_integer_cdf_fast(logits: torch.Tensor) -> np.ndarray:
    """Vectorized PyTorch implementation of monotonic integer CDF generation.

    Computes frequencies and cumulative sums inside PyTorch before a single
    contiguous int64 buffer view, avoiding multi-step NumPy conversions on
    large vocabularies.
    """
    probs = torch.softmax(logits.to(torch.float64), dim=-1)
    vocab_size = probs.shape[0]
    remaining_mass = TOTAL_FREQ - vocab_size
    if remaining_mass <= 0:
        raise ValueError(f"TOTAL_FREQ ({TOTAL_FREQ}) must exceed vocab_size ({vocab_size})")

    freqs = torch.floor(probs * remaining_mass).to(torch.int64) + 1
    remainder = TOTAL_FREQ - int(freqs.sum().item())
    max_idx = int(torch.argmax(probs).item())
    freqs[max_idx] += remainder

    cdf = torch.zeros(vocab_size + 1, dtype=torch.int64, device=probs.device)
    torch.cumsum(freqs, dim=0, out=cdf[1:])
    return cdf.cpu().numpy()


@torch.no_grad()
def select_optimal_domain_mdl(
    model: Any,
    tokenizer: Any,
    plaintext: str,
    device: str,
    bos_token_id: int,
) -> Tuple[int, str, float]:
    """Minimum Description Length (MDL) Auto-Router (Rissanen, 1978).

    Evaluates the exact Shannon cross-entropy bit cost of `plaintext` across all
    registered DOMAIN_PRIORS in batched single-pass mode and returns
    (best_domain_id, domain_name, expected_bits_per_byte).
    """
    target_ids = tokenizer.encode(plaintext, add_special_tokens=False)
    if not target_ids:
        return 0, "raw", 0.0

    num_bytes = max(len(plaintext.encode("utf-8")), 1)
    best_id = 0
    best_name = "raw"
    best_bpb = float("inf")
    ln2 = float(np.log(2.0))

    for dom_id, (dom_name, preamble) in DOMAIN_PRIORS.items():
        preamble_ids = tokenizer.encode(preamble, add_special_tokens=False) if preamble else []
        prefix_ids = [bos_token_id] + preamble_ids
        seq_ids = prefix_ids + target_ids

        input_tensor = torch.tensor([seq_ids], device=device)
        outputs = model(input_tensor)
        logits = outputs.logits[0].to(torch.float32) if hasattr(outputs, "logits") else outputs[0].to(torch.float32)

        # Evaluate cross-entropy strictly on the target slice
        start_idx = len(prefix_ids) - 1
        end_idx = len(seq_ids) - 1
        target_logits = logits[start_idx:end_idx, :]
        target_labels = torch.tensor(target_ids, device=device)

        loss_nats = torch.nn.functional.cross_entropy(
            target_logits, target_labels, reduction="sum"
        ).item()
        bpb = (loss_nats / ln2) / num_bytes

        if bpb < best_bpb:
            best_bpb = bpb
            best_id = dom_id
            best_name = dom_name

    return best_id, best_name, round(best_bpb, 3)

# end of file: unplow/src/unplow/conditioning.py
