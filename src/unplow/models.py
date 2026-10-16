# Path: unplow/src/unplow/models.py
"""Model substrate, discovery, and loader utilities for unplow.

Coordinates local LLM weights, HuggingFace snapshots, GGUF binaries, Ollama
endpoints, and provides deterministic mock models for fast offline testing.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("unplow.models")

# Known default local snapshot locations for Qwen2.5-1.5B
LOCAL_QWEN_SNAPSHOTS = [
    Path("models/qwen2.5-1.5b-instruct"),
    Path("/home/fekerr/src/unplow/models/qwen2.5-1.5b-instruct"),
    Path(
        "/home/fekerr/src/local-code-plower/workspace/models/cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    ),
]


def resolve_model_path(model_identifier: Optional[str] = None) -> str:
    """Resolve a model identifier or path to an active local weight directory."""
    if model_identifier and model_identifier not in ("auto", "qwen", "default"):
        p = Path(model_identifier).expanduser().resolve()
        if p.exists():
            return str(p)
        return model_identifier

    # Check known local snapshots
    for snap in LOCAL_QWEN_SNAPSHOTS:
        try:
            resolved = snap.resolve()
            if resolved.exists() and (resolved / "config.json").exists():
                logger.info("Auto-resolved local model snapshot: %s", resolved)
                return str(resolved)
        except Exception:
            continue

    # Fallback to standard HuggingFace identifier
    return "Qwen/Qwen2.5-1.5B"


def load_causal_lm(
    model_identifier: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Any, Any, str]:
    """Load AutoTokenizer and AutoModelForCausalLM from local weights."""
    target_path = resolve_model_path(model_identifier)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)

    logger.info("Loading tokenizer from %s...", target_path)
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=Path(target_path).exists())

    logger.info("Loading causal LM model from %s on %s...", target_path, device)
    model = AutoModelForCausalLM.from_pretrained(
        target_path,
        dtype=dtype,
        local_files_only=Path(target_path).exists(),
    ).to(device)
    model.eval()

    return tokenizer, model, target_path


def audit_system_models() -> List[Dict[str, Any]]:
    """Scan and return all local LLM artifacts available on the system."""
    models: List[Dict[str, Any]] = []

    # 1. HuggingFace / Safetensors snapshots
    for snap in LOCAL_QWEN_SNAPSHOTS:
        try:
            r = snap.resolve()
            if r.exists() and (r / "config.json").exists():
                models.append({
                    "type": "safetensors_snapshot",
                    "name": "Qwen2.5-1.5B-Instruct",
                    "path": str(r),
                    "active_link": str(snap),
                    "parameters": "1.54B",
                    "status": "READY",
                })
                break
        except Exception:
            pass

    # 2. Local GGUF files in ~/ai_models
    ai_models_dir = Path("/home/fekerr/ai_models")
    if ai_models_dir.exists():
        for gguf_file in ai_models_dir.glob("*.gguf"):
            models.append({
                "type": "gguf",
                "name": gguf_file.name,
                "path": str(gguf_file.resolve()),
                "size_mb": round(gguf_file.stat().st_size / (1024 * 1024), 2),
                "status": "READY",
            })

    # 3. Ollama local daemon
    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        req = urllib.request.Request(f"{ollama_host}/api/tags", headers={"User-Agent": "unplow"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for m in data.get("models", []):
                models.append({
                    "type": "ollama",
                    "name": m.get("name"),
                    "size_mb": round(m.get("size", 0) / (1024 * 1024), 2),
                    "digest": m.get("digest", "")[:12],
                    "status": "ONLINE",
                })
    except Exception:
        pass

    return models


# ------------------------------------------------------------------------------
# Deterministic Mock Components for High-Speed Unit Testing
# ------------------------------------------------------------------------------

class MockByteTokenizer:
    """Deterministic byte-level tokenizer for zero-dependency unit tests."""

    def __init__(self, vocab_size: int = 260):
        self.vocab_size = vocab_size
        self.bos_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = [b + 2 for b in text.encode("utf-8")]
        if add_special_tokens:
            return [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, tokens: List[int], clean_up_tokenization_spaces: bool = False, skip_special_tokens: bool = False) -> str:
        clean = []
        for t in tokens:
            if skip_special_tokens and t in (self.bos_token_id, self.eos_token_id):
                continue
            if 2 <= t < self.vocab_size:
                clean.append((t - 2) % 256)
        return bytes(clean).decode("utf-8", errors="replace")


class MockModelOutput:
    """Mock container for model logits and past key values."""

    def __init__(self, logits: torch.Tensor, past_key_values: Any = None):
        self.logits = logits
        self.past_key_values = past_key_values


class MockCausalLM(nn.Module):
    """Deterministic causal language model for unit tests."""

    def __init__(self, vocab_size: int = 260, hidden_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        # Deterministic weights
        nn.init.orthogonal_(self.embed.weight)
        nn.init.orthogonal_(self.head.weight)
        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
    ) -> MockModelOutput:
        x = self.embed(input_ids)
        logits = self.head(x)
        return MockModelOutput(logits=logits, past_key_values=past_key_values)

# end of file: unplow/src/unplow/models.py
