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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_identifier == "mock":
        tokenizer = MockByteTokenizer(vocab_size=260)
        model = MockCausalLM(vocab_size=260, hidden_dim=32).to(device)
        return tokenizer, model, "mock"

    target_path = resolve_model_path(model_identifier)
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


def load_local_model(model_identifier: Optional[str] = None) -> Tuple[Any, Any]:
    """Convenience helper returning (model, tokenizer) from local weights."""
    tokenizer, model, _ = load_causal_lm(model_identifier=model_identifier)
    return model, tokenizer


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

class MockBatchEncoding:
    """Mock container providing .input_ids for HuggingFace tokenizer output."""

    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


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

    def decode(
        self,
        tokens: List[int],
        clean_up_tokenization_spaces: bool = False,
        skip_special_tokens: bool = False,
    ) -> str:
        clean = []
        for t in tokens:
            if skip_special_tokens and t in (self.bos_token_id, self.eos_token_id):
                continue
            if 2 <= t < self.vocab_size:
                clean.append((t - 2) % 256)
        return bytes(clean).decode("utf-8", errors="replace")

    def __call__(self, text: str, return_tensors: Optional[str] = None, **kwargs: Any) -> Any:
        tokens = self.encode(text)
        if return_tensors == "pt":
            return MockBatchEncoding(input_ids=torch.tensor([tokens], dtype=torch.long))
        return {"input_ids": tokens}


class DeterministicMockTokenizer:
    """Deterministic mock tokenizer supporting multi-character tokens for mechanistic tests."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.id_to_token: Dict[int, str] = {0: "<bos>", 1: "<eos>"}
        self.token_to_id: Dict[str, int] = {"<bos>": 0, "<eos>": 1}
        sample_words = [
            "alpha", "beta", "gamma", "delta", "epsilon", "def", "class", "return",
            "self", "import", "tensor", "model", "cache", "basin", "circuit", "trace",
            "layer", "neuron", "weight", "lossless", "unplow", "stream", "norm", "gain",
        ]
        for i, w in enumerate(sample_words):
            if i + 2 < self.vocab_size:
                idx = i + 2
                self.id_to_token[idx] = w
                self.token_to_id[w] = idx
        for i in range(len(sample_words) + 2, self.vocab_size):
            w = f"tok_{i}"
            self.id_to_token[i] = w
            self.token_to_id[w] = i

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens: List[int] = []
        for word in text.replace("\n", " \n ").split(" "):
            if not word:
                continue
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                idx = 2 + (abs(hash(word)) % (self.vocab_size - 2))
                self.token_to_id[word] = idx
                self.id_to_token[idx] = word
                tokens.append(idx)
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(
        self,
        tokens: List[int],
        clean_up_tokenization_spaces: bool = False,
        skip_special_tokens: bool = False,
    ) -> str:
        words: List[str] = []
        for t in tokens:
            if skip_special_tokens and t in (self.bos_token_id, self.eos_token_id):
                continue
            words.append(self.id_to_token.get(t, f"tok_{t}"))
        return " ".join(words)

    def __call__(self, text: str, return_tensors: Optional[str] = None, **kwargs: Any) -> Any:
        tokens = self.encode(text)
        if return_tensors == "pt":
            return MockBatchEncoding(input_ids=torch.tensor([tokens], dtype=torch.long))
        return {"input_ids": tokens}


class MockModelOutput:
    """Mock container for model logits, past key values, and hidden states."""

    def __init__(
        self,
        logits: torch.Tensor,
        past_key_values: Any = None,
        hidden_states: Any = None,
    ):
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states


class MockMLP(nn.Module):
    def __init__(self, hidden_dim: int = 32, ffn_dim: int = 64):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.gate_proj.weight)
        nn.init.orthogonal_(self.up_proj.weight)
        nn.init.orthogonal_(self.down_proj.weight)


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int = 32, ffn_dim: int = 64):
        super().__init__()
        self.mlp = MockMLP(hidden_dim, ffn_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden_dim)


class MockModelBody(nn.Module):
    def __init__(self, vocab_size: int = 260, hidden_dim: int = 32, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([MockTransformerLayer(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.orthogonal_(self.embed_tokens.weight)


class MockCausalLM(nn.Module):
    """Deterministic causal language model with mock transformer layers for unit tests."""

    def __init__(self, vocab_size: int = 260, hidden_dim: int = 32):
        super().__init__()
        torch.manual_seed(42)
        self.vocab_size = vocab_size
        self.model = MockModelBody(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=4)
        self.embed = self.model.embed_tokens
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head = self.head
        nn.init.orthogonal_(self.head.weight)
        self.eval()

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
    ) -> MockModelOutput:
        x = self.embed(input_ids)
        hidden_states = None
        if output_hidden_states:
            h_states = [x]
            curr_h = x
            for layer in self.model.layers:
                delta = layer.mlp.down_proj(layer.mlp.up_proj(layer.post_attention_layernorm(curr_h)))
                curr_h = curr_h + delta
                h_states.append(curr_h)
            hidden_states = tuple(h_states)

        x = self.model.norm(x)
        logits = self.head(x)
        new_past = past_key_values
        if use_cache:
            seq_len = input_ids.shape[1]
            if past_key_values is None:
                new_past = tuple(
                    (torch.zeros(1, 2, seq_len, 8), torch.zeros(1, 2, seq_len, 8))
                    for _ in range(len(self.model.layers))
                )
            elif isinstance(past_key_values, tuple):
                new_past = tuple(
                    (
                        torch.cat([k, torch.zeros(1, 2, seq_len, 8)], dim=2),
                        torch.cat([v, torch.zeros(1, 2, seq_len, 8)], dim=2),
                    )
                    for k, v in past_key_values
                )
        return MockModelOutput(
            logits=logits,
            past_key_values=new_past,
            hidden_states=hidden_states,
        )


# Aliases for explicit mechanistic unit testing
DeterministicMockCausalLM = MockCausalLM

# end of file: unplow/src/unplow/models.py
