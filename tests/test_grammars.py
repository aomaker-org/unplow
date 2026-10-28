# Path: tests/test_grammars.py
# Unit tests for DomainGrammarEngine and keyword logit biasing
# Max Line Length: 120 Chars

from pathlib import Path
import pytest
import torch

from unplow.grammars import DomainGrammarEngine


class DummyTokenizer:
    """Mock tokenizer mapping vocabulary words to fixed token IDs."""

    def __init__(self):
        self.vocab = {
            "typedef": 10,
            "struct": 11,
            "uint32_t": 12,
            "return": 13,
            "pub": 20,
            "fn": 21,
            "impl": 22,
        }

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self.vocab:
            return [self.vocab[text]]
        return [999]


def test_grammar_engine_loading():
    """Verify that EBNF grammar files for c and rust are loaded on initialization."""
    engine = DomainGrammarEngine()
    assert "c" in engine.grammars
    assert "rust" in engine.grammars
    assert len(engine.grammars["c"]) > 0
    assert len(engine.grammars["rust"]) > 0


def test_grammar_engine_build_mask_fn_missing():
    """Verify build_mask_fn returns None when domain is unknown or tokenizer is missing."""
    engine = DomainGrammarEngine(tokenizer=None)
    assert engine.build_mask_fn("c") is None

    tokenizer = DummyTokenizer()
    engine_with_tok = DomainGrammarEngine(tokenizer=tokenizer)
    assert engine_with_tok.build_mask_fn("unknown_domain") is None


def test_grammar_engine_logit_biasing():
    """Verify that build_mask_fn applies +1.5 soft bias to legal keywords and leaves others unchanged."""
    tokenizer = DummyTokenizer()
    engine = DomainGrammarEngine(tokenizer=tokenizer)

    mask_fn = engine.build_mask_fn("c")
    assert mask_fn is not None

    vocab_size = 1000
    batch_size = 1
    input_ids = torch.tensor([[1, 2, 3]])
    initial_logits = torch.zeros((batch_size, vocab_size))

    # Apply mask_fn to cloned logits
    biased_logits = mask_fn(input_ids, initial_logits.clone())

    # Check that known C keywords defined in DummyTokenizer received +1.5 bias
    # e.g., "typedef" (10), "struct" (11), "uint32_t" (12), "return" (13)
    for kw_id in [10, 11, 12, 13]:
        assert biased_logits[0, kw_id].item() == pytest.approx(1.5, rel=1e-5)

    # Check that unrelated token IDs remain 0.0
    for other_id in [0, 1, 5, 20, 50, 100]:
        assert biased_logits[0, other_id].item() == pytest.approx(0.0, rel=1e-5)


def test_grammar_engine_empty_logits():
    """Verify mask_fn gracefully handles empty logits tensor."""
    tokenizer = DummyTokenizer()
    engine = DomainGrammarEngine(tokenizer=tokenizer)
    mask_fn = engine.build_mask_fn("c")
    assert mask_fn is not None

    empty_logits = torch.empty((1, 0))
    result = mask_fn(torch.tensor([[1]]), empty_logits)
    assert result.shape[-1] == 0

# <!-- end of file: tests/test_grammars.py -->
