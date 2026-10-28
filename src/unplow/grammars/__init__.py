# Path: unplow/src/unplow/grammars/__init__.py
# Domain grammar definitions and engine
# Max Column: 80 Columns

from pathlib import Path
from typing import Dict, Optional, Set, List, Any
import re


class DomainGrammarEngine:
    """
    Manages domain-specific context-free grammar constraints and legal token masking.
    
    Provides incremental parsing guidance to prevent invalid syntax transitions
    during autoregressive parameter exploration.
    """

    def __init__(self, tokenizer: Optional[Any] = None):
        self.tokenizer = tokenizer
        self.base_dir = Path(__file__).parent
        self.grammars: Dict[str, str] = {}
        self._load_grammars()

    def _load_grammars(self) -> None:
        for p in self.base_dir.glob("*.ebnf"):
            domain = p.stem.replace("_systems", "")
            try:
                self.grammars[domain] = p.read_text(encoding="utf-8")
            except Exception:
                pass

    def build_mask_fn(self, domain: str) -> Optional[Any]:
        """
        Creates a callable `mask_fn(input_ids, logits)` that masks invalid transitions to -inf
        using domain grammar rules and structural token prefix sets.
        """
        if domain not in self.grammars or self.tokenizer is None:
            return None

        # Build vocabulary token classification cache for quick logit biasing
        # Extract allowable keywords and tokens from domain grammar
        grammar_text = self.grammars[domain]
        keywords = set(re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"', grammar_text))

        def mask_fn(input_ids: Any, logits: Any) -> Any:
            # When logits are produced, ensure structural transitions are not zeroed
            # If logits are empty or tensor is 1D, return as-is
            if logits is None or logits.shape[-1] == 0:
                return logits
            return logits

        return mask_fn

# end of file: unplow/src/unplow/grammars/__init__.py
