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
        Creates a callable `mask_fn(input_ids, logits)` that biases or masks invalid transitions
        using domain grammar rules and structural keyword token sets.
        """
        if domain not in self.grammars or self.tokenizer is None:
            return None

        # Extract allowable keywords and symbols from domain grammar
        grammar_text = self.grammars[domain]
        keywords = set(re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"', grammar_text))
        if not keywords:
            return None

        # Build set of legal token IDs matching domain keywords
        legal_keyword_ids = set()
        for kw in keywords:
            tok_ids = self.tokenizer.encode(kw, add_special_tokens=False)
            if tok_ids:
                legal_keyword_ids.add(tok_ids[0])

        def mask_fn(input_ids: Any, logits: Any) -> Any:
            """
            Applies soft grammar biasing: boosts legal syntax tokens by +1.5 logit units
            to prioritize legitimate syntactic transitions while allowing valid variable names.
            """
            if logits is None or logits.shape[-1] == 0:
                return logits

            # Apply positive bias to recognized keywords in the domain grammar
            for kid in legal_keyword_ids:
                if kid < logits.shape[-1]:
                    logits[:, kid] = logits[:, kid] + 1.5

            return logits

        return mask_fn

# end of file: unplow/src/unplow/grammars/__init__.py
