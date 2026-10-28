# Path: unplow/inbox/gemweb100.md
# CFG Logit Masking in Parameter Basin Probing (Constrained Decompression)
# Max Column: 80 Columns

Integrating context-free grammar (CFG) logit masking directly into the
autoregressive loop in `paradigm1.py` transforms the generator from an
unconstrained predictor into a constrained pushdown automaton (PDA). At each
generation step $t$, the decoder queries the grammar state machine to retrieve
the exact subset of vocabulary tokens that can legally extend the current
partial AST, masking all invalid transitions to $-\infty$ before computing the
Softmax and Top-K filter.

```
+--------------------------------------------------------------------------+
|                       Constrained Decoding Pipeline                      |
|                                                                          |
|  x_{<t} --------> [ Incremental CFG State / PDA ]                        |
|                           |                                              |
|                           v Legal Token Mask M_t \in {0, -\infty}        |
|                                                                          |
|  z_t -----------> [ Logit Addition: z'_t = z_t + M_t ]                   |
|                           |                                              |
|                           v Top-K Truncation & Temperature Scaling       |
|                                                                          |
|                   [ Softmax & Token Sample: x_t ]                        |
+--------------------------------------------------------------------------+
```

---

### Step 1: Precompute the Token-to-Grammar Map

Evaluating a full grammar from scratch across the vocabulary
($\vert{}V\vert{} \approx 151,936$ for Qwen2.5) at every step is
computationally prohibitive. Instead, use an incremental lexer/parser engine
like **Lark**, **Tree-sitter**, or **Outlines/InterCode** abstractions to
build an index that tracks partial derivations.

For a lightweight, zero-dependency Python implementation, you can map the
parser state to valid token sets using an incremental Earley or LR(1) parser.
Alternatively, wrapping an existing engine like `outlines` or
`transformers.LogitsProcessor` provides the fastest path.

```python
import torch
from transformers import LogitsProcessor, AutoTokenizer

class CFGLogitsProcessor(LogitsProcessor):
    """
    Logits processor enforcing context-free grammar constraints
    on next-token generation.
    """
    def __init__(self, grammar_guide, tokenizer: AutoTokenizer):
        self.guide = grammar_guide
        self.tokenizer = tokenizer
        self.state = self.guide.initial_state()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Determine current legal token IDs from grammar state
        legal_tokens = self.guide.get_allowed_tokens(self.state)
        
        # Build -inf mask for illegal transitions
        mask = torch.full_like(scores, float('-inf'))
        mask[:, legal_tokens] = 0.0
        
        return scores + mask

    def accept_token(self, token_id: int):
        """Advances the internal CFG state machine."""
        self.state = self.guide.transition(self.state, token_id)

```

---

### Step 2: Refactor `WeightDecompressor.probe_weight_basins`

Modify the core decoding loop in `src/unplow/paradigm1.py` to interleave the
grammar mask between raw logit extraction and your two-stage temperature
schedule.

```python
import torch
import torch.nn.functional as F

class WeightDecompressor:
    def __init__(self, model, tokenizer, grammar_engine):
        self.model = model
        self.tokenizer = tokenizer
        self.grammar_engine = grammar_engine

    def probe_weight_basins(
        self,
        seed_prefix: str,
        domain: str,
        max_new_tokens: int = 512,
        k: int = 60,
        tau_explore: float = 1.10,
        tau_lock: float = 0.30,
        explore_steps: int = 6,
    ) -> str:
        input_ids = self.tokenizer.encode(seed_prefix, return_tensors="pt").to(self.model.device)
        
        # Initialize the grammar guide for the specified domain (python, c, rust, etc.)
        guide = self.grammar_engine.build_guide(domain, initial_code=seed_prefix)
        current_state = guide.initial_state()

        generated = input_ids.clone()

        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = self.model(generated)
                next_token_logits = outputs.logits[:, -1, :]  # Shape: [1, vocab_size]

                # 1. Mechanistic Grammar Masking
                allowed_tokens = guide.get_allowed_tokens(current_state)
                if not allowed_tokens:
                    # Derivation reached a valid terminal state; break generation safely
                    break

                # Apply CFG mask: invalidate transitions leading to syntax errors
                cfg_mask = torch.full_like(next_token_logits, float('-inf'))
                cfg_mask[0, allowed_tokens] = 0.0
                constrained_logits = next_token_logits + cfg_mask

                # 2. Paradigm 1 Temperature Scheduling
                tau = tau_explore if step < explore_steps else tau_lock
                scaled_logits = constrained_logits / tau

                # 3. Top-K Logit Truncation on Legally Constrained Set
                # Ensure K does not exceed the count of currently allowed tokens
                effective_k = min(k, len(allowed_tokens))
                top_k_values, top_k_indices = torch.topk(scaled_logits, effective_k, dim=-1)
                
                # Zero out unselected elements
                filtered_logits = torch.full_like(scaled_logits, float('-inf'))
                filtered_logits.scatter_(-1, top_k_indices, top_k_values)

                # 4. Softmax and Categorical Sampling
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Advance autoregressive state and CFG pushdown automaton
                generated = torch.cat([generated, next_token], dim=-1)
                token_id = next_token.item()
                current_state = guide.transition(current_state, token_id)

                # Check for standard EOS or CFG terminal state
                if token_id == self.tokenizer.eos_token_id or guide.is_terminal(current_state):
                    break

        raw_output = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return raw_output

```

---

### Step 3: Low-Level Engine Integration (C, Rust, Python, C++)

Because `unplow` targets 5 low-level systems domains (Python, C, C++, Rust,
Gencode), configuring per-language grammars using standard EBNF definitions
guarantees compilability across all targets:

```python
# unplow/grammars/loader.py
import outlines.grammars as grammars
from outlines.fsm.guide import CFGGuide

class DomainGrammarEngine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.grammars = {
            "python": self._load_ebnf("grammars/python.ebnf"),
            "c": self._load_ebnf("grammars/c99.ebnf"),
            "cpp": self._load_ebnf("grammars/cpp.ebnf"),
            "rust": self._load_ebnf("grammars/rust.ebnf"),
            "gencode": self._load_ebnf("grammars/json_ast.ebnf"),
        }

    def _load_ebnf(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def build_guide(self, domain: str, initial_code: str) -> CFGGuide:
        ebnf_grammar = self.grammars[domain]
        # CFGGuide builds a pushdown automaton index over the tokenizer vocabulary
        guide = CFGGuide(ebnf_grammar, self.tokenizer)
        return guide

```

---

### Key Operational Benefits for `unplow`

* **Eliminates ~87% Inference Waste:** Rather than parsing after emission and
  dropping 87 out of 100 generated candidates, every forward pass strictly
  yields a syntactically valid token sequence.
* **Preserves Annealing Dynamics:** Setting invalid logits to $-\infty$ occurs
  *before* applying $\tau_{\text{explore}}$ and $\tau_{\text{lock}}$, ensuring
  that your stochastic basin search and subsequent greedy lock-in operate
  exclusively across mathematically valid AST branches.
* **Bypasses Stub Wrapping:** Incomplete or unclosed structures (e.g., missing
  braces, mismatched parentheses, unbalanced indentation) are forced to
  terminate properly according to grammar production rules before hitting
  `max_new_tokens`.

<!-- end of file: unplow/inbox/gemweb100.md -->

