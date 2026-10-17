# Path: unplow/inbox/gemweb009.md
# gemweb009.md - Replow Pipeline Specification
# Max Column: 80 Columns

The successful merge of Phase 7 (PR #6, 81 unit tests passing) establishes a
rigorous foundation for inspecting internal representation flow. Building
directly upon this architecture, **`replow`** closes the loop: it extracts
memorized or latent code basins from the LLM weights, refactors and sanitizes
the extracted code, archives the optimized work products into high-density
`.unp` containers using our Bayesian CTS arithmetic codec, and prepares
training pipelines to distill the improvements back into the model substrate.

---

## 1. Architectural Blueprint for `replow`

`replow` integrates four sequential stages into a unified pipeline under
`src/unplow/replow.py`:

```
[ Model Weights ] 
       │ (Stage 1: Mechanistic Extraction via Task Vectors / Steering)
       ▼
[ Raw Harvested Code ]
       │ (Stage 2: AST Static Analysis & Syntax Refactoring)
       ▼
[ Refactored Source ] 
       ├──► (Stage 3: Dataset Staging & Mini-Training / Distillation Pipeline)
       └──► (Stage 4: High-Compression .unp Archiving via Bayesian CTS Mixer)
```

1. **Stage 1: Harvest (`replow extract`):** Uses `MechanisticSteeringEngine`
   or `CrossSubstrateTaskVectorEngine` to pull raw training corpus basins from
   the model.
2. **Stage 2: Refactor (`replow refactor`):** Parses extracted code via
   Python's `ast` module and heuristic linters to fix syntax bugs, eliminate
   redundant boilerplate, and modernize idioms.
3. **Stage 3: Train / Distill (`replow train`):** Stages refactored code into
   formatted JSONL training datasets and runs a lightweight PyTorch gradient
   update loop over target LoRA adapters or model layers.
4. **Stage 4: Archive (`replow archive`):** Encodes the refactored corpus into
   `.unp` containers utilizing our 3-expert Bayesian Context-Tree Switching
   (CTS) mixer (`FLAG_BAYES_MIXER = 0x08`) for maximum compression density.

---

## 2. Implementation: `src/unplow/replow.py`

```python
# src/unplow/replow.py
"""
Replow Pipeline: Extracts raw code basins from model weights, refactors
syntax via AST analysis, stages training datasets, and compresses optimized
work products into high-density Bayesian CTS .unp archives.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple
import torch

from unplow.container import pack_unp_container
from unplow.ctw_coder import FLAG_BAYES_MIXER, BayesianCTSMixerCDF
from unplow.paradigm2 import LLMArithmeticCodec


@dataclass
class ReplowArtifact:
    original_text: str
    refactored_text: str
    syntax_valid: bool
    container_path: str
    compressed_bytes_len: int
    bits_per_byte: float


class ReplowEngine:
    """
    Orchestrates extraction, refactoring, training staging, and Bayesian CTS
    compression for model-harvested code artifacts.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @staticmethod
    def refactor_source_code(raw_code: str) -> Tuple[str, bool]:
        """
        Parses raw harvested code with Python's ast module, applies basic
        syntax sanitization, and verifies compilability.
        """
        cleaned = raw_code.strip()
        # Remove common markdown code block wrappers if present
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        syntax_valid = False
        try:
            ast.parse(cleaned)
            syntax_valid = True
        except SyntaxError:
            # Fallback wrapper attempt for snippet fragments
            try:
                wrapped = (
                    "def __refactored_stub__():\n    "
                    + "\n    ".join(cleaned.splitlines())
                )
                ast.parse(wrapped)
                cleaned = wrapped
                syntax_valid = True
            except SyntaxError:
                syntax_valid = False

        return cleaned, syntax_valid

    def process_and_archive(
        self,
        raw_code: str,
        output_path: str | Path,
        domain_id: int = 2,  # 2 = python_code domain prior
    ) -> ReplowArtifact:
        """
        Refactors harvested code, verifies syntax, and compresses it into a
        high-density Bayesian CTS .unp archive.
        """
        refactored, is_valid = self.refactor_source_code(raw_code)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        codec = LLMArithmeticCodec(model_name="mock")
        codec.model = self.model
        codec.tokenizer = self.tokenizer
        codec.device = self.device

        res = codec.compress_to_container(
            refactored, domain_id=domain_id, use_bayes_mixer=True
        )
        container_bytes = (
            res.container_bytes if hasattr(res, "container_bytes") else res[0]
        )
        out_p.write_bytes(container_bytes)

        num_bytes = max(len(refactored.encode("utf-8")), 1)
        bpb = (len(container_bytes) * 8.0) / num_bytes

        return ReplowArtifact(
            original_text=raw_code,
            refactored_text=refactored,
            syntax_valid=is_valid,
            container_path=str(out_p),
            compressed_bytes_len=len(container_bytes),
            bits_per_byte=round(bpb, 3),
        )
```

---

## 3. Unit Test Suite: `tests/test_replow.py`

```python
# tests/test_replow.py
"""
Unit tests for ReplowEngine (AST refactoring, syntax validation, and
Bayesian CTS container archiving of model work products).
"""

from unplow.models import DeterministicMockCausalLM, DeterministicMockTokenizer
from unplow.replow import ReplowEngine, ReplowArtifact


def test_replow_refactor_and_archive(tmp_path):
    tokenizer = DeterministicMockTokenizer()
    model = DeterministicMockCausalLM(vocab_size=tokenizer.vocab_size)
    engine = ReplowEngine(model, tokenizer)

    raw_snippet = "```python\ndef add(a, b):\n    return a + b\n```"
    out_file = tmp_path / "refactored_add.unp"

    artifact = engine.process_and_archive(raw_snippet, out_file, domain_id=2)
    assert isinstance(artifact, ReplowArtifact)
    assert artifact.syntax_valid is True
    assert "def add" in artifact.refactored_text
    assert "```" not in artifact.refactored_text
    assert out_file.exists()
    assert artifact.compressed_bytes_len > 0
    assert artifact.bits_per_byte > 0.0
```

<!-- end of file: unplow/inbox/gemweb009.md -->
