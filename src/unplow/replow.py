# Path: unplow/src/unplow/replow.py
"""
Replow Pipeline: Extracts raw code basins from model weights, refactors
syntax via AST analysis, stages training datasets, and compresses optimized
work products into high-density Bayesian CTS .unp archives.
"""

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from unplow.container import pack_unp_container
from unplow.ctw_coder import FLAG_BAYES_MIXER, BayesianCTSMixerCDF
from unplow.paradigm2 import LLMArithmeticCodec


@dataclass
class ReplowArtifact:
    """Represents a sanitized, refactored, and archived code artifact."""

    original_text: str
    refactored_text: str
    syntax_valid: bool
    container_path: str
    compressed_bytes_len: int
    bits_per_byte: float

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "refactored_text": self.refactored_text,
            "syntax_valid": self.syntax_valid,
            "container_path": self.container_path,
            "compressed_bytes_len": self.compressed_bytes_len,
            "bits_per_byte": self.bits_per_byte,
        }


class ReplowEngine:
    """
    Orchestrates extraction, AST refactoring, distillation training staging,
    and Bayesian CTS compression for model-harvested code artifacts.
    """

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        try:
            self.device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            self.device = torch.device("cpu")

    @staticmethod
    def refactor_source_code(raw_code: str) -> Tuple[str, bool]:
        """
        Parses raw harvested code with Python's ast module, applies basic syntax
        sanitization, strips markdown code fences, and verifies compilability.
        """
        """
        Parses raw harvested code with Python's ast module, applies basic syntax
        sanitization, strips markdown code fences, and verifies compilability.
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

        cleaned = textwrap.dedent(cleaned).strip()

        syntax_valid = False
        try:
            ast.parse(cleaned)
            compile(cleaned, "<string>", "exec")
            syntax_valid = True
        except SyntaxError:
            # Fallback wrapper attempt for snippet fragments (e.g. return outside func)
            try:
                wrapped = "def __refactored_stub__():\n    " + "\n    ".join(cleaned.splitlines())
                ast.parse(wrapped)
                compile(wrapped, "<string>", "exec")
                cleaned = wrapped
                syntax_valid = True
            except SyntaxError:
                syntax_valid = False

        return cleaned, syntax_valid

    @staticmethod
    def compute_code_metrics(code: str, domain: str = "python") -> Dict[str, Any]:
        """Calculates complexity, quality, and structural metrics for extracted inputs.

        Metrics provide quantitative criteria for future destination models to filter
        or curriculum-weight training inputs.
        """
        lines = [line for line in code.splitlines() if line.strip()]
        num_lines = len(lines)
        num_chars = len(code)

        words = code.split()
        token_count = len(words)

        ast_node_count = 0
        cyclomatic_complexity = 1
        syntax_valid = False

        if domain == "python":
            try:
                tree = ast.parse(code)
                syntax_valid = True
                nodes = list(ast.walk(tree))
                ast_node_count = len(nodes)
                for node in nodes:
                    if isinstance(
                        node,
                        (
                            ast.If,
                            ast.While,
                            ast.For,
                            ast.AsyncFor,
                            ast.ExceptHandler,
                            ast.With,
                            ast.AsyncWith,
                            ast.Assert,
                            ast.comprehension,
                        ),
                    ):
                        cyclomatic_complexity += 1
                    elif isinstance(node, ast.BoolOp):
                        cyclomatic_complexity += len(node.values) - 1
            except SyntaxError:
                syntax_valid = False
        else:
            syntax_valid = bool(num_lines > 0)
            branch_keywords = {
                "if", "else", "for", "while", "switch", "case", "match", "catch"
            }
            for word in words:
                cleaned_word = word.strip("():{};,")
                if cleaned_word in branch_keywords:
                    cyclomatic_complexity += 1
            ast_node_count = token_count

        unique_tokens = len(set(words))
        token_diversity = (unique_tokens / token_count) if token_count > 0 else 0.0

        quality_score = 0.0
        if syntax_valid:
            quality_score += 0.50
        if num_lines >= 3:
            quality_score += 0.20
        quality_score += min(0.30, round(token_diversity * 0.30, 4))
        quality_score = round(min(1.0, quality_score), 4)

        return {
            "num_lines": num_lines,
            "num_chars": num_chars,
            "token_count": token_count,
            "ast_node_count": ast_node_count,
            "cyclomatic_complexity": cyclomatic_complexity,
            "token_diversity": round(token_diversity, 4),
            "syntax_valid": syntax_valid,
            "quality_score": quality_score,
        }

    def stage_training_dataset(
        self,
        samples: List[str],
        output_path: Union[str, Path],
    ) -> Path:
        """
        Stages refactored code snippets into a formatted JSONL training dataset.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for s in samples:
                clean_s, _ = self.refactor_source_code(s)
                f.write(json.dumps({"text": clean_s}) + "\n")
        return out_p

    def train_step(
        self,
        text_samples: List[str],
        learning_rate: float = 1e-4,
        steps: int = 1,
    ) -> float:
        """
        Runs a lightweight PyTorch gradient update loop over trainable model
        parameters using cross-entropy next-token prediction loss.
        """
        # Memory safety guard: for large models (> 10M params) on CPU,
        # freeze the backbone and isolate the final transformer layer / adapter
        total_params = sum(p.numel() for p in self.model.parameters())
        if total_params > 10_000_000:
            for p in self.model.parameters():
                p.requires_grad = False
            unfrozen = False
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers") and self.model.model.layers:
                for p in self.model.model.layers[-1].parameters():
                    p.requires_grad = True
                unfrozen = True
            elif hasattr(self.model, "layers") and self.model.layers:
                for p in self.model.layers[-1].parameters():
                    p.requires_grad = True
                unfrozen = True
            if not unfrozen:
                for p in list(self.model.parameters())[-4:]:
                    p.requires_grad = True

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            return 0.0

        optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
        total_loss = 0.0
        step_count = 0
        self.model.train()

        for _ in range(steps):
            for sample in text_samples:
                tok_out = self.tokenizer(sample, return_tensors="pt")
                input_ids = (
                    tok_out.input_ids.to(self.device)
                    if hasattr(tok_out, "input_ids")
                    else tok_out["input_ids"].to(self.device)
                )
                if input_ids.shape[1] < 2:
                    continue

                optimizer.zero_grad()
                outputs = self.model(input_ids)
                logits = outputs.logits[:, :-1, :].contiguous()
                targets = input_ids[:, 1:].contiguous()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                step_count += 1

        self.model.eval()
        return round(total_loss / max(step_count, 1), 4)

    def process_and_archive(
        self,
        raw_code: str,
        output_path: Union[str, Path],
        domain_id: int = 2,  # 2 = python_code domain prior
    ) -> ReplowArtifact:
        """
        Refactors harvested code, verifies syntax, and compresses it into a
        high-density Bayesian CTS .unp archive.
        """
        refactored, is_valid = self.refactor_source_code(raw_code)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        codec = LLMArithmeticCodec(
            model=self.model,
            tokenizer=self.tokenizer,
            device=str(self.device),
        )

        res = codec.compress_to_container(
            refactored,
            domain_id=domain_id,
            use_bayes_mixer=True,
        )
        if hasattr(res, "container_bytes"):
            container_bytes = res.container_bytes
            bpb = res.stats.bits_per_byte
        elif isinstance(res, tuple):
            container_bytes = res[0]
            bpb = res[2].bits_per_byte if len(res) > 2 and hasattr(res[2], "bits_per_byte") else 0.0
        else:
            container_bytes = res
            num_bytes = max(len(refactored.encode("utf-8")), 1)
            bpb = (len(container_bytes) * 8.0) / num_bytes

        out_p.write_bytes(container_bytes)

        return ReplowArtifact(
            original_text=raw_code,
            refactored_text=refactored,
            syntax_valid=is_valid,
            container_path=str(out_p),
            compressed_bytes_len=len(container_bytes),
            bits_per_byte=round(bpb, 3),
        )

    def run_pipeline(
        self,
        raw_code: str,
        output_unp_path: Union[str, Path],
        train_steps: int = 0,
        domain_id: int = 2,
    ) -> Tuple[ReplowArtifact, Optional[float]]:
        """
        Full replow pipeline:
        Stage 1 & 2: Clean and refactor code.
        Stage 3: Optional distillation train step.
        Stage 4: Bayesian CTS .unp archive generation.
        """
        artifact = self.process_and_archive(raw_code, output_unp_path, domain_id=domain_id)
        loss = None
        if train_steps > 0:
            loss = self.train_step([artifact.refactored_text], steps=train_steps)
        return artifact, loss


# end of file: unplow/src/unplow/replow.py
