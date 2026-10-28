# Path: copilot-pr-comments.md
# copilot-pr-comments.md - Extracted GitHub Copilot PR Review Comments
# Max Column: 80 Columns

# Copilot Pull Request Review Comments & Findings

Total Extracted Copilot Artifacts: 101

## PR #29: docs(inbox): ingest gemweb102-106 execution distillation, exercism harnesses, and DPO
- **URL**: https://github.com/aomaker-org/unplow/pull/29
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T18:49:16Z
- **File**: `inbox/gemweb102.md` (Line: 5)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/29#discussion_r4404732178

### Content
```markdown
The PR description states these files "Enforce strict max 80-column formatting," and each file's own header declares `# Max Column: 80 Columns` (line 3). However, the ingested content is largely unwrapped and massively exceeds 80 columns. Measured maximum line widths are: gemweb102.md 448 cols (this line), gemweb103.md 260, gemweb104.md 216, gemweb105.md 305, gemweb106.md 326, with dozens of prose lines in the 120-270 col range across the five files. This contradicts both the per-file header and the PR description; even the "code max 120" allowance from AI.md Directive 04 does not cover these prose lines. Either reflow the prose to 80 columns or correct the header/description so the stated convention matches the actual content.
```

---

## PR #29: docs(inbox): ingest gemweb102-106 execution distillation, exercism harnesses, and DPO
- **URL**: https://github.com/aomaker-org/unplow/pull/29
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-28T18:49:16Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/29#pullrequestreview-5687979027

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

An unresolved finding shows the ingested prose far exceeds the 80-column limit asserted by both the file headers and PR description, leaving a contradiction that must be corrected before approval.

<details open>
<summary><strong>1 open issue</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Prose exceeds 80-column limit despite file headers](https://github.com/aomaker-org/unplow/pull/29#discussion_r4404732178) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR ingests five new documentation files into `inbox/` (`gemweb102.md`–`gemweb106.md`), continuing the repository's practice of recording externally-sourced ("gemweb") research notes as plain markdown. The content surveys and specifies a prompt-driven, execution-verified distillation pipeline for training compact systems-programming models: architecture overview, benchmark/problem-set survey, an Exercism C extraction harness, an async batch distillation runner, and a DPO triplet-generation stage. These are reference/knowledge documents rather than wired-in package code; no `src/` modules, tests, or indexes are modified.

**Changes:**
- Adds a closed-loop execution-distillation architecture note (gemweb102) and a systems-programming benchmark survey (gemweb103).
- Adds illustrative Python harnesses: Exercism C extraction to JSON (gemweb104), an async inference + ProcessPool verification runner (gemweb105), and DPO `(prompt, chosen, rejected)` generation (gemweb106).
- All files carry the standard `# Path:` header and `<!-- end of file: ... -->` footer, but their content is not wrapped to the declared 80-column limit (see finding).

| File | Description |
| ---- | ----------- |
| inbox/​gemweb102.md | Execution-distillation pipeline overview, problem-bank schema, teacher-harvest/student-train/DPO snippets. |
| inbox/​gemweb103.md | Survey of execution-verified benchmarks and systems-specific test-suite sources. |
| inbox/​gemweb104.md | Exercism C track extraction script parsing instructions/headers/Unity harnesses into a JSON manifest. |
| inbox/​gemweb105.md | Async batch runner decoupling OpenAI-compatible inference from a `gcc` ProcessPoolExecutor sandbox. |
| inbox/​gemweb106.md | DPO triplet generation prioritizing runtime assertion failures over compile errors, plus a TRL training script. |

Notes: Header/footer conventions are correctly applied across all five files, and no inbox index/manifest needed updating. The one recorded finding is that the PR description's and per-file headers' "strict max 80-column" claim does not match the ingested content, which has many unwrapped lines (up to 448 columns).
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #28: feat(extraction): add CFG grammar engine and logit masking hook to parameter basin probing
- **URL**: https://github.com/aomaker-org/unplow/pull/28
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T18:40:06Z
- **File**: `src/unplow/grammars/__init__.py` (Line: 37)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/28#discussion_r4404633613

### Content
```markdown
`build_mask_fn` unconditionally returns `None`, so the core grammar-masking feature introduced by this PR never actually runs. Because `mask_fn` is always `None`, the `grammar_mask_fn` hook added in `probe_weight_basins` is skipped, the loaded `self.grammars` are never consulted, and the `c_systems.ebnf`/`rust_systems.ebnf` definitions have no effect. The docstring also states this method "Creates a callable `mask_fn(input_ids, logits)` that masks invalid transitions to -inf", which does not match the `return None` behavior. Consider implementing the masking logic (or clearly marking this as a non-functional scaffold / NotImplemented stub) so the feature behaves as described.
```

---

## PR #28: feat(extraction): add CFG grammar engine and logit masking hook to parameter basin probing
- **URL**: https://github.com/aomaker-org/unplow/pull/28
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T18:40:06Z
- **File**: `src/unplow/incremental_pipeline.py` (Line: 275)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/28#discussion_r4404633695

### Content
```markdown
The PR description states the grammar engine is connected to `extract_and_stage_from_model()`, but no such method exists in this file — the integration is actually in `extract_domain_training_data()`. Please align the PR description with the actual method name to avoid confusion for future readers.
```

---

## PR #28: feat(extraction): add CFG grammar engine and logit masking hook to parameter basin probing
- **URL**: https://github.com/aomaker-org/unplow/pull/28
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-28T18:40:06Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/28#pullrequestreview-5687857101

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

A critical unresolved issue remains: `build_mask_fn` always returns `None`, leaving the PR's core grammar-masking feature non-functional despite its documented behavior.

<details open>
<summary><strong>2 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.png" alt="High severity" width="62" height="18" align="texttop"></picture> [build_mask_fn is a no-op, so grammar masking never activates](https://github.com/aomaker-org/unplow/pull/28#discussion_r4404633613) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [PR description cites the wrong extraction hook name](https://github.com/aomaker-org/unplow/pull/28#discussion_r4404633695) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR scaffolds a Context-Free Grammar (CFG) constraint system for the weight-extraction pipeline. It introduces a new `unplow.grammars` package containing a `DomainGrammarEngine` plus scoped EBNF grammars for C and Rust, and wires a `grammar_mask_fn` hook into `WeightDecompressor.probe_weight_basins()` so logits can (in principle) be masked to `-inf` for illegal grammar transitions before temperature scaling and Top-K filtering. The engine is instantiated inside `extract_domain_training_data()` in the incremental pipeline.

The most important caveat: `DomainGrammarEngine.build_mask_fn()` currently returns `None` unconditionally, so the masking hook is never actually exercised — the loaded grammars and `.ebnf` files have no runtime effect yet. The hook plumbing in `paradigm1.py` is correct and safely no-ops when `mask_fn is None`, so nothing breaks, but the feature is effectively a scaffold rather than the functional masking the description implies.

**Changes:**
- Added `unplow.grammars` package: `DomainGrammarEngine` (loads `*.ebnf` by domain) plus `c_systems.ebnf` and `rust_systems.ebnf`.
- Added an optional `grammar_mask_fn(input_ids, logits)` hook to `probe_weight_basins()`, applied per step before temperature/Top-K.
- Connected the engine into `extract_domain_training_data()` in `incremental_pipeline.py`.

| File | Description |
| ---- | ----------- |
| `src/​unplow/​grammars/​__init__.py` | New `DomainGrammarEngine`; loads EBNF grammars, but `build_mask_fn` is a stub returning `None`. |
| `src/​unplow/​grammars/​c_systems.ebnf` | New C systems-subset EBNF grammar (currently unused at runtime). |
| `src/​unplow/​grammars/​rust_systems.ebnf` | New Rust systems-subset EBNF grammar (currently unused at runtime). |
| `src/​unplow/​paradigm1.py` | Adds optional `grammar_mask_fn` hook applied to per-step logits; no-ops when `None`. |
| `src/​unplow/​incremental_pipeline.py` | Imports and instantiates `DomainGrammarEngine`, passing its (currently `None`) mask fn into probing. |

Key review notes recorded as comments:
1. `build_mask_fn` always returns `None`, so the grammar-masking feature — the stated purpose of the PR — never runs, and its docstring overstates its behavior.
2. The PR description references `extract_and_stage_from_model()`, which does not exist; the actual integration point is `extract_domain_training_data()`.
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #27: docs(inbox): ingest gemweb101 scoped systems-subset EBNF grammars for C and Rust
- **URL**: https://github.com/aomaker-org/unplow/pull/27
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-28T18:36:27Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/27#pullrequestreview-5687810241

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

Unresolved active findings remain, including a misleading comment describing parser behavior the code does not actually perform and an EBNF line exceeding the mandated column limit.

<details open>
<summary><strong>2 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Wrap overlong EBNF rule to satisfy 120-character limit](https://github.com/aomaker-org/unplow/pull/27#discussion_r4404596567) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Comment claims initial_code is consumed but code never uses it](https://github.com/aomaker-org/unplow/pull/27#discussion_r4404596652) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR ingests a new reference document, `inbox/gemweb101.md`, into the `inbox/` staging area. The doc provides scoped "systems-subset" EBNF grammars for C and Rust intended to constrain autoregressive decoding (logit masking) so generated routines stay syntactically valid, and sketches a `DomainGrammarEngine` loader pattern (using `outlines`' `CFGGuide`) to be wired into `paradigm1.py`. It is purely additive documentation and follows the repo's `# Path:` header / `<!-- end of file: -->` footer convention, consistent with the sibling `gemweb100.md`.

**Changes:**
- Adds `c_systems.ebnf` and `rust_systems.ebnf` grammar specs covering low-level types, structs, pointers/borrowing, loops, bitwise logic, and lifetimes.
- Documents a Python `DomainGrammarEngine.build_guide` loading pattern for grammar-constrained generation.
- Applies the standard path header/footer and 80-column prose formatting.

| File | Description |
| ---- | ----------- |
| inbox/​gemweb101.md | New ingested doc with C/Rust EBNF grammars and a `CFGGuide` loader example; one EBNF line exceeds the 120-char code limit, and the loader's inline comment claims `initial_code` is consumed when it is unused. |

Notes for the author (non-blocking): the `build_guide` comment describes shifting across `initial_code`, but that parameter is never passed to `CFGGuide` (same pattern as `gemweb100.md`); and line 57 is 124 columns, over the Directive 04 code limit. Both are flagged inline with suggested fixes.
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #27: docs(inbox): ingest gemweb101 scoped systems-subset EBNF grammars for C and Rust
- **URL**: https://github.com/aomaker-org/unplow/pull/27
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T18:36:26Z
- **File**: `inbox/gemweb101.md` (Line: 57)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/27#discussion_r4404596567

### Content
```markdown
This EBNF line is 124 columns, exceeding the 120-char limit for code mandated by AI.md Directive 04 (summarized in AGENTS.md). Lark rule definitions can continue across lines, so wrapping the alternation keeps it within the limit.
```

---

## PR #27: docs(inbox): ingest gemweb101 scoped systems-subset EBNF grammars for C and Rust
- **URL**: https://github.com/aomaker-org/unplow/pull/27
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T18:36:26Z
- **File**: `inbox/gemweb101.md` (Line: 218)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/27#discussion_r4404596652

### Content
```markdown
The comment states the parser "immediately shifts across the tokens in `initial_code`" to guarantee subsequent predictions complete the AST, but `initial_code` is never used in this method: `CFGGuide(grammar_str, self.tokenizer)` is constructed only from the grammar and tokenizer. As written, the comment describes behavior the code does not perform, which is misleading for readers adopting this pattern. Either reword the comment to reflect that `initial_code` is not yet consumed, or pre-shift the guide across `initial_code` as described.
```

---

## PR #26: docs(inbox): ingest gemweb100 CFG logit masking in parameter basin probing
- **URL**: https://github.com/aomaker-org/unplow/pull/26
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-28T18:31:57Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/26#pullrequestreview-5687754617

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟢 Approval recommended

The review surfaced no confident, objective defects or unresolved issues in this documentation-only reference note, leaving no blocking concerns outstanding.

**No open issues** 🎉

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR adds a single ingested reference note, `inbox/gemweb100.md`, to the project's `inbox/` collection of Gemini-web-sourced design memos (`gemweb001`–`gemweb016`). The document describes a proposed technique for integrating Context-Free Grammar (CFG) logit masking into the autoregressive decoding loop of `WeightDecompressor.probe_weight_basins` in `src/unplow/paradigm1.py`, turning unconstrained generation into a constrained pushdown automaton (PDA) so that invalid token transitions are masked to `-∞` before temperature scaling and Top-K filtering. It is documentation only — no source code is modified in this PR; the referenced `paradigm1.py` change is presented as a future refactor proposal.

**Changes:**
- Adds `inbox/gemweb100.md`: a reference memo covering a token-to-grammar map (via `outlines`/Lark/Tree-sitter or a zero-dependency parser), a proposed refactor of the `probe_weight_basins` decode loop, per-language EBNF grammar loading, and operational benefits.
- Follows the repository's file conventions: `# Path:` header, `<!-- end of file: ... -->` footer, and prose kept within 80 columns.

| File | Description |
| ---- | ----------- |
| `inbox/​gemweb100.md` | New ingested reference note proposing CFG logit masking in the parameter-basin decode loop; conforms to header/footer and column conventions. |

I reviewed the document for convention adherence (header/footer format verified against sibling inbox files, prose within 80 columns), internal consistency, spelling/grammar, and technical accuracy of the described decoding pipeline. The parameter differences between the memo's proposed `probe_weight_basins` signature and the current implementation are intentional (it is an explicit refactor proposal), so they are not flagged as a discrepancy. I did not find any confident, objective defects warranting inline comments on this reference material.
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #25: docs: input model extraction methodology and architecture
- **URL**: https://github.com/aomaker-org/unplow/pull/25
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-28T18:18:47Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/25#pullrequestreview-5687587012

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

A documentation accuracy issue remains unresolved, where the Quality Score formula and prose in the docs do not match the terms actually implemented in compute_code_metrics().

<details open>
<summary><strong>1 open issue</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Docs misstate quality score formula and token-diversity cap](https://github.com/aomaker-org/unplow/pull/25#discussion_r4404418490) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR adds a new documentation file, `docs/input_model_extraction_method.md`, that describes `unplow`'s mechanistic training-data extraction methodology. It fits into the project's existing `docs/` governance (AI.md directives, path header/footer conventions) and explains the pipeline implemented across `paradigm1.py` (weight basin probing), `incremental_pipeline.py` (domain seeds, additive corpus storage), and `replow.py` (AST refactoring and metric scoring).

**Changes:**
- Documents the two-stage temperature schedule (explore τ=1.10 for steps 1–6, lock τ=0.30 thereafter) and Top-K=60 truncation used for parameter basin probing.
- Describes the 5 domain seed anchors, AST-guided refactoring/compilability verification, and complexity/quality metrics.
- Details additive corpus storage layout under `workspace/till_corpus/{domain}/` and the unified `corpus_manifest.jsonl` log.

| File | Description |
| ---- | ----------- |
| docs/​input_model_extraction_method.md | New methodology/architecture doc; accurately reflects the extraction pipeline except for the Quality Score formula, which cites a nonexistent `ast_nodes/20` node-density term instead of the code's `num_lines ≥ 3` line-count bonus and capped token-diversity term. |

Note: the temperature values and step thresholds in the doc match the actual `incremental_pipeline.py` caller (τ=1.1/0.3, `explore_steps=6`), so those are correct even though they differ from the method's generic defaults in `paradigm1.py` (1.15/0.25/12). The single finding is the Quality Score formula inconsistency, which I've stored as a documentation comment with a suggested correction.
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #25: docs: input model extraction methodology and architecture
- **URL**: https://github.com/aomaker-org/unplow/pull/25
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T18:18:46Z
- **File**: `docs/input_model_extraction_method.md` (Line: 134)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/25#discussion_r4404418490

### Content
```markdown
The documented Quality Score formula does not match the implementation in `compute_code_metrics()` (`src/unplow/replow.py:156-162`). The code computes `Q = 0.50·[syntax_valid] + 0.20·[num_lines ≥ 3] + min(0.30, 0.30·token_diversity)`. There is no `ast_nodes/20` node-density term — the 0.2 weight is a minimum line-count bonus — and the token-diversity contribution is capped at 0.30. The prose "weighting ... node density" on line 132 is likewise inaccurate. Consider correcting the formula and description to reflect the actual terms.
```

---

## PR #24: docs: input model extraction coverage and telemetry audit
- **URL**: https://github.com/aomaker-org/unplow/pull/24
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T18:16:16Z
- **File**: `docs/extraction_coverage_audit.md` (Line: 42)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/24#discussion_r4404393312

### Content
```markdown
This file declares "# Max Column: 80 Columns" (line 3) and the repo mandates an 80-column limit for text/Markdown (AGENTS.md / AI.md Directive 04). The existing docs adhere strictly — `docs/architecture.md`, `docs/decompression_guide.md`, and `docs/scratch_model_training.md` have zero lines over 80 columns, using ASCII-box layouts instead of wide Markdown tables. This table header is 119 columns and the total row (line 42) is 90; several other lines also exceed 80: 61, 67, 68, 72, 76, 79, 93, 98, and 99. Please reflow these to stay within 80 columns (e.g. use the boxed layout seen in `docs/scratch_model_training.md`, abbreviate the column headers, or split the table).

This issue also appears on line 98 of the same file.
```

---

## PR #24: docs: input model extraction coverage and telemetry audit
- **URL**: https://github.com/aomaker-org/unplow/pull/24
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-28T18:16:16Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/24#pullrequestreview-5687555895

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

An unresolved finding remains, as the audit doc exceeds the repo-mandated 80-column limit on multiple lines, breaking consistency with the other docs that adhere to it.

<details open>
<summary><strong>1 open issue</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Documentation table exceeds the repo&#39;s 80-column limit](https://github.com/aomaker-org/unplow/pull/24#discussion_r4404393312) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR adds a new documentation file, `docs/extraction_coverage_audit.md`, that reports extraction-coverage and telemetry metrics for the training pipeline. It audits 454 mechanistic extractions harvested from the substrate model `Qwen/Qwen2.5-Coder-0.5B` across 5 language domains (Python, C, C++, Rust, Gencode), summarizing raw vs. refactored AST validity, cyclomatic complexity, token counts, per-seed coverage, and learning-evolution dynamics across 187 step-comparison checkpoints. It is a docs-only change that complements the existing `scratch_model_training.md` / `architecture.md` and reflects runtime artifacts generated by `src/unplow/incremental_pipeline.py`.

**Changes:**
- Adds an executive summary and per-domain extraction breakdown table (samples, seeds, syntax validity, complexity, quality).
- Documents per-domain seed-anchor coverage tied to `DOMAIN_EXTRACTION_SEEDS` and artifact-retention file naming.
- Describes refactoring/quality dynamics and learning-evolution phases over training steps.

| File | Description |
| ---- | ----------- |
| docs/​extraction_coverage_audit.md | New audit doc; totals and seed sums are internally consistent, but the refactored-file count (474) conflicts with the 454 total used everywhere else, and several lines exceed the repo-mandated 80-column limit. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #23: feat(training): scratch model training from dynamic Qwen Coder extraction
- **URL**: https://github.com/aomaker-org/unplow/pull/23
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T18:13:45Z
- **File**: `src/unplow/models.py` (Line: 150)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/23#discussion_r4404368074

### Content
```markdown
The `"parameters"` field now holds a layer count (e.g. `"4L"`) instead of a parameter count. The CLI (`cli.py` `models` command) renders this field as the model's size, so users will see something like "(safetensors_snapshot, 24L)" where they previously saw a parameter total ("1.54B"). Storing a layer count under a key named `parameters` is misleading; consider reporting an actual parameter estimate, or using a distinct key/label (e.g. `num_layers`) for the layer count.
```

---

## PR #23: feat(training): scratch model training from dynamic Qwen Coder extraction
- **URL**: https://github.com/aomaker-org/unplow/pull/23
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-28T18:13:45Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/23#pullrequestreview-5687524444

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

An unresolved moderate defect leaves the model stuck in eval mode mid-epoch after comparisons, silently disabling train-time behavior, alongside other outstanding issues.

<details open>
<summary><strong>2 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Restore training mode after model comparison](https://github.com/aomaker-org/unplow/pull/23#discussion_r4404367983) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Rename parameters field to reflect layer count](https://github.com/aomaker-org/unplow/pull/23#discussion_r4404368074) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR adds a "train from scratch" training path to `unplow`'s incremental pipeline. Instead of fine-tuning an isolated layer of a heavy backbone, it instantiates a compact 4/6-layer transformer that mirrors a target model's tokenizer and projection geometry (e.g. Qwen2.5-Coder-0.5B), then trains it on code dynamically harvested from the target model's weights. It also layers in corpus quality/complexity scoring, WSL host-disk telemetry, Directive 17 heartbeats, periodic side-by-side generation comparisons, and automatic checkpoint resume.

**Changes:**
- Adds `create_scratch_model()` and a `MockCausalLM.generate()` helper, plus config-driven model auditing.
- Adds `ReplowEngine.compute_code_metrics()` and persists raw inputs + per-iteration/unified JSON(L) manifests for downstream filtering.
- Extends `IncrementalTrainer` with heartbeats, step-interval model comparisons, scratch-training guard, auto-resume, new CLI flags, and extends `DiskTelemetry`/`check_disk` with WSL host-drive probing.

| File | Description |
| ---- | ----------- |
| src/​unplow/​models.py | New `create_scratch_model`, `MockCausalLM.generate`, config-based audit fields |
| src/​unplow/​incremental_pipeline.py | Heartbeats, model comparisons, metrics/manifests, scratch-training, auto-resume, CLI flags |
| src/​unplow/​replow.py | New `compute_code_metrics`; adds a duplicated docstring |
| src/​unplow/​till_guard.py | WSL detection and Windows host-drive telemetry in `DiskTelemetry`/`check_disk` |
| tests/​test_incremental_pipeline.py | Tests for scratch model training and prompt comparison |
| tests/​test_till_guard.py | Asserts WSL host telemetry fields when running under WSL |
| docs/​scratch_model_training.md | New documentation of the scratch-training pipeline |
| .gitignore | Ignores evaluation logs and the new coder-0.5b model dir |

Key findings I flagged:
- **Bug (resume):** the auto-resume glob `adapter_model_rev*.pt` never matches the base `adapter_model.pt` written first for each domain, so auto-resume typically finds nothing.
- **Bug (eval mode):** `compare_models_on_test_prompts` leaves the destination model in `eval()` mode, so mid-epoch comparisons disable train-mode behavior for the rest of the epoch.
- **Maintainability:** duplicated docstring in `refactor_source_code`.
- **error_message / naming:** stale "layer 27" help text; `"parameters"` audit field now carries a layer count.
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #23: feat(training): scratch model training from dynamic Qwen Coder extraction
- **URL**: https://github.com/aomaker-org/unplow/pull/23
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T18:13:44Z
- **File**: `src/unplow/incremental_pipeline.py` (Line: 187)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/23#discussion_r4404367983

### Content
```markdown
`compare_models_on_test_prompts` switches `self.model` into eval mode here but never restores its previous (training) mode before returning. Since this method is invoked from inside the per-step training loop (`train_domain_incrementally` only calls `self.model.train()` once at the start of each epoch), any comparison that triggers mid-epoch leaves the destination model in eval mode for all remaining training steps of that epoch, silently disabling dropout/train-time behavior until the next epoch begins. Consider capturing the prior training state at the start and restoring it before returning (e.g. `self.model.train()` if it was training), so training-mode semantics are preserved after the comparison.

This issue also appears in the following locations of the same file:
- line 746
- line 819
```

---

## PR #22: feat(visualizer): eliminate timelapse blanking with homogeneous clustering and canvas player (Phase 23)
- **URL**: https://github.com/aomaker-org/unplow/pull/22
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:59:11Z
- **File**: `src/unplow/visualizer.py` (Line: 1050)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/22#discussion_r4304684239

### Content
```markdown
In dashboard mode this re-loads the checkpoint with `torch.load` even though the same file was already loaded during the pre-computation pass (line 1003), where `t_q_c` was extracted and pooled to build `dq`. This doubles checkpoint deserialization I/O for every dashboard frame. Consider storing the raw W_Q panel (e.g. `pool_tensor_2d(t_q_c, panel_size)`) in `loaded_data` during the first pass and reusing it here instead of reloading. This is an optional improvement.
```

---

## PR #22: feat(visualizer): eliminate timelapse blanking with homogeneous clustering and canvas player (Phase 23)
- **URL**: https://github.com/aomaker-org/unplow/pull/22
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T21:59:11Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/22#pullrequestreview-5560290363

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

The new `/api/timelapse/frames` endpoint reads files from an unsanitized user-supplied filename, allowing directory-traversal reads that the existing `/images/` handler already guards against.

<details open>
<summary><strong>2 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.png" alt="High severity" width="62" height="18" align="texttop"></picture> [Sanitize frames filename to prevent path traversal](https://github.com/aomaker-org/unplow/pull/22#discussion_r4304684220) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Avoid reloading checkpoint during dashboard rendering](https://github.com/aomaker-org/unplow/pull/22#discussion_r4304684239) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR (Phase 23) overhauls the checkpoint timelapse video pipeline to remove frame blanking and brightness pulsing, and replaces the browser-native APNG `<img>` playback with a double-buffered HTML5 canvas player in the Web Studio. It touches the zero-dependency visualizer codec, the web server/endpoints and client JS, the CLI, tests, and architecture docs.

**Changes:**
- Rendering: adds `filter_homogeneous_checkpoints` (size-cluster filtering), a two-pass global normalization controlled by a new `smooth_scale`/`--no-smooth` option, and drops the all-zeros frame 0 in `delta`/`step` modes by evolving from the first learned revision against a separate baseline.
- Decoding/serving: adds `extract_apng_frames` (standalone PNG extraction) and a `GET /api/timelapse/frames` endpoint returning base64 pre-decoded frames.
- Web UI: replaces `canvas-freeze` with a double-buffered `canvas-player`, with play/pause, scrub, A/B blink, and diff-view controls.

| File | Description |
| ---- | ----------- |
| src/​unplow/​visualizer.py | Adds `extract_apng_frames`, `filter_homogeneous_checkpoints`, and reworks `render_checkpoint_timelapse` into a two-pass globally-normalized renderer. |
| src/​unplow/​web.py | Adds `/api/timelapse/frames` endpoint and rewrites the client-side video player to a canvas-based frame cache. |
| src/​unplow/​cli.py | Wires the new `--no-smooth` flag into `render_checkpoint_timelapse`. |
| docs/​architecture.md | Documents the Phase 23 flicker-free synthesis and canvas studio section. |
| tests/​test_visualizer.py | Adds tests for the two new helpers and updates expected delta frame count. |
| tests/​test_web.py | Adds coverage for the frames endpoint. |
| tests/​test_cli.py | Asserts the `--no-smooth` flag appears in `video` help. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #22: feat(visualizer): eliminate timelapse blanking with homogeneous clustering and canvas player (Phase 23)
- **URL**: https://github.com/aomaker-org/unplow/pull/22
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:59:10Z
- **File**: `src/unplow/web.py` (Line: 1468)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/22#discussion_r4304684220

### Content
```markdown
The new `/api/timelapse/frames` endpoint builds the target path directly from the user-supplied `filename` query parameter (`out_dir / filename`) without stripping directory components. A value containing `../` (or an absolute path, which `pathlib` resolves by discarding `out_dir`) escapes `workspace/visualizations`, letting a client on the loopback interface read arbitrary PNG files from the filesystem. The `/images/` handler already guards against this by reducing the request to its base name via `Path(path).name` (web.py:1552); the frames endpoint should apply the same sanitization.
```

---

## PR #21: feat(visualizer): implement RFC 7689 APNG checkpoint timelapse video studio (Phase 22)
- **URL**: https://github.com/aomaker-org/unplow/pull/21
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:38:09Z
- **File**: `src/unplow/visualizer.py` (Line: 603)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/21#discussion_r4304617312

### Content
```markdown
APNG is not defined by RFC 7689. RFC 7689 is "Signaling Extensions for Wavelength Switched Optical Networks" and is unrelated to image formats; in fact APNG is not specified by any IETF RFC (base PNG is RFC 2083). This incorrect citation is repeated in the PR title, `docs/architecture.md` section 18, and here. Recommend referring to the "APNG specification" instead of "RFC 7689".
```

---

## PR #21: feat(visualizer): implement RFC 7689 APNG checkpoint timelapse video studio (Phase 22)
- **URL**: https://github.com/aomaker-org/unplow/pull/21
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:38:09Z
- **File**: `src/unplow/web.py` (Line: 34)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/21#discussion_r4304617320

### Content
```markdown
`compile_images_to_apng` and `encode_apng_rgb` are imported here but never used in `web.py` (only `render_checkpoint_timelapse` is called). These unused imports should be removed to avoid lint warnings and dead imports.
```

---

## PR #21: feat(visualizer): implement RFC 7689 APNG checkpoint timelapse video studio (Phase 22)
- **URL**: https://github.com/aomaker-org/unplow/pull/21
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T21:38:09Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/21#pullrequestreview-5560218902

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

It introduces a sizable hand-rolled binary APNG format plus web endpoints, and two newly added web tests depend on git-ignored checkpoints so they fail outside a developer's trained environment, requiring human verification.

<details open>
<summary><strong>5 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.png" alt="High severity" width="62" height="18" align="texttop"></picture> [Timelapse test depends on missing non-hermetic checkpoints](https://github.com/aomaker-org/unplow/pull/21#discussion_r4304617244) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Playback controls do not actually affect APNG frames](https://github.com/aomaker-org/unplow/pull/21#discussion_r4304617271) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Incorrect RFC 7689 reference in APNG docs](https://github.com/aomaker-org/unplow/pull/21#discussion_r4304617291) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Incorrect RFC 7689 citation for APNG](https://github.com/aomaker-org/unplow/pull/21#discussion_r4304617312) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Unused APNG imports in web.py](https://github.com/aomaker-org/unplow/pull/21#discussion_r4304617320) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR adds "Phase 22": a pure-Python, zero-dependency Animated PNG (APNG) engine plus a checkpoint-evolution timelapse generator, and wires it into the existing visualizer, web studio, CLI, and Makefile. It extends the air-gapped (Directive 18) tooling so users can render how LoRA adapter weights evolve across `adapter_model_rev*.pt` revisions as a self-contained animated PNG, viewable in-browser without `ffmpeg`.

**Changes:**
- New APNG codec and renderer in `visualizer.py` (`make_png_chunk`, `encode_apng_rgb`, `decode_png_rgb`, `compile_images_to_apng`, `pool_tensor_2d`, `render_checkpoint_timelapse`) supporting `delta`/`step`/`raw`/`dashboard` modes.
- Web studio: new Timelapse tab, GET/POST `/api/timelapse` endpoints, and a video toolbar (play/pause freeze-frame, scrubber, speed selector).
- CLI `video`/`timelapse` subcommand, `make video`/`make timelapse` targets, architecture docs sections 16–18, and new visualizer/web/CLI tests.

| File | Description |
| ---- | ----------- |
| src/​unplow/​visualizer.py | Adds pure-Python APNG encode/decode, image/checkpoint timelapse rendering; docstring mislabels APNG as "RFC 7689". |
| src/​unplow/​web.py | Adds timelapse generation branch, `/api/timelapse` routes, and video UI; two unused imports and non-functional playback controls. |
| src/​unplow/​cli.py | Adds `video`/`timelapse` subcommand with input-dir and checkpoint-domain modes. |
| Makefile | Adds `video`/`timelapse` targets and help text. |
| docs/​architecture.md | Documents visualizer/web/timelapse phases; repeats incorrect "RFC 7689" attribution. |
| tests/​test_visualizer.py | Adds hermetic APNG/timelapse unit tests using `tmp_path`. |
| tests/​test_web.py | Adds timelapse tests that are non-hermetic and fail in a clean checkout. |
| tests/​test_cli.py | Adds `video` help and execution tests (hermetic, using `tmp_path`). |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #21: feat(visualizer): implement RFC 7689 APNG checkpoint timelapse video studio (Phase 22)
- **URL**: https://github.com/aomaker-org/unplow/pull/21
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:38:08Z
- **File**: `tests/test_web.py` (Line: 185)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/21#discussion_r4304617244

### Content
```markdown
`models/till-python-lora/` is git-ignored (`.gitignore`: `/models/till-*/`) and no fixture creates it in the server's working directory, so in a clean checkout/CI this endpoint falls through to the "no checkpoints" branch and returns HTTP 404 (`{"status":"error",...}`). `urllib.request.urlopen` raises `HTTPError` on 404, so this block throws before reaching the asserts and the test fails. Unlike the other tests here, it doesn't use the `mock_checkpoints`/`tmp_path` fixtures, making it non-hermetic. Consider seeding a timelapse PNG under `workspace/visualizations/` (or real checkpoints under a temp CWD) before hitting the endpoint.

This issue also appears on line 188 of the same file.
```

---

## PR #21: feat(visualizer): implement RFC 7689 APNG checkpoint timelapse video studio (Phase 22)
- **URL**: https://github.com/aomaker-org/unplow/pull/21
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:38:08Z
- **File**: `src/unplow/web.py` (Line: 1105)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/21#discussion_r4304617271

### Content
```markdown
Optional: `changePlaybackSpeed()` is a no-op (the speed dropdown in the toolbar has no effect), and the scrubber/prev/next controls (`onScrubVideo`, `stepFrame`) only update the "Frame X / Y" label without changing the displayed APNG frame, since an APNG in an `<img>` cannot be seeked or re-timed. This diverges from the PR description, which advertises functional frame scrubbing and playback-speed control. Consider either disabling/removing these controls or clearly labeling them as non-functional to avoid misleading users.
```

---

## PR #21: feat(visualizer): implement RFC 7689 APNG checkpoint timelapse video studio (Phase 22)
- **URL**: https://github.com/aomaker-org/unplow/pull/21
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:38:08Z
- **File**: `docs/architecture.md` (Line: 393)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/21#discussion_r4304617291

### Content
```markdown
APNG is not defined by RFC 7689. RFC 7689 is "Signaling Extensions for Wavelength Switched Optical Networks" and has nothing to do with images; APNG is not an IETF RFC at all (base PNG is RFC 2083). This mislabel also appears in the `encode_apng_rgb` docstring and the PR title. Recommend dropping the "RFC 7689" reference and calling it the APNG specification.
```

---

## PR #20: feat(phase21): support headless server remote access and SSH tunneling from Windows 11
- **URL**: https://github.com/aomaker-org/unplow/pull/20
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:01:22Z
- **File**: `src/unplow/web.py` (Line: 558)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/20#discussion_r4304487640

### Content
```markdown
This sentence hardcodes the hostname `fg26`, but this element has no `id` and is never updated by the status-fetch JavaScript (unlike `server-host-badge`, `ssh-cmd-text`, and the LAN links). On any machine whose hostname is not `fg26`, the modal will display a stale, incorrect hostname, which undermines the dynamic detection this PR adds. Since the hostname is already shown dynamically elsewhere in the modal and header, drop the hardcoded hostname from this static text.
```

---

## PR #20: feat(phase21): support headless server remote access and SSH tunneling from Windows 11
- **URL**: https://github.com/aomaker-org/unplow/pull/20
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T21:01:22Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/20#pullrequestreview-5560070551

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

The default change to bind an unauthenticated server on all interfaces (`0.0.0.0`) is a significant security posture shift that conflicts with the repository's loopback-only air-gap mandate and warrants human judgment.

<details open>
<summary><strong>5 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.png" alt="High severity" width="62" height="18" align="texttop"></picture> [Keep default host bound to loopback](https://github.com/aomaker-org/unplow/pull/20#discussion_r4304487578) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.png" alt="High severity" width="62" height="18" align="texttop"></picture> [Avoid exposing unauthenticated server on all interfaces by default](https://github.com/aomaker-org/unplow/pull/20#discussion_r4304487596) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Close UDP probe socket on all error paths](https://github.com/aomaker-org/unplow/pull/20#discussion_r4304487613) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Revert unexplained active_iteration fallback change](https://github.com/aomaker-org/unplow/pull/20#discussion_r4304487624) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Remove hardcoded hostname from status modal text](https://github.com/aomaker-org/unplow/pull/20#discussion_r4304487640) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR extends the standard-library web visualizer (`src/unplow/web.py`) to support remote access from a Windows 11 host against a headless Linux server. It changes the default bind address from `127.0.0.1` to `0.0.0.0` (configurable via `UNPLOW_HOST`), adds `get_headless_status()` and `get_lan_ip_addresses()` helpers, enriches the `/api/status` response with server/network metadata, prints SSH-tunnel and LAN connection instructions at startup, and embeds an interactive "SSH Remote Guide" modal in the SPA.

**Changes:**
- Bind the web server to all interfaces by default and surface SSH/LAN connection guidance in the terminal banner and a browser modal.
- Add headless-display and LAN-IP detection helpers, and expose hostname/username/LAN IPs/SSH command through `/api/status`.
- Update `cli.py` `--host` default and the `Makefile` `web` target to `0.0.0.0`, with tests for the new helpers and status fields.

| File | Description |
| ---- | ----------- |
| src/​unplow/​web.py | Adds headless/LAN helpers, 0.0.0.0 default bind, status server metadata, startup banner, and SSH guide modal. |
| src/​unplow/​cli.py | Changes `web --host` default to `0.0.0.0` (via `UNPLOW_HOST`). |
| Makefile | `make web` now binds `--host 0.0.0.0 --port 8765`. |
| tests/​test_web.py | Asserts new `server` status fields and tests `get_headless_status`/`get_lan_ip_addresses`. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #20: feat(phase21): support headless server remote access and SSH tunneling from Windows 11
- **URL**: https://github.com/aomaker-org/unplow/pull/20
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:01:21Z
- **File**: `src/unplow/cli.py` (Line: 494)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/20#discussion_r4304487578

### Content
```markdown
The `--host` default was also changed to `0.0.0.0`, which binds the CLI-launched server to all interfaces with no authentication. For the same reasons as the module default (and to respect the loopback-only air-gap mandate in `AI.md` Directive 18), keep the default on loopback and let operators opt into LAN/SSH exposure via `UNPLOW_HOST` or an explicit `--host 0.0.0.0`.
```

---

## PR #20: feat(phase21): support headless server remote access and SSH tunneling from Windows 11
- **URL**: https://github.com/aomaker-org/unplow/pull/20
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:01:21Z
- **File**: `src/unplow/web.py` (Line: 40)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/20#discussion_r4304487596

### Content
```markdown
Changing the default bind address to `0.0.0.0` exposes the unauthenticated web server to every host on the local network by default. This server offers static file serving, a `/api/generate` endpoint that triggers visualization work, and a `/api/status` endpoint that now discloses the hostname, OS username, and LAN IPs — all without any authentication. This also conflicts with the air-gap mandate in `AI.md` Directive 18, which restricts operations to loopback addresses (`127.0.0.1`, `localhost`). Since `UNPLOW_HOST` already lets an operator opt into LAN/SSH binding, keep the default on loopback so remote exposure is an explicit, deliberate choice. Note that `src/unplow/cli.py:493` and the `Makefile` `web` target make the same change and should be reconciled with this policy.
```

---

## PR #20: feat(phase21): support headless server remote access and SSH tunneling from Windows 11
- **URL**: https://github.com/aomaker-org/unplow/pull/20
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:01:21Z
- **File**: `src/unplow/web.py` (Line: 59)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/20#discussion_r4304487613

### Content
```markdown
The UDP probe socket is only closed on the success path (`s.close()` at line 55). If `connect()` or `getsockname()` raises, the socket is never closed and leaks a file descriptor. Because `get_lan_ip_addresses()` is invoked on every `/api/status` poll, repeated failures could accumulate open descriptors over the server's lifetime. Use a context manager so the socket is always released.
```

---

## PR #20: feat(phase21): support headless server remote access and SSH tunneling from Windows 11
- **URL**: https://github.com/aomaker-org/unplow/pull/20
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T21:01:21Z
- **File**: `src/unplow/web.py` (Line: 1065)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/20#discussion_r4304487624

### Content
```markdown
The fallback for `active_iteration` was silently changed from `6` to `8`. This is an unexplained magic number, and the change is unrelated to the headless/remote-access scope described in the PR. It alters what the UI displays when no iteration is found in the logs. Consider reverting this to keep the PR focused, or extract a named constant with a comment explaining the value.
```

---

## PR #19: feat(phase20): interactive visualizer web studio and embedded PNG comparison metadata
- **URL**: https://github.com/aomaker-org/unplow/pull/19
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T20:56:28Z
- **File**: `src/unplow/web.py` (Line: 901)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/19#discussion_r4304472666

### Content
```markdown
The fallback `active_iter or 6` reports a hard-coded "Iteration #6" whenever no iteration is parsed from the logs (including when no pipeline has ever run). This surfaces a misleading, unexplained magic value in the UI telemetry (`Pipeline Iteration #6`). Prefer reporting the real value (0 / unknown) so the dashboard does not fabricate pipeline activity.
```

---

## PR #19: feat(phase20): interactive visualizer web studio and embedded PNG comparison metadata
- **URL**: https://github.com/aomaker-org/unplow/pull/19
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T20:56:28Z
- **File**: `src/unplow/web.py` (Line: 22)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/19#discussion_r4304472682

### Content
```markdown
Several of these imports are unused in this module: `os`, `sys`, `threading`, `time`, and the `Tuple`/`Union` typing names are never referenced. Removing them keeps the import list clean.
```

---

## PR #19: feat(phase20): interactive visualizer web studio and embedded PNG comparison metadata
- **URL**: https://github.com/aomaker-org/unplow/pull/19
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T20:56:28Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/19#pullrequestreview-5560054921

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

It introduces a network-facing server that performs pickle-based `torch.load` and unsanitized filename-driven file writes, which warrants human security review beyond the specific findings raised.

<details open>
<summary><strong>4 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.png" alt="High severity" width="62" height="18" align="texttop"></picture> [Unsanitized output_filename allows path traversal writes](https://github.com/aomaker-org/unplow/pull/19#discussion_r4304472648) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Status endpoint rescans entire tree on every poll](https://github.com/aomaker-org/unplow/pull/19#discussion_r4304472656) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Fallback iteration value shows misleading hard-coded #6](https://github.com/aomaker-org/unplow/pull/19#discussion_r4304472666) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Remove unused imports from module](https://github.com/aomaker-org/unplow/pull/19#discussion_r4304472682) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR adds a Phase-20 feature: an interactive, standard-library-only ("air-gapped", Directive 18) web studio for browsing and generating neural-weight visualizations, plus embedded PNG `tEXt` provenance metadata and pure-Python bitmap-font banners on rendered heatmaps. It fits into the existing visualizer/CLI stack by extending `visualizer.py` renderers with metadata + banners and exposing a new `unplow web` entrypoint that serves an SPA calling back into those renderers.

**Changes:**
- New `src/unplow/web.py`: a `BaseHTTPRequestHandler`-based threaded server with `/api/checkpoints`, `/api/images`, `/api/status`, `/api/generate`, image serving, and an embedded SPA.
- `visualizer.py`: 5x7 ASCII font + `draw_text_rgb`/`draw_rect_rgb`, PNG `tEXt` write/read (`encode_png_rgb`/`read_png_metadata`), `compute_tensor_metrics`, and banner/metadata support across the three renderers.
- CLI `web`/`ui`/`studio` subcommand and `--no-banner`; `make web`/`make ui` targets; `.gitignore` entry and `.gitkeep` for `workspace/visualizations/`; new tests.

| File | Description |
| ---- | ----------- |
| src/​unplow/​web.py | New web server/SPA; main source of the performance, security, and cleanliness findings |
| src/​unplow/​visualizer.py | Adds font, draw helpers, PNG metadata I/O, metrics, and banner rendering |
| src/​unplow/​cli.py | Adds `web` subcommand and wires new renderer params/metadata printing |
| tests/​test_web.py | New unit/integration tests for scanning, endpoints, generation |
| tests/​test_visualizer.py | Tests PNG metadata round-trip, text/rect drawing, delta banner metadata |
| tests/​test_cli.py | Adds `web --help` smoke test |
| Makefile | Adds `web`/`ui` targets; wraps a long line |
| .gitignore | Ignores `workspace/visualizations/*` except `.gitkeep` |
| workspace/​visualizations/​.gitkeep | Placeholder to keep output dir tracked |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #19: feat(phase20): interactive visualizer web studio and embedded PNG comparison metadata
- **URL**: https://github.com/aomaker-org/unplow/pull/19
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T20:56:27Z
- **File**: `src/unplow/web.py` (Line: 138)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/19#discussion_r4304472648

### Content
```markdown
`output_filename` from the request body is used to build the output path without sanitization, so a value such as `../../evil.png` resolves outside `workspace/visualizations/` and lets a client write arbitrary files on the host. Since the server can be bound to a non-loopback interface via `--host`, this is a path-traversal write. Strip the value to its basename before use.
```

---

## PR #19: feat(phase20): interactive visualizer web studio and embedded PNG comparison metadata
- **URL**: https://github.com/aomaker-org/unplow/pull/19
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T20:56:27Z
- **File**: `src/unplow/web.py` (Line: 898)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/19#discussion_r4304472656

### Content
```markdown
The `/api/status` handler recursively globs the entire working tree with `Path(".").glob("**/*.log")` and reads every matching log file in full on each request. Since the SPA polls this endpoint every 5 seconds (`setInterval(fetchStatus, 5000)`), this repeatedly walks large/ignored directories (e.g. `models/`, `workspace/`, `.git/`) and re-reads growing logs, which is an avoidable recurring cost. Consider scoping the scan to the known logs location (the repo keeps logs under `logs/`) and/or caching the result. Also note `log_path` on line 889 is assigned but never used and can be removed.
```

---

## PR #18: feat(phase19): zero-dependency checkpoint visualizer and differential heatmaps
- **URL**: https://github.com/aomaker-org/unplow/pull/18
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T20:37:49Z
- **File**: `Makefile` (Line: 135)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/18#discussion_r4304402228

### Content
```markdown
The new `visualize` target is not registered in the `.PHONY` list at the top of the Makefile (lines 8-12), unlike every other recipe-only target in this file (e.g. `probe`, `incremental-pipeline`, `full-pipeline`). Without this, `make visualize` would be skipped if a file or directory named `visualize` ever exists in the repo root. Add it to the `.PHONY` declarations (ideally the group at line 12).
```

---

## PR #18: feat(phase19): zero-dependency checkpoint visualizer and differential heatmaps
- **URL**: https://github.com/aomaker-org/unplow/pull/18
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T20:37:49Z
- **File**: `src/unplow/cli.py` (Line: 462)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/18#discussion_r4304402245

### Content
```markdown
This help text is inaccurate. When `--tensor` is omitted, the default-selection logic (cli.py:1173, 1183) picks the first state-dict key containing both `q_proj` and `weight` — i.e. the earliest layer (typically layer 0), not "Layer 27". Update the help text to describe the actual behavior to avoid confusing users.
```

---

## PR #18: feat(phase19): zero-dependency checkpoint visualizer and differential heatmaps
- **URL**: https://github.com/aomaker-org/unplow/pull/18
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T20:37:49Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/18#pullrequestreview-5559968150

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

Concrete defects beyond minor nits — an unused/ignored `colormap` parameter, inaccurate CLI help text, and a `.PHONY` convention omission — should be addressed before approval.

<details open>
<summary><strong>4 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.png" alt="High severity" width="62" height="18" align="texttop"></picture> [torch.load should use weights_only for safety](https://github.com/aomaker-org/unplow/pull/18#discussion_r4304402183) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Unused colormap argument is ignored in rendering](https://github.com/aomaker-org/unplow/pull/18#discussion_r4304402204) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Add visualize target to .PHONY list](https://github.com/aomaker-org/unplow/pull/18#discussion_r4304402228) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Help text wrongly says default tensor is Layer 27](https://github.com/aomaker-org/unplow/pull/18#discussion_r4304402245) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR adds a zero-dependency checkpoint visualizer (`unplow.visualizer`) that renders neural weight matrices, parameter-update deltas, and a 4-quadrant attention dashboard as self-contained PNGs using only stdlib `zlib`/`struct` (no Pillow/matplotlib), in keeping with the air-gapped Directive 18. It wires the renderer into a new `unplow visualize` CLI subcommand (with `vis` alias), adds a `make visualize` convenience target, and provides unit/CLI test coverage.

**Changes:**
- New `visualizer.py` with a pure-Python PNG encoder, inferno/diverging colormaps, and tensor/delta/dashboard renderers.
- New `visualize` CLI subcommand dispatch and a `make visualize` target that auto-selects the two latest LoRA revisions.
- New tests in `tests/test_visualizer.py` and `tests/test_cli.py`.

| File | Description |
| ---- | ----------- |
| src/​unplow/​visualizer.py | Core renderer: PNG encoder, colormaps, heatmap/delta/dashboard functions |
| src/​unplow/​cli.py | Adds `visualize`/`vis` subparser and dispatch branch |
| Makefile | Adds `visualize` target selecting latest two checkpoints |
| tests/​test_visualizer.py | Unit tests for PNG encoding, colormaps, and renderers |
| tests/​test_cli.py | CLI help + end-to-end execution tests for the subcommand |

Key issues flagged: an unused `colormap` parameter in `render_tensor_heatmap` (always uses inferno), inaccurate "Layer 27" help text (default actually selects the first `q_proj` weight), the `visualize` Makefile target missing from `.PHONY`, plus `weights_only=False` (security) and per-pixel Python loops (performance) notes.
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #18: feat(phase19): zero-dependency checkpoint visualizer and differential heatmaps
- **URL**: https://github.com/aomaker-org/unplow/pull/18
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T20:37:48Z
- **File**: `src/unplow/visualizer.py` (Line: 193)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/18#discussion_r4304402183

### Content
```markdown
Loading checkpoints with `weights_only=False` re-enables arbitrary code execution via pickle during `torch.load`. Since PyTorch 2.6 the default is `weights_only=True` precisely to mitigate this when loading untrusted `.pt` files, and these call sites load user-supplied checkpoint paths. Because the checkpoints here are plain tensor state dicts (as produced by the pipeline and in the tests), `weights_only=True` is sufficient and safer. The same pattern appears in cli.py:1171-1172 and cli.py:1182. Consider switching to `weights_only=True` unless the checkpoints are known to contain non-tensor objects.
```

---

## PR #18: feat(phase19): zero-dependency checkpoint visualizer and differential heatmaps
- **URL**: https://github.com/aomaker-org/unplow/pull/18
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T20:37:48Z
- **File**: `src/unplow/visualizer.py` (Line: 99)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/18#discussion_r4304402204

### Content
```markdown
The `colormap` parameter is accepted but never used — the pixel loop below always calls `colormap_inferno` (line 123) regardless of what is passed. A caller requesting, e.g., `colormap="diverging"` would be silently ignored, which is misleading. Either remove the unused parameter or branch on it to select the colormap (note the two palettes use different input normalizations: inferno expects `[0, 1]`, diverging expects `[-1, 1]`).

This issue also appears on line 242 of the same file.
```

---

## PR #17: feat(phase18): extract training code from model weights with review and AST refactoring
- **URL**: https://github.com/aomaker-org/unplow/pull/17
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T19:33:47Z
- **File**: `tests/test_incremental_pipeline.py` (Line: 195)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/17#discussion_r4304175460

### Content
```markdown
This test exercises `extract_domain_training_data`, which writes to the hardcoded CWD-relative path `workspace/till_corpus/{domain}/` (see `incremental_pipeline.py:163`). Although the trainer is given `tmp_path`-based `base_checkpoint_dir`/`base_target_dir`, the extraction ignores those and writes into the repository's gitignored `workspace/` tree instead. Because `AdditiveFileHandler.get_next_revision_path` never overwrites, each run leaves new `extracted_iter*.py` (and `_revNNN`) files behind, so the test is not hermetic and accumulates artifacts on repeated local runs. Consider redirecting the working directory to `tmp_path` so the writes stay isolated.
```

---

## PR #17: feat(phase18): extract training code from model weights with review and AST refactoring
- **URL**: https://github.com/aomaker-org/unplow/pull/17
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T19:33:47Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/17#pullrequestreview-5559688658

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

The new extraction path is correct and its APIs verified, but the added test is non-hermetic (writes into the repo's `workspace/` tree rather than `tmp_path`) and the `num_samples` parameter is misleading since only the first record is ever returned.

<details open>
<summary><strong>2 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [probe_weight_basins ignores num_samples and returns one result](https://github.com/aomaker-org/unplow/pull/17#discussion_r4304175425) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Test writes extracted files into the repo workspace tree](https://github.com/aomaker-org/unplow/pull/17#discussion_r4304175460) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR (phase 18) adds a dynamic training-data extraction path to the incremental multi-domain training pipeline. Instead of only reading staged corpus files, `IncrementalTrainer` can now probe the loaded model's weights (via `WeightDecompressor.probe_weight_basins`) around domain-specific seed prefixes, review/sanitize the result through `ReplowEngine.refactor_source_code` (AST validation, fence stripping), and additively persist each iteration's sample under `workspace/till_corpus/{domain}/`. It fits into the broader "LLM decompression / latent manifold extraction" architecture of the codebase by feeding extracted code back into per-domain `till-*` training loops.

**Changes:**
- New `extract_domain_training_data` method plus `DOMAIN_EXTRACTION_SEEDS`/`DOMAIN_EXTENSIONS` tables, and an extraction branch wired into `train_domain_incrementally` (now accepting `corpus_texts=None`, `iteration`, and `extract_from_model`).
- New CLI flags `--extract-from-model` (default True) and `--no-extract`, with `main()` reworked to only load staged corpus files when extraction is disabled.
- Version bump to `0.17.0` (`pyproject.toml`, `uv.lock`) and a new unit test for iteration-distinct extraction.

| File | Description |
| ---- | ----------- |
| `src/​unplow/​incremental_pipeline.py` | Adds weight-based extraction method, seed/extension tables, extended training signature, and CLI toggle logic. |
| `tests/​test_incremental_pipeline.py` | Adds a test asserting iteration-1 vs iteration-2 extraction produce distinct, non-empty samples. |
| `pyproject.toml` | Bumps package version to `0.17.0`. |
| `uv.lock` | Reflects the `0.17.0` version bump. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #17: feat(phase18): extract training code from model weights with review and AST refactoring
- **URL**: https://github.com/aomaker-org/unplow/pull/17
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T19:33:46Z
- **File**: `src/unplow/incremental_pipeline.py` (Line: 141)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/17#discussion_r4304175425

### Content
```markdown
The `num_samples` parameter is passed through to `probe_weight_basins`, but only `records[0].text` is ever consumed and the method always returns a single-element list (`return [clean_code]`). As a result, callers passing `num_samples > 1` would silently get just one sample back while still paying the full extraction cost for the discarded ones. Consider either iterating over all returned records to build the result list, or dropping the parameter to avoid the misleading interface.
```

---

## PR #16: feat(phase17): persist trained model checkpoints, status reporting, and continuous loop
- **URL**: https://github.com/aomaker-org/unplow/pull/16
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T19:05:10Z
- **File**: `src/unplow/incremental_pipeline.py` (Line: 383)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/16#discussion_r4304068347

### Content
```markdown
The literal `%` in this help string breaks `--help`. argparse applies `%`-formatting to help strings, so `15.0%)` raises `ValueError: unsupported format character ')'` when the help is rendered (e.g. `python incremental_pipeline.py --help`). Escape it as `%%` so it renders as a literal percent sign.
```

---

## PR #16: feat(phase17): persist trained model checkpoints, status reporting, and continuous loop
- **URL**: https://github.com/aomaker-org/unplow/pull/16
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T19:05:10Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/16#pullrequestreview-5559545296

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

Two confirmed bugs in the new CLI handling—`LOOP=0`/`CONTINUOUS=0` still enables looping, and the literal `%` in the `--min-free-pct` help string crashes `--help`—should be fixed before approval.

<details open>
<summary><strong>3 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Initialize metrics before the epoch loop](https://github.com/aomaker-org/unplow/pull/16#discussion_r4304068307) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Treat LOOP=0 as false in continuous mode](https://github.com/aomaker-org/unplow/pull/16#discussion_r4304068333) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Escape % in argparse help text](https://github.com/aomaker-org/unplow/pull/16#discussion_r4304068347) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR extends the Phase 17 incremental training pipeline (`src/unplow/incremental_pipeline.py`) so that each `till-*` domain model now persists trained artifacts, emits structured status reports, and can run continuously. It builds on existing infrastructure (`AdditiveFileHandler` for additive revision-safe writes, `ReplowEngine` for `.unp` container packing, and `TillDiskGuardian` for disk telemetry) and wires new CLI/env controls into `main()`. The package version is bumped from `0.15.0` to `0.16.0`.

**Changes:**
- Adds checkpoint persistence (`adapter_model.pt` / `_revNNN.pt`), `adapter_config.json` metadata, and `.unp` corpus archival per domain, plus a NaN/Inf loss guard.
- Adds a structured ASCII `print_status_summary` report and a continuous iteration loop with `--loop`/`LOOP`/`CONTINUOUS`, `--max-iterations`, and `--min-free-pct` controls.
- Adds a persistence/status test and passes `PIPELINE_ARGS` through the `incremental-pipeline` Make target.

| File | Description |
| ---- | ----------- |
| src/​unplow/​incremental_pipeline.py | Core changes: checkpoint/container persistence, status report, NaN guard, continuous loop, new CLI/env args |
| tests/​test_incremental_pipeline.py | New test validating checkpoint/config persistence and `print_status_summary` |
| Makefile | Forwards `$(PIPELINE_ARGS)` to the incremental-pipeline target |
| pyproject.toml | Version bump to 0.16.0 |
| uv.lock | Lockfile version bump to 0.16.0 |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #16: feat(phase17): persist trained model checkpoints, status reporting, and continuous loop
- **URL**: https://github.com/aomaker-org/unplow/pull/16
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T19:05:09Z
- **File**: `src/unplow/incremental_pipeline.py` (Line: 255)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/16#discussion_r4304068307

### Content
```markdown
These metrics (`avg_step_loss`, `loss_val`, `ppl_val`, and `elapsed` below) are only bound inside the `for epoch in range(1, total_epochs + 1)` loop. The new checkpoint-persistence block now reads them after the loop, so when `total_epochs <= 0` (reachable via `--epochs 0` or `EPOCHS=0`, which are not validated) the loop body never executes and these names are undefined, raising `NameError`. Previously the function could return cleanly with no epochs. Consider initializing these metric variables before the loop (or validating that `total_epochs >= 1`).
```

---

## PR #16: feat(phase17): persist trained model checkpoints, status reporting, and continuous loop
- **URL**: https://github.com/aomaker-org/unplow/pull/16
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T19:05:09Z
- **File**: `src/unplow/incremental_pipeline.py` (Line: 370)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/16#discussion_r4304068333

### Content
```markdown
The `--loop` default treats any non-empty value of `LOOP`/`CONTINUOUS` as truthy, so `LOOP=0` (or `CONTINUOUS=0`) enables continuous looping instead of disabling it. `bool(os.environ.get("LOOP"))` is `True` for the string `"0"`. Since the PR documents `LOOP=1` as the enabling value, a user setting `LOOP=0` to turn looping off would be surprised to find it still runs forever. Consider parsing the value against an explicit truthy set.
```

---

## PR #15: feat(phase16): display training data snippets in incremental pipeline
- **URL**: https://github.com/aomaker-org/unplow/pull/15
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T18:44:24Z
- **File**: `src/unplow/incremental_pipeline.py` (Line: 80)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/15#discussion_r4303988578

### Content
```markdown
The loop variable `l` is an ambiguous single-character name (easily confused with `1`/`I`, flagged as PEP8 E741) and is the only such usage in the codebase. Additionally, the inner loop appends every non-empty line of a corpus file before the 20-line limit is checked at the outer level, so a single large corpus file is scanned in full even though only 20 lines are displayed. Renaming to `line` and breaking the inner loop once the limit is reached addresses both.
```

---

## PR #15: feat(phase16): display training data snippets in incremental pipeline
- **URL**: https://github.com/aomaker-org/unplow/pull/15
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T18:44:24Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/15#pullrequestreview-5559433056

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

The new snippet loop uses the ambiguous variable name `l` and scans entire large corpus files instead of stopping at the 20-line display limit.

<details open>
<summary><strong>1 open issue</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Ambiguous loop variable and missing early exit in corpus scan](https://github.com/aomaker-org/unplow/pull/15#discussion_r4303988578) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR adds observability to the incremental multi-domain training pipeline by printing a formatted 10–20 line snippet of the input training data for each `till-*` domain before fine-tuning begins during `make incremental-pipeline`. It also lowers the default epoch/step counts to CPU-friendly values with environment-variable overrides, and bumps the package version to `0.15.0`.

**Changes:**
- Render a formatted, line-numbered snippet (up to 20 non-empty lines) of `corpus_texts` at the start of `train_domain_incrementally`.
- Make `--epochs`/`--steps` defaults configurable via `EPOCHS`/`STEPS` env vars with reduced CPU-safe fallbacks (1 and 2).
- Bump the package version from `0.14.0` to `0.15.0` in `pyproject.toml` and `uv.lock`.

| File | Description |
| ---- | ----------- |
| src/​unplow/​incremental_pipeline.py | Adds training-data snippet display and env-var-overridable epoch/step defaults. |
| pyproject.toml | Bumps package version to 0.15.0. |
| uv.lock | Updates locked `unplow` version to 0.15.0 to match pyproject. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #14: feat(phase15): incremental multi-domain training loop with progress tracking and storage guardrails
- **URL**: https://github.com/aomaker-org/unplow/pull/14
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T18:32:55Z
- **File**: `src/unplow/incremental_pipeline.py` (Line: 30)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/14#discussion_r4303949172

### Content
```markdown
`load_local_model` is imported here but never used in this module (only `load_causal_lm` is referenced, at line 213 and in `main`). This appears to be a leftover from the spec's example snippet. Consider dropping it to avoid an unused import.
```

---

## PR #14: feat(phase15): incremental multi-domain training loop with progress tracking and storage guardrails
- **URL**: https://github.com/aomaker-org/unplow/pull/14
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T18:32:55Z
- **File**: `src/unplow/incremental_pipeline.py` (Line: 159)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/14#discussion_r4303949196

### Content
```markdown
This breach message hardcodes "<15%", but the threshold is configurable via `min_free_pct` (`self.guardian.min_free_pct`) and can differ from 15 (e.g. the tests use 0.01 and 99.99). When a non-default threshold is in effect, this log line is misleading. Consider interpolating the actual threshold. Optional/minor.
```

---

## PR #14: feat(phase15): incremental multi-domain training loop with progress tracking and storage guardrails
- **URL**: https://github.com/aomaker-org/unplow/pull/14
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T18:32:55Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/14#pullrequestreview-5559376460

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

The new module contains an unused import and an operator-facing log message that hardcodes "<15%" despite a configurable threshold, both minor but worth addressing before approval.

<details open>
<summary><strong>2 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Remove unused load_local_model import](https://github.com/aomaker-org/unplow/pull/14#discussion_r4303949172) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Log the configured free-space threshold instead of 15%](https://github.com/aomaker-org/unplow/pull/14#discussion_r4303949196) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR implements Phase 15 of the `unplow` project: a standalone incremental multi-domain training loop with per-step loss/perplexity tracking and disk-telemetry guardrails. It ingests the spec `inbox/gemweb016.md` into a new `IncrementalTrainer` module that runs discrete epoch increments with AdamW, evaluates perplexity via the existing `PerplexityTracker`, and enforces the free-disk floor between increments via `TillDiskGuardian` (with optional `rclone` offload). It wires the module into the Makefile and bumps the package to `0.14.0`. The dependency contracts (`TillDiskGuardian(manifest_path=..., min_free_pct=...)`, `check_disk`, `enforce_space`, `PerplexityTracker.compute_corpus_loss`, `TILL_DOMAINS[...]["seed_code"]`) all match their call sites, and the CLI bootstrap mirrors the established `full_pipeline.py` pattern.

**Changes:**
- Add `IncrementalTrainer` (`src/unplow/incremental_pipeline.py`) with large-model parameter isolation, per-epoch disk auditing/offload, and a standalone CLI runner.
- Add `tests/test_incremental_pipeline.py` covering roundtrip epochs, storage-floor breach halting, and large-model layer isolation.
- Register the `incremental-pipeline` Make target and bump the version to `0.14.0` (`pyproject.toml`, `uv.lock`).

| File | Description |
| ---- | ----------- |
| src/​unplow/​incremental_pipeline.py | New `IncrementalTrainer` module + CLI; core of the feature (one unused import, one hardcoded-threshold log message noted). |
| tests/​test_incremental_pipeline.py | Unit tests mirroring `test_full_pipeline.py` conventions; validates epochs, halting, and isolation. |
| inbox/​gemweb016.md | Source spec ingested for this phase; reference doc only. |
| Makefile | Adds `incremental-pipeline` target to `.PHONY` and help menu. |
| pyproject.toml | Version bump `0.13.0` → `0.14.0`. |
| uv.lock | Lockfile version bump to match `pyproject.toml`. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #13: feat(phase14): closed-loop multi-domain training pipeline orchestrator and storage offload CLI
- **URL**: https://github.com/aomaker-org/unplow/pull/13
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T18:20:41Z
- **File**: `Makefile` (Line: 12)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/13#discussion_r4303907397

### Content
```markdown
The `till-offload` alias is defined as a make target on line 125 (and advertised via `make till-offload` in the help text) but, unlike its `till-guard-offload` sibling, it is not listed in `.PHONY`. Add `till-offload` to the `.PHONY` list so the alias still runs if a file named `till-offload` ever exists in the directory. This is a minor/optional consistency fix.
```

---

## PR #13: feat(phase14): closed-loop multi-domain training pipeline orchestrator and storage offload CLI
- **URL**: https://github.com/aomaker-org/unplow/pull/13
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T18:20:41Z
- **File**: `tests/test_full_pipeline.py` (Line: 44)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/13#discussion_r4303907418

### Content
```markdown
This assertion (and the pipeline it drives) writes to `workspace/targets/till_python.unp` relative to the current working directory rather than `tmp_path`, so the test leaves artifacts in the repository working tree and is not hermetic or order-independent. Other tests in this suite keep output under `tmp_path` (e.g. `test_cli_till_train` at tests/test_cli.py:326). Consider adding the `monkeypatch` fixture and calling `monkeypatch.chdir(tmp_path)` before invoking `execute_full_pipeline`, so the generated `workspace/` tree is created inside the temp directory and this assertion can target that path.
```

---

## PR #13: feat(phase14): closed-loop multi-domain training pipeline orchestrator and storage offload CLI
- **URL**: https://github.com/aomaker-org/unplow/pull/13
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T18:20:41Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/13#pullrequestreview-5559315288

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

The new `till-guard-offload` enqueue actions have no corresponding `worker.py` dispatch handler, so any queued offload task fails at runtime, contradicting the PR description.

<details open>
<summary><strong>3 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Enqueued offload action lacks worker dispatch handler](https://github.com/aomaker-org/unplow/pull/13#discussion_r4303907372) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [till-offload make target is missing .PHONY](https://github.com/aomaker-org/unplow/pull/13#discussion_r4303907397) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Pipeline test writes output outside tmp_path](https://github.com/aomaker-org/unplow/pull/13#discussion_r4303907418) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR implements "Phase 14" of the `unplow` workspace: a closed-loop, multi-domain `till-*` training pipeline orchestrator plus a storage-offload CLI surface. It ingests the spec in `inbox/gemweb015.md` and turns it into runnable code — `execute_full_pipeline()` enforces the 15% free-disk quota via `TillDiskGuardian`, auto-stages corpora, runs distillation/Bayesian CTS compression across the five domains, and re-checks disk space after each domain, while a new `till-guard-offload` subcommand exposes the guardian's rclone offload via the CLI, Makefile, and async queue registration.

**Changes:**
- Add `src/unplow/full_pipeline.py` orchestrator with a standalone CLI runner and `sys.path` un-shadowing for direct script execution.
- Add the `till-guard-offload`/`till-offload` CLI subcommand, register it as an enqueue action, and add `till-offload`/`full-pipeline` Makefile targets.
- Add unit tests (`tests/test_full_pipeline.py`, `test_cli_till_guard_offload`) and bump the version to `0.13.0`.

| File | Description |
| ---- | ----------- |
| src/​unplow/​full_pipeline.py | New closed-loop multi-domain pipeline orchestrator and CLI entry point. |
| src/​unplow/​cli.py | Adds `till-guard-offload` subcommand, dispatch branch, and enqueue action registration. |
| Makefile | Adds `till-offload`/`full-pipeline` targets and help text; `till-offload` alias missing from `.PHONY`. |
| tests/​test_full_pipeline.py | New pipeline test; writes artifacts to the repo tree instead of `tmp_path`. |
| tests/​test_cli.py | Adds `test_cli_till_guard_offload` dry-run JSON test. |
| pyproject.toml | Version bump to `0.13.0`. |
| uv.lock | Regenerated lockfile reflecting the version bump. |
| inbox/​gemweb015.md | Source spec document for the Phase 14 pipeline. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #13: feat(phase14): closed-loop multi-domain training pipeline orchestrator and storage offload CLI
- **URL**: https://github.com/aomaker-org/unplow/pull/13
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T18:20:40Z
- **File**: `src/unplow/cli.py` (Line: 486)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/13#discussion_r4303907372

### Content
```markdown
The `till-guard-offload` / `till_guard_offload` / `till-offload` / `till_offload` actions are added as valid `enqueue` choices here, but `worker.py`'s `process_task` has no dispatch branch for them (handlers stop at `till-train`/`till_train`). Every other action registered in this list has a matching worker handler, so an enqueued offload task falls through to the final `else` branch and fails with `Unknown decompression action 'till-guard-offload'`. This contradicts the PR description's claim that the action was registered "in asynchronous worker queue dispatch actions." Either add a dispatch branch in `worker.py` (e.g. invoking `TillDiskGuardian.enforce_space`) or remove these from the enqueue choices.
```

---

## PR #12: docs(phase13): ingest operational execution and verification checklist for till models
- **URL**: https://github.com/aomaker-org/unplow/pull/12
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T18:02:39Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/12#pullrequestreview-5559238615

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟢 Approval recommended

It is a documentation-only ingestion that follows the repository's header/footer and 80-column conventions, and every referenced API, file, and command was verified to exist.

**No open issues** 🎉

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR ingests a new inbox document, `inbox/gemweb014.md`, containing the Phase 13 Operational Execution & Verification Checklist for the `till-*` specialized models. It is a documentation-only change that transcribes the incoming checklist (evaluation harness runs, storage guardian verification, and multi-domain build/test synchronization) while conforming to the repository's Directive 04 formatting standards. It fits into the existing `inbox/` convention of ingesting gemini-web messages verbatim.

**Changes:**
- Adds `inbox/gemweb014.md` with the Phase 13 verification checklist (eval harness, `TillDiskGuardian`, and clean build/test steps).
- Follows the `# Path:` header / `<!-- end of file: -->` footer convention and stays within 80 columns.

| File | Description |
| ---- | ----------- |
| inbox/​gemweb014.md | New ingested Phase 13 operational checklist documenting eval, storage-guardian, and build/test verification commands. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #11: feat(phase12): operational command reference, domain till training recipes, and parameter isolation guard
- **URL**: https://github.com/aomaker-org/unplow/pull/11
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T17:51:29Z
- **File**: `src/unplow/replow.py` (Line: 123)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/11#discussion_r4303807187

### Content
```markdown
The inline comment and the PR description both state this guard is meant for large models "on CPU", but the condition only checks the parameter count and never checks the device. If the model is loaded on a GPU (e.g. CUDA available), the guard will still freeze the entire backbone and train only the final layer, silently narrowing the fine-tuning scope beyond the documented CPU-only intent. Consider gating on the device as well so the behavior matches the stated scope.
```

---

## PR #11: feat(phase12): operational command reference, domain till training recipes, and parameter isolation guard
- **URL**: https://github.com/aomaker-org/unplow/pull/11
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T17:51:29Z
- **File**: `src/unplow/replow.py` (Line: 137)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/11#discussion_r4303807204

### Content
```markdown
This new memory-safety guard (freeze backbone / isolate last layer / 4-parameter fallback) is not exercised by any test. The existing train_step tests in tests/test_replow.py use DeterministicMockCausalLM, which has far fewer than 10M parameters, so the `total_params > 10_000_000` branch and its sub-branches never run. Consider adding a test that drives a model above the threshold (or makes the threshold injectable) to cover the freezing/unfreezing logic.
```

---

## PR #11: feat(phase12): operational command reference, domain till training recipes, and parameter isolation guard
- **URL**: https://github.com/aomaker-org/unplow/pull/11
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T17:51:29Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/11#pullrequestreview-5559186427

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

The new `train_step` guard applies regardless of device despite being documented as CPU-only, and its branching logic lacks test coverage.

<details open>
<summary><strong>2 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Guard ignores device and may freeze GPU models](https://github.com/aomaker-org/unplow/pull/11#discussion_r4303807187) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Large-model freeze fallback lacks test coverage](https://github.com/aomaker-org/unplow/pull/11#discussion_r4303807204) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR delivers "Phase 12" of the `unplow` project. It ingests a new operational command-reference document, expands the `Makefile` with per-domain and aggregate `till-train` targets, adds a CPU-memory-safety parameter-isolation guard to `ReplowEngine.train_step`, and bumps the package version to `0.12.0`. The code change fits into the existing till-domain distillation pipeline (`till_guard.train_till_domain` → `ReplowEngine.train_step` → Bayesian CTS archiving) by preventing AdamW optimizer-state explosion when fine-tuning large LLM backbones on CPU.

**Changes:**
- Add a memory-safety guard in `train_step` that freezes the backbone and trains only the last transformer layer/adapter for models with >10M parameters.
- Add `Makefile` targets `till-train-c/cpp/rust/gencode` plus aggregate `till-train-all`, matching the CLI's valid `--domain` choices.
- Add the `inbox/gemweb013.md` command reference and bump version to `0.12.0` (`pyproject.toml`, `uv.lock`).

| File | Description |
| ---- | ----------- |
| `src/​unplow/​replow.py` | Adds the large-model parameter-isolation guard in `train_step`; the guard is documented as CPU-only but is not gated on device, and its branches are untested. |
| `Makefile` | Adds four per-domain training targets and an aggregate `till-train-all`; domains align with CLI choices and all are declared `.PHONY`. |
| `inbox/​gemweb013.md` | New operational command reference; Directive 04 header/footer and <=80-column formatting verified. |
| `pyproject.toml` | Version bump `0.11.0` → `0.12.0`. |
| `uv.lock` | Regenerated lock reflecting the `0.12.0` version. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #10: feat(phase11): disk space telemetry, rclone offload manifest, and till domain model distillation staging
- **URL**: https://github.com/aomaker-org/unplow/pull/10
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T17:34:06Z
- **File**: `src/unplow/till_guard.py` (Line: 216)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/10#discussion_r4303749071

### Content
```markdown
`till_guard.py` emits its diagnostics via `print()` (9 call sites across `enforce_space`), whereas every other library module under `src/unplow/` (e.g. `bundler.py`, `replow.py`, `worker.py`, `eval.py`) produces no direct `print()` output and relies on the module `logger` / CLI-side formatting. This is also internally inconsistent: the module already defines `logger = logging.getLogger("unplow.till_guard")` but only uses it in `load_manifest`. When this runs inside the worker daemon, these prints bypass logging and go straight to stdout. Consider routing these messages through `logger` (gating detail on `verbose`) to stay consistent with the rest of the codebase. Optional.
```

---

## PR #10: feat(phase11): disk space telemetry, rclone offload manifest, and till domain model distillation staging
- **URL**: https://github.com/aomaker-org/unplow/pull/10
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T17:34:06Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/10#pullrequestreview-5559109471

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

`till_guard.py` imports stdlib `tomllib`, which does not exist on Python 3.10 despite the package declaring `requires-python = ">=3.10"`, breaking package import for a supported interpreter.

<details open>
<summary><strong>4 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.png" alt="High severity" width="62" height="18" align="texttop"></picture> [Python 3.10 import fails without tomllib fallback](https://github.com/aomaker-org/unplow/pull/10#discussion_r4303749024) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Manifest space settings are parsed but never used](https://github.com/aomaker-org/unplow/pull/10#discussion_r4303749041) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Staging setup can trigger real rclone moves](https://github.com/aomaker-org/unplow/pull/10#discussion_r4303749055) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Disk guard diagnostics use print instead of logger](https://github.com/aomaker-org/unplow/pull/10#discussion_r4303749071) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

Phase 11 adds a Disk Space Telemetry & Offload Guardian plus a "till" domain-model distillation/staging pipeline. A new `till_guard.py` module monitors free disk space against a 15% threshold, offloads configured cache/checkpoint/archive targets to Google Drive via `rclone move`, stages isolated per-language training corpora (`till-python/c/cpp/rust/gencode`), and runs closed-loop distillation that archives corpora into Bayesian CTS `.unp` containers. It is wired into the CLI (`till-init`, `till-train`), the background worker/queue, the Makefile, package exports, and docs, with the package version bumped to 0.11.0.

**Changes:**
- New `TillDiskGuardian`/`DiskTelemetry`, `init_till_staging`, and `train_till_domain` with manifest-driven offload config (`workspace/till_manifest.toml`).
- CLI subcommands, worker task handlers, Makefile recipes, and `__init__` exports for the till workflow.
- Version bump to 0.11.0, new/updated tests, and documentation across README/architecture/decompression guide.

| File | Description |
| ---- | ----------- |
| src/​unplow/​till_guard.py | Core guardian, telemetry, staging, and distillation logic (main review focus). |
| src/​unplow/​worker.py | Adds `till-init`/`till-train` task handlers. |
| src/​unplow/​cli.py | Adds `till-init`/`till-train` subcommands and queue actions. |
| src/​unplow/​__init__.py | Exports new symbols; bumps `__version__` to 0.11.0. |
| workspace/​till_manifest.toml | Offload target configuration manifest. |
| tests/​test_till_guard.py, test_cli.py, test_worker.py | Unit/CLI/worker coverage for the new pipeline. |
| docs/​architecture.md, docs/​decompression_guide.md, README.md | Phase 11 documentation. |
| Makefile, pyproject.toml, uv.lock | `till-*` recipes and 0.11.0 version bump. |
| inbox/​gemweb012.md, .gitignore | Design note and ignore rules for till artifacts/checkpoints. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #10: feat(phase11): disk space telemetry, rclone offload manifest, and till domain model distillation staging
- **URL**: https://github.com/aomaker-org/unplow/pull/10
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T17:34:05Z
- **File**: `src/unplow/till_guard.py` (Line: 15)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/10#discussion_r4303749024

### Content
```markdown
`import tomllib` will break the package on Python 3.10. `tomllib` was only added to the standard library in Python 3.11, but this project declares `requires-python = ">=3.10"` (pyproject.toml) and `uv.lock` still resolves a 3.10 environment (it carries `python_full_version < '3.11'` markers). Because `src/unplow/__init__.py` imports from `till_guard`, this makes the entire `unplow` package fail to import with `ModuleNotFoundError: No module named 'tomllib'` on any supported 3.10 install. Either bump `requires-python` to `>=3.11`, or add a `tomli` fallback (declare `tomli; python_version < "3.11"` in dependencies and `try: import tomllib except ModuleNotFoundError: import tomli as tomllib`).
```

---

## PR #10: feat(phase11): disk space telemetry, rclone offload manifest, and till domain model distillation staging
- **URL**: https://github.com/aomaker-org/unplow/pull/10
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T17:34:05Z
- **File**: `src/unplow/till_guard.py` (Line: 152)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/10#discussion_r4303749041

### Content
```markdown
The manifest's `[manifest]` settings `target_free_percentage` and `rclone_destination` are parsed by `load_manifest()` but never applied: `TillDiskGuardian` always uses the hard-coded constructor defaults (`min_free_pct=15.0`, `gdrive_dest="gdrive:transfer/till-20261017/"`). As a result, editing those values in `workspace/till_manifest.toml` has no effect at runtime (they only coincidentally match the defaults today). Consider reading `target_free_percentage` and `rclone_destination` from the loaded manifest to override these fields, otherwise the manifest keys are misleading dead configuration.
```

---

## PR #10: feat(phase11): disk space telemetry, rclone offload manifest, and till domain model distillation staging
- **URL**: https://github.com/aomaker-org/unplow/pull/10
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T17:34:05Z
- **File**: `src/unplow/till_guard.py` (Line: 271)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/10#discussion_r4303749055

### Content
```markdown
`init_till_staging` (and `train_till_domain` at line 377) invoke `enforce_space(...)` with the default `dry_run=False`, so when free space on `/` drops below the threshold this will actually run `rclone move` against the real offload targets declared in `workspace/till_manifest.toml` (`~/.cache/uv`, `~/.cache/pip`, `models/qwen2.5-1.5b-instruct`, etc.). The staging/test paths (`test_init_till_staging`, `test_cli_till_init`, `test_worker_till_tasks`) all reach this code against the real default manifest, so on a nearly-full CI/dev machine these "setup" calls could silently relocate a developer's caches and model checkpoints to Google Drive. Consider plumbing a `dry_run` flag through `init_till_staging`/`train_till_domain` (defaulting to a safe value and passing `dry_run=True` from tests) so disk enforcement isn't triggered as a side effect of staging.
```

---

## PR #9: feat(phase10): hierarchical workspace bundler, recursive tree archiving, and Bayesian CTS container streaming
- **URL**: https://github.com/aomaker-org/unplow/pull/9
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T17:06:16Z
- **File**: `src/unplow/bundler.py` (Line: 21)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/9#discussion_r4303632724

### Content
```markdown
`BUNDLE_MAGIC` and `BUNDLE_VERSION` are defined but never used anywhere in the module (or exported). The bundle payload is wrapped in a standard `.unp` container with no bundle-specific magic or version byte, so there is no way to distinguish a workspace bundle from an ordinary single-text container at unbundle time. Either incorporate these constants into the serialized format (so `unbundle_archive` can validate it was handed an actual bundle) or remove them to avoid dead, misleading state.
```

---

## PR #9: feat(phase10): hierarchical workspace bundler, recursive tree archiving, and Bayesian CTS container streaming
- **URL**: https://github.com/aomaker-org/unplow/pull/9
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T17:06:16Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/9#pullrequestreview-5558963147

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

It introduces a new lossless archival/compression path whose correctness depends on tokenizer round-tripping of arbitrary bytes, and it contains a confirmed directory-exclusion bug that can silently drop all files, warranting human verification.

<details open>
<summary><strong>4 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.png" alt="High severity" width="62" height="18" align="texttop"></picture> [Exclude paths using relative tree paths](https://github.com/aomaker-org/unplow/pull/9#discussion_r4303632649) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Remove trailing blank line after footer](https://github.com/aomaker-org/unplow/pull/9#discussion_r4303632685) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Remove trailing blank line after footer](https://github.com/aomaker-org/unplow/pull/9#discussion_r4303632705) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Unused bundle magic and version constants](https://github.com/aomaker-org/unplow/pull/9#discussion_r4303632724) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR adds **Phase 10: the Hierarchical Workspace Bundler**, a new `WorkspaceBundler` (`src/unplow/bundler.py`) that recursively packages a directory tree into a single self-verifying `.unp` container using the existing 3-expert Bayesian CTS arithmetic codec, and restores it with per-file SHA-256 verification. It fits on top of the existing single-payload `LLMArithmeticCodec`, exposing the capability through new `bundle`/`unbundle` CLI subcommands, worker queue handlers, docs, and tests. Supporting changes add convenience accessor properties on the codec result types and a small hardening fix to the container header packer. The version is bumped 0.9.0 → 0.10.0.

**Changes:**
- New `bundler.py` with `BundledFileEntry`, `BundleManifestReport`, and `WorkspaceBundler` (bundle/unbundle + manifest reporting).
- Wiring: CLI `bundle`/`unbundle` commands, worker task handlers, `__init__` exports, Makefile `bundle` target, README/architecture/decompression docs.
- Codec support: `container_bytes`/`text`/`meta`/`is_valid` properties on result types, and `pack_unp_container` now coerces `max_ctx_len is None` to 512.

| File | Description |
| ---- | ----------- |
| src/​unplow/​bundler.py | New core bundler engine; contains the exclusion-filter bug and unused magic/version constants |
| src/​unplow/​cli.py | Adds `bundle`/`unbundle` subcommands and dispatch |
| src/​unplow/​worker.py | Adds `bundle`/`unbundle` queue task handlers |
| src/​unplow/​paradigm2.py | Adds accessor properties (`container_bytes`, `text`, `meta`, `is_valid`) used by the bundler |
| src/​unplow/​container.py | Hardens `pack_unp_container` against a `None` `max_ctx_len` |
| src/​unplow/​__init__.py | Exports new bundler symbols; version bump to 0.10.0 |
| tests/​test_bundler.py | New unit tests for bundle/unbundle round-trip and reporting |
| tests/​test_cli.py | CLI help + bundle/unbundle mock round-trip tests |
| tests/​test_worker.py | Worker bundle-task test |
| docs/​architecture.md | Documents the bundler; adds trailing blank line after footer |
| docs/​decompression_guide.md | Adds Phase 10 section; adds trailing blank line after footer |
| README.md | Adds bundler feature/usage/tree entries |
| Makefile | Adds `bundle` target and `.PHONY` entry |
| inbox/​gemweb011.md | Phase 10 specification document |
| pyproject.toml /​ uv.lock | Version bump to 0.10.0 |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #9: feat(phase10): hierarchical workspace bundler, recursive tree archiving, and Bayesian CTS container streaming
- **URL**: https://github.com/aomaker-org/unplow/pull/9
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T17:06:15Z
- **File**: `src/unplow/bundler.py` (Line: 146)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/9#discussion_r4303632649

### Content
```markdown
The directory exclusion filter checks `p.parts`, which for an `rglob` result rooted at the resolved `root_dir` includes the absolute path prefix. As a result, if the source directory's absolute path contains any excluded name anywhere above the tree (e.g. `build`, `dist`, `venv`, `logs`, or `workspace`), every file is silently skipped and an empty archive is produced. This is a realistic failure here since the project's own docs and Makefile write/read bundles under `workspace/` — bundling a directory located under `workspace/` would drop all files with no error. Match the excludes against the path relative to `root_dir` instead.
```

---

## PR #9: feat(phase10): hierarchical workspace bundler, recursive tree archiving, and Bayesian CTS container streaming
- **URL**: https://github.com/aomaker-org/unplow/pull/9
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T17:06:15Z
- **File**: `docs/architecture.md` (Line: 322)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/9#discussion_r4303632685

### Content
```markdown
A blank line is added after the end-of-file footer. By convention in this repo the `<!-- end of file: ... -->` footer is the final line of the file (see README.md, AI.md, AGENTS.md). Please drop the trailing blank line(s) so the footer remains last.
```

---

## PR #9: feat(phase10): hierarchical workspace bundler, recursive tree archiving, and Bayesian CTS container streaming
- **URL**: https://github.com/aomaker-org/unplow/pull/9
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T17:06:15Z
- **File**: `docs/decompression_guide.md` (Line: 371)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/9#discussion_r4303632705

### Content
```markdown
A blank line is added after the end-of-file footer. By convention in this repo the `<!-- end of file: ... -->` footer is the final line of the file (see README.md, AI.md, AGENTS.md). Please drop the trailing blank line(s) so the footer remains last.
```

---

## PR #8: feat(phase9): automated evaluation harness, functional correctness sandboxing, and perplexity shift tracking
- **URL**: https://github.com/aomaker-org/unplow/pull/8
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T16:41:19Z
- **File**: `src/unplow/__init__.py` (Line: 216)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/8#discussion_r4303526262

### Content
```markdown
The `__all__` list follows a consistent two-section ordering: capitalized class/constant names sorted A–Z first, followed by lowercase function names sorted A–Z (starting at `"audit_system_models"`). These two lowercase function names are inserted in the middle of the capitalized section (between `"Task"` and `"TaskStatus"`), breaking that convention. They should be moved into the lowercase section (alphabetically `evaluate_artifact` fits after `"compute_str_sha256"` and `format_summary_table` right after it).
```

---

## PR #8: feat(phase9): automated evaluation harness, functional correctness sandboxing, and perplexity shift tracking
- **URL**: https://github.com/aomaker-org/unplow/pull/8
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T16:41:19Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/8#pullrequestreview-5558833984

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

It integrates sandboxed `exec`-based arbitrary code execution and multiprocessing into the background worker daemon, which is operationally/security-sensitive and warrants human review in addition to the convention issues flagged.

<details open>
<summary><strong>4 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Use normalized action in task dispatch](https://github.com/aomaker-org/unplow/pull/8#discussion_r4303526209) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Remove blank line after file footer](https://github.com/aomaker-org/unplow/pull/8#discussion_r4303526236) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Remove extra blank line after file footer](https://github.com/aomaker-org/unplow/pull/8#discussion_r4303526248) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Move utility functions to the lowercase __all__ section](https://github.com/aomaker-org/unplow/pull/8#discussion_r4303526262) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR implements "Phase 9" for `unplow`, adding an automated evaluation harness (`src/unplow/eval.py`) that pairs a sandboxed functional-correctness runner (executing refactored code against unit-test assertions in isolated multiprocessing namespaces with timeout guards) with a perplexity/cross-entropy shift tracker measured over a validation corpus. It wires this into the CLI (`unplow eval`), the background worker daemon, the `Makefile`, docs, and bumps the version to `0.9.0`. Integration points were verified against the codebase: `load_causal_lm` returns `(tokenizer, model, path)`, `decompress_from_container` unpacks to a 3-tuple, and `LLMArithmeticCodec` accepts `model=`/`tokenizer=` — all used consistently.

**Changes:**
- Add `eval.py` (`FunctionalCorrectnessRunner`, `PerplexityTracker`, `evaluate_artifact`, `format_summary_table`) plus data classes and public exports.
- Add the `unplow eval` CLI subcommand (alias `replow-eval`) and `eval`/`replow_eval` worker task handlers, with a `make eval` target.
- Add unit/CLI/worker tests and `fib_solution.py`/`fib_tests.py` fixtures; document in README/architecture/decompression guides; bump version to `0.9.0`.

| File | Description |
| ---- | ----------- |
| src/​unplow/​eval.py | New evaluation harness: sandbox runner + perplexity tracker + reporting |
| src/​unplow/​cli.py | Adds `eval`/`replow-eval` subcommand and dispatch; consolidates `load_causal_lm` import |
| src/​unplow/​worker.py | Adds `eval`/`replow_eval` task branch (uses `task.action` instead of normalized `action`) |
| src/​unplow/​__init__.py | Exports new eval symbols (two lowercase names misplaced in `__all__`); version bump |
| tests/​test_eval.py, test_cli.py, test_worker.py | New coverage for runner, tracker, CLI, and worker eval task |
| tests/​data/​fib_solution.py, fib_tests.py | Fixtures for functional evaluation |
| README.md, docs/​architecture.md, docs/​decompression_guide.md | Phase 9 docs (trailing blank lines added after footers) |
| Makefile, pyproject.toml, uv.lock, inbox/​gemweb010.md | `make eval` target, version bump, ingested spec |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #8: feat(phase9): automated evaluation harness, functional correctness sandboxing, and perplexity shift tracking
- **URL**: https://github.com/aomaker-org/unplow/pull/8
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T16:41:18Z
- **File**: `src/unplow/worker.py` (Line: 485)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/8#discussion_r4303526209

### Content
```markdown
This branch tests `task.action` directly, while every other branch in `process_task` dispatches on the normalized `action = task.action.lower()` (defined at the top of the method; see e.g. the `("replow", "replow_pipeline")` branch). As a result, an enqueued action such as `"Eval"` or `"REPLOW_EVAL"` would be handled case-insensitively by every other branch but fall through to the `else` `ValueError` here. Use the normalized `action` for consistency.
```

---

## PR #8: feat(phase9): automated evaluation harness, functional correctness sandboxing, and perplexity shift tracking
- **URL**: https://github.com/aomaker-org/unplow/pull/8
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T16:41:18Z
- **File**: `docs/architecture.md` (Line: 299)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/8#discussion_r4303526236

### Content
```markdown
This adds a trailing blank line after the `<!-- end of file: ... -->` footer, so the footer is no longer the final line of the file. Per the repo convention (Directive 04, mandatory file footers; e.g. `src/unplow/container.py` and `src/unplow/mechanistic.py` both end exactly on their footer line), the footer should be the last line. Remove the trailing blank line.
```

---

## PR #8: feat(phase9): automated evaluation harness, functional correctness sandboxing, and perplexity shift tracking
- **URL**: https://github.com/aomaker-org/unplow/pull/8
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T16:41:18Z
- **File**: `docs/decompression_guide.md` (Line: 332)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/8#discussion_r4303526248

### Content
```markdown
This adds another trailing blank line after the `<!-- end of file: ... -->` footer, leaving the footer two lines before the end of the file. Per the repo convention (Directive 04, mandatory file footers; e.g. `src/unplow/container.py` and `src/unplow/mechanistic.py` end exactly on their footer line), the footer should be the final line. Remove the trailing blank line(s) after the footer.
```

---

## PR #7: feat(phase8): replow closed-loop refactoring, distillation staging, and Bayesian CTS archiving
- **URL**: https://github.com/aomaker-org/unplow/pull/7
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T16:22:56Z
- **File**: `src/unplow/replow.py` (Line: 5)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/7#discussion_r4303452509

### Content
```markdown
This docstring (and the matching PR description / architecture docs) states that the module "Extracts raw code basins from model weights" via mechanistic steering/task-vectors as Stage 1, but `ReplowEngine` does not implement any weight extraction — it only refactors, stages, trains on, and compresses code that is supplied by the caller (via `--raw`, a file, or stdin). No steering/task-vector/mechanistic code is imported or invoked here. Consider updating the wording to reflect that harvesting is an upstream/manual step, or implementing the extraction stage, so the documentation matches the behavior.
```

---

## PR #7: feat(phase8): replow closed-loop refactoring, distillation staging, and Bayesian CTS archiving
- **URL**: https://github.com/aomaker-org/unplow/pull/7
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T16:22:56Z
- **File**: `src/unplow/replow.py` (Line: 19)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/7#discussion_r4303452530

### Content
```markdown
These three imports are unused in this module. `pack_unp_container`, `FLAG_BAYES_MIXER`, and `BayesianCTSMixerCDF` are never referenced—compression is delegated entirely to `LLMArithmeticCodec.compress_to_container(..., use_bayes_mixer=True)`. The unused CTS-mixer imports in particular can mislead readers into thinking the mixer is invoked directly here. Consider keeping only the import that is actually used.
```

---

## PR #7: feat(phase8): replow closed-loop refactoring, distillation staging, and Bayesian CTS archiving
- **URL**: https://github.com/aomaker-org/unplow/pull/7
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T16:22:56Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/7#pullrequestreview-5558744082

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

It adds a new engine integrating the arithmetic codec, background worker, and distillation training whose runtime behavior cannot be fully verified here, and it carries several concrete defects (a dead bits-per-byte branch, documentation examples that would fail, and a docstring/PR claim of weight extraction not implemented) that warrant human review.

<details open>
<summary><strong>5 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Use stats.bits_per_byte from compress result](https://github.com/aomaker-org/unplow/pull/7#discussion_r4303452436) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Fix replow CLI example arguments](https://github.com/aomaker-org/unplow/pull/7#discussion_r4303452465) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Correct run_pipeline API example](https://github.com/aomaker-org/unplow/pull/7#discussion_r4303452488) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Docstring overstates code weight extraction](https://github.com/aomaker-org/unplow/pull/7#discussion_r4303452509) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Remove unused CTS-mixer imports](https://github.com/aomaker-org/unplow/pull/7#discussion_r4303452530) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR implements "Phase 8" of `unplow`, adding a `replow` closed-loop pipeline. It introduces a new `ReplowEngine`/`ReplowArtifact` module that sanitizes/refactors source code via Python's `ast`, optionally stages a JSONL training dataset and runs lightweight PyTorch next-token distillation steps, and compresses the refactored code into a `.unp` container using the existing Bayesian CTS arithmetic codec. The capability is exposed through a new `unplow replow` CLI subcommand, a background-worker handler, a `make replow` target, a test suite, and documentation, with a version bump to `0.8.0`.

**Changes:**
- New `src/unplow/replow.py` engine (AST refactor → optional dataset staging/distillation → Bayesian CTS `.unp` archiving), re-exported from `__init__.py`.
- CLI subcommand `replow` and `DecompressionWorker` `replow`/`replow_pipeline` task handlers.
- Tests (`test_replow.py`, additions to `test_cli.py`/`test_worker.py`), docs, `Makefile` target, spec ingest, and `0.8.0` version bump.

| File | Description |
| ---- | ----------- |
| src/​unplow/​replow.py | New `ReplowEngine`/`ReplowArtifact`; refactoring, staging, train step, and container archiving (has unused imports, a dead bpb branch, and a docstring overstating weight extraction). |
| src/​unplow/​worker.py | Adds `replow`/`replow_pipeline` worker handler invoking `run_pipeline`. |
| src/​unplow/​cli.py | Adds `replow` subcommand (positional `file`, int `--domain`) and run dispatch. |
| src/​unplow/​__init__.py | Re-exports `ReplowArtifact`/`ReplowEngine`; bumps `__version__` to 0.8.0. |
| tests/​test_replow.py | New unit tests for refactoring, fallback, staging, train step, pipeline. |
| tests/​test_cli.py | CLI help + mock-model end-to-end `replow` tests. |
| tests/​test_worker.py | Worker `replow` task test. |
| docs/​decompression_guide.md | Phase 8 section; CLI/Python examples contain argument/attribute mismatches. |
| docs/​architecture.md | Section 12 architecture overview of `replow`. |
| README.md | Key feature bullet, quickstart item 20, repo-structure entry. |
| Makefile | Adds `replow` target. |
| inbox/​gemweb009.md | Ingested Phase 8 specification. |
| pyproject.toml /​ uv.lock | Version bump to 0.8.0. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #7: feat(phase8): replow closed-loop refactoring, distillation staging, and Bayesian CTS archiving
- **URL**: https://github.com/aomaker-org/unplow/pull/7
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T16:22:55Z
- **File**: `src/unplow/replow.py` (Line: 187)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/7#discussion_r4303452436

### Content
```markdown
The first branch here is unreachable: `compress_to_container` always returns a `ContainerResult`, which is a `bytes` subclass exposing the payload directly and carrying `.meta`/`.stats` attributes (it has no `container_bytes` attribute, and it is not a `tuple`). As a result both the `hasattr(res, "container_bytes")` and `isinstance(res, tuple)` branches never execute, so the accurate `res.stats.bits_per_byte` is never used. The code always falls into the `else` fallback, which computes bpb over the *entire container* (header + payload) rather than just the compressed bitstream. This overstates `bits_per_byte` for small inputs and diverges from how the rest of the codebase reports it (e.g. `cli.py` unpacks the result and prints `stats.bits_per_byte`). Consider reading the density from the result's stats.
```

---

## PR #7: feat(phase8): replow closed-loop refactoring, distillation staging, and Bayesian CTS archiving
- **URL**: https://github.com/aomaker-org/unplow/pull/7
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T16:22:55Z
- **File**: `docs/decompression_guide.md` (Line: 260)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/7#discussion_r4303452465

### Content
```markdown
This CLI example will not run as written. There is no `--file` option on the `replow` subcommand; the input file is a positional argument (`p_replow.add_argument("file", nargs="?", ...)`), so `--file extracted_snippet.py` raises an "unrecognized arguments" error. Additionally, `--domain` is declared with `type=int`, so `--domain python` fails argparse parsing; the python-code domain prior is id `2`. Update the example to use the positional file argument and an integer domain id.
```

---

## PR #7: feat(phase8): replow closed-loop refactoring, distillation staging, and Bayesian CTS archiving
- **URL**: https://github.com/aomaker-org/unplow/pull/7
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T16:22:55Z
- **File**: `docs/decompression_guide.md` (Line: 283)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/7#discussion_r4303452488

### Content
```markdown
This Python API example does not match the actual `run_pipeline` signature and would raise `TypeError`/`AttributeError` if executed. The keyword arguments are wrong: the method accepts `raw_code`, `output_unp_path`, `train_steps`, and `domain_id` (an int) — there is no `raw_source`, `output_path`, `dataset_path`, or string `domain` parameter (dataset staging is a separate call to `stage_training_dataset`). `run_pipeline` also returns a `(artifact, loss)` tuple rather than a bare artifact. Finally, the attribute names are wrong: `ReplowArtifact` exposes `refactored_text` and `compressed_bytes_len`, not `refactored_code`/`compressed_size`.
```

---

## PR #6: feat(phase7): unified layer-wise internal representation auditor and convergence flow
- **URL**: https://github.com/aomaker-org/unplow/pull/6
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T16:02:07Z
- **File**: `src/unplow/auditor.py` (Line: 9)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/6#discussion_r4303371948

### Content
```markdown
`math` and `Optional` are imported here but never used in this module: Shannon entropy is computed with `torch.log2` (not `math`), and no annotation in the file uses `Optional`. Removing these unused imports keeps the module clean and avoids lint warnings.
```

---

## PR #6: feat(phase7): unified layer-wise internal representation auditor and convergence flow
- **URL**: https://github.com/aomaker-org/unplow/pull/6
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T16:02:07Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/6#pullrequestreview-5558648191

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

It spans new ML logit-lens logic plus CLI/worker/model integration whose live-model behavior cannot be fully verified in this environment, warranting human review.

<details open>
<summary><strong>2 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Rename misnamed FFN weight norm field](https://github.com/aomaker-org/unplow/pull/6#discussion_r4303371914) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Remove unused math and Optional imports](https://github.com/aomaker-org/unplow/pull/6#discussion_r4303371948) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR implements "Phase 7" of `unplow`: a layer-wise internal representation auditor that applies the Logit Lens technique across every Transformer layer. For each layer it projects the final-token residual hidden state through the terminal norm + unembedding head, then records the top predicted token, logit/probability, Shannon entropy, residual L2 norm, and the FFN down-projection weight norm. The feature is surfaced through a new `auditor` module, a CLI subcommand, a background-worker task handler, and docs, fitting alongside the existing mechanistic/steering inspection tooling.

**Changes:**
- Adds `src/unplow/auditor.py` (`ModelInternalAuditor`, `LayerAuditReport`) plus package exports and a `load_local_model` helper.
- Enhances `MockCausalLM`/`MockModelOutput` with `output_hidden_states=True` for deterministic testing.
- Wires up the audit flow via the `audit` CLI subcommand, a worker `layer_audit` handler, a `make audit` target, docs, and a `0.7.0` version bump.

| File | Description |
| ---- | ----------- |
| src/​unplow/​auditor.py | New logit-lens layer auditor and report dataclass (core feature). |
| src/​unplow/​models.py | Adds `output_hidden_states` to the mock model and a `load_local_model` helper. |
| src/​unplow/​cli.py | Adds `audit` subcommand with `--top-k`, `--json`, `--output` options. |
| src/​unplow/​worker.py | Adds `layer_audit` task handler writing a JSON work product. |
| src/​unplow/​__init__.py | Exports new symbols and bumps `__version__` to 0.7.0. |
| tests/​test_auditor.py | New unit tests for the auditor flow, empty prompt, and error path. |
| tests/​test_cli.py | Adds CLI audit help and mock-execution tests. |
| tests/​test_worker.py | Adds a worker audit-task test. |
| pyproject.toml /​ uv.lock | Version bump to 0.7.0. |
| README.md /​ docs/​*.md | Documents the Phase 7 auditor and CLI usage. |
| Makefile | Adds `audit` target. |
| inbox/​gemweb008.md | Design spec note for the auditor. |

Two minor issues were flagged inline (unused `math`/`Optional` imports; the misleadingly named `ffn_output_l2_norm` field that actually holds a static weight-matrix norm). No functional bugs were found; the mock/real logit-lens flow, alias dispatch, and worker integration are consistent.
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #6: feat(phase7): unified layer-wise internal representation auditor and convergence flow
- **URL**: https://github.com/aomaker-org/unplow/pull/6
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T16:02:06Z
- **File**: `src/unplow/auditor.py` (Line: 22)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/6#discussion_r4303371914

### Content
```markdown
The field `ffn_output_l2_norm` actually stores the L2 norm of the layer's `down_proj.weight` matrix (computed at line 133), which is a static, prompt-independent weight norm — not the L2 norm of the FFN output activation that the name implies. This is misleading to consumers of `to_dict()` and the emitted JSON report (the architecture doc itself describes it as the weight scale ‖W_down‖₂). Consider renaming it to something like `ffn_down_weight_l2_norm` and updating `to_dict()` and the tests accordingly. Optional/minor.
```

---

## PR #5: feat(phase6): cross-substrate task vectors, differential SVD steering, and online Bayesian CTS mixture coding
- **URL**: https://github.com/aomaker-org/unplow/pull/5
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T01:26:31Z
- **File**: `src/unplow/models.py` (Line: 282)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/5#discussion_r4300720809

### Content
```markdown
Calling `torch.manual_seed(42)` inside the `MockCausalLM` constructor mutates the global PyTorch RNG as a side effect of model construction. Since `MockCausalLM`/`DeterministicMockCausalLM` is instantiated widely across the test suite (including via `load_causal_lm("mock")`), any test that constructs a mock and later relies on global randomness will have its RNG state silently reset, creating hidden test-ordering coupling. If reproducible weights are the goal, prefer seeding a local `torch.Generator` and passing it to the init routines, or reset the RNG state after construction, rather than reseeding the global RNG in `__init__`.
```

---

## PR #5: feat(phase6): cross-substrate task vectors, differential SVD steering, and online Bayesian CTS mixture coding
- **URL**: https://github.com/aomaker-org/unplow/pull/5
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T01:26:31Z
- **File**: `src/unplow/cli.py` (Line: 33)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/5#discussion_r4300720833

### Content
```markdown
`FLAG_BAYES_MIXER` (line 32) and `BayesianCTSMixerCDF` (line 33) are imported here but never used anywhere in `cli.py`. Additionally, the module-level import of `CrossSubstrateTaskVectorEngine` (line 40) is redundant because the `task-vector` branch re-imports it locally (line 659). Removing the unused imports keeps the module clean.
```

---

## PR #5: feat(phase6): cross-substrate task vectors, differential SVD steering, and online Bayesian CTS mixture coding
- **URL**: https://github.com/aomaker-org/unplow/pull/5
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T01:26:31Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/5#pullrequestreview-5555352503

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

It adds large, correctness-critical numerical modules (lossless arithmetic coder, SVD steering, and an untested Q6_K dequantizer) whose failure modes are silent, warranting human verification.

<details open>
<summary><strong>3 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Add tests for Q6_K dequantization](https://github.com/aomaker-org/unplow/pull/5#discussion_r4300720788) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Mock CausalLM reseeds global RNG](https://github.com/aomaker-org/unplow/pull/5#discussion_r4300720809) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Remove unused CLI imports](https://github.com/aomaker-org/unplow/pull/5#discussion_r4300720833) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR implements "Phase 6" of `unplow`, an air-gapped LLM-based arithmetic compression / weight-forensics toolkit. It adds two substantial new capabilities and wires them through the container format, codec, CLI, and worker. It bridges on-disk GGUF weights with live Safetensors models (cross-substrate task vectors + SVD steering), adds an online Bayesian mixture arithmetic coder, and completes the final flag bit of the 18-byte `.unp` container header.

**Changes:**
- New `task_vectors.py` (cross-substrate ΔW SVD extraction, Logit Lens projection, forward-hook residual steering) and `ctw_coder.py` (online Bayesian CTS 3-expert mixture CDF), each with unit tests.
- Container Bit 3 (`FLAG_BAYES_MIXER = 0x08`) added across `container.py`/`conditioning.py`, and `paradigm2.py` selects the Bayesian mixer by precedence (bayes > bigram > unigram); CLI/worker gain `--bayes-mix` and a `task-vector` subcommand.
- Q6_K (GGML type 14) vectorized block dequantizer in `gguf_inspector.py`; version bump to `0.6.0`; README/docs updates.

| File | Description |
| ---- | ----------- |
| src/​unplow/​task_vectors.py | New cross-substrate task-vector engine (SVD, Logit Lens, steering hooks). |
| src/​unplow/​ctw_coder.py | New online Bayesian CTS mixture CDF coder. |
| src/​unplow/​gguf_inspector.py | Adds Q6_K block dequantizer and type registration. |
| src/​unplow/​paradigm2.py | Wires `enable_bayes_mixer` through compress/decompress/container paths. |
| src/​unplow/​container.py | Adds `FLAG_BAYES_MIXER` and metadata accessors. |
| src/​unplow/​conditioning.py | Extends `pack_flags_with_domain` with bit 3. |
| src/​unplow/​cli.py | Adds `task-vector` command and `--bayes-mix`; has unused imports. |
| src/​unplow/​worker.py | Adds `task_vector` and `bayes_mixture` task handling. |
| src/​unplow/​models.py | Adds `torch.manual_seed(42)` in `MockCausalLM.__init__` (global side effect). |
| src/​unplow/​__init__.py | Exports new symbols; version `0.6.0`. |
| tests/​* | New tests for task vectors, CTS mixer, CLI/worker/conditioning flag. |
| README.md /​ docs/​*.md /​ Makefile /​ pyproject.toml /​ uv.lock /​ inbox/​gemweb007.md | Docs, targets, version, and planning spec. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #5: feat(phase6): cross-substrate task vectors, differential SVD steering, and online Bayesian CTS mixture coding
- **URL**: https://github.com/aomaker-org/unplow/pull/5
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T01:26:30Z
- **File**: `src/unplow/gguf_inspector.py` (Line: 275)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/5#discussion_r4300720788

### Content
```markdown
The new Q6_K block dequantizer (GGML type 14) has no unit test, whereas the other dequantization paths in this file are tested (F32/Q8_0 via `build_synthetic_gguf_v3_bytes`, and Q4_0 via a hand-built block passed directly to `dequantize_buffer` in `tests/test_gguf_inspector.py`). Because `build_synthetic_gguf_v3_bytes` only supports F32/Q8_0, this complex vectorized path (ql/qh/scale unpacking, sign offset, per-sub-block scale indexing) is entirely unexercised. Consider adding a direct `dequantize_buffer(..., GGML_TYPE_Q6_K, 256)` test with a hand-constructed 210-byte block, mirroring the existing Q4_0 test.
```

---

## PR #4: feat(phase5): GGUF binary inspection, block dequantizers, SVD subspace steering, and Order-1 Markov bigram cache coding
- **URL**: https://github.com/aomaker-org/unplow/pull/4
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T00:52:59Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/4#pullrequestreview-5555183165

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

It is a large change that modifies the correctness-critical lossless arithmetic-codec path and adds substantial new binary-parsing and quantization logic, so final human verification is warranted.

**No open issues** 🎉

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR implements "Phase 5" of `unplow`, extending the LLM-as-compressor toolkit with three new capabilities: offline GGUF binary forensics, multi-neuron SVD subspace activation steering, and a higher-order (Order-1 Markov bigram) adaptive cache that plugs into the existing lossless arithmetic codec. It adds two new modules plus wiring through the container flags, codec, CLI, worker, docs, and package exports, and bumps the version to `0.5.0`.

**Changes:**
- Adds `gguf_inspector.py`: a pure-Python `mmap` GGUF v2/v3 parser, vectorized F32/F16/Q8_0/Q4_0 dequantizers, and task-vector divergence forensics against PyTorch reference tensors.
- Adds `subspace_coder.py`: an SVD-based rank-1 subspace steering engine and `Order1BigramCacheCDF`, a new bigram/unigram/LLM mixture CDF gated by container flag `FLAG_BIGRAM_CACHE = 0x04`.
- Wires the new bigram cache and both subcommands (`gguf-inspect`, `subspace-steer`, plus `--bigram-mix`) through the codec, container flags, CLI, worker, Makefile, docs, and `__init__` exports, with new test suites.

| File | Description |
| ---- | ----------- |
| src/​unplow/​gguf_inspector.py | New GGUF parser, block dequantizers, and task-vector delta report. |
| src/​unplow/​subspace_coder.py | New SVD subspace steering engine and Order-1 bigram mixture CDF. |
| src/​unplow/​paradigm2.py | Integrates bigram cache into compress/decompress/container paths. |
| src/​unplow/​container.py | Adds `FLAG_BIGRAM_CACHE` flag, properties, and pack support. |
| src/​unplow/​conditioning.py | Extends `pack_flags_with_domain` with the bigram-cache bit. |
| src/​unplow/​cli.py | Adds `gguf-inspect`/`subspace-steer` subcommands and `--bigram-mix`. |
| src/​unplow/​worker.py | Adds `gguf_inspect`/`subspace_steer` task actions and `bigram_mix` passthrough. |
| src/​unplow/​__init__.py | Exports new symbols and bumps `__version__` to 0.5.0. |
| tests/​test_gguf_inspector.py | New tests for GGUF parsing/dequant/delta comparison. |
| tests/​test_subspace_coder.py | New tests for SVD steering, hook cleanup, bigram CDF, roundtrip. |
| tests/​test_worker.py | Adds worker tests for the two new task actions. |
| tests/​test_cli.py | Adds CLI help tests for the new subcommands/options. |
| tests/​test_conditioning.py | Extends flag-packing roundtrip test with the bigram bit. |
| README.md /​ docs/​architecture.md /​ docs/​decompression_guide.md | Document Phase 5 features and references. |
| inbox/​gemweb006.md | New free-form design note/blueprint for Phase 5. |
| pyproject.toml /​ uv.lock /​ Makefile | Version bump to 0.5.0 and new make targets. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #3: feat(phase4): activation steering, folded circuits, and cache mixture arithmetic coding
- **URL**: https://github.com/aomaker-org/unplow/pull/3
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-17T00:24:57Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/3#pullrequestreview-5555051763

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

A compression tool's lossless guarantee is silently broken for non-default `cache_weight` (not persisted in the container), and the core codec return path relies on fragile CPython bytecode introspection, so human review is warranted.

<details open>
<summary><strong>2 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/high-v2-light.png" alt="High severity" width="62" height="18" align="texttop"></picture> [Cache weight omitted from container breaks roundtrip](https://github.com/aomaker-org/unplow/pull/3#discussion_r4300472902) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Bytecode introspection in __iter__ is fragile and slow](https://github.com/aomaker-org/unplow/pull/3#discussion_r4300472929) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR implements "Phase 4" of the `unplow` LLM-compression toolkit (per `inbox/gemweb005.md`/`gemweb004.md`), adding two additive modules and wiring them into the codec, CLI, and worker. It extends the mechanistic-interpretability and arithmetic-coding capabilities without changing the frozen 18-byte `.unp` header layout, using a previously-unused flag bit (`0x02`, `FLAG_CACHE_MIXTURE`).

**Changes:**
- New `steering.py` (RMSNorm-folded SwiGLU circuit lens + causal forward-hook activation steering of `down_proj` neurons) and new `cache_coder.py` (online prefix-cache mixture CDF + per-token Shannon surprisal profiler).
- Codec integration in `paradigm2.py` (optional cache mixture in `compress`/`decompress`/container APIs) plus `ContainerResult`/`DecompressResult` wrappers; flag plumbing in `container.py`/`conditioning.py`.
- New CLI subcommands (`steer`, `profile`/`surprisal`, `circuits --folded`, `--cache-mix`), worker task handlers, mock test doubles, tests, docs, and a `0.3.0 → 0.4.0` version bump.

| File | Description |
| ---- | ----------- |
| src/​unplow/​steering.py | New folded-circuit lens + activation-steering engine (core new logic). |
| src/​unplow/​cache_coder.py | New online prefix-cache mixture CDF + surprisal profiler. |
| src/​unplow/​paradigm2.py | Wires cache mixture into codec; adds bytecode-introspecting result wrappers. |
| src/​unplow/​container.py | Adds `FLAG_CACHE_MIXTURE` flag + metadata properties. |
| src/​unplow/​conditioning.py | Extends `pack_flags_with_domain` with `cache_mixture` bit. |
| src/​unplow/​models.py | Adds `mock` load path + `DeterministicMock*`/`MockBatchEncoding` doubles. |
| src/​unplow/​cli.py | Adds `steer`/`profile` commands, `--folded`/`--cache-mix` flags. |
| src/​unplow/​worker.py | Adds `steer`/`profile` task handlers; `model_name` usage. |
| src/​unplow/​__init__.py | Exports new symbols; version bump to 0.4.0. |
| tests/​* | New tests for steering, cache coder, CLI help, worker tasks. |
| README.md, docs/​*.md, Makefile, pyproject.toml, uv.lock | Docs, targets, metadata, and version updates. |
| inbox/​gemweb004.md, inbox/​gemweb005.md | Received Phase 4 specification dumps. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #3: feat(phase4): activation steering, folded circuits, and cache mixture arithmetic coding
- **URL**: https://github.com/aomaker-org/unplow/pull/3
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T00:24:56Z
- **File**: `src/unplow/paradigm2.py` (Line: 391)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/3#discussion_r4300472902

### Content
```markdown
The `cache_weight` used at compression time is never persisted in the `.unp` container. `compress_to_container` forwards `cache_weight` to `compress` (which feeds it into `OnlinePrefixCacheCDF`), but `pack_unp_container` only records the single `FLAG_CACHE_MIXTURE` bit — not the weight. On the decode side, `decompress_from_container` calls `self.decompress(...)` without any `cache_weight`, so it always falls back to the default `0.08`.

As a result, calling `compress_to_container(..., enable_cache_mixture=True, cache_weight=X)` with any `X != 0.08` produces a container that `decompress_from_container` cannot losslessly reconstruct: the encoder and decoder build different mixed CDFs, the arithmetic decoder desyncs, and the output is corrupted (the checksum will report a mismatch, but the plaintext is already wrong). The existing roundtrip test only passes because both sides use the default weight.

Consider either persisting the weight (e.g., in reserved header bits / a quantized field) so the decoder can recover it, exposing a matching `cache_weight` on `decompress_from_container`, or treating `cache_weight` as a fixed protocol constant and removing it from the public container API.
```

---

## PR #3: feat(phase4): activation steering, folded circuits, and cache mixture arithmetic coding
- **URL**: https://github.com/aomaker-org/unplow/pull/3
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-17T00:24:56Z
- **File**: `src/unplow/paradigm2.py` (Line: 88)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/3#discussion_r4300472929

### Content
```markdown
This `__iter__` (and the sibling `DecompressResult.__iter__` below) decides how many values to yield by introspecting the caller's bytecode via `sys._getframe(1)`, `frame.f_lasti`, and scanning for an `UNPACK_SEQUENCE` opcode. This is a fragile anti-pattern:

- It depends on CPython-specific bytecode and frame internals (`f_lasti` alignment, `UNPACK_SEQUENCE` opname/argval). It is not guaranteed to hold on other interpreters (PyPy, etc.) or across future CPython bytecode changes, and it silently falls back to byte-iteration (yielding ints) in any context it doesn't recognize, which can mask bugs.
- `dis.get_instructions(frame.f_code)` disassembles the entire calling function on every unpack, an avoidable cost on hot paths.
- The behavior is inconsistent: `ContainerResult` handles arity 1/2/3 but `DecompressResult` handles only 2/3, so a 1-target unpack falls through to the 3-tuple iterator.

Since the goal is to let a single return value serve both as `bytes`/data and as a `(payload, meta, stats)` tuple, a clearer and more robust approach is an explicit container type (e.g., a `NamedTuple`/dataclass with named fields, or returning a fixed-arity tuple and having callers index into it) rather than caller-bytecode inspection.
```

---

## PR #2: feat(phase3): ingest gemweb003.md, coupled SwiGLU circuits, and zero-bit domain conditioning
- **URL**: https://github.com/aomaker-org/unplow/pull/2
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-16T23:52:29Z
- **File**: `src/unplow/worker.py` (Line: 226)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/2#discussion_r4300340674

### Content
```markdown
`harvest_associative_seeds` internally re-invokes `extract_coupled_circuits` with its default parameters (circuits.py:221-224), re-running the entire static-weight scan already performed at lines 216-221. This doubles the work for every circuits task, and the second scan also ignores the configured `candidates`/`top_tokens` (it uses the defaults 24/5). Consider deriving the harvested seeds from the already-computed `circuits` list instead of re-extracting.
```

---

## PR #2: feat(phase3): ingest gemweb003.md, coupled SwiGLU circuits, and zero-bit domain conditioning
- **URL**: https://github.com/aomaker-org/unplow/pull/2
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-16T23:52:29Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/2#pullrequestreview-5554858884

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

It alters the `.unp` binary header format and the lossless compress/decompress roundtrip plus adds numerically sensitive mechanistic code that cannot be fully verified in this offline environment, warranting human review.

<details open>
<summary><strong>3 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [--harvest flag is ignored when seeds default is nonzero](https://github.com/aomaker-org/unplow/pull/2#discussion_r4300340614) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Circuit decompressor uses the wrong device default](https://github.com/aomaker-org/unplow/pull/2#discussion_r4300340646) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Seed harvesting repeats the full circuit scan](https://github.com/aomaker-org/unplow/pull/2#discussion_r4300340674) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR ingests `inbox/gemweb003.md` and implements its "Phase 3" proposals on top of the existing `unplow` LLM compression engine. It adds static mechanistic circuit extraction, zero-bit domain conditioning for the lossless arithmetic codec, a faster PyTorch integer CDF, and an MDL-based domain auto-router, wiring all of this into the CLI and the async worker. The changes touch correctness-sensitive areas (the `.unp` binary header format and the lossless compress/decompress roundtrip), but domain data is packed into previously-unused `flags` nibble bits so existing archives remain compatible, and the roundtrip stays deterministic because compress and decompress derive the same preamble from the stored domain id.

**Changes:**
- New `circuits.py` (coupled SwiGLU key→value circuit extraction with anisotropic mean-centering and selectivity-margin filtering) and new `conditioning.py` (nibble packing, KV-cache pre-warming, fast int64 CDF, MDL auto-router).
- `container.py`/`paradigm2.py` extended to store a 4-bit domain id in `flags[4:8]` and pre-warm the KV-cache during compress/decompress; `models.py` mock upgraded to a SwiGLU MLP with a KV-cache-shaped interface.
- CLI `circuits` subcommand plus `--domain`/`--auto-domain` on `archive`, matching worker actions, docs, README, Makefile, and `0.3.0` version bumps.

| File | Description |
| ---- | ----------- |
| src/​unplow/​circuits.py | New SwiGLU circuit decompressor; extracts associative trigger→target rules from static weights. |
| src/​unplow/​conditioning.py | New domain-conditioning module: flag packing, KV pre-warm, fast CDF, MDL router. |
| src/​unplow/​container.py | Adds `domain_id` property and packs domain into the `flags` upper nibble. |
| src/​unplow/​paradigm2.py | Threads `domain_id` through compress/decompress/container; swaps in fast CDF. |
| src/​unplow/​models.py | Mock model gains gate/up/down SwiGLU MLP, embed_tokens, and KV-cache output. |
| src/​unplow/​worker.py | Adds `circuits` task and `domain_id` for archive; missing device propagation + redundant re-scan. |
| src/​unplow/​cli.py | Adds `circuits` subcommand and `--domain`/`--auto-domain`; harvest gating is always-on. |
| src/​unplow/​__init__.py | Re-exports new symbols; version bump to 0.3.0. |
| tests/​test_circuits.py, tests/​test_conditioning.py, tests/​test_cli.py, tests/​test_worker.py | Unit/integration coverage for the new modules and subcommands. |
| README.md, docs/​architecture.md, docs/​decompression_guide.md, Makefile | Document and expose the new Phase 3 capabilities. |
| inbox/​gemweb003.md | Ingested source specification driving the Phase 3 work. |
| pyproject.toml, uv.lock | Version bump to 0.3.0. |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #2: feat(phase3): ingest gemweb003.md, coupled SwiGLU circuits, and zero-bit domain conditioning
- **URL**: https://github.com/aomaker-org/unplow/pull/2
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-16T23:52:28Z
- **File**: `src/unplow/cli.py` (Line: 258)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/2#discussion_r4300340614

### Content
```markdown
The `--harvest` flag has no effect here: because `--seeds` defaults to 12 (line 89), `getattr(args, "seeds", 0) > 0` is always true, so this block runs on every invocation regardless of `--harvest`. This is inconsistent with the `inspect` command, which gates purely on `if args.harvest:` (line 228). Beyond making the flag misleading, it forces `harvest_associative_seeds` to re-run the full `extract_coupled_circuits` scan already executed on line 241 (see circuits.py:221-224), doubling the static-weight analysis every time. Consider gating on `args.harvest` to honor the flag, or derive the seeds from the already-computed `circuits` list to avoid the second scan. Note the README example (`circuits ... --seeds 4`) implies seeds should print without `--harvest`, so clarify the intended gating before changing it.
```

---

## PR #2: feat(phase3): ingest gemweb003.md, coupled SwiGLU circuits, and zero-bit domain conditioning
- **URL**: https://github.com/aomaker-org/unplow/pull/2
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-16T23:52:28Z
- **File**: `src/unplow/worker.py` (Line: 211)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/2#discussion_r4300340646

### Content
```markdown
The circuits decompressor is instantiated without `device=self.device`, unlike the Paradigm 1/2 codecs which both propagate the worker's configured device (lines 66 and 72). `SwiGLUCircuitDecompressor` otherwise defaults to `cuda` when available, which can diverge from the device the worker was configured to use. Pass `device=self.device` for consistency.
```

---

## PR #1: feat(phase2): ingest gemweb002.md, mechanistic logit lens, and .unp container
- **URL**: https://github.com/aomaker-org/unplow/pull/1
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-16T23:52:23Z
- **File**: `src/unplow/paradigm1.py` (Line: 134)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/1#discussion_r4300340198

### Content
```markdown
The same default seed-prefix list is now duplicated twice in this block (once as the fallback when mechanistic harvesting returns nothing, and again in the `else` branch). This makes future edits error-prone since the two copies can drift apart. Consider defining the defaults once and referencing them in both branches.
```

---

## PR #1: feat(phase2): ingest gemweb002.md, mechanistic logit lens, and .unp container
- **URL**: https://github.com/aomaker-org/unplow/pull/1
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-16T23:52:23Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/1#pullrequestreview-5554858214

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🔵 Needs a closer look

The changes center on a correctness-sensitive lossless arithmetic codec with new KV-cache truncation and direct manipulation of real model internals whose end-to-end behavior could not be fully verified here, warranting human review.

<details open>
<summary><strong>3 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/medium-v2-light.png" alt="Medium severity" width="62" height="18" align="texttop"></picture> [Missing worker test coverage for archive/unarchive actions](https://github.com/aomaker-org/unplow/pull/1#discussion_r4300340142) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Package version mismatch between pyproject and __version__](https://github.com/aomaker-org/unplow/pull/1#discussion_r4300340174) · New
- <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)" srcset="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.svg"><img src="https://github.githubassets.com/static/images/icons/copilot-code-review/low-v2-light.png" alt="Low severity" width="62" height="18" align="texttop"></picture> [Duplicate default seed-prefix list in both branches](https://github.com/aomaker-org/unplow/pull/1#discussion_r4300340198) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR implements "Phase 2" of the `unplow` LLM-as-compressor engine by ingesting the `inbox/gemweb002.md` spec and translating it into working modules. It adds (1) a mechanistic "logit lens" inspector that decodes memorized vocabulary clusters directly from static MLP `down_proj` weights, (2) a self-verifying `.unp` binary container format with a bounded-memory sliding KV-cache horizon, and (3) CLI/worker subcommands (`inspect`, `archive`, `unarchive`) plus supporting docs and tests. It fits into the existing dual-paradigm architecture (Paradigm 1 weight basin probing, Paradigm 2 arithmetic coding) by extending both paradigms and wiring the new capabilities through the CLI and async worker.

**Changes:**
- New `mechanistic.py` (logit-lens MLP inspection + seed harvesting) and `container.py` (`.unp` framing, checksum verification, `slice_kv_cache`).
- `paradigm2.py` gains `max_ctx_len` sliding-window support and `compress_to_container`/`decompress_from_container`; `paradigm1.py` can seed from mechanistic harvesting; `models.py` mock model grows transformer-layer structure for inspection.
- CLI, worker, Makefile, docs, `__init__`, and tests updated for `inspect`/`archive`/`unarchive`.

| File | Description |
| ---- | ----------- |
| src/​unplow/​container.py | New `.unp` header pack/unpack, checksum verify, and KV-cache slicing helper |
| src/​unplow/​mechanistic.py | New logit-lens weight inspector and seed harvester |
| src/​unplow/​paradigm2.py | Sliding-window `max_ctx_len` and container compress/decompress methods |
| src/​unplow/​paradigm1.py | Optional mechanistic seed harvesting (introduces duplicated default list) |
| src/​unplow/​models.py | Mock model extended with MLP/layer/norm structure for inspection |
| src/​unplow/​cli.py | Adds `inspect`, `archive`, `unarchive` subcommands and `--max-ctx-len` |
| src/​unplow/​worker.py | Adds `archive`/`unarchive` task dispatch (untested) |
| src/​unplow/​__init__.py | Re-exports new symbols; bumps `__version__` to 0.2.0 (diverges from pyproject) |
| tests/​test_container.py, test_mechanistic.py, test_sliding_window.py, test_cli.py | New/updated tests for container, inspector, sliding window, CLI help |
| docs/​architecture.md, docs/​decompression_guide.md, Makefile, inbox/​gemweb002.md | Documentation, targets, and ingested spec |
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #1: feat(phase2): ingest gemweb002.md, mechanistic logit lens, and .unp container
- **URL**: https://github.com/aomaker-org/unplow/pull/1
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-16T23:52:22Z
- **File**: `src/unplow/worker.py` (Line: 224)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/1#discussion_r4300340142

### Content
```markdown
The new `archive`/`unarchive` worker actions add non-trivial glue (base64 round-tripping, work-product paths, `complete_task`/`processed_count` bookkeeping) but are not covered by any test, whereas the sibling `probe`, `compress`, and `decompress` actions are exercised in `tests/test_worker.py` (e.g. `test_worker_compress_and_decompress`). Consider adding a worker-level test that enqueues an `archive` task and then an `unarchive` task to guard this dispatch path.
```

---

## PR #1: feat(phase2): ingest gemweb002.md, mechanistic logit lens, and .unp container
- **URL**: https://github.com/aomaker-org/unplow/pull/1
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-16T23:52:22Z
- **File**: `src/unplow/__init__.py` (Line: 75)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/1#discussion_r4300340174

### Content
```markdown
The package version is bumped to `0.2.0` here, but `pyproject.toml` still declares `version = "0.1.0"` (the source of truth for the hatchling build). As a result the installed distribution metadata (`pip show unplow`) will report `0.1.0` while `unplow.__version__` reports `0.2.0`. Please bump `pyproject.toml` to match (or derive `__version__` from the package metadata) so the two stay in sync.
```

---

<!-- end of file: copilot-pr-comments.md -->
