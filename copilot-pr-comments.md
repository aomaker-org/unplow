# Path: copilot-pr-comments.md
# copilot-pr-comments.md - Extracted GitHub Copilot PR Review Comments

# Copilot Pull Request Review Comments & Findings

Total Extracted Copilot Artifacts: 15

## PR #30: fix(refactor): address Copilot review feedback on PR #20-#29
- **URL**: https://github.com/aomaker-org/unplow/pull/30
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T19:26:25Z
- **File**: `tools/review_copilot_comments.py` (Line: 31)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/30#discussion_r4405123299

### Content
```markdown
These `elif` conditions are 137 and 131 columns, over the 120-char code limit
declared in this file's header (`# Max Line Length: 120 Chars`) and AI.md
Directive 04. Collapsing each chained `or` into an `any(...)` membership test
keeps them within the limit.
```

---

## PR #30: fix(refactor): address Copilot review feedback on PR #20-#29
- **URL**: https://github.com/aomaker-org/unplow/pull/30
- **Type**: `review_summary`
- **User**: `copilot-pull-request-reviewer[bot]`
- **Created At**: 2026-10-28T19:26:25Z
- **Review State**: `COMMENTED`
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/30#pullrequestreview-5688483194

### Content
```markdown
<!-- ccr-overview-v2 -->

### 🟡 Changes recommended

A critical unresolved finding remains, as the returned `mask_fn` is still a no-
op that masks no transitions, contradicting its docstring and the PR's
functionality claim.

<details open>
<summary><strong>5 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/high-v2-light.svg"><img
src="https://github.githubassets.com/static/images/icons/copilot-code-
review/high-v2-light.png" alt="High severity" width="62" height="18"
align="texttop"></picture> [Mask function never masks invalid
transitions](https://github.com/aomaker-
org/unplow/pull/30#discussion_r4405123091) · New
- <picture><source media="(prefers-color-scheme: dark)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.svg"><img
src="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.png" alt="Low severity" width="62" height="18"
align="texttop"></picture> [Generated report violates declared 80-column
limit](https://github.com/aomaker-org/unplow/pull/30#discussion_r4405123167) ·
New
- <picture><source media="(prefers-color-scheme: dark)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.svg"><img
src="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.png" alt="Low severity" width="62" height="18"
align="texttop"></picture> [Generated synthesis breaks its 80-column
limit](https://github.com/aomaker-org/unplow/pull/30#discussion_r4405123215) ·
New
- <picture><source media="(prefers-color-scheme: dark)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.svg"><img
src="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.png" alt="Low severity" width="62" height="18"
align="texttop"></picture> [Remove unused os import](https://github.com/aomaker-
org/unplow/pull/30#discussion_r4405123255) · New
- <picture><source media="(prefers-color-scheme: dark)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.svg"><img
src="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.png" alt="Low severity" width="62" height="18"
align="texttop"></picture> [Chained OR conditions exceed the line-length
limit](https://github.com/aomaker-org/unplow/pull/30#discussion_r4405123299) ·
New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR applies a batch of fixes responding to earlier Copilot review feedback
spanning PRs #20–#29. The bulk of the change is mechanical prose reflow of
ingested `inbox/gemweb10x.md` notes and documentation to the repo's 80-column
convention (Directive 04), plus a handful of small code fixes (socket cleanup,
path-traversal hardening, training-state restoration, a grammar-mask hook, and a
documented formula correction). It also adds two helper scripts and two large
generated report files that captured the Copilot feedback being addressed.

**Changes:**
- Reflows prose in `inbox/gemweb101–106.md`,
`docs/extraction_coverage_audit.md`, and corrects the Quality Score formula + a
parser comment to match `src/unplow/replow.py`.
- Hardens runtime code: socket context manager in `get_lan_ip_addresses()`,
strict path containment on `/api/timelapse/frames`, and a `try/finally`
training-state restore in `compare_models_on_test_prompts()`.
- Adds `tools/extract_copilot_comments.py` / `tools/review_copilot_comments.py`
and the generated `copilot-pr-comments.md` / `copilot-pr-review.md`, plus a
`build_mask_fn()` body in `grammars/__init__.py`.

| File | Description |
| ---- | ----------- |
| src/​unplow/​grammars/​__init__.py | `build_mask_fn` now returns a callable,
but the callable is still a no-op (both branches return `logits` unchanged);
`keywords` unused — flagged |
| src/​unplow/​web.py | Socket wrapped in context manager;
`/api/timelapse/frames` basename + containment check — correct |
| src/​unplow/​incremental_pipeline.py | `try/finally` restores
`self.model.train()` after comparison — correct |
| tools/​extract_copilot_comments.py | New extractor; unused `os` import and a
136-col line flagged |
| tools/​review_copilot_comments.py | New categorizer; two `elif` lines exceed
the declared 120-col limit |
| docs/​input_model_extraction_method.md | Quality Score formula reconciled with
`compute_code_metrics()` |
| docs/​extraction_coverage_audit.md | Prose/table rows reflowed within changed
hunks |
| inbox/​gemweb101–106.md | Prose reflowed to 80 columns; one parser comment
corrected |
| copilot-pr-comments.md | Generated dump; declares 80-col header but 644 lines
exceed it — flagged |
| copilot-pr-review.md | Generated synthesis; declares 80-col header but 73
lines exceed it — flagged |

Key finding: `build_mask_fn()` is presented as "functional" in the PR
description, but the hook it returns performs no masking, so the grammar-
constraint feature still has no runtime effect. The remaining comments cover
unused imports and Directive 04 column-limit violations (including the two new
generated files that declare an 80-column header yet exceed it substantially).
</details>

<sub>Review effort: Balanced</sub>
```

---

## PR #30: fix(refactor): address Copilot review feedback on PR #20-#29
- **URL**: https://github.com/aomaker-org/unplow/pull/30
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T19:26:24Z
- **File**: `copilot-pr-comments.md` (Line: 3)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/30#discussion_r4405123167

### Content
```markdown
This generated file declares `# Max Column: 80 Columns`, but 644 of its lines
exceed 80 columns (up to 862). That contradicts both this header and the PR's
stated goal of making files "strictly obey the 80-column limit (Directive 04)."
Since the body is emitted by `generate_markdown_report`, either wrap the content
on generation, correct/drop the 80-column header, or avoid committing this
generated artifact.
```

---

## PR #30: fix(refactor): address Copilot review feedback on PR #20-#29
- **URL**: https://github.com/aomaker-org/unplow/pull/30
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T19:26:24Z
- **File**: `copilot-pr-review.md` (Line: 3)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/30#discussion_r4405123215

### Content
```markdown
This generated synthesis declares `# Max Column: 80 Columns`, but 73 of its
lines exceed 80 columns (up to 316), because the per-issue body is truncated to
300 chars and emitted as a single line in `generate_review_report`. This
violates the declared header and the PR's stated 80-column goal. Consider
wrapping the emitted body lines, or dropping the 80-column header on this
generated artifact.
```

---

## PR #30: fix(refactor): address Copilot review feedback on PR #20-#29
- **URL**: https://github.com/aomaker-org/unplow/pull/30
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T19:26:24Z
- **File**: `tools/extract_copilot_comments.py` (Line: 8)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/30#discussion_r4405123255

### Content
```markdown
`os` is imported but never used anywhere in this module. Removing it keeps the
import list clean — the exact "unused import" class of finding this tooling is
meant to surface.

This issue also appears on line 20 of the same file.
```

---

## PR #30: fix(refactor): address Copilot review feedback on PR #20-#29
- **URL**: https://github.com/aomaker-org/unplow/pull/30
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T19:26:23Z
- **File**: `src/unplow/grammars/__init__.py` (Line: 50)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/30#discussion_r4405123091

### Content
```markdown
The returned `mask_fn` is still a no-op: both branches of the inner function
`return logits` unchanged, so no transitions are ever masked to -inf.
Consequently the `keywords` set built on line 43 is never used, the loaded
`self.grammars` have no runtime effect, and the behavior contradicts both the
docstring ("masks invalid transitions to -inf ... using domain grammar rules")
and the PR description's claim that `build_mask_fn()` is now "functional."
Either implement the actual grammar/vocab-based masking (bias disallowed tokens
to -inf using `keywords` and the tokenizer) or clearly mark this as a non-
functional stub so callers are not misled.
```

---

## PR #29: docs(inbox): ingest gemweb102-106 execution distillation, exercism harnesses, and DPO
- **URL**: https://github.com/aomaker-org/unplow/pull/29
- **Type**: `inline_comment`
- **User**: `Copilot`
- **Created At**: 2026-10-28T18:49:16Z
- **File**: `inbox/gemweb102.md` (Line: 5)
- **Comment URL**: https://github.com/aomaker-org/unplow/pull/29#discussion_r4404732178

### Content
```markdown
The PR description states these files "Enforce strict max 80-column formatting,"
and each file's own header declares `# Max Column: 80 Columns` (line 3).
However, the ingested content is largely unwrapped and massively exceeds 80
columns. Measured maximum line widths are: gemweb102.md 448 cols (this line),
gemweb103.md 260, gemweb104.md 216, gemweb105.md 305, gemweb106.md 326, with
dozens of prose lines in the 120-270 col range across the five files. This
contradicts both the per-file header and the PR description; even the "code max
120" allowance from AI.md Directive 04 does not cover these prose lines. Either
reflow the prose to 80 columns or correct the header/description so the stated
convention matches the actual content.
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

An unresolved finding shows the ingested prose far exceeds the 80-column limit
asserted by both the file headers and PR description, leaving a contradiction
that must be corrected before approval.

<details open>
<summary><strong>1 open issue</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.svg"><img
src="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.png" alt="Low severity" width="62" height="18"
align="texttop"></picture> [Prose exceeds 80-column limit despite file
headers](https://github.com/aomaker-org/unplow/pull/29#discussion_r4404732178) ·
New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR ingests five new documentation files into `inbox/`
(`gemweb102.md`–`gemweb106.md`), continuing the repository's practice of
recording externally-sourced ("gemweb") research notes as plain markdown. The
content surveys and specifies a prompt-driven, execution-verified distillation
pipeline for training compact systems-programming models: architecture overview,
benchmark/problem-set survey, an Exercism C extraction harness, an async batch
distillation runner, and a DPO triplet-generation stage. These are
reference/knowledge documents rather than wired-in package code; no `src/`
modules, tests, or indexes are modified.

**Changes:**
- Adds a closed-loop execution-distillation architecture note (gemweb102) and a
systems-programming benchmark survey (gemweb103).
- Adds illustrative Python harnesses: Exercism C extraction to JSON (gemweb104),
an async inference + ProcessPool verification runner (gemweb105), and DPO
`(prompt, chosen, rejected)` generation (gemweb106).
- All files carry the standard `# Path:` header and `<!-- end of file: ... -->`
footer, but their content is not wrapped to the declared 80-column limit (see
finding).

| File | Description |
| ---- | ----------- |
| inbox/​gemweb102.md | Execution-distillation pipeline overview, problem-bank
schema, teacher-harvest/student-train/DPO snippets. |
| inbox/​gemweb103.md | Survey of execution-verified benchmarks and systems-
specific test-suite sources. |
| inbox/​gemweb104.md | Exercism C track extraction script parsing
instructions/headers/Unity harnesses into a JSON manifest. |
| inbox/​gemweb105.md | Async batch runner decoupling OpenAI-compatible
inference from a `gcc` ProcessPoolExecutor sandbox. |
| inbox/​gemweb106.md | DPO triplet generation prioritizing runtime assertion
failures over compile errors, plus a TRL training script. |

Notes: Header/footer conventions are correctly applied across all five files,
and no inbox index/manifest needed updating. The one recorded finding is that
the PR description's and per-file headers' "strict max 80-column" claim does not
match the ingested content, which has many unwrapped lines (up to 448 columns).
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
`build_mask_fn` unconditionally returns `None`, so the core grammar-masking
feature introduced by this PR never actually runs. Because `mask_fn` is always
`None`, the `grammar_mask_fn` hook added in `probe_weight_basins` is skipped,
the loaded `self.grammars` are never consulted, and the
`c_systems.ebnf`/`rust_systems.ebnf` definitions have no effect. The docstring
also states this method "Creates a callable `mask_fn(input_ids, logits)` that
masks invalid transitions to -inf", which does not match the `return None`
behavior. Consider implementing the masking logic (or clearly marking this as a
non-functional scaffold / NotImplemented stub) so the feature behaves as
described.
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
The PR description states the grammar engine is connected to
`extract_and_stage_from_model()`, but no such method exists in this file — the
integration is actually in `extract_domain_training_data()`. Please align the PR
description with the actual method name to avoid confusion for future readers.
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

A critical unresolved issue remains: `build_mask_fn` always returns `None`,
leaving the PR's core grammar-masking feature non-functional despite its
documented behavior.

<details open>
<summary><strong>2 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/high-v2-dark.svg"><source media="(prefers-color-scheme: light)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/high-v2-light.svg"><img
src="https://github.githubassets.com/static/images/icons/copilot-code-
review/high-v2-light.png" alt="High severity" width="62" height="18"
align="texttop"></picture> [build_mask_fn is a no-op, so grammar masking never
activates](https://github.com/aomaker-org/unplow/pull/28#discussion_r4404633613)
· New
- <picture><source media="(prefers-color-scheme: dark)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.svg"><img
src="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.png" alt="Low severity" width="62" height="18"
align="texttop"></picture> [PR description cites the wrong extraction hook
name](https://github.com/aomaker-org/unplow/pull/28#discussion_r4404633695) ·
New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR scaffolds a Context-Free Grammar (CFG) constraint system for the weight-
extraction pipeline. It introduces a new `unplow.grammars` package containing a
`DomainGrammarEngine` plus scoped EBNF grammars for C and Rust, and wires a
`grammar_mask_fn` hook into `WeightDecompressor.probe_weight_basins()` so logits
can (in principle) be masked to `-inf` for illegal grammar transitions before
temperature scaling and Top-K filtering. The engine is instantiated inside
`extract_domain_training_data()` in the incremental pipeline.

The most important caveat: `DomainGrammarEngine.build_mask_fn()` currently
returns `None` unconditionally, so the masking hook is never actually exercised
— the loaded grammars and `.ebnf` files have no runtime effect yet. The hook
plumbing in `paradigm1.py` is correct and safely no-ops when `mask_fn is None`,
so nothing breaks, but the feature is effectively a scaffold rather than the
functional masking the description implies.

**Changes:**
- Added `unplow.grammars` package: `DomainGrammarEngine` (loads `*.ebnf` by
domain) plus `c_systems.ebnf` and `rust_systems.ebnf`.
- Added an optional `grammar_mask_fn(input_ids, logits)` hook to
`probe_weight_basins()`, applied per step before temperature/Top-K.
- Connected the engine into `extract_domain_training_data()` in
`incremental_pipeline.py`.

| File | Description |
| ---- | ----------- |
| `src/​unplow/​grammars/​__init__.py` | New `DomainGrammarEngine`; loads EBNF
grammars, but `build_mask_fn` is a stub returning `None`. |
| `src/​unplow/​grammars/​c_systems.ebnf` | New C systems-subset EBNF grammar
(currently unused at runtime). |
| `src/​unplow/​grammars/​rust_systems.ebnf` | New Rust systems-subset EBNF
grammar (currently unused at runtime). |
| `src/​unplow/​paradigm1.py` | Adds optional `grammar_mask_fn` hook applied to
per-step logits; no-ops when `None`. |
| `src/​unplow/​incremental_pipeline.py` | Imports and instantiates
`DomainGrammarEngine`, passing its (currently `None`) mask fn into probing. |

Key review notes recorded as comments:
1. `build_mask_fn` always returns `None`, so the grammar-masking feature — the
stated purpose of the PR — never runs, and its docstring overstates its
behavior.
2. The PR description references `extract_and_stage_from_model()`, which does
not exist; the actual integration point is `extract_domain_training_data()`.
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

Unresolved active findings remain, including a misleading comment describing
parser behavior the code does not actually perform and an EBNF line exceeding
the mandated column limit.

<details open>
<summary><strong>2 open issues</strong></summary>

- <picture><source media="(prefers-color-scheme: dark)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.svg"><img
src="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.png" alt="Low severity" width="62" height="18"
align="texttop"></picture> [Wrap overlong EBNF rule to satisfy 120-character
limit](https://github.com/aomaker-org/unplow/pull/27#discussion_r4404596567) ·
New
- <picture><source media="(prefers-color-scheme: dark)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-dark.svg"><source media="(prefers-color-scheme: light)"
srcset="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.svg"><img
src="https://github.githubassets.com/static/images/icons/copilot-code-
review/low-v2-light.png" alt="Low severity" width="62" height="18"
align="texttop"></picture> [Comment claims initial_code is consumed but code
never uses it](https://github.com/aomaker-
org/unplow/pull/27#discussion_r4404596652) · New
</details>

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR ingests a new reference document, `inbox/gemweb101.md`, into the
`inbox/` staging area. The doc provides scoped "systems-subset" EBNF grammars
for C and Rust intended to constrain autoregressive decoding (logit masking) so
generated routines stay syntactically valid, and sketches a
`DomainGrammarEngine` loader pattern (using `outlines`' `CFGGuide`) to be wired
into `paradigm1.py`. It is purely additive documentation and follows the repo's
`# Path:` header / `<!-- end of file: -->` footer convention, consistent with
the sibling `gemweb100.md`.

**Changes:**
- Adds `c_systems.ebnf` and `rust_systems.ebnf` grammar specs covering low-level
types, structs, pointers/borrowing, loops, bitwise logic, and lifetimes.
- Documents a Python `DomainGrammarEngine.build_guide` loading pattern for
grammar-constrained generation.
- Applies the standard path header/footer and 80-column prose formatting.

| File | Description |
| ---- | ----------- |
| inbox/​gemweb101.md | New ingested doc with C/Rust EBNF grammars and a
`CFGGuide` loader example; one EBNF line exceeds the 120-char code limit, and
the loader's inline comment claims `initial_code` is consumed when it is unused.
|

Notes for the author (non-blocking): the `build_guide` comment describes
shifting across `initial_code`, but that parameter is never passed to `CFGGuide`
(same pattern as `gemweb100.md`); and line 57 is 124 columns, over the Directive
04 code limit. Both are flagged inline with suggested fixes.
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
This EBNF line is 124 columns, exceeding the 120-char limit for code mandated by
AI.md Directive 04 (summarized in AGENTS.md). Lark rule definitions can continue
across lines, so wrapping the alternation keeps it within the limit.
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
The comment states the parser "immediately shifts across the tokens in
`initial_code`" to guarantee subsequent predictions complete the AST, but
`initial_code` is never used in this method: `CFGGuide(grammar_str,
self.tokenizer)` is constructed only from the grammar and tokenizer. As written,
the comment describes behavior the code does not perform, which is misleading
for readers adopting this pattern. Either reword the comment to reflect that
`initial_code` is not yet consumed, or pre-shift the guide across `initial_code`
as described.
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

The review surfaced no confident, objective defects or unresolved issues in this
documentation-only reference note, leaving no blocking concerns outstanding.

**No open issues** 🎉

<details>
<summary><strong>What changed in this PR</strong></summary>

This PR adds a single ingested reference note, `inbox/gemweb100.md`, to the
project's `inbox/` collection of Gemini-web-sourced design memos
(`gemweb001`–`gemweb016`). The document describes a proposed technique for
integrating Context-Free Grammar (CFG) logit masking into the autoregressive
decoding loop of `WeightDecompressor.probe_weight_basins` in
`src/unplow/paradigm1.py`, turning unconstrained generation into a constrained
pushdown automaton (PDA) so that invalid token transitions are masked to `-∞`
before temperature scaling and Top-K filtering. It is documentation only — no
source code is modified in this PR; the referenced `paradigm1.py` change is
presented as a future refactor proposal.

**Changes:**
- Adds `inbox/gemweb100.md`: a reference memo covering a token-to-grammar map
(via `outlines`/Lark/Tree-sitter or a zero-dependency parser), a proposed
refactor of the `probe_weight_basins` decode loop, per-language EBNF grammar
loading, and operational benefits.
- Follows the repository's file conventions: `# Path:` header, `<!-- end of
file: ... -->` footer, and prose kept within 80 columns.

| File | Description |
| ---- | ----------- |
| `inbox/​gemweb100.md` | New ingested reference note proposing CFG logit
masking in the parameter-basin decode loop; conforms to header/footer and column
conventions. |

I reviewed the document for convention adherence (header/footer format verified
against sibling inbox files, prose within 80 columns), internal consistency,
spelling/grammar, and technical accuracy of the described decoding pipeline. The
parameter differences between the memo's proposed `probe_weight_basins`
signature and the current implementation are intentional (it is an explicit
refactor proposal), so they are not flagged as a discrepancy. I did not find any
confident, objective defects warranting inline comments on this reference
material.
</details>

<sub>Review effort: Balanced</sub>
```

---

<!-- end of file: copilot-pr-comments.md -->
