# Path: copilot-pr-review.md
# copilot-pr-review.md - Comprehensive Review and Actionable Refactoring Plan from Copilot Feedback
# Max Column: 80 Columns

# Copilot Pull Request Review & Refactoring Synthesis

This document synthesizes and prioritizes 72 inline review comments and 29
review summaries extracted across all repository PRs.

## Summary of Findings by Category

- **General Logic & Design Hygiene**: 33 actionable findings
- **Documentation & Implementation Discrepancies**: 11 actionable findings
- **Dead Code, Unused Imports & Stubs**: 10 actionable findings
- **Test Coverage & Fragility**: 10 actionable findings
- **Formatting & Column Limits (Directive 04)**: 3 actionable findings
- **Resource Management, State Leaks & Performance**: 3 actionable findings
- **Security, Path Traversal & Isolation**: 2 actionable findings

## General Logic & Design Hygiene

### [PR #28] `src/unplow/incremental_pipeline.py` (Line 275)
- **Issue**: The PR description states the grammar engine is connected to `extract_and_stage_from_model()`, but no such method exists in this file — the integration is actually in `extract_domain_training_data()`. Please align the PR description with the actual method name to avoid confusion for future readers.

### [PR #22] `src/unplow/web.py` (Line 1468)
- **Issue**: The new `/api/timelapse/frames` endpoint builds the target path directly from the user-supplied `filename` query parameter (`out_dir / filename`) without stripping directory components. A value containing `../` (or an absolute path, which `pathlib` resolves by discarding `out_dir`) escapes `workspac...

### [PR #21] `docs/architecture.md` (Line 393)
- **Issue**: APNG is not defined by RFC 7689. RFC 7689 is "Signaling Extensions for Wavelength Switched Optical Networks" and has nothing to do with images; APNG is not an IETF RFC at all (base PNG is RFC 2083). This mislabel also appears in the `encode_apng_rgb` docstring and the PR title. Recommend dropping th...

### [PR #21] `src/unplow/visualizer.py` (Line 603)
- **Issue**: APNG is not defined by RFC 7689. RFC 7689 is "Signaling Extensions for Wavelength Switched Optical Networks" and is unrelated to image formats; in fact APNG is not specified by any IETF RFC (base PNG is RFC 2083). This incorrect citation is repeated in the PR title, `docs/architecture.md` section 18...

### [PR #20] `src/unplow/cli.py` (Line 494)
- **Issue**: The `--host` default was also changed to `0.0.0.0`, which binds the CLI-launched server to all interfaces with no authentication. For the same reasons as the module default (and to respect the loopback-only air-gap mandate in `AI.md` Directive 18), keep the default on loopback and let operators opt ...

### [PR #20] `src/unplow/web.py` (Line 1065)
- **Issue**: The fallback for `active_iteration` was silently changed from `6` to `8`. This is an unexplained magic number, and the change is unrelated to the headless/remote-access scope described in the PR. It alters what the UI displays when no iteration is found in the logs. Consider reverting this to keep t...

### [PR #20] `src/unplow/web.py` (Line 558)
- **Issue**: This sentence hardcodes the hostname `fg26`, but this element has no `id` and is never updated by the status-fetch JavaScript (unlike `server-host-badge`, `ssh-cmd-text`, and the LAN links). On any machine whose hostname is not `fg26`, the modal will display a stale, incorrect hostname, which underm...

### [PR #19] `src/unplow/web.py` (Line 138)
- **Issue**: `output_filename` from the request body is used to build the output path without sanitization, so a value such as `../../evil.png` resolves outside `workspace/visualizations/` and lets a client write arbitrary files on the host. Since the server can be bound to a non-loopback interface via `--host`,...

### [PR #19] `src/unplow/web.py` (Line 22)
- **Issue**: Several of these imports are unused in this module: `os`, `sys`, `threading`, `time`, and the `Tuple`/`Union` typing names are never referenced. Removing them keeps the import list clean.

### [PR #18] `Makefile` (Line 135)
- **Issue**: The new `visualize` target is not registered in the `.PHONY` list at the top of the Makefile (lines 8-12), unlike every other recipe-only target in this file (e.g. `probe`, `incremental-pipeline`, `full-pipeline`). Without this, `make visualize` would be skipped if a file or directory named `visuali...

### [PR #16] `src/unplow/incremental_pipeline.py` (Line 255)
- **Issue**: These metrics (`avg_step_loss`, `loss_val`, `ppl_val`, and `elapsed` below) are only bound inside the `for epoch in range(1, total_epochs + 1)` loop. The new checkpoint-persistence block now reads them after the loop, so when `total_epochs <= 0` (reachable via `--epochs 0` or `EPOCHS=0`, which are n...

### [PR #16] `src/unplow/incremental_pipeline.py` (Line 370)
- **Issue**: The `--loop` default treats any non-empty value of `LOOP`/`CONTINUOUS` as truthy, so `LOOP=0` (or `CONTINUOUS=0`) enables continuous looping instead of disabling it. `bool(os.environ.get("LOOP"))` is `True` for the string `"0"`. Since the PR documents `LOOP=1` as the enabling value, a user setting `...

### [PR #16] `src/unplow/incremental_pipeline.py` (Line 383)
- **Issue**: The literal `%` in this help string breaks `--help`. argparse applies `%`-formatting to help strings, so `15.0%)` raises `ValueError: unsupported format character ')'` when the help is rendered (e.g. `python incremental_pipeline.py --help`). Escape it as `%%` so it renders as a literal percent sign.

### [PR #15] `src/unplow/incremental_pipeline.py` (Line 80)
- **Issue**: The loop variable `l` is an ambiguous single-character name (easily confused with `1`/`I`, flagged as PEP8 E741) and is the only such usage in the codebase. Additionally, the inner loop appends every non-empty line of a corpus file before the 20-line limit is checked at the outer level, so a single ...

### [PR #13] `src/unplow/cli.py` (Line 486)
- **Issue**: The `till-guard-offload` / `till_guard_offload` / `till-offload` / `till_offload` actions are added as valid `enqueue` choices here, but `worker.py`'s `process_task` has no dispatch branch for them (handlers stop at `till-train`/`till_train`). Every other action registered in this list has a matchin...

### [PR #13] `Makefile` (Line 12)
- **Issue**: The `till-offload` alias is defined as a make target on line 125 (and advertised via `make till-offload` in the help text) but, unlike its `till-guard-offload` sibling, it is not listed in `.PHONY`. Add `till-offload` to the `.PHONY` list so the alias still runs if a file named `till-offload` ever e...

### [PR #11] `src/unplow/replow.py` (Line 123)
- **Issue**: The inline comment and the PR description both state this guard is meant for large models "on CPU", but the condition only checks the parameter count and never checks the device. If the model is loaded on a GPU (e.g. CUDA available), the guard will still freeze the entire backbone and train only the...

### [PR #10] `src/unplow/till_guard.py` (Line 15)
- **Issue**: `import tomllib` will break the package on Python 3.10. `tomllib` was only added to the standard library in Python 3.11, but this project declares `requires-python = ">=3.10"` (pyproject.toml) and `uv.lock` still resolves a 3.10 environment (it carries `python_full_version < '3.11'` markers). Becaus...

### [PR #10] `src/unplow/till_guard.py` (Line 216)
- **Issue**: `till_guard.py` emits its diagnostics via `print()` (9 call sites across `enforce_space`), whereas every other library module under `src/unplow/` (e.g. `bundler.py`, `replow.py`, `worker.py`, `eval.py`) produces no direct `print()` output and relies on the module `logger` / CLI-side formatting. This...

### [PR #9] `src/unplow/bundler.py` (Line 146)
- **Issue**: The directory exclusion filter checks `p.parts`, which for an `rglob` result rooted at the resolved `root_dir` includes the absolute path prefix. As a result, if the source directory's absolute path contains any excluded name anywhere above the tree (e.g. `build`, `dist`, `venv`, `logs`, or `workspa...

### [PR #9] `docs/architecture.md` (Line 322)
- **Issue**: A blank line is added after the end-of-file footer. By convention in this repo the `<!-- end of file: ... -->` footer is the final line of the file (see README.md, AI.md, AGENTS.md). Please drop the trailing blank line(s) so the footer remains last.

### [PR #9] `docs/decompression_guide.md` (Line 371)
- **Issue**: A blank line is added after the end-of-file footer. By convention in this repo the `<!-- end of file: ... -->` footer is the final line of the file (see README.md, AI.md, AGENTS.md). Please drop the trailing blank line(s) so the footer remains last.

### [PR #8] `docs/architecture.md` (Line 299)
- **Issue**: This adds a trailing blank line after the `<!-- end of file: ... -->` footer, so the footer is no longer the final line of the file. Per the repo convention (Directive 04, mandatory file footers; e.g. `src/unplow/container.py` and `src/unplow/mechanistic.py` both end exactly on their footer line), t...

### [PR #8] `docs/decompression_guide.md` (Line 332)
- **Issue**: This adds another trailing blank line after the `<!-- end of file: ... -->` footer, leaving the footer two lines before the end of the file. Per the repo convention (Directive 04, mandatory file footers; e.g. `src/unplow/container.py` and `src/unplow/mechanistic.py` end exactly on their footer line)...

### [PR #8] `src/unplow/__init__.py` (Line 216)
- **Issue**: The `__all__` list follows a consistent two-section ordering: capitalized class/constant names sorted A–Z first, followed by lowercase function names sorted A–Z (starting at `"audit_system_models"`). These two lowercase function names are inserted in the middle of the capitalized section (between `"...

### [PR #7] `docs/decompression_guide.md` (Line 260)
- **Issue**: This CLI example will not run as written. There is no `--file` option on the `replow` subcommand; the input file is a positional argument (`p_replow.add_argument("file", nargs="?", ...)`), so `--file extracted_snippet.py` raises an "unrecognized arguments" error. Additionally, `--domain` is declared...

### [PR #7] `src/unplow/replow.py` (Line 5)
- **Issue**: This docstring (and the matching PR description / architecture docs) states that the module "Extracts raw code basins from model weights" via mechanistic steering/task-vectors as Stage 1, but `ReplowEngine` does not implement any weight extraction — it only refactors, stages, trains on, and compress...

### [PR #7] `src/unplow/replow.py` (Line 19)
- **Issue**: These three imports are unused in this module. `pack_unp_container`, `FLAG_BAYES_MIXER`, and `BayesianCTSMixerCDF` are never referenced—compression is delegated entirely to `LLMArithmeticCodec.compress_to_container(..., use_bayes_mixer=True)`. The unused CTS-mixer imports in particular can mislead r...

### [PR #3] `src/unplow/paradigm2.py` (Line 88)
- **Issue**: This `__iter__` (and the sibling `DecompressResult.__iter__` below) decides how many values to yield by introspecting the caller's bytecode via `sys._getframe(1)`, `frame.f_lasti`, and scanning for an `UNPACK_SEQUENCE` opcode. This is a fragile anti-pattern:  - It depends on CPython-specific bytecod...

### [PR #2] `src/unplow/worker.py` (Line 211)
- **Issue**: The circuits decompressor is instantiated without `device=self.device`, unlike the Paradigm 1/2 codecs which both propagate the worker's configured device (lines 66 and 72). `SwiGLUCircuitDecompressor` otherwise defaults to `cuda` when available, which can diverge from the device the worker was conf...

### [PR #2] `src/unplow/worker.py` (Line 226)
- **Issue**: `harvest_associative_seeds` internally re-invokes `extract_coupled_circuits` with its default parameters (circuits.py:221-224), re-running the entire static-weight scan already performed at lines 216-221. This doubles the work for every circuits task, and the second scan also ignores the configured ...

### [PR #1] `src/unplow/__init__.py` (Line 75)
- **Issue**: The package version is bumped to `0.2.0` here, but `pyproject.toml` still declares `version = "0.1.0"` (the source of truth for the hatchling build). As a result the installed distribution metadata (`pip show unplow`) will report `0.1.0` while `unplow.__version__` reports `0.2.0`. Please bump `pypro...

### [PR #1] `src/unplow/paradigm1.py` (Line 134)
- **Issue**: The same default seed-prefix list is now duplicated twice in this block (once as the fallback when mechanistic harvesting returns nothing, and again in the `else` branch). This makes future edits error-prone since the two copies can drift apart. Consider defining the defaults once and referencing th...

## Documentation & Implementation Discrepancies

### [PR #28] `src/unplow/grammars/__init__.py` (Line 37)
- **Issue**: `build_mask_fn` unconditionally returns `None`, so the core grammar-masking feature introduced by this PR never actually runs. Because `mask_fn` is always `None`, the `grammar_mask_fn` hook added in `probe_weight_basins` is skipped, the loaded `self.grammars` are never consulted, and the `c_systems....

### [PR #25] `docs/input_model_extraction_method.md` (Line 134)
- **Issue**: The documented Quality Score formula does not match the implementation in `compute_code_metrics()` (`src/unplow/replow.py:156-162`). The code computes `Q = 0.50·[syntax_valid] + 0.20·[num_lines ≥ 3] + min(0.30, 0.30·token_diversity)`. There is no `ast_nodes/20` node-density term — the 0.2 weight is ...

### [PR #23] `src/unplow/models.py` (Line 150)
- **Issue**: The `"parameters"` field now holds a layer count (e.g. `"4L"`) instead of a parameter count. The CLI (`cli.py` `models` command) renders this field as the model's size, so users will see something like "(safetensors_snapshot, 24L)" where they previously saw a parameter total ("1.54B"). Storing a lay...

### [PR #19] `src/unplow/web.py` (Line 901)
- **Issue**: The fallback `active_iter or 6` reports a hard-coded "Iteration #6" whenever no iteration is parsed from the logs (including when no pipeline has ever run). This surfaces a misleading, unexplained magic value in the UI telemetry (`Pipeline Iteration #6`). Prefer reporting the real value (0 / unknown...

### [PR #18] `src/unplow/cli.py` (Line 462)
- **Issue**: This help text is inaccurate. When `--tensor` is omitted, the default-selection logic (cli.py:1173, 1183) picks the first state-dict key containing both `q_proj` and `weight` — i.e. the earliest layer (typically layer 0), not "Layer 27". Update the help text to describe the actual behavior to avoid ...

### [PR #17] `src/unplow/incremental_pipeline.py` (Line 141)
- **Issue**: The `num_samples` parameter is passed through to `probe_weight_basins`, but only `records[0].text` is ever consumed and the method always returns a single-element list (`return [clean_code]`). As a result, callers passing `num_samples > 1` would silently get just one sample back while still paying t...

### [PR #14] `src/unplow/incremental_pipeline.py` (Line 159)
- **Issue**: This breach message hardcodes "<15%", but the threshold is configurable via `min_free_pct` (`self.guardian.min_free_pct`) and can differ from 15 (e.g. the tests use 0.01 and 99.99). When a non-default threshold is in effect, this log line is misleading. Consider interpolating the actual threshold. O...

### [PR #10] `src/unplow/till_guard.py` (Line 152)
- **Issue**: The manifest's `[manifest]` settings `target_free_percentage` and `rclone_destination` are parsed by `load_manifest()` but never applied: `TillDiskGuardian` always uses the hard-coded constructor defaults (`min_free_pct=15.0`, `gdrive_dest="gdrive:transfer/till-20261017/"`). As a result, editing tho...

### [PR #7] `docs/decompression_guide.md` (Line 283)
- **Issue**: This Python API example does not match the actual `run_pipeline` signature and would raise `TypeError`/`AttributeError` if executed. The keyword arguments are wrong: the method accepts `raw_code`, `output_unp_path`, `train_steps`, and `domain_id` (an int) — there is no `raw_source`, `output_path`, `...

### [PR #6] `src/unplow/auditor.py` (Line 22)
- **Issue**: The field `ffn_output_l2_norm` actually stores the L2 norm of the layer's `down_proj.weight` matrix (computed at line 133), which is a static, prompt-independent weight norm — not the L2 norm of the FFN output activation that the name implies. This is misleading to consumers of `to_dict()` and the e...

### [PR #2] `src/unplow/cli.py` (Line 258)
- **Issue**: The `--harvest` flag has no effect here: because `--seeds` defaults to 12 (line 89), `getattr(args, "seeds", 0) > 0` is always true, so this block runs on every invocation regardless of `--harvest`. This is inconsistent with the `inspect` command, which gates purely on `if args.harvest:` (line 228)....

## Dead Code, Unused Imports & Stubs

### [PR #27] `inbox/gemweb101.md` (Line 218)
- **Issue**: The comment states the parser "immediately shifts across the tokens in `initial_code`" to guarantee subsequent predictions complete the AST, but `initial_code` is never used in this method: `CFGGuide(grammar_str, self.tokenizer)` is constructed only from the grammar and tokenizer. As written, the co...

### [PR #21] `src/unplow/web.py` (Line 1105)
- **Issue**: Optional: `changePlaybackSpeed()` is a no-op (the speed dropdown in the toolbar has no effect), and the scrubber/prev/next controls (`onScrubVideo`, `stepFrame`) only update the "Frame X / Y" label without changing the displayed APNG frame, since an APNG in an `<img>` cannot be seeked or re-timed. T...

### [PR #21] `src/unplow/web.py` (Line 34)
- **Issue**: `compile_images_to_apng` and `encode_apng_rgb` are imported here but never used in `web.py` (only `render_checkpoint_timelapse` is called). These unused imports should be removed to avoid lint warnings and dead imports.

### [PR #19] `src/unplow/web.py` (Line 898)
- **Issue**: The `/api/status` handler recursively globs the entire working tree with `Path(".").glob("**/*.log")` and reads every matching log file in full on each request. Since the SPA polls this endpoint every 5 seconds (`setInterval(fetchStatus, 5000)`), this repeatedly walks large/ignored directories (e.g....

### [PR #18] `src/unplow/visualizer.py` (Line 99)
- **Issue**: The `colormap` parameter is accepted but never used — the pixel loop below always calls `colormap_inferno` (line 123) regardless of what is passed. A caller requesting, e.g., `colormap="diverging"` would be silently ignored, which is misleading. Either remove the unused parameter or branch on it to ...

### [PR #14] `src/unplow/incremental_pipeline.py` (Line 30)
- **Issue**: `load_local_model` is imported here but never used in this module (only `load_causal_lm` is referenced, at line 213 and in `main`). This appears to be a leftover from the spec's example snippet. Consider dropping it to avoid an unused import.

### [PR #9] `src/unplow/bundler.py` (Line 21)
- **Issue**: `BUNDLE_MAGIC` and `BUNDLE_VERSION` are defined but never used anywhere in the module (or exported). The bundle payload is wrapped in a standard `.unp` container with no bundle-specific magic or version byte, so there is no way to distinguish a workspace bundle from an ordinary single-text container...

### [PR #7] `src/unplow/replow.py` (Line 187)
- **Issue**: The first branch here is unreachable: `compress_to_container` always returns a `ContainerResult`, which is a `bytes` subclass exposing the payload directly and carrying `.meta`/`.stats` attributes (it has no `container_bytes` attribute, and it is not a `tuple`). As a result both the `hasattr(res, "c...

### [PR #6] `src/unplow/auditor.py` (Line 9)
- **Issue**: `math` and `Optional` are imported here but never used in this module: Shannon entropy is computed with `torch.log2` (not `math`), and no annotation in the file uses `Optional`. Removing these unused imports keeps the module clean and avoids lint warnings.

### [PR #5] `src/unplow/cli.py` (Line 33)
- **Issue**: `FLAG_BAYES_MIXER` (line 32) and `BayesianCTSMixerCDF` (line 33) are imported here but never used anywhere in `cli.py`. Additionally, the module-level import of `CrossSubstrateTaskVectorEngine` (line 40) is redundant because the `task-vector` branch re-imports it locally (line 659). Removing the unu...

## Test Coverage & Fragility

### [PR #21] `tests/test_web.py` (Line 185)
- **Issue**: `models/till-python-lora/` is git-ignored (`.gitignore`: `/models/till-*/`) and no fixture creates it in the server's working directory, so in a clean checkout/CI this endpoint falls through to the "no checkpoints" branch and returns HTTP 404 (`{"status":"error",...}`). `urllib.request.urlopen` rais...

### [PR #17] `tests/test_incremental_pipeline.py` (Line 195)
- **Issue**: This test exercises `extract_domain_training_data`, which writes to the hardcoded CWD-relative path `workspace/till_corpus/{domain}/` (see `incremental_pipeline.py:163`). Although the trainer is given `tmp_path`-based `base_checkpoint_dir`/`base_target_dir`, the extraction ignores those and writes i...

### [PR #13] `tests/test_full_pipeline.py` (Line 44)
- **Issue**: This assertion (and the pipeline it drives) writes to `workspace/targets/till_python.unp` relative to the current working directory rather than `tmp_path`, so the test leaves artifacts in the repository working tree and is not hermetic or order-independent. Other tests in this suite keep output unde...

### [PR #11] `src/unplow/replow.py` (Line 137)
- **Issue**: This new memory-safety guard (freeze backbone / isolate last layer / 4-parameter fallback) is not exercised by any test. The existing train_step tests in tests/test_replow.py use DeterministicMockCausalLM, which has far fewer than 10M parameters, so the `total_params > 10_000_000` branch and its sub...

### [PR #10] `src/unplow/till_guard.py` (Line 271)
- **Issue**: `init_till_staging` (and `train_till_domain` at line 377) invoke `enforce_space(...)` with the default `dry_run=False`, so when free space on `/` drops below the threshold this will actually run `rclone move` against the real offload targets declared in `workspace/till_manifest.toml` (`~/.cache/uv`,...

### [PR #8] `src/unplow/worker.py` (Line 485)
- **Issue**: This branch tests `task.action` directly, while every other branch in `process_task` dispatches on the normalized `action = task.action.lower()` (defined at the top of the method; see e.g. the `("replow", "replow_pipeline")` branch). As a result, an enqueued action such as `"Eval"` or `"REPLOW_EVAL"...

### [PR #5] `src/unplow/gguf_inspector.py` (Line 275)
- **Issue**: The new Q6_K block dequantizer (GGML type 14) has no unit test, whereas the other dequantization paths in this file are tested (F32/Q8_0 via `build_synthetic_gguf_v3_bytes`, and Q4_0 via a hand-built block passed directly to `dequantize_buffer` in `tests/test_gguf_inspector.py`). Because `build_synt...

### [PR #5] `src/unplow/models.py` (Line 282)
- **Issue**: Calling `torch.manual_seed(42)` inside the `MockCausalLM` constructor mutates the global PyTorch RNG as a side effect of model construction. Since `MockCausalLM`/`DeterministicMockCausalLM` is instantiated widely across the test suite (including via `load_causal_lm("mock")`), any test that construct...

### [PR #3] `src/unplow/paradigm2.py` (Line 391)
- **Issue**: The `cache_weight` used at compression time is never persisted in the `.unp` container. `compress_to_container` forwards `cache_weight` to `compress` (which feeds it into `OnlinePrefixCacheCDF`), but `pack_unp_container` only records the single `FLAG_CACHE_MIXTURE` bit — not the weight. On the decod...

### [PR #1] `src/unplow/worker.py` (Line 224)
- **Issue**: The new `archive`/`unarchive` worker actions add non-trivial glue (base64 round-tripping, work-product paths, `complete_task`/`processed_count` bookkeeping) but are not covered by any test, whereas the sibling `probe`, `compress`, and `decompress` actions are exercised in `tests/test_worker.py` (e.g...

## Formatting & Column Limits (Directive 04)

### [PR #29] `inbox/gemweb102.md` (Line 5)
- **Issue**: The PR description states these files "Enforce strict max 80-column formatting," and each file's own header declares `# Max Column: 80 Columns` (line 3). However, the ingested content is largely unwrapped and massively exceeds 80 columns. Measured maximum line widths are: gemweb102.md 448 cols (this...

### [PR #27] `inbox/gemweb101.md` (Line 57)
- **Issue**: This EBNF line is 124 columns, exceeding the 120-char limit for code mandated by AI.md Directive 04 (summarized in AGENTS.md). Lark rule definitions can continue across lines, so wrapping the alternation keeps it within the limit.

### [PR #24] `docs/extraction_coverage_audit.md` (Line 42)
- **Issue**: This file declares "# Max Column: 80 Columns" (line 3) and the repo mandates an 80-column limit for text/Markdown (AGENTS.md / AI.md Directive 04). The existing docs adhere strictly — `docs/architecture.md`, `docs/decompression_guide.md`, and `docs/scratch_model_training.md` have zero lines over 80 ...

## Resource Management, State Leaks & Performance

### [PR #23] `src/unplow/incremental_pipeline.py` (Line 187)
- **Issue**: `compare_models_on_test_prompts` switches `self.model` into eval mode here but never restores its previous (training) mode before returning. Since this method is invoked from inside the per-step training loop (`train_domain_incrementally` only calls `self.model.train()` once at the start of each epo...

### [PR #22] `src/unplow/visualizer.py` (Line 1050)
- **Issue**: In dashboard mode this re-loads the checkpoint with `torch.load` even though the same file was already loaded during the pre-computation pass (line 1003), where `t_q_c` was extracted and pooled to build `dq`. This doubles checkpoint deserialization I/O for every dashboard frame. Consider storing the...

### [PR #20] `src/unplow/web.py` (Line 59)
- **Issue**: The UDP probe socket is only closed on the success path (`s.close()` at line 55). If `connect()` or `getsockname()` raises, the socket is never closed and leaks a file descriptor. Because `get_lan_ip_addresses()` is invoked on every `/api/status` poll, repeated failures could accumulate open descrip...

## Security, Path Traversal & Isolation

### [PR #20] `src/unplow/web.py` (Line 40)
- **Issue**: Changing the default bind address to `0.0.0.0` exposes the unauthenticated web server to every host on the local network by default. This server offers static file serving, a `/api/generate` endpoint that triggers visualization work, and a `/api/status` endpoint that now discloses the hostname, OS u...

### [PR #18] `src/unplow/visualizer.py` (Line 193)
- **Issue**: Loading checkpoints with `weights_only=False` re-enables arbitrary code execution via pickle during `torch.load`. Since PyTorch 2.6 the default is `weights_only=True` precisely to mitigate this when loading untrusted `.pt` files, and these call sites load user-supplied checkpoint paths. Because the ...

<!-- end of file: copilot-pr-review.md -->
