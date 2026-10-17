# Path: unplow/Makefile
# Makefile for unplow workspace

SHELL := /bin/bash
PYTHON := uv run python
PYTEST := uv run pytest

.PHONY: help venv install test test-cov lint status models inspect circuits steer
.PHONY: profile probe compress decompress archive unarchive manifold benchmark
.PHONY: gguf-inspect subspace-steer task-vector audit replow eval bundle
.PHONY: till-init till-train till-train-c till-train-cpp till-train-rust till-train-gencode till-train-all
.PHONY: daemon view watch sync clean

help:
	@echo "unplow Development Targets:"
	@echo "  make venv       - Create .venv & install package in editable mode"
	@echo "  make test       - Run test suite via uv run pytest"
	@echo "  make test-cov   - Run test suite with coverage report"
	@echo "  make status     - Check queue status and backend availability"
	@echo "  make models     - Audit local LLMs available for decompression"
	@echo "  make inspect    - Direct mechanistic MLP weight inspection via Logit Lens"
	@echo "  make circuits   - Extract coupled SwiGLU circuits with mean-centered logit lens"
	@echo "  make steer      - Activation-steered neuron basin decompression (Phase 4)"
	@echo "  make profile    - Per-token Shannon surprisal & cache profiling (Phase 4)"
	@echo "  make gguf-inspect - Parse GGUF v2/v3 header and dequantize tensors (Phase 5)"
	@echo "  make subspace-steer - Multi-neuron SVD subspace steering (Phase 5)"
	@echo "  make task-vector - Cross-substrate task-vector SVD steering (Phase 6)"
	@echo "  make audit      - Layer-wise internal information & Logit Lens audit (Phase 7)"
	@echo "  make replow     - Refactor code basins, stage datasets, and archive (Phase 8)"
	@echo "  make eval       - Sandboxed functional tests & perplexity evaluation (Phase 9)"
	@echo "  make bundle     - Hierarchical multi-file directory tree bundling (Phase 10)"
	@echo "  make till-init  - Initialize disk guard telemetry and stage domain corpora (Phase 11)"
	@echo "  make till-train - Run till-python distillation and Bayesian CTS archiving (Phase 11)"
	@echo "  make till-train-all - Train and archive all 5 specialized till-* domains"
	@echo "  make probe      - Run static weight decompression probe (Paradigm 1)"
	@echo "  make compress   - Run arithmetic text compression (Paradigm 2)"
	@echo "  make decompress - Run arithmetic text decompression (Paradigm 2)"
	@echo "  make archive    - Pack text/file into self-verifying .unp container"
	@echo "  make unarchive  - Unpack and verify .unp container"
	@echo "  make manifold   - Decompress arbitrary seed through LLM manifold"
	@echo "  make benchmark  - Run decompression benchmark (bpb, speed, RSS)"
	@echo "  make daemon     - Run background worker daemon"
	@echo "  make view       - View generated artifacts and executions"
	@echo "  make watch      - Stream live completed tasks and audit logs"
	@echo "  make sync       - Run turn-sync handover audit (Directive 08)"
	@echo "  make clean      - Clean temporary artifacts and cache"

venv:
	@echo "[*] Creating virtual environment with uv..."
	uv venv
	@echo "[*] Installing unplow in editable mode..."
	uv pip install -e ".[dev]"
	@echo "[+] Virtual environment ready."

install: venv

test:
	@echo "[*] Running unplow test suite..."
	$(PYTEST) -v tests/

test-cov:
	$(PYTEST) --cov=unplow tests/

status:
	uv run unplow status

models:
	uv run unplow models

inspect:
	uv run unplow inspect --harvest

circuits:
	uv run unplow circuits --harvest --seeds 8

steer:
	uv run unplow steer --layer 16 --neuron 467 --gain 4.5 --seq-len 32

profile:
	uv run unplow profile --text "unplow_token unplow_token lossless cache mixture verification."

gguf-inspect:
	uv run unplow gguf-inspect /home/fekerr/ai_models/qwen2-1.5b.gguf

subspace-steer:
	uv run unplow subspace-steer --layer 16 --neurons 467,468,469 --gain 4.0 --seq-len 32

task-vector:
	uv run unplow task-vector --gguf /home/fekerr/ai_models/qwen2.5-coder-1.5b.gguf --layer 16 --steer --prompt "/* "

audit:
	uv run unplow audit "def fibonacci(n):"

replow:
	uv run unplow replow --raw "def add(a, b):\n    return a + b" -o workspace/targets/sample_replow.unp

eval:
	uv run unplow eval --artifact tests/data/fib_solution.py --test-suite tests/data/fib_tests.py

bundle:
	uv run unplow bundle tests/data/ -o workspace/targets/tests_data.unp

till-init:
	uv run unplow till-init --manifest workspace/till_manifest.toml

till-train:
	uv run unplow till-train --domain python --corpus workspace/till_corpus/python/ -o workspace/targets/till_python.unp

till-train-c:
	uv run unplow till-train --domain c --corpus workspace/till_corpus/c/ -o workspace/targets/till_c.unp

till-train-cpp:
	uv run unplow till-train --domain cpp --corpus workspace/till_corpus/cpp/ -o workspace/targets/till_cpp.unp

till-train-rust:
	uv run unplow till-train --domain rust --corpus workspace/till_corpus/rust/ -o workspace/targets/till_rust.unp

till-train-gencode:
	uv run unplow till-train --domain gencode --corpus workspace/till_corpus/gencode/ -o workspace/targets/till_gencode.unp

till-train-all: till-train till-train-c till-train-cpp till-train-rust till-train-gencode

probe:
	uv run unplow probe --samples 2 --seq-len 32

compress:
	uv run unplow compress --text "Information theory connects sequence modeling with data compression."

decompress:
	@echo "Run: uv run unplow decompress <payload_file> --tokens <num_tokens>"

archive:
	uv run unplow archive \
		--text "Information theory connects sequence modeling with data compression." \
		-o workspace/targets/sample.unp

unarchive:
	uv run unplow unarchive workspace/targets/sample.unp

manifold:
	uv run unplow manifold --seed "unplow_latent_manifold" --tokens 20

benchmark:
	uv run unplow benchmark --runs 2

daemon:
	uv run unplow daemon --interval 2.0

view:
	@./tools/unplow_viewer.sh summary

watch:
	@./tools/unplow_viewer.sh follow

sync: test
	@echo "=== [Directive 08] unplow Turn-Sync Audit ==="
	@git status
	@echo "[+] Turn-sync audit completed cleanly."

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

# end of file: unplow/Makefile
