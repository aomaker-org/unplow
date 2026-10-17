# Path: unplow/src/unplow/cli.py
"""Command-line interface for unplow.

Provides interactive, queued, forensic auditing, and inference benchmarking subcommands:
  - probe       : Unpack static weight basins & rank memorization (Paradigm 1)
  - inspect     : Direct MLP weight inspection via Logit Lens (Geva et al., 2021)
  - compress    : Lossless autoregressive arithmetic compression (Paradigm 2)
  - decompress  : Exact lossless arithmetic decompression (Paradigm 2)
  - archive     : Pack file/text into self-verifying .unp binary container
  - unarchive   : Unpack and verify .unp binary container
  - manifold    : Generative seed decompression through LLM manifold (Paradigm 2)
  - benchmark   : Measure compression ratio, bits/byte, tok/s, RSS memory
  - models      : Probe and audit local LLM models available on the system
  - status      : Display task queue counts, telemetry, and daemon heartbeat
  - enqueue     : Add asynchronous decompression task to atomic file queue
  - daemon      : Run background worker polling loop with verbose telemetry
  - forensics   : Inspect cryptographic audit journal and execution traces
"""

import argparse
import base64
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional

from .bundler import WorkspaceBundler
from .circuits import SwiGLUCircuitDecompressor
from .conditioning import DOMAIN_PRIORS
from .container import FLAG_BAYES_MIXER, unpack_unp_container
from .eval import evaluate_artifact, format_summary_table
from .forensics import ForensicAuditLogger, get_telemetry_snapshot
from .mechanistic import MechanisticWeightInspector
from .models import audit_system_models, load_causal_lm, resolve_model_path
from .paradigm1 import WeightDecompressor
from .paradigm2 import LLMArithmeticCodec
from .queue import FileTaskQueue
from .task_vectors import CrossSubstrateTaskVectorEngine
from .till_guard import TillDiskGuardian, init_till_staging, train_till_domain
from .worker import DecompressionWorker


def setup_cli_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="unplow",
        description="unplow - Autonomous LLM Decompression Engine & Arithmetic Codec",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose forensic diagnostic logging")
    subparsers = parser.add_subparsers(dest="command", help="unplow subcommands")

    # Command: probe (Paradigm 1)
    p_probe = subparsers.add_parser("probe", aliases=["p1"], help="Probe weight basins (Paradigm 1)")
    p_probe.add_argument("-n", "--samples", type=int, default=4, help="Number of basin probes (default: 4)")
    p_probe.add_argument("-l", "--seq-len", type=int, default=64, help="Sequence length per probe (default: 64)")
    p_probe.add_argument("--explore-steps", type=int, default=12, help="High-temp exploration steps (default: 12)")
    p_probe.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_probe.add_argument("--prefix", action="append", help="Custom seed prefix (can repeat)")
    p_probe.add_argument("--mechanistic", action="store_true", help="Harvest seeds directly from MLP weights")
    p_probe.add_argument("-o", "--output", help="Output path for JSONL results")

    # Command: inspect (Mechanistic Logit Lens)
    p_insp = subparsers.add_parser(
        "inspect", aliases=["mechanistic"], help="Direct MLP weight inspection via Logit Lens"
    )
    p_insp.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_insp.add_argument("-k", "--neurons", type=int, default=6, help="Top neurons per layer to inspect (default: 6)")
    p_insp.add_argument("--top-tokens", type=int, default=5, help="Top tokens per neuron (default: 5)")
    p_insp.add_argument("--harvest", action="store_true", help="Harvest and display seed prefixes")
    p_insp.add_argument("--seeds", type=int, default=12, help="Max seed prefixes to harvest")

    # Command: circuits (Coupled SwiGLU Circuits & Anisotropic Logit Lens)
    p_circ = subparsers.add_parser(
        "circuits", aliases=["circuit", "swiglu"], help="Extract coupled SwiGLU circuits"
    )
    p_circ.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_circ.add_argument("-k", "--candidates", type=int, default=24, help="Candidate neurons per layer (default: 24)")
    p_circ.add_argument("--top-tokens", type=int, default=5, help="Top tokens per circuit key/value (default: 5)")
    p_circ.add_argument("--layer", type=int, action="append", help="Specific layer index to inspect (repeatable)")
    p_circ.add_argument("--allow-numerals", action="store_true", help="Include numeric and punctuation circuits")
    p_circ.add_argument("--harvest", action="store_true", help="Harvest and display synthesized associative seeds")
    p_circ.add_argument("--seeds", type=int, default=12, help="Max seed prefixes to harvest")
    p_circ.add_argument("--folded", action="store_true", help="Extract RMSNorm-folded SwiGLU circuits")
    p_circ.add_argument("-o", "--output", help="Output path for JSON results")

    # Command: compress (Paradigm 2)
    p_comp = subparsers.add_parser(
        "compress", aliases=["p2-compress"], help="Lossless arithmetic compression (Paradigm 2)"
    )
    p_comp.add_argument("text", nargs="?", help="Text to compress (or read from stdin/--file)")
    p_comp.add_argument("-f", "--file", help="File containing text to compress")
    p_comp.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_comp.add_argument("--max-ctx-len", type=int, default=None, help="Context horizon window")
    p_comp.add_argument(
        "--cache-mix",
        "--cache-mixture",
        dest="cache_mix",
        action="store_true",
        help="Enable online prefix-cache mixture coding",
    )
    p_comp.add_argument(
        "--bigram-mix",
        "--bigram-mixture",
        dest="bigram_mix",
        action="store_true",
        help="Enable Order-1 Markov bigram cache mixture coding",
    )
    p_comp.add_argument(
        "--bayes-mix",
        "--bayes-mixture",
        dest="bayes_mix",
        action="store_true",
        help="Enable online Bayesian Context-Tree Switching mixture coding (Bit 3)",
    )
    p_comp.add_argument("-o", "--output", help="Output binary file path")

    # Command: decompress (Paradigm 2)
    p_decomp = subparsers.add_parser(
        "decompress", aliases=["p2-decompress"], help="Lossless arithmetic decompression (Paradigm 2)"
    )
    p_decomp.add_argument("payload_file", help="Path to compressed payload file (.bin)")
    p_decomp.add_argument("-t", "--tokens", type=int, required=True, help="Exact number of tokens in original text")
    p_decomp.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_decomp.add_argument("--max-ctx-len", type=int, default=None, help="Context horizon window")
    p_decomp.add_argument(
        "--cache-mix",
        "--cache-mixture",
        dest="cache_mix",
        action="store_true",
        help="Enable online prefix-cache mixture coding",
    )
    p_decomp.add_argument(
        "--bigram-mix",
        "--bigram-mixture",
        dest="bigram_mix",
        action="store_true",
        help="Enable Order-1 Markov bigram cache mixture coding",
    )
    p_decomp.add_argument(
        "--bayes-mix",
        "--bayes-mixture",
        dest="bayes_mix",
        action="store_true",
        help="Enable online Bayesian Context-Tree Switching mixture coding (Bit 3)",
    )
    p_decomp.add_argument("-o", "--output", help="Output file path for reconstructed text")

    # Command: archive (Paradigm 2 .unp Container)
    p_arch = subparsers.add_parser(
        "archive", aliases=["pack"], help="Pack file/text into self-verifying .unp container"
    )
    p_arch.add_argument("file", nargs="?", help="Path to input text/source file")
    p_arch.add_argument("--text", help="Direct text to pack into container")
    p_arch.add_argument("-o", "--output", help="Output .unp archive path")
    p_arch.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_arch.add_argument("--max-ctx-len", type=int, default=512, help="Context horizon window (default: 512)")
    p_arch.add_argument(
        "--domain",
        default=None,
        help="Domain conditioning prior (0..15, prose_tech, python_code, etc. or 'auto')",
    )
    p_arch.add_argument(
        "--auto-domain",
        action="store_true",
        help="Auto-select optimal domain prior using MDL cross-entropy",
    )
    p_arch.add_argument(
        "--cache-mix",
        "--cache-mixture",
        dest="cache_mix",
        action="store_true",
        help="Enable online prefix-cache mixture arithmetic coding (Bit 1)",
    )
    p_arch.add_argument(
        "--bigram-mix",
        "--bigram-mixture",
        dest="bigram_mix",
        action="store_true",
        help="Enable Order-1 Markov bigram cache mixture coding (Bit 2)",
    )
    p_arch.add_argument(
        "--bayes-mix",
        "--bayes-mixture",
        dest="bayes_mix",
        action="store_true",
        help="Enable online Bayesian Context-Tree Switching mixture coding (Bit 3)",
    )

    # Command: steer (Phase 4 Activation Steering)
    p_steer = subparsers.add_parser(
        "steer", help="Activation-steered neuron basin decompression (Turner et al., 2023)"
    )
    p_steer.add_argument("--layer", type=int, required=True, help="Layer index to steer")
    p_steer.add_argument("--neuron", type=int, required=True, help="Neuron index to steer")
    p_steer.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_steer.add_argument("--gain", type=float, default=4.5, help="Steering impulse gain (default: 4.5)")
    p_steer.add_argument("--prompt", default="/* ", help="Seed prompt (default: '/* ')")
    p_steer.add_argument("--seq-len", type=int, default=48, help="Sequence length to generate (default: 48)")
    p_steer.add_argument("--steps", type=int, default=16, help="Number of forward passes to steer (default: 16)")
    p_steer.add_argument(
        "--temp",
        "--temperature",
        dest="temperature",
        type=float,
        default=0.25,
        help="Sampling temperature (default: 0.25)",
    )
    p_steer.add_argument("-o", "--output", help="Output JSON path for steered record")

    # Command: profile / surprisal (Phase 4 Shannon Surprisal Profiler)
    p_prof = subparsers.add_parser(
        "profile",
        aliases=["surprisal"],
        help="Per-token Shannon surprisal ledger and cache savings profiler",
    )
    p_prof.add_argument("text", nargs="?", help="Text to profile")
    p_prof.add_argument("-t", "--text", dest="opt_text", help="Text to profile")
    p_prof.add_argument("-f", "--file", "-i", "--input", dest="file", help="File containing text to profile")
    p_prof.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_prof.add_argument(
        "--cache-weight",
        type=float,
        default=0.08,
        help="Prefix cache mixture weight (default: 0.08)",
    )
    p_prof.add_argument("-o", "--output", help="Output JSON path for surprisal profile")

    # Command: gguf-inspect (Phase 5 GGUF v2/v3 Binary Inspector & Task-Vector Forensics)
    p_gguf = subparsers.add_parser(
        "gguf-inspect",
        aliases=["gguf"],
        help="Parse GGUF v2/v3 header, dequantize tensors, and compute task vectors",
    )
    p_gguf.add_argument("path", help="Path to .gguf model file")
    p_gguf.add_argument(
        "--compare",
        "--compare-layer",
        dest="compare_layer",
        type=int,
        default=None,
        help="Transformer layer index to compare with reference model",
    )
    p_gguf.add_argument(
        "--tensor",
        default=None,
        help="Specific GGUF tensor name to inspect/compare",
    )
    p_gguf.add_argument(
        "-m", "--model", default="auto", help="Model name or local snapshot path for reference comparison"
    )
    p_gguf.add_argument("-o", "--output", help="Output JSON path for GGUF summary or delta report")

    # Command: subspace-steer (Phase 5 Multi-Neuron SVD Subspace Steering)
    p_sub = subparsers.add_parser(
        "subspace-steer",
        aliases=["subspace"],
        help="Multi-neuron SVD principal subspace steering (Elhage et al., 2022)",
    )
    p_sub.add_argument("--layer", type=int, required=True, help="Layer index to steer")
    p_sub.add_argument(
        "--neurons",
        type=str,
        required=True,
        help="Comma-separated neuron indices (e.g., '467,468,469')",
    )
    p_sub.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_sub.add_argument("--gain", type=float, default=5.0, help="Steering impulse gain (default: 5.0)")
    p_sub.add_argument("--prompt", default="/* ", help="Seed prompt (default: '/* ')")
    p_sub.add_argument("--seq-len", type=int, default=48, help="Sequence length to generate (default: 48)")
    p_sub.add_argument("--steps", type=int, default=16, help="Number of forward passes to steer (default: 16)")
    p_sub.add_argument(
        "--temp",
        "--temperature",
        dest="temperature",
        type=float,
        default=0.25,
        help="Sampling temperature (default: 0.25)",
    )
    p_sub.add_argument("-o", "--output", help="Output JSON path for subspace record")

    # Command: task-vector (Phase 6 Cross-Substrate Task Vectors & Differential Steering)
    p_tv = subparsers.add_parser(
        "task-vector",
        aliases=["tv", "taskvector"],
        help="Cross-substrate task-vector forensics & differential SVD steering (Ilharco et al., 2023)",
    )
    p_tv.add_argument("--gguf", required=True, help="Path to .gguf model file")
    p_tv.add_argument("--layer", type=int, default=16, help="Layer index to analyze / steer (default: 16)")
    p_tv.add_argument("-m", "--model", default="auto", help="Reference model name or local snapshot path")
    p_tv.add_argument("--steer", action="store_true", help="Steer generation along cross-substrate task vector")
    p_tv.add_argument("--prompt", default="def ", help="Seed prompt for steered generation (default: 'def ')")
    p_tv.add_argument("--gain", type=float, default=4.5, help="Steering impulse gain (default: 4.5)")
    p_tv.add_argument("--seq-len", type=int, default=48, help="Sequence length to generate (default: 48)")
    p_tv.add_argument("--steps", type=int, default=16, help="Number of forward passes to steer (default: 16)")
    p_tv.add_argument(
        "--temp",
        "--temperature",
        dest="temperature",
        type=float,
        default=0.25,
        help="Sampling temperature (default: 0.25)",
    )
    p_tv.add_argument("--top-tokens", type=int, default=6, help="Top tokens to decode via Logit Lens (default: 6)")
    p_tv.add_argument("--top-neurons", type=int, default=6, help="Top divergent neurons to analyze (default: 6)")
    p_tv.add_argument("-o", "--output", help="Output JSON path for report or steered record")

    # Command: audit (Phase 7 Layer-Wise Information & Surprisal Audit)
    p_audit = subparsers.add_parser(
        "audit",
        aliases=["layer-audit", "inspect-layers"],
        help="Layer-wise internal information & Logit Lens audit across Transformer depth",
    )
    p_audit.add_argument("prompt", help="Input prompt / sequence to audit across layers")
    p_audit.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_audit.add_argument("--top-k", type=int, default=5, help="Number of top token candidates per layer (default: 5)")
    p_audit.add_argument("-o", "--output", help="Output JSON path for layer audit report")
    p_audit.add_argument("--json", action="store_true", help="Print output in JSON format instead of table")

    # Command: replow (Phase 8 Refactoring, Distillation Staging, and Bayesian CTS Archiving)
    p_replow = subparsers.add_parser(
        "replow",
        help="Refactor code basins, stage training datasets, and archive with Bayesian CTS",
    )
    p_replow.add_argument("file", nargs="?", default=None, help="Input raw code file or script to process")
    p_replow.add_argument("--raw", default=None, help="Direct raw code string to process")
    p_replow.add_argument(
        "-o", "--output",
        default="workspace/targets/replow_artifact.unp",
        help="Output .unp archive path (default: workspace/targets/replow_artifact.unp)",
    )
    p_replow.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_replow.add_argument("--domain", type=int, default=2, help="Domain prior ID (default: 2 for python_code)")
    p_replow.add_argument(
        "--train-steps", type=int, default=0, help="Optional distillation update steps (default: 0)"
    )
    p_replow.add_argument("--stage-dataset", default=None, help="Optional JSONL path to stage training dataset")
    p_replow.add_argument("--json", action="store_true", help="Output artifact details as JSON")

    # Command: eval (Phase 9 Replow Evaluation Harness)
    p_eval = subparsers.add_parser(
        "eval",
        aliases=["replow-eval"],
        help="Run sandboxed functional correctness tests and perplexity shift tracking",
    )
    p_eval.add_argument("--artifact", required=True, help="Path to refactored .unp container or .py script")
    p_eval.add_argument("--test-suite", default=None, help="Python file containing test assertions or functions")
    p_eval.add_argument("--corpus", default=None, help="Optional text file or string for perplexity evaluation")
    p_eval.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_eval.add_argument("--timeout", type=float, default=3.0, help="Per-test timeout in seconds (default: 3.0)")
    p_eval.add_argument("-o", "--output", help="Output JSON path for evaluation report")
    p_eval.add_argument("--json", action="store_true", help="Output evaluation report as JSON")

    # Command: bundle (Phase 10 Hierarchical Workspace Bundling)
    p_bundle = subparsers.add_parser(
        "bundle", help="Bundle directory tree into hierarchical .unp archive"
    )
    p_bundle.add_argument("source", help="Source directory path to bundle")
    p_bundle.add_argument("-o", "--output", required=True, help="Output .unp archive path")
    p_bundle.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_bundle.add_argument("--domain", type=int, default=2, help="Domain prior ID (default: 2 for python_code)")
    p_bundle.add_argument("--json", action="store_true", help="Output bundle manifest as JSON")

    # Command: unbundle (Phase 10 Hierarchical Workspace Unbundling)
    p_unbundle = subparsers.add_parser(
        "unbundle", help="Unpack hierarchical .unp archive into directory tree"
    )
    p_unbundle.add_argument("archive", help="Input .unp archive path")
    p_unbundle.add_argument("-o", "--output", required=True, help="Target restore directory")
    p_unbundle.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_unbundle.add_argument("--json", action="store_true", help="Output unbundle manifest as JSON")

    # Command: till-init (Phase 11 Staging & Disk Telemetry)
    p_tinit = subparsers.add_parser(
        "till-init",
        help="Initialize disk guard telemetry and stage till-* domain datasets",
    )
    p_tinit.add_argument(
        "--manifest",
        default="workspace/till_manifest.toml",
        help="Path to till offload manifest (default: workspace/till_manifest.toml)",
    )
    p_tinit.add_argument(
        "--corpus-dir",
        default="workspace/till_corpus",
        help="Base directory for domain training corpora (default: workspace/till_corpus)",
    )
    p_tinit.add_argument("--json", action="store_true", help="Output staging details as JSON")

    # Command: till-train (Phase 11 Specialized Domain Distillation & Archiving)
    p_ttrain = subparsers.add_parser(
        "till-train",
        help="Run domain-specific closed-loop distillation and Bayesian CTS archiving",
    )
    p_ttrain.add_argument(
        "--domain",
        required=True,
        choices=["python", "c", "cpp", "rust", "gencode"],
        help="Target specialization domain",
    )
    p_ttrain.add_argument(
        "--corpus",
        default=None,
        help="Input corpus directory (defaults to workspace/till_corpus/<domain>/)",
    )
    p_ttrain.add_argument(
        "-o", "--output",
        default=None,
        help="Output .unp archive path (defaults to workspace/targets/till_<domain>.unp)",
    )
    p_ttrain.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_ttrain.add_argument(
        "--train-steps", type=int, default=1, help="Distillation training update steps (default: 1)"
    )
    p_ttrain.add_argument("--json", action="store_true", help="Output train report as JSON")

    # Command: till-guard-offload (Phase 14 Storage Guardian Offload)
    p_toff = subparsers.add_parser(
        "till-guard-offload",
        aliases=["till-offload"],
        help="Enforce disk threshold and trigger rclone offload to Google Drive",
    )
    p_toff.add_argument(
        "--manifest",
        default="workspace/till_manifest.toml",
        help="Path to till offload manifest (default: workspace/till_manifest.toml)",
    )
    p_toff.add_argument(
        "--target-dirs",
        nargs="*",
        default=None,
        help="Specific directories to evaluate for offload (defaults to manifest targets)",
    )
    p_toff.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate offload without executing rclone move",
    )
    p_toff.add_argument("--json", action="store_true", help="Output offload result as JSON")

    # Command: unarchive (Paradigm 2 .unp Container)
    p_unarch = subparsers.add_parser("unarchive", aliases=["unpack"], help="Unpack and verify .unp container")
    p_unarch.add_argument("archive_file", help="Path to .unp archive file")
    p_unarch.add_argument("-o", "--output", help="Output file path for reconstructed text")
    p_unarch.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")

    # Command: manifold (Paradigm 2)
    p_mani = subparsers.add_parser(
        "manifold", aliases=["p2-manifold"], help="Generative seed decompression (Paradigm 2)"
    )
    p_mani.add_argument("-s", "--seed", default="unplow_latent_manifold", help="Seed string or arbitrary byte text")
    p_mani.add_argument("-t", "--tokens", type=int, default=30, help="Number of tokens to generate (default: 30)")
    p_mani.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_mani.add_argument("-o", "--output", help="Output file path")

    # Command: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run decompression benchmarks (bpb, speed, RSS)")
    p_bench.add_argument("-r", "--runs", type=int, default=2, help="Number of benchmark iterations (default: 2)")
    p_bench.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")

    # Command: models
    subparsers.add_parser("models", help="Audit local LLM models available on the system")

    # Command: status
    subparsers.add_parser("status", help="Display queue status, telemetry, and heartbeat")

    # Command: enqueue
    p_enq = subparsers.add_parser("enqueue", help="Add asynchronous task to file queue")
    p_enq.add_argument(
        "action",
        choices=[
            "probe", "compress", "decompress", "manifold", "archive", "unarchive",
            "circuits", "steer", "profile", "surprisal", "audit", "layer-audit",
            "replow", "replow_pipeline", "eval", "replow_eval", "bundle", "unbundle",
            "till-init", "till-train", "till_init", "till_train",
            "till-guard-offload", "till_guard_offload", "till-offload", "till_offload",
        ],
        help="Task action",
    )
    p_enq.add_argument("--payload", default="", help="Text or payload string")
    p_enq.add_argument("--extra", default="{}", help="JSON dictionary of extra parameters")

    # Command: daemon
    p_daemon = subparsers.add_parser("daemon", help="Run background worker queue processor")
    p_daemon.add_argument("-i", "--interval", type=float, default=2.0, help="Polling interval in seconds")
    p_daemon.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_daemon.add_argument("--max-iterations", type=int, default=None, help="Stop after N iterations (for testing)")

    # Command: forensics
    p_for = subparsers.add_parser("forensics", help="Inspect cryptographic audit journal")
    p_for.add_argument("-n", "--limit", type=int, default=20, help="Max entries to display")

    args = parser.parse_args()
    setup_cli_logging(verbose=getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # --------------------------------------------------------------------------
    # Subcommand Execution
    # --------------------------------------------------------------------------
    if args.command in ("probe", "p1"):
        print(f"[*] Initializing Paradigm 1 Weight Decompressor (model: {args.model})...")
        decompressor = WeightDecompressor(model_name=args.model)
        prefixes = args.prefix if args.prefix else None
        print(f"[*] Probing {args.samples} weight basins "
              f"(seq_len={args.seq_len}, explore_steps={args.explore_steps})...")
        records = decompressor.probe_weight_basins(
            num_samples=args.samples,
            seq_len=args.seq_len,
            explore_steps=args.explore_steps,
            seed_prefixes=prefixes,
            use_mechanistic_seeds=getattr(args, "mechanistic", False),
        )

        print("\n=== [Paradigm 1] Decompressed Weight Basins (Ranked by Carlini Memorization Score) ===")
        for idx, rec in enumerate(records, 1):
            print(f"\n--- Basin Probe #{idx} ---")
            print(f"Memorization Score : {rec.memorization_score:.3f}")
            print(f"Model Complexity   : {rec.model_bits_per_byte:.3f} bits/byte")
            print(f"Zlib Entropy       : {rec.zlib_bits_per_byte:.3f} bits/byte")
            print(f"Casing Sensitivity : {rec.casing_sensitivity:.3f}")
            print(f"Extracted Text:\n{rec.text}")

        if args.output:
            out_p = Path(args.output)
            decompressor.save_records_to_jsonl(records, out_p)
            print(f"\n[+] Saved {len(records)} records to {out_p}")

    elif args.command in ("inspect", "mechanistic"):
        print(f"[*] Initializing Mechanistic Weight Inspector (model: {args.model})...")
        inspector = MechanisticWeightInspector(model_name=args.model)
        traces = inspector.extract_mlp_memory_clusters(
            top_neurons_per_layer=args.neurons,
            top_k_tokens=args.top_tokens,
        )
        print(f"\n=== Mechanistic MLP Down-Projection Value Clusters ({len(traces)} neurons analyzed) ===")
        for tr in traces[:15]:
            toks_str = ", ".join(repr(t) for t in tr.top_tokens)
            print(f"- Layer {tr.layer_idx:02d} Neuron #{tr.neuron_idx:04d} (L2: {tr.l2_norm:.4f}) -> [{toks_str}]")

        if args.harvest:
            seeds = inspector.harvest_seed_prefixes(num_seeds=args.seeds)
            print(f"\n[+] Harvested {len(seeds)} Mechanistic Seed Prefixes:")
            for s in seeds:
                print(f"    {repr(s)}")

    elif args.command in ("circuits", "circuit", "swiglu"):
        layers = args.layer if args.layer else None
        if getattr(args, "folded", False):
            from .steering import MechanisticSteeringEngine
            print(f"[*] Initializing Mechanistic Steering Engine for folded circuits (model: {args.model})...")
            engine = MechanisticSteeringEngine(model_name=args.model)
            print(f"[*] Extracting RMSNorm-folded SwiGLU circuits (candidates={args.candidates})...")
            traces = engine.extract_rmsnorm_folded_circuits(
                layer_indices=layers,
                candidates_per_layer=args.candidates,
                top_k_tokens=args.top_tokens,
            )
            print(f"\n=== [RMSNorm-Folded SwiGLU Circuits (Dar et al., 2023)] ({len(traces)} traces found) ===")
            print(f"{'Layer':<6} {'Neuron':<8} {'Centered L2':<13} {'Selectivity':<13} "
                  f"{'Folded Trigger Key':<24} {'Target Value':<20}")
            print("-" * 88)
            for tr in traces[:25]:
                k_disp = str(tr.folded_trigger_keys[:3])
                v_disp = str(tr.target_values[:3])
                print(f"{tr.layer_idx:<6} {tr.neuron_idx:<8} {tr.centered_l2_norm:<13.4f} "
                      f"{tr.selectivity_margin:<13.4f} {k_disp:<24} {v_disp:<20}")
            if args.output:
                out_p = Path(args.output)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                with open(out_p, "w", encoding="utf-8") as f:
                    json.dump([t.to_dict() for t in traces], f, indent=2)
                print(f"\n[+] Exported {len(traces)} folded traces to {out_p}")
            return

        print(f"[*] Initializing SwiGLU Circuit Decompressor (model: {args.model})...")
        decompressor = SwiGLUCircuitDecompressor(model_name=args.model)
        suppress_num = not getattr(args, "allow_numerals", False)
        print(f"[*] Extracting coupled SwiGLU circuits "
              f"(candidates={args.candidates}, suppress_numerals={suppress_num})...")
        circuits = decompressor.extract_coupled_circuits(
            layer_indices=layers,
            candidates_per_layer=args.candidates,
            top_k_tokens=args.top_tokens,
            suppress_numerals=suppress_num,
        )

        print(f"\n=== [Coupled SwiGLU Circuits & Anisotropic Logit Lens] ({len(circuits)} circuits found) ===")
        print(f"{'Layer':<6} {'Neuron':<8} {'Centered L2':<13} {'Selectivity':<13} "
              f"{'Trigger Key':<20} {'Target Value':<20}")
        print("-" * 84)
        for c in circuits[:25]:
            k_disp = str(c.trigger_key_tokens[:3])
            v_disp = str(c.target_value_tokens[:3])
            print(f"{c.layer_idx:<6} {c.neuron_idx:<8} {c.centered_l2_norm:<13.4f} {c.selectivity_margin:<13.4f} "
                  f"{k_disp:<20} {v_disp:<20}")

        if args.harvest or getattr(args, "seeds", 0) > 0:
            seeds = decompressor.harvest_associative_seeds(
                max_seeds=args.seeds,
                layer_indices=layers,
                suppress_numerals=suppress_num,
            )
            print(f"\n[+] Harvested {len(seeds)} Associative Seeds for Weight Probing:")
            for s in seeds:
                print(f"  - {s!r}")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in circuits], f, indent=2)
            print(f"\n[+] Exported {len(circuits)} circuit traces to {out_p}")

    elif args.command == "steer":
        from .steering import MechanisticSteeringEngine
        print(f"[*] Decompressing training basin via activation steering: "
              f"Layer {args.layer}, Neuron {args.neuron} (gain={args.gain}, model={args.model})...")
        engine = MechanisticSteeringEngine(model_name=args.model)
        rec = engine.decompress_neuron_basin(
            layer_idx=args.layer,
            neuron_idx=args.neuron,
            seed_prompt=args.prompt,
            seq_len=args.seq_len,
            steering_gain=args.gain,
            steering_steps=args.steps,
            temperature=args.temperature,
        )
        print("\n=== Steered Basin Decompression Record ===")
        print(f"Layer / Neuron     : Layer {rec.layer_idx}, Neuron #{rec.neuron_idx:04d}")
        print(f"Steering Gain      : {rec.steering_gain}")
        print(f"Model Cross-Entropy: {rec.model_bpb} bpb")
        print(f"Zlib Syntactic BPB : {rec.zlib_bpb} bpb")
        print(f"Memorization Score : {rec.memorization_score}x")
        print("\n--- Extracted Basin Plaintext ---")
        print(rec.extracted_text)

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(rec.to_dict(), f, indent=2)
            print(f"\n[+] Exported steered record to {out_p}")

    elif args.command in ("profile", "surprisal"):
        from .cache_coder import profile_sequence_surprisal
        text = getattr(args, "opt_text", None) or args.text
        if args.file:
            text = Path(args.file).read_text(encoding="utf-8")
        elif not text and not sys.stdin.isatty():
            text = sys.stdin.read()

        if not text:
            print("[-] Error: No text supplied for surprisal profiling.", file=sys.stderr)
            sys.exit(1)

        print(f"[*] Profiling sequence surprisal (model={args.model}, cache_weight={args.cache_weight})...")
        tok, mod, mod_path = load_causal_lm(model_identifier=args.model)
        bos_id = getattr(tok, "bos_token_id", 0) or 0
        report = profile_sequence_surprisal(
            model=mod,
            tokenizer=tok,
            plaintext=text,
            bos_token_id=bos_id,
            cache_weight=args.cache_weight,
        )
        print(f"\n=== Per-Token Shannon Surprisal Profile "
              f"({report.total_tokens} tokens, {report.total_bytes} bytes) ===")
        print(f"Ideal Total Bits   : {report.ideal_total_bits:.3f} bits")
        print(f"Ideal Bits Per Byte: {report.ideal_bits_per_byte:.3f} bpb")
        print(f"Cache Bits Saved   : {report.cache_bits_saved:.3f} bits")
        header_cols = (
            f"{'Step':<6} {'Token ID':<10} {'Token Str':<18} "
            f"{'P(LLM)':<12} {'P(Mix)':<12} {'Bits':<8} {'Cum Bits':<10}"
        )
        print(f"\n{header_cols}")
        print("-" * 78)
        for e in report.entries[:30]:
            tok_rep = repr(e.token_str)
            print(f"{e.step_idx:<6} {e.token_id:<10} {tok_rep:<18} {e.prob_llm:<12.6f} {e.prob_mixed:<12.6f} "
                  f"{e.ideal_shannon_bits:<8.3f} {e.cumulative_bits:<10.3f}")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"\n[+] Exported surprisal report to {out_p}")

    elif args.command in ("gguf-inspect", "gguf"):
        from .gguf_inspector import GGUFBinaryInspector
        import torch

        print(f"[*] Parsing GGUF binary container: {args.path}...")
        inspector = GGUFBinaryInspector(args.path)
        summary = inspector.parse_header()
        arch = summary.metadata.get("general.architecture", "unknown")
        print("\n=== GGUF Container Header Summary ===")
        print(f"File Path         : {summary.file_path}")
        print(f"File Size         : {summary.file_size_bytes / (1024 * 1024):.2f} MB")
        print(f"GGUF Version      : v{summary.version}")
        print(f"Architecture      : {arch}")
        print(f"Tensor Count      : {summary.tensor_count}")
        print(f"Metadata Key-Vals : {summary.metadata_kv_count}")
        print(f"Alignment         : {summary.alignment} bytes")
        print(f"Data Offset       : {summary.data_offset} bytes")

        delta_report = None
        if args.compare_layer is not None:
            layer_idx = args.compare_layer
            tensor_name = args.tensor or f"blk.{layer_idx}.ffn_down.weight"
            print(f"\n[*] Comparing GGUF tensor '{tensor_name}' with reference model (layer {layer_idx})...")
            _, model, _ = load_causal_lm(model_identifier=args.model, device="cpu", dtype=torch.float32)
            ref_tensor = model.model.layers[layer_idx].mlp.down_proj.weight
            delta_report = inspector.compare_with_reference_tensor(
                gguf_tensor_name=tensor_name,
                ref_tensor=ref_tensor,
                top_k_cols=5,
            )
            print("\n=== Cross-Substrate Task-Vector Delta Report ===")
            print(f"GGUF Tensor       : {delta_report.gguf_tensor_name} ({delta_report.ggml_type_name})")
            print(f"Tensor Shape      : {delta_report.shape}")
            print(f"GGUF L2 Norm      : {delta_report.gguf_l2_norm:.4f}")
            print(f"Ref PyTorch Norm  : {delta_report.ref_l2_norm:.4f}")
            print(f"Cosine Similarity : {delta_report.cosine_similarity:.6f}")
            print(f"Relative Frobenius: {delta_report.relative_frobenius_delta:.6f}")
            print(f"Top Divergent Cols: {delta_report.top_divergent_columns}")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            export_data = delta_report.to_dict() if delta_report is not None else summary.to_dict()
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
            print(f"\n[+] Exported report to {out_p}")

    elif args.command in ("subspace-steer", "subspace"):
        from .subspace_coder import SVDSubspaceSteeringEngine
        import torch

        neuron_list = [int(x.strip()) for x in args.neurons.split(",") if x.strip()]
        if not neuron_list:
            print("[-] Error: No neuron indices provided in --neurons.", file=sys.stderr)
            sys.exit(1)

        print(f"[*] Loading model for SVD Subspace Steering: {args.model}...")
        tokenizer, model, _ = load_causal_lm(model_identifier=args.model, dtype=torch.float32)
        engine = SVDSubspaceSteeringEngine(model, tokenizer)

        print(f"[*] Steering Layer {args.layer} along principal subspace of neurons {neuron_list} "
              f"(gain: {args.gain}, steps: {args.steps}, seq_len: {args.seq_len})...")
        rec = engine.steer_with_subspace(
            layer_idx=args.layer,
            neuron_indices=neuron_list,
            seed_prompt=args.prompt,
            seq_len=args.seq_len,
            steering_gain=args.gain,
            steering_steps=args.steps,
            temperature=args.temperature,
        )

        print("\n=== Multi-Neuron SVD Subspace Steering Basin ===")
        print(f"Target Layer      : Layer {rec.layer_idx}")
        print(f"Neuron Ensemble   : {rec.neuron_indices}")
        print(f"SV Energy Ratio   : {rec.singular_value_ratio:.4f}")
        print(f"Steering Gain     : {rec.steering_gain}")
        print(f"Model Loss (CE)   : {rec.model_bpb:.3f} bpb")
        print(f"zlib Compression  : {rec.zlib_bpb:.3f} bpb")
        print(f"Memorization Score: {rec.memorization_score:.3f}x")
        print("\n--- Extracted Subspace Basin Text ---")
        print(rec.extracted_text)

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(rec.to_dict(), f, indent=2)
            print(f"\n[+] Exported subspace steering record to {out_p}")

    elif args.command in ("task-vector", "tv", "taskvector"):
        from .task_vectors import CrossSubstrateTaskVectorEngine
        import torch

        gguf_p = Path(args.gguf)
        if not gguf_p.exists():
            print(f"[-] Error: GGUF model file '{args.gguf}' not found.", file=sys.stderr)
            sys.exit(1)

        print(f"[*] Loading reference model for Cross-Substrate Task Vector: {args.model}...")
        tokenizer, model, _ = load_causal_lm(model_identifier=args.model, dtype=torch.float32)
        engine = CrossSubstrateTaskVectorEngine(model, tokenizer)

        print(f"[*] Extracting Task Vector Delta from {gguf_p.name} at Layer {args.layer}...")
        vec, report = engine.compute_layer_task_vector(
            gguf_path=gguf_p,
            layer_idx=args.layer,
            top_k_tokens=args.top_tokens,
            top_k_neurons=args.top_neurons,
        )

        print("\n=== Cross-Substrate Task-Vector Spectrum Report ===")
        print(f"GGUF Source       : {report.gguf_tensor_name}")
        print(f"Target Layer      : Layer {report.layer_idx}")
        print(f"Frobenius Delta   : {report.frobenius_delta_norm:.4f}")
        print(f"Cosine Similarity : {report.cosine_similarity:.6f}")
        print(f"Top Singular Value: {report.top_singular_value:.4f}")
        print(f"SV Variance Ratio : {report.explained_variance_ratio:.4f}")
        print(f"Promoted Tokens   : {report.promoted_tokens}")
        print(f"Suppressed Tokens : {report.suppressed_tokens}")
        print(f"Top Divergent Cols: {report.top_divergent_neurons}")

        steered_record = None
        if args.steer:
            print(
                f"\n[*] Steering live model along task vector (gain: {args.gain}, "
                f"steps: {args.steps}, seq_len: {args.seq_len})..."
            )
            steered_record = engine.steer_with_gguf_task_vector(
                gguf_path=gguf_p,
                layer_idx=args.layer,
                seed_prompt=args.prompt,
                seq_len=args.seq_len,
                steering_gain=args.gain,
                steering_steps=args.steps,
                temperature=args.temperature,
            )
            print("\n=== Steered Cross-Substrate Generation Result ===")
            print(f"Extracted Text    : {steered_record.extracted_text}")
            print(f"Model Loss (CE)   : {steered_record.model_bpb:.3f} bpb")
            print(f"zlib Compression  : {steered_record.zlib_bpb:.3f} bpb")
            print(f"Memorization Score: {steered_record.memorization_score:.3f}x")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            data = steered_record.to_dict() if steered_record is not None else report.to_dict()
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"\n[+] Exported task vector results to {out_p}")

    elif args.command in ("audit", "layer-audit", "inspect-layers"):
        from .auditor import ModelInternalAuditor

        is_json = getattr(args, "json", False)
        if not is_json:
            print(f"[*] Loading model for Layer-Wise Internal Information Audit: {args.model}...")
        tokenizer, model, _ = load_causal_lm(model_identifier=args.model)
        auditor = ModelInternalAuditor(model, tokenizer)

        if not is_json:
            print(f"[*] Auditing internal representations for prompt: {repr(args.prompt)}...")
        reports = auditor.audit_sequence(args.prompt, top_k=args.top_k)

        if getattr(args, "json", False):
            payload = {
                "prompt": args.prompt,
                "layer_count": len(reports),
                "layers": [r.to_dict() for r in reports],
            }
            print(json.dumps(payload, indent=2))
        else:
            print("\n" + auditor.format_audit_table(reports, prompt=args.prompt))

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "prompt": args.prompt,
                "layer_count": len(reports),
                "layers": [r.to_dict() for r in reports],
            }
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"\n[+] Exported layer audit report to {out_p}")

    elif args.command == "replow":
        from .replow import ReplowEngine

        raw_code = args.raw
        if args.file:
            fp = Path(args.file)
            if not fp.exists():
                print(f"[-] Error: File '{args.file}' not found.", file=sys.stderr)
                sys.exit(1)
            raw_code = fp.read_text(encoding="utf-8")
        elif not raw_code and not sys.stdin.isatty():
            raw_code = sys.stdin.read()

        if not raw_code:
            print("[-] Error: No raw code provided to replow (use --raw, file path, or stdin).", file=sys.stderr)
            sys.exit(1)

        is_json = getattr(args, "json", False)
        if not is_json:
            print(f"[*] Initializing ReplowEngine with model: {args.model}...")

        tokenizer, model, _ = load_causal_lm(model_identifier=args.model)
        engine = ReplowEngine(model, tokenizer)

        if args.stage_dataset:
            ds_p = engine.stage_training_dataset([raw_code], args.stage_dataset)
            if not is_json:
                print(f"[+] Staged training dataset at: {ds_p}")

        if not is_json:
            print(f"[*] Processing AST refactoring and Bayesian CTS archiving -> {args.output}...")

        artifact, loss = engine.run_pipeline(
            raw_code=raw_code,
            output_unp_path=args.output,
            train_steps=args.train_steps,
            domain_id=args.domain,
        )

        if is_json:
            data = artifact.to_dict()
            if loss is not None:
                data["distillation_loss"] = loss
            print(json.dumps(data, indent=2))
        else:
            print("\n=== Replow Pipeline Artifact Summary ===")
            print(f"Syntax Valid      : {artifact.syntax_valid}")
            print(f"Container Archive : {artifact.container_path} ({artifact.compressed_bytes_len} bytes)")
            print(f"Bits per Byte     : {artifact.bits_per_byte:.3f} bpb")
            if loss is not None:
                print(f"Distillation Loss : {loss:.4f}")
            print("\n--- Refactored Code ---")
            print(artifact.refactored_text)

    elif args.command in ("eval", "replow-eval"):
        art_path = Path(args.artifact)
        if not art_path.exists():
            print(f"[-] Error: Artifact '{args.artifact}' not found.", file=sys.stderr)
            sys.exit(1)

        is_json = getattr(args, "json", False)
        model, tokenizer = None, None

        corpus_texts = None
        if args.corpus:
            cp = Path(args.corpus)
            if cp.exists():
                corpus_texts = [
                    line.strip() for line in cp.read_text(encoding="utf-8").splitlines() if line.strip()
                ]
            else:
                corpus_texts = [args.corpus]

        if art_path.suffix.lower() == ".unp" or corpus_texts:
            if not is_json:
                print(f"[*] Loading model substrate '{args.model}' for evaluation...")
            tokenizer, model, _ = load_causal_lm(model_identifier=args.model)

        summary = evaluate_artifact(
            artifact_path=str(art_path),
            test_suite_path=args.test_suite,
            corpus_texts=corpus_texts,
            model=model,
            tokenizer=tokenizer,
            timeout_sec=args.timeout,
        )

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
            if not is_json:
                print(f"[+] Evaluation report saved to {out_p}")

        if is_json:
            print(json.dumps(summary.to_dict(), indent=2))
        else:
            print(format_summary_table(summary))

    elif args.command == "bundle":
        src_path = Path(args.source)
        if not src_path.exists() or not src_path.is_dir():
            print(
                f"[-] Error: Source directory '{args.source}' not found or not a directory.",
                file=sys.stderr,
            )
            sys.exit(1)

        is_json = getattr(args, "json", False)
        if not is_json:
            print(f"[*] Initializing WorkspaceBundler with model: {args.model}...")

        tokenizer, model = None, None
        if args.model != "mock":
            tokenizer, model, _ = load_causal_lm(model_identifier=args.model)

        bundler = WorkspaceBundler(model=model, tokenizer=tokenizer, model_name=args.model)
        report = bundler.bundle_directory(
            source_dir=src_path,
            output_unp_path=args.output,
            domain_id=args.domain,
        )

        if is_json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.format_manifest_table())

    elif args.command == "unbundle":
        arc_path = Path(args.archive)
        if not arc_path.exists():
            print(f"[-] Error: Archive '{args.archive}' not found.", file=sys.stderr)
            sys.exit(1)

        is_json = getattr(args, "json", False)
        if not is_json:
            print(f"[*] Initializing WorkspaceBundler for unbundle with model: {args.model}...")

        tokenizer, model = None, None
        if args.model != "mock":
            tokenizer, model, _ = load_causal_lm(model_identifier=args.model)

        bundler = WorkspaceBundler(model=model, tokenizer=tokenizer, model_name=args.model)
        report = bundler.unbundle_archive(
            archive_path=arc_path,
            target_dir=args.output,
        )

        if is_json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.format_manifest_table())

    elif args.command == "till-init":
        is_json = getattr(args, "json", False)
        if not is_json:
            print("[*] Initializing till model staging & disk telemetry guard...")
        res = init_till_staging(
            manifest_path=args.manifest,
            base_corpus_dir=args.corpus_dir,
            verbose=not is_json,
        )
        if is_json:
            print(json.dumps(res, indent=2))
        else:
            telem = res["telemetry"]
            print(
                f"[+] Disk Telemetry: Total={telem['total_gb']} GB, "
                f"Free={telem['free_gb']} GB ({telem['free_percent']}%)"
            )
            print(f"[+] Staged {len(res['domains'])} domain corpora under {res['base_corpus_dir']}:")
            for dom, info in res["domains"].items():
                print(f"    - till-{dom:<8}: corpus={info['corpus_dir']} (Domain ID: {info['domain_id']})")

    elif args.command == "till-train":
        is_json = getattr(args, "json", False)
        if not is_json:
            print(f"[*] Starting till-{args.domain} distillation and Bayesian CTS archiving...")
        report = train_till_domain(
            domain=args.domain,
            corpus_dir=args.corpus,
            output_unp_path=args.output,
            model_name=args.model,
            train_steps=args.train_steps,
            verbose=not is_json,
        )
        if is_json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.format_table())

    elif args.command in ("till-guard-offload", "till-offload"):
        is_json = getattr(args, "json", False)
        guardian = TillDiskGuardian(manifest_path=args.manifest)
        telem = guardian.check_disk()
        if not is_json:
            print(
                f"[DiskGuardian] Partition Telemetry: {telem.free_gb} GB free "
                f"({telem.free_percent}%)"
            )
        satisfied = guardian.enforce_space(
            target_dirs=args.target_dirs,
            dry_run=args.dry_run,
            verbose=not is_json,
        )
        post_telem = guardian.check_disk()
        if is_json:
            print(json.dumps({
                "status": "SATISFIED" if satisfied else "OFFLOAD_FAILED",
                "initial_telemetry": telem.to_dict(),
                "post_telemetry": post_telem.to_dict(),
                "dry_run": args.dry_run,
            }, indent=2))
        else:
            if satisfied:
                print(f"[+] Storage threshold satisfied (>= {guardian.min_free_pct}%).")
            else:
                print("[-] Warning: Storage threshold not satisfied.", file=sys.stderr)

    elif args.command in ("archive", "pack"):
        text = args.text
        if args.file:
            p = Path(args.file)
            if not p.exists():
                print(f"[-] Error: Target file '{args.file}' not found.", file=sys.stderr)
                sys.exit(1)
            text = p.read_text(encoding="utf-8")
        elif not text and not sys.stdin.isatty():
            text = sys.stdin.read()

        if not text:
            print("[-] Error: No text supplied for .unp archive.", file=sys.stderr)
            sys.exit(1)

        domain_choice = "auto" if getattr(args, "auto_domain", False) else (args.domain or 0)
        cache_mix = getattr(args, "cache_mix", False)
        bigram_mix = getattr(args, "bigram_mix", False)
        bayes_mix = getattr(args, "bayes_mix", False)
        print(f"[*] Packing into .unp container using model: {args.model} "
              f"(domain: {domain_choice}, max_ctx: {args.max_ctx_len}, "
              f"cache_mix: {cache_mix}, bigram_mix: {bigram_mix}, bayes_mix: {bayes_mix})...")
        codec = LLMArithmeticCodec(model_name=args.model)
        container_bytes, meta, stats = codec.compress_to_container(
            text=text,
            max_ctx_len=args.max_ctx_len,
            verify_hash=True,
            domain_id=domain_choice,
            enable_cache_mixture=cache_mix,
            enable_bigram_cache=bigram_mix,
            enable_bayes_mixer=bayes_mix,
        )

        out_path = Path(args.output) if args.output else Path(f"{args.file or 'payload'}.unp")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(container_bytes)

        dom_name = DOMAIN_PRIORS.get(meta.domain_id, ("unknown", ""))[0]
        cache_mode = "DISABLED"
        if meta.bayes_mixer:
            cache_mode = "BAYESIAN CTS (Bit 3)"
        elif meta.bigram_cache:
            cache_mode = "BIGRAM (Bit 2)"
        elif meta.cache_mixture:
            cache_mode = "UNIGRAM (Bit 1)"

        print("\n=== .unp Binary Archive Created ===")
        print(f"Destination       : {out_path} ({len(container_bytes)} bytes)")
        print(f"Original Text     : {stats.original_bytes} bytes ({stats.num_tokens} tokens)")
        print(f"Bits per Byte     : {stats.bits_per_byte} bpb")
        print(f"Compression Ratio : {stats.compression_ratio}:1 ({stats.savings_percent}% reduction)")
        print(f"Domain Prior      : ID {meta.domain_id} ({dom_name})")
        print(f"Cache Mixture     : {cache_mode}")
        print(f"Header SHA256 Pre : {meta.sha256_prefix_hex}")

    elif args.command in ("unarchive", "unpack"):
        p = Path(args.archive_file)
        if not p.exists():
            print(f"[-] Error: Archive '{args.archive_file}' not found.", file=sys.stderr)
            sys.exit(1)

        container_bytes = p.read_bytes()
        print(f"[*] Unpacking .unp container from {p} (model: {args.model})...")
        codec = LLMArithmeticCodec(model_name=args.model)
        t0 = time.time()
        text, meta, is_valid = codec.decompress_from_container(container_bytes)
        elapsed = time.time() - t0

        dom_name = DOMAIN_PRIORS.get(meta.domain_id, ("unknown", ""))[0]
        cache_mode = "DISABLED"
        if meta.bayes_mixer:
            cache_mode = "BAYESIAN CTS (Bit 3)"
        elif meta.bigram_cache:
            cache_mode = "BIGRAM (Bit 2)"
        elif meta.cache_mixture:
            cache_mode = "UNIGRAM (Bit 1)"

        print(f"\n=== .unp Archive Decompression Verified ===")
        print(f"Tokens Decoded    : {meta.num_tokens} tokens in {elapsed:.2f}s "
              f"({meta.num_tokens / max(elapsed, 0.001):.1f} tok/s)")
        print(f"Context Stride    : {meta.max_ctx_len}")
        print(f"Domain Prior      : ID {meta.domain_id} ({dom_name})")
        print(f"Cache Mixture     : {cache_mode}")
        print(f"SHA-256 Checksum  : {'VERIFIED MATCH' if is_valid else 'CHECKSUM MISMATCH'}")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(text, encoding="utf-8")
            print(f"[+] Reconstructed plaintext saved to {out_p}")
        else:
            print("\n--- Plaintext Output ---")
            print(text)

    elif args.command in ("compress", "p2-compress"):
        text = args.text
        if args.file:
            text = Path(args.file).read_text(encoding="utf-8")
        elif not text and not sys.stdin.isatty():
            text = sys.stdin.read()

        if not text:
            print("[-] Error: No text supplied for compression.", file=sys.stderr)
            sys.exit(1)

        cache_mix = getattr(args, "cache_mix", False)
        bigram_mix = getattr(args, "bigram_mix", False)
        bayes_mix = getattr(args, "bayes_mix", False)
        print(f"[*] Initializing Paradigm 2 Arithmetic Codec (model: {args.model}, "
              f"cache_mix: {cache_mix}, bigram_mix: {bigram_mix}, bayes_mix: {bayes_mix})...")
        codec = LLMArithmeticCodec(model_name=args.model)
        payload, n_tokens, stats = codec.compress(
            text,
            max_ctx_len=args.max_ctx_len,
            enable_cache_mixture=cache_mix,
            enable_bigram_cache=bigram_mix,
            enable_bayes_mixer=bayes_mix,
        )

        print("\n=== [Paradigm 2] Lossless Arithmetic Compression Results ===")
        print(f"Original Size     : {stats.original_bytes} bytes ({stats.num_tokens} tokens)")
        print(f"Compressed Size   : {stats.compressed_bytes} bytes")
        print(f"Bits per Byte     : {stats.bits_per_byte} bpb")
        print(f"Compression Ratio : {stats.compression_ratio}:1 ({stats.savings_percent}% savings)")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_bytes(payload)
            print(f"[+] Written {len(payload)} compressed bytes to {out_p}")
        else:
            b64_str = base64.b64encode(payload).decode("ascii")
            print(f"Base64 Payload    : {b64_str}")
            print(f"Tokens Flag       : --tokens {n_tokens}")

    elif args.command in ("decompress", "p2-decompress"):
        payload_p = Path(args.payload_file)
        if not payload_p.exists():
            print(f"[-] Error: Payload file '{args.payload_file}' not found.", file=sys.stderr)
            sys.exit(1)

        cache_mix = getattr(args, "cache_mix", False)
        bigram_mix = getattr(args, "bigram_mix", False)
        bayes_mix = getattr(args, "bayes_mix", False)
        payload_bytes = payload_p.read_bytes()
        print(f"[*] Initializing Paradigm 2 Arithmetic Codec (model: {args.model}, "
              f"cache_mix: {cache_mix}, bigram_mix: {bigram_mix}, bayes_mix: {bayes_mix})...")
        codec = LLMArithmeticCodec(model_name=args.model)
        print(f"[*] Decompressing {len(payload_bytes)} bytes into {args.tokens} tokens...")
        t0 = time.time()
        reconstructed = codec.decompress(
            payload_bytes,
            num_tokens=args.tokens,
            max_ctx_len=args.max_ctx_len,
            enable_cache_mixture=cache_mix,
            enable_bigram_cache=bigram_mix,
            enable_bayes_mixer=bayes_mix,
        )
        elapsed = time.time() - t0

        print("\n=== [Paradigm 2] Lossless Reconstructed Text ===")
        print(reconstructed)
        print(f"\n[+] Decompressed in {elapsed:.2f}s ({args.tokens / max(elapsed, 0.001):.1f} tok/s)")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(reconstructed, encoding="utf-8")
            print(f"[+] Saved reconstructed text to {out_p}")

    elif args.command in ("manifold", "p2-manifold"):
        print(f"[*] Initializing Paradigm 2 Manifold Decompressor (model: {args.model})...")
        codec = LLMArithmeticCodec(model_name=args.model)
        print(f"[*] Decompressing seed '{args.seed}' across model manifold ({args.tokens} tokens)...")
        t0 = time.time()
        unpacked = codec.decompress_seed(args.seed, num_tokens=args.tokens)
        elapsed = time.time() - t0

        print("\n=== [Paradigm 2] Manifold Generative Decompression Result ===")
        print(unpacked)
        print(f"\n[+] Generated in {elapsed:.2f}s")

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(unpacked, encoding="utf-8")
            print(f"[+] Saved manifold text to {out_p}")

    elif args.command == "benchmark":
        print(f"[*] Running unplow Decompression Benchmark (model: {args.model}, runs: {args.runs})...")
        codec = LLMArithmeticCodec(model_name=args.model)
        sample = (
            "Claude Shannon proved that the optimal code length for a symbol given prior "
            "context under an autoregressive model is -log2 P(x_t | x_<t) bits. Minimizing "
            "cross-entropy loss during pre-training is mathematically identical to minimizing "
            "the expected bit-length required to losslessly encode the training dataset."
        )
        print(f"[*] Sample payload: {len(sample.encode('utf-8'))} bytes")

        times_comp = []
        times_decomp = []
        for r in range(args.runs):
            t0 = time.time()
            payload, n_tokens, stats = codec.compress(sample)
            times_comp.append(time.time() - t0)

            t1 = time.time()
            recon = codec.decompress(payload, num_tokens=n_tokens)
            times_decomp.append(time.time() - t1)
            assert recon == sample, "Benchmark exact match assertion failed!"

        telem = get_telemetry_snapshot()
        avg_comp = sum(times_comp) / len(times_comp)
        avg_decomp = sum(times_decomp) / len(times_decomp)

        print("\n=== unplow Benchmark Summary ===")
        print(f"Model Identifier      : {codec.model_path}")
        print(f"Original Text Size    : {stats.original_bytes} bytes ({stats.num_tokens} tokens)")
        print(f"Compressed Payload    : {stats.compressed_bytes} bytes")
        print(f"Bits per Byte (bpb)   : {stats.bits_per_byte} bpb")
        print(f"Compression Ratio     : {stats.compression_ratio}:1 ({stats.savings_percent}% savings)")
        print(f"Avg Compression Time  : {avg_comp:.3f}s ({stats.num_tokens / avg_comp:.1f} tok/s)")
        print(f"Avg Decompression Time: {avg_decomp:.3f}s ({stats.num_tokens / avg_decomp:.1f} tok/s)")
        print(f"Process Resident RSS  : {telem['rss_mb']} MB")

    elif args.command == "models":
        print("[*] Auditing local models available on the system...")
        models = audit_system_models()
        print("\n=== Local Model Substrates & Coordinates ===")
        for m in models:
            m_type = m.get("type", "unknown")
            name = m.get("name", "unnamed")
            status = m.get("status", "UNKNOWN")
            path = m.get("path") or m.get("active_link", "")
            size = f"{m['size_mb']} MB" if "size_mb" in m else m.get("parameters", "")
            print(f"- [{status}] {name} ({m_type}, {size}) -> {path}")

    elif args.command == "status":
        queue = FileTaskQueue()
        counts = queue.get_counts()
        telem = get_telemetry_snapshot()
        hb_file = Path("logs/heartbeat/unplow.json")
        hb_data = {}
        if hb_file.exists():
            try:
                hb_data = json.loads(hb_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        print("=== unplow System Status ===")
        print(f"Hostname         : {telem['hostname']}")
        print(f"Current PID      : {telem['pid']}")
        print(f"Current RSS      : {telem['rss_mb']} MB")
        print(f"Task Queue       : incoming={counts['incoming']}, processing={counts['processing']}, "
              f"done={counts['done']}, failed={counts['failed']}")
        if hb_data:
            print(f"Daemon Heartbeat : status={hb_data.get('status')}, "
                  f"last_seen={hb_data.get('timestamp')}, "
                  f"processed={hb_data.get('processed_tasks')}, "
                  f"failed={hb_data.get('failed_tasks')}")
        else:
            print("Daemon Heartbeat : No active worker heartbeat recorded.")

    elif args.command == "enqueue":
        queue = FileTaskQueue()
        extra = {}
        if args.extra:
            try:
                extra = json.loads(args.extra)
            except json.JSONDecodeError as e:
                print(f"[-] Invalid JSON for --extra: {e}", file=sys.stderr)
                sys.exit(1)
        task = queue.enqueue(action=args.action, payload=args.payload, extra=extra)
        print(f"[+] Task enqueued: {task.task_id} (action: {task.action})")

    elif args.command == "daemon":
        print(f"[*] Starting unplow DecompressionWorker daemon (interval={args.interval}s, model={args.model})...")
        worker = DecompressionWorker(model_name=args.model)
        worker.run_loop(interval=args.interval, max_iterations=args.max_iterations)

    elif args.command == "forensics":
        forensics = ForensicAuditLogger()
        entries = forensics.get_recent_entries(limit=args.limit)
        print(f"=== Recent Forensic Ledger Entries ({len(entries)} items) ===")
        for entry in entries:
            ts = entry.get("timestamp_utc", "")
            action = entry.get("action", "")
            task_id = entry.get("task_id", "")
            phase = entry.get("phase", "")
            print(f"[{ts}] [{phase}] task={task_id} action={action} details={entry.get('details')}")


if __name__ == "__main__":
    main()

# end of file: unplow/src/unplow/cli.py
