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

from .container import unpack_unp_container
from .forensics import ForensicAuditLogger, get_telemetry_snapshot
from .mechanistic import MechanisticWeightInspector
from .models import audit_system_models, resolve_model_path
from .paradigm1 import WeightDecompressor
from .paradigm2 import LLMArithmeticCodec
from .queue import FileTaskQueue
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
    p_insp = subparsers.add_parser("inspect", aliases=["mechanistic"], help="Direct MLP weight inspection via Logit Lens")
    p_insp.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_insp.add_argument("-k", "--neurons", type=int, default=6, help="Top neurons per layer to inspect (default: 6)")
    p_insp.add_argument("--top-tokens", type=int, default=5, help="Top tokens per neuron (default: 5)")
    p_insp.add_argument("--harvest", action="store_true", help="Harvest and display seed prefixes")
    p_insp.add_argument("--seeds", type=int, default=12, help="Max seed prefixes to harvest")

    # Command: compress (Paradigm 2)
    p_comp = subparsers.add_parser("compress", aliases=["p2-compress"], help="Lossless arithmetic compression (Paradigm 2)")
    p_comp.add_argument("text", nargs="?", help="Text to compress (or read from stdin/--file)")
    p_comp.add_argument("-f", "--file", help="File containing text to compress")
    p_comp.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_comp.add_argument("--max-ctx-len", type=int, default=None, help="Context horizon window")
    p_comp.add_argument("-o", "--output", help="Output binary file path")

    # Command: decompress (Paradigm 2)
    p_decomp = subparsers.add_parser("decompress", aliases=["p2-decompress"], help="Lossless arithmetic decompression (Paradigm 2)")
    p_decomp.add_argument("payload_file", help="Path to compressed payload file (.bin)")
    p_decomp.add_argument("-t", "--tokens", type=int, required=True, help="Exact number of tokens in original text")
    p_decomp.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_decomp.add_argument("--max-ctx-len", type=int, default=None, help="Context horizon window")
    p_decomp.add_argument("-o", "--output", help="Output file path for reconstructed text")

    # Command: archive (Paradigm 2 .unp Container)
    p_arch = subparsers.add_parser("archive", aliases=["pack"], help="Pack file/text into self-verifying .unp container")
    p_arch.add_argument("file", nargs="?", help="Path to input text/source file")
    p_arch.add_argument("--text", help="Direct text to pack into container")
    p_arch.add_argument("-o", "--output", help="Output .unp archive path")
    p_arch.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")
    p_arch.add_argument("--max-ctx-len", type=int, default=512, help="Context horizon window (default: 512)")

    # Command: unarchive (Paradigm 2 .unp Container)
    p_unarch = subparsers.add_parser("unarchive", aliases=["unpack"], help="Unpack and verify .unp container")
    p_unarch.add_argument("archive_file", help="Path to .unp archive file")
    p_unarch.add_argument("-o", "--output", help="Output file path for reconstructed text")
    p_unarch.add_argument("-m", "--model", default="auto", help="Model name or local snapshot path")

    # Command: manifold (Paradigm 2)
    p_mani = subparsers.add_parser("manifold", aliases=["p2-manifold"], help="Generative seed decompression (Paradigm 2)")
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
    p_enq.add_argument("action", choices=["probe", "compress", "decompress", "manifold", "archive", "unarchive"], help="Task action")
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
        print(f"[*] Probing {args.samples} weight basins (seq_len={args.seq_len}, explore_steps={args.explore_steps})...")
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

        print(f"[*] Packing into .unp container using model: {args.model} (max_ctx: {args.max_ctx_len})...")
        codec = LLMArithmeticCodec(model_name=args.model)
        container_bytes, meta, stats = codec.compress_to_container(
            text=text,
            max_ctx_len=args.max_ctx_len,
            verify_hash=True,
        )

        out_path = Path(args.output) if args.output else Path(f"{args.file or 'payload'}.unp")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(container_bytes)

        print("\n=== .unp Binary Archive Created ===")
        print(f"Destination       : {out_path} ({len(container_bytes)} bytes)")
        print(f"Original Text     : {stats.original_bytes} bytes ({stats.num_tokens} tokens)")
        print(f"Bits per Byte     : {stats.bits_per_byte} bpb")
        print(f"Compression Ratio : {stats.compression_ratio}:1 ({stats.savings_percent}% reduction)")
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

        print(f"\n=== .unp Archive Decompression Verified ===")
        print(f"Tokens Decoded    : {meta.num_tokens} tokens in {elapsed:.2f}s ({meta.num_tokens / max(elapsed, 0.001):.1f} tok/s)")
        print(f"Context Stride    : {meta.max_ctx_len}")
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

        print(f"[*] Initializing Paradigm 2 Arithmetic Codec (model: {args.model})...")
        codec = LLMArithmeticCodec(model_name=args.model)
        payload, n_tokens, stats = codec.compress(text, max_ctx_len=args.max_ctx_len)

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

        payload_bytes = payload_p.read_bytes()
        print(f"[*] Initializing Paradigm 2 Arithmetic Codec (model: {args.model})...")
        codec = LLMArithmeticCodec(model_name=args.model)
        print(f"[*] Decompressing {len(payload_bytes)} bytes into {args.tokens} tokens...")
        t0 = time.time()
        reconstructed = codec.decompress(payload_bytes, num_tokens=args.tokens, max_ctx_len=args.max_ctx_len)
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
