# Path: unplow/inbox/gemweb105.md
# Batch Distillation Runner with Decoupled Process Pool
# Max Column: 80 Columns

Here is an asynchronous, worker-bounded batch pipeline designed to query a local
instance of `Qwen/Qwen2.5-Coder` (via its OpenAI-compatible completions API
endpoint, matching local runners like `vLLM`, `llama.cpp` server, or `Ollama`)
and evaluate generated candidate solutions in parallel sub-processes.

The pipeline applies an **async inference pool** decoupled from a **process
execution pool**, so heavy C compilation (`gcc`) and execution runs do not block
network I/O or token streaming.

```python
#!/usr/bin/env python3
"""
batch_distill_runner.py

Async/parallel execution runner for Exercism C problems:
1. Batches queries to an OpenAI-compatible Qwen2.5-Coder endpoint.
2. Isolates compilation (gcc) and Unity harness execution in a ProcessPoolExecutor.
3. Collects passing (prompt, completion) pairs additively into a training manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ---------------------------------------------------------------------------
# Sandbox Worker: Compilation & Unity Test Harness Execution
# ---------------------------------------------------------------------------

def _run_unity_test_sync(
    task_dict: Dict[str, Any],
    candidate_c_code: str,
    compile_timeout: float = 6.0,
    run_timeout: float = 3.0,
) -> Tuple[bool, str]:
    """
    Synchronous worker executed in a separate process.
    Compiles candidate code against Unity harness and executes binary.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Sanitize markdown code fences if the model emitted them
        clean_code = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", candidate_c_code, flags=re.MULTILINE)
        clean_code = re.sub(r"^```\s*$", "", clean_code, flags=re.MULTILINE).strip()

        # 1. Write student candidate solution
        impl_path = tmp / "solution.c"
        impl_path.write_text(clean_code, encoding="utf-8")

        # 2. Write required interface header
        header_path = tmp / task_dict["header_filename"]
        header_path.write_text(task_dict["declarations"], encoding="utf-8")

        # 3. Write test harness and Unity framework sources
        test_path = tmp / task_dict["test_filename"]
        test_path.write_text(task_dict["test_harness"], encoding="utf-8")

        has_unity_c = False
        for fname, content in task_dict.get("unity_support", {}).items():
            fpath = tmp / fname
            fpath.write_text(content, encoding="utf-8")
            if fname == "unity.c":
                has_unity_c = True

        bin_path = tmp / "test_runner"

        # 4. Construct compilation command
        compile_cmd = [
            "gcc",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            str(impl_path),
            str(test_path),
        ]
        if has_unity_c:
            compile_cmd.append(str(tmp / "unity.c"))

        compile_cmd.extend(["-I", str(tmp), "-o", str(bin_path)])

        try:
            c_res = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=compile_timeout,
            )
            if c_res.returncode != 0:
                return False, f"Compile Error: {c_res.stderr.strip()[:200]}"
        except subprocess.TimeoutExpired:
            return False, "Compile Timeout"

        # 5. Run compiled test binary
        try:
            r_res = subprocess.run(
                [str(bin_path)],
                capture_output=True,
                text=True,
                timeout=run_timeout,
            )
            if r_res.returncode == 0:
                return True, clean_code
            return False, f"Assertion Failure: {r_res.stdout.strip()[:200]}"
        except subprocess.TimeoutExpired:
            return False, "Execution Timeout"
        except Exception as e:
            return False, f"Runtime Error: {str(e)}"


# ---------------------------------------------------------------------------
# Async Generation & Batch Coordination
# ---------------------------------------------------------------------------

async def generate_candidate(
    client: httpx.AsyncClient,
    api_base: str,
    model_name: str,
    prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 1024,
) -> Optional[str]:
    """Issues an async completion request to the OpenAI-compatible engine."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "stop": ["/* Task:", "\n\n// Task:", "\nint main()"],
    }
    try:
        resp = await client.post(
            f"{api_base}/completions",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["text"]
    except Exception as exc:
        print(f"[!] API call failed: {exc}")
        return None


async def process_task(
    task: Dict[str, Any],
    client: httpx.AsyncClient,
    executor: ProcessPoolExecutor,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Generates and evaluates solutions up to max_attempts per problem."""
    loop = asyncio.get_running_loop()
    task_id = task["task_id"]

    for attempt in range(1, args.max_attempts + 1):
        async with semaphore:
            temp = 0.2 if attempt == 1 else 0.7  # Greedy first, explore later
            completion = await generate_candidate(
                client=client,
                api_base=args.api_base,
                model_name=args.model,
                prompt=task["prompt"],
                temperature=temp,
                max_tokens=args.max_tokens,
            )

        if not completion:
            continue

        # Offload compilation & execution to the ProcessPool
        passed, result_or_err = await loop.run_in_executor(
            executor,
            _run_unity_test_sync,
            task,
            completion,
        )

        if passed:
            print(f"[✓] {task_id} passed on attempt {attempt}")
            return {
                "task_id": task_id,
                "slug": task.get("slug", ""),
                "domain": "c",
                "prompt": task["prompt"],
                "completion": result_or_err,
                "attempts_used": attempt,
            }
        else:
            print(f"[-] {task_id} attempt {attempt} failed: {result_or_err}")

    return None


async def run_pipeline(args: argparse.Namespace):
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        tasks: List[Dict[str, Any]] = json.load(f)

    print(f"[*] Loaded {len(tasks)} tasks from {manifest_path}")

    # Set up bounded worker resources
    semaphore = asyncio.Semaphore(args.concurrency)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    passed_records: List[Dict[str, Any]] = []

    # Open persistent HTTP client and process execution pool
    limits = httpx.Limits(max_keepalive_connections=args.concurrency, max_connections=args.concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            coros = [
                process_task(t, client, executor, semaphore, args)
                for t in tasks
            ]

            # Stream tasks as they complete and flush output incrementally
            for fut in asyncio.as_completed(coros):
                result = await fut
                if result:
                    passed_records.append(result)
                    with open(output_path, "w", encoding="utf-8") as out_f:
                        json.dump(passed_records, out_f, indent=2)

    yield_rate = (len(passed_records) / len(tasks)) * 100 if tasks else 0.0
    print(
        f"\n[DONE] Saved {len(passed_records)} verified training pairs to '{output_path}' "
        f"({yield_rate:.1f}% yield)."
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Async Exercism C verification runner for Qwen2.5-Coder.")
    parser.add_argument("--manifest", type=str, default="exercism_c_problems.json", help="Path to input tasks JSON")
    parser.add_argument("--output", type=str, default="workspace/till_corpus/exercism_c_verified.json", help="Output JSON path")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible server URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Target model identifier")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent API calls")
    parser.add_argument("--workers", type=int, default=4, help="Max compilation processes (CPU cores)")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max generation retries per problem")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max completion tokens")
    args = parser.parse_args()

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()

```

---

### Key Operational Characteristics

1. **Decoupled Workloads:**
* Inference uses `httpx.AsyncClient` throttled by
  `asyncio.Semaphore(args.concurrency)`, preventing connection drops or
  GPU/server queue overruns.
* Verification uses
  `concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)`, isolating
  compilation timeouts and crash isolation away from the main event loop.


2. **Adaptive Temperature Scheduling:**
* Attempt 1 defaults to $\tau = 0.20$ to exploit low-entropy, high-likelihood
  routines.
* If the first attempt fails compilation or an assertion, attempt 2 and 3 bump
  temperature to $\tau = 0.70$ to diverge out of bad parameter basins.


3. **Additive Manifest Flushes:**
* The file at `--output` updates after every passed verification via
  `asyncio.as_completed()`, so progress is preserved immediately even if a batch
  run is terminated early.

<!-- end of file: unplow/inbox/gemweb105.md -->
