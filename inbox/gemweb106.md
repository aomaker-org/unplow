# Path: unplow/inbox/gemweb106.md
# Direct Preference Optimization (DPO) Triplet Generation
# Max Column: 80 Columns

To format your data for Direct Preference Optimization (DPO), each training instance must consist of a triplet: `(prompt, chosen, rejected)`.

In systems code generation, a clear signal hierarchy exists:

1. **`chosen` (Positive):** Must strictly pass both `gcc` compilation and 100% of the Unity test harness assertions.
2. **`rejected` (Negative):** A candidate that **failed either compilation or runtime assertions**, but is **not empty or trivial noise**. (Prioritizing runtime/assertion failures over syntax errors provides a stronger gradient for code generation, as the student learns semantic correctness rather than basic token validity).

---

### 1. Updated Execution Return Payload

Modify `_run_unity_test_sync` so it returns the sanitized code alongside the error category (`compile_error`, `runtime_error`, or `pass`):

```python
def _run_unity_test_sync(
    task_dict: Dict[str, Any],
    candidate_c_code: str,
    compile_timeout: float = 6.0,
    run_timeout: float = 3.0,
) -> Tuple[bool, str, str]:
    """
    Returns:
        (passed: bool, clean_code: str, error_detail: str)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Strip markdown fences
        clean_code = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", candidate_c_code, flags=re.MULTILINE)
        clean_code = re.sub(r"^```\s*$", "", clean_code, flags=re.MULTILINE).strip()

        if not clean_code:
            return False, "", "empty_generation"

        impl_path = tmp / "solution.c"
        impl_path.write_text(clean_code, encoding="utf-8")

        header_path = tmp / task_dict["header_filename"]
        header_path.write_text(task_dict["declarations"], encoding="utf-8")

        test_path = tmp / task_dict["test_filename"]
        test_path.write_text(task_dict["test_harness"], encoding="utf-8")

        has_unity_c = False
        for fname, content in task_dict.get("unity_support", {}).items():
            fpath = tmp / fname
            fpath.write_text(content, encoding="utf-8")
            if fname == "unity.c":
                has_unity_c = True

        bin_path = tmp / "test_runner"

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
            c_res = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=compile_timeout)
            if c_res.returncode != 0:
                return False, clean_code, f"compile_error: {c_res.stderr.strip()[:160]}"
        except subprocess.TimeoutExpired:
            return False, clean_code, "compile_timeout"

        try:
            r_res = subprocess.run([str(bin_path)], capture_output=True, text=True, timeout=run_timeout)
            if r_res.returncode == 0:
                return True, clean_code, "pass"
            return False, clean_code, f"assertion_failure: {r_res.stdout.strip()[:160]}"
        except subprocess.TimeoutExpired:
            return False, clean_code, "execution_timeout"
        except Exception as e:
            return False, clean_code, f"runtime_error: {str(e)}"

```

---

### 2. Refactored `process_task` for DPO Pair Assembly

Instead of returning immediately on the first pass or failure, generate candidates across attempts to locate **both** a valid solution (`chosen`) and a high-signal invalid attempt (`rejected`).

```python
async def process_task_dpo(
    task: Dict[str, Any],
    client: httpx.AsyncClient,
    executor: ProcessPoolExecutor,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    task_id = task["task_id"]

    chosen_candidate: Optional[str] = None
    rejected_candidates: List[Tuple[str, str]] = []  # (code, error_type)

    for attempt in range(1, args.max_attempts + 1):
        # Escalate temperature to sample alternative paths
        temp = 0.2 if attempt == 1 else (0.5 if attempt == 2 else 0.8)

        async with semaphore:
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

        passed, clean_code, status = await loop.run_in_executor(
            executor,
            _run_unity_test_sync,
            task,
            completion,
        )

        if passed and not chosen_candidate:
            print(f"[✓] {task_id}: Found CHOSEN candidate (attempt {attempt})")
            chosen_candidate = clean_code
        elif not passed and clean_code:
            print(f"[-] {task_id}: Recorded REJECTED candidate (attempt {attempt}) -> {status}")
            rejected_candidates.append((clean_code, status))

        # Early exit if we already have both a winner and at least one negative sample
        if chosen_candidate and rejected_candidates:
            break

    # DPO requires both a positive and a negative sample for the exact same prompt
    if chosen_candidate and rejected_candidates:
        # Prioritize runtime assertion failures over compile errors for better DPO signal
        assertion_fails = [c for c, s in rejected_candidates if "assertion_failure" in s]
        selected_rejected = assertion_fails[0] if assertion_fails else rejected_candidates[0][0]

        return {
            "task_id": task_id,
            "slug": task.get("slug", ""),
            "domain": "c",
            "prompt": task["prompt"],
            "chosen": chosen_candidate,
            "rejected": selected_rejected,
            "total_attempts": attempt
        }

    return None

```

---

### 3. Output Schema for DPO Training (Hugging Face / TRL Compatible)

The records saved to `dpo_manifest.json` map directly to standard DPO trainers (such as Hugging Face TRL’s `DPOTrainer`):

```json
[
  {
    "task_id": "exercism_c_circular_buffer",
    "slug": "circular-buffer",
    "domain": "c",
    "prompt": "/* Task: circular-buffer */\n/* Specification:\n...",
    "chosen": "#include \"circular_buffer.h\"\n#include <stdlib.h>\n...",
    "rejected": "#include \"circular_buffer.h\"\n// Failed buffer wrap-around assertion\n...",
    "total_attempts": 2
  }
]

```

---

### 4. Direct DPO Training Script Using TRL

Train your compact destination model on these pairs using this lightweight fine-tuning script:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

def train_dpo(
    dpo_data_path: str,
    model_name: str,
    output_dir: str = "./dpo_destination_model"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the generated JSON file
    dataset = load_dataset("json", data_files=dpo_data_path, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Implicitly freezes an in-memory copy of 'model' as reference
        args=training_args,
        beta=0.1,        # DPO temperature parameter (higher = stricter KL constraint)
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=1024,
        max_length=2048,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

```

<!-- end of file: unplow/inbox/gemweb106.md -->
