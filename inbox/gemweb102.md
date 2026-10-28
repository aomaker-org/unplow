# Path: unplow/inbox/gemweb102.md
# Prompt-Driven Execution Distillation Pipeline
# Max Column: 80 Columns

Transitioning from unconstrained weight decompressing to **prompt-driven
execution distillation** closes the loop on functional correctness. Instead of
sampling arbitrary routines from the source model's weights and checking syntax
post-hoc, you prompt the source (input/teacher) model with concrete programming
tasks, evaluate its output against unit tests or AST verifiers, and supervise
the destination (student) model on the verified solutions.

```
+--------------------------------------------------------------------------+
|                 Closed-Loop Execution Distillation Pipeline              |
|                                                                          |
|  [ Task Seeds / Specs ] ---> [ Input Model (Teacher) ]                   |
|                                      |                                   |
|                                      v Raw Completion                    |
|                        [ Test Harness / Compiler / Val ]                 |
|                                      |                                   |
|                          Pass? ------+------ Fail?                       |
|                            |                   |                         |
|                            v                   v                         |
|                    [ Dataset Cache ]        [ Discard / Retry ]          |
|                            |                                             |
|                            v Supervised Fine-Tuning / DPO                |
|                    [ Destination Model (Student) ]                       |
+--------------------------------------------------------------------------+

```

---

### 1. Generating & Structuring the Problem Bank

To train a compact destination model effectively, your problem inputs need
consistent structural boundaries so the model learns deterministic formatting,
reasoning, and implementation phases.

A minimal schema for problem definitions:

```json
{
  "task_id": "c_crc32_bitwise",
  "domain": "c",
  "prompt": "Implement an optimized, bitwise CRC-32 routine matching the IEEE 802.3 standard polynomial 0xEDB88320. Function signature: uint32_t compute_crc32(const uint8_t *data, size_t length);",
  "test_harness": "#include <assert.h>\n#include <stdint.h>\n#include <stddef.h>\n\nuint32_t compute_crc32(const uint8_t *data, size_t length);\n\nint main() {\n    const uint8_t test[] = \"123456789\";\n    assert(compute_crc32(test, 9) == 0xCBF43926);\n    return 0;\n}"
}

```

---

### 2. Teacher Harvesting Script with Execution Filtering

Using execution feedback (via compilation and unit testing) ensures that 100% of
the training tokens given to the destination model represent working systems
code.

```python
import subprocess
import tempfile
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_c_solution(code_body: str, test_harness: str) -> bool:
    """Compiles and executes the solution with the test harness via clang/gcc."""
    full_source = f"{code_body}\n\n{test_harness}"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "test.c")
        bin_path = os.path.join(tmpdir, "test_bin")
        
        with open(src_path, "w") as f:
            f.write(full_source)
            
        # 1. Compilation check (strict flags)
        compile_res = subprocess.run(
            ["gcc", "-O2", "-Wall", "-Werror", src_path, "-o", bin_path],
            capture_output=True,
            timeout=5
        )
        if compile_res.returncode != 0:
            return False
            
        # 2. Execution / Assertion check
        try:
            exec_res = subprocess.run([bin_path], capture_output=True, timeout=3)
            return exec_res.returncode == 0
        except subprocess.TimeoutExpired:
            return False

def harvest_problem_solutions(
    teacher_model,
    tokenizer,
    problems_file: str,
    output_manifest: str,
    samples_per_task: int = 4
):
    with open(problems_file, "r") as f:
        problems = json.load(f)

    successful_pairs = []

    for item in problems:
        task_id = item["task_id"]
        prompt_text = item["prompt"]
        harness = item["test_harness"]

        formatted_prompt = f"/* Problem: {prompt_text} */\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(teacher_model.device)

        for _ in range(samples_per_task):
            with torch.no_grad():
                output_ids = teacher_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.4,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Strip the prompt to isolate generated code
            generated_code = tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Execution filter: only retain passes
            if evaluate_c_solution(generated_code, harness):
                successful_pairs.append({
                    "task_id": task_id,
                    "prompt": formatted_prompt,
                    "completion": generated_code
                })
                break  # Move to next task once verified solution is captured

    with open(output_manifest, "w") as f:
        json.dump(successful_pairs, f, indent=2)

```

---

### 3. Training the Destination Model

With verified task-solution pairs, you train the destination model using
standard **Causal LM Cross-Entropy with Target-Only Masking** (masking prompt
tokens so the student only incurs loss on predicting the solution):

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

class DistillationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_len: int = 512):
        with open(data_path, "r") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item["prompt"]
        completion = item["completion"]

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        comp_ids = self.tokenizer.encode(completion, add_special_tokens=False) + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + comp_ids
        # Mask out prompt positions (-100 tells PyTorch cross_entropy to ignore)
        labels = [-100] * len(prompt_ids) + comp_ids

        # Truncate
        input_ids = input_ids[:self.max_len]
        labels = labels[:self.max_len]

        # Pad to max_len
        pad_len = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def train_student_epoch(student_model, dataloader, optimizer, device):
    student_model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = student_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

```

---

### 4. Direct Preference Optimization (DPO) Bonus

Because you test the input model's attempts with an execution harness, you will
often obtain **both passed and failed outputs** for the exact same prompt.

Instead of discarding the failures, pair them into a preference dataset:

* `chosen`: Output passing the unit test.
* `rejected`: Output that failed compilation or assertion.

Feeding these pairs into a **DPO (Direct Preference Optimization)** loss
function during a secondary fine-tuning phase sharply suppresses common failure
modes (such as off-by-one pointer errors or unhandled buffer bounds) in the
minimized destination model.

<!-- end of file: unplow/inbox/gemweb102.md -->
