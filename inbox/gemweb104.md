# Path: unplow/inbox/gemweb104.md
# Exercism C Track Extraction Harness Specification
# Max Column: 80 Columns

Exercism's C track repository structure stores exercises under
`exercises/practice/<slug>/` (or `exercises/concept/<slug>/`). Each exercise
contains:

* `.docs/instructions.md`: The problem specification and requirements.
* `<slug>.h`: Function declarations and type signatures.
* `test_<slug>.c`: Unity-based test harness invoking the functions.
* `vendor/unity.h` and `vendor/unity.c`: The lightweight C unit test framework.

The script below clones or walks a local clone of `exercism/c`, parses the
instructions, captures the function prototypes from the header file, and bundles
the Unity test code into a standalone JSON problem manifest.

```python
#!/usr/bin/env python3
"""
parse_exercism_c.py

Extracts Exercism C practice exercises into a unified JSON manifest
for prompt-driven LLM distillation and execution-verified test harnesses.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_files_by_pattern(directory: Path, pattern: str) -> List[Path]:
    """Return matching files within a directory non-recursively."""
    return sorted(list(directory.glob(pattern)))


def extract_header_signatures(header_path: Path) -> str:
    """
    Extracts typedefs, structs, and function prototypes from a C header,
    stripping inclusion guards.
    """
    if not header_path.exists():
        return ""

    content = header_path.read_text(encoding="utf-8", errors="replace")

    # Strip include guards (#ifndef / #define / #endif)
    lines = content.splitlines()
    filtered_lines: List[str] = []
    
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#(ifndef|define)\s+[A-Za-z0-9_]+_H\b", stripped):
            continue
        if stripped == "#endif" and any("#ifndef" in l for l in lines):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


def sanitize_instructions(instructions_md: str) -> str:
    """Cleans instructional markdown down to concise functional requirements."""
    # Strip Exercism markdown advisory callouts
    cleaned = re.sub(r"\{~.*?~\}", "", instructions_md, flags=re.DOTALL)
    # Strip HTML comment blocks
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def parse_exercise_directory(exercise_dir: Path, unity_src: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Parses a single practice exercise directory into structured task data."""
    slug = exercise_dir.name
    
    # 1. Locate problem instructions
    instructions_file = exercise_dir / ".docs" / "instructions.md"
    if not instructions_file.exists():
        # Fallback to legacy structure
        instructions_file = exercise_dir / "README.md"
        if not instructions_file.exists():
            return None

    raw_instructions = instructions_file.read_text(encoding="utf-8", errors="replace")
    instructions = sanitize_instructions(raw_instructions)

    # 2. Locate header file for signatures
    headers = find_files_by_pattern(exercise_dir, "*.h")
    header_content = ""
    target_header = None
    for h in headers:
        if "test" not in h.name and "vendor" not in str(h):
            target_header = h
            header_content = extract_header_signatures(h)
            break

    # 3. Locate test harness file
    test_files = find_files_by_pattern(exercise_dir, "test_*.c")
    if not test_files:
        test_files = [f for f in find_files_by_pattern(exercise_dir, "*.c") if "test" in f.name]
    
    if not test_files:
        return None

    test_harness_c = test_files[0].read_text(encoding="utf-8", errors="replace")

    # Construct the base prompt presented to the teacher/student model
    prompt_builder = [
        f"/* Task: {slug} */",
        "/* Specification:",
        instructions,
        "*/\n",
    ]
    if header_content:
        prompt_builder.append("/* Required signatures and interface definitions: */")
        prompt_builder.append(header_content)
        prompt_builder.append("\n/* Implement the required routine(s) below. */\n")

    full_prompt = "\n".join(prompt_builder)

    return {
        "task_id": f"exercism_c_{slug}",
        "slug": slug,
        "domain": "c",
        "prompt": full_prompt,
        "declarations": header_content,
        "header_filename": target_header.name if target_header else f"{slug}.h",
        "test_harness": test_harness_c,
        "test_filename": test_files[0].name,
        "unity_support": unity_src
    }


def main():
    parser = argparse.ArgumentParser(description="Parse Exercism C track into JSON problem sets.")
    parser.add_argument("--repo-dir", type=Path, default=Path("c"), help="Path to local exercism/c clone")
    parser.add_argument("--output", type=Path, default=Path("exercism_c_problems.json"), help="Output JSON path")
    parser.add_argument("--clone", action="store_true", help="Shallow clone repo if directory missing")
    args = parser.parse_args()

    repo_dir: Path = args.repo_dir

    if args.clone and not repo_dir.exists():
        print(f"[*] Shallow cloning exercism/c into {repo_dir}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/exercism/c.git", str(repo_dir)],
            check=True
        )

    if not repo_dir.exists() or not repo_dir.is_dir():
        sys.exit(f"Error: Directory {repo_dir} does not exist. Run with --clone or provide valid path.")

    # Cache Unity test framework files if present in the track
    unity_src = {}
    unity_dir = repo_dir / "exercises" / "shared" / "vendor"
    if not unity_dir.exists():
        # Alternate shared vendor location in older structures
        unity_dir = repo_dir / "vendor"

    for u_file in ["unity.h", "unity.c", "unity_internals.h"]:
        target = unity_dir / u_file
        if target.exists():
            unity_src[u_file] = target.read_text(encoding="utf-8", errors="replace")

    practice_dir = repo_dir / "exercises" / "practice"
    if not practice_dir.exists():
        practice_dir = repo_dir / "exercises"

    exercises = sorted([d for d in practice_dir.iterdir() if d.is_dir()])
    print(f"[*] Processing {len(exercises)} potential exercise directories...")

    manifest: List[Dict[str, Any]] = []
    for ex_dir in exercises:
        parsed = parse_exercise_directory(ex_dir, unity_src)
        if parsed:
            manifest.append(parsed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[✓] Extracted {len(manifest)} valid problem sets into '{args.output}'.")


if __name__ == "__main__":
    main()

```

---

### Integration into the Execution Test Runner

Because Exercism C tests use the Unity framework, compile each generated
solution in a sandbox directory containing the cached `unity.c` and header
stubs:

```python
def compile_and_run_unity_test(task_dict: dict, generated_c_code: str) -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        # 1. Write the model's implementation
        impl_path = tmp / "solution.c"
        impl_path.write_text(generated_c_code)
        
        # 2. Write the required interface header
        header_path = tmp / task_dict["header_filename"]
        header_path.write_text(task_dict["declarations"])

        # 3. Write Unity framework and test harness files
        (tmp / task_dict["test_filename"]).write_text(task_dict["test_harness"])
        for fname, content in task_dict.get("unity_support", {}).items():
            (tmp / fname).write_text(content)

        bin_path = tmp / "test_runner"

        # 4. Compile with gcc
        cmd = [
            "gcc", "-O2", "-Wall", "-Wextra",
            str(impl_path),
            str(tmp / task_dict["test_filename"]),
            str(tmp / "unity.c") if (tmp / "unity.c").exists() else "",
            "-I", str(tmp),
            "-o", str(bin_path)
        ]
        cmd = [c for c in cmd if c]  # prune empty strings

        res = subprocess.run(cmd, capture_output=True, timeout=5)
        if res.returncode != 0:
            return False

        # 5. Execute test runner
        exec_res = subprocess.run([str(bin_path)], capture_output=True, timeout=3)
        return exec_res.returncode == 0

```

<!-- end of file: unplow/inbox/gemweb104.md -->
