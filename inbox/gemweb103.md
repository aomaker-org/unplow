# Path: unplow/inbox/gemweb103.md
# Systems-Programming Problem Sets and Benchmarks Survey
# Max Column: 80 Columns

For training and distilling compact systems-programming models (targeting C,
C++, Rust, Python, and assembly/low-level tasks), the best problem sets combine
**rigorous unit test harnesses**, **deterministic specifications**, and **varied
algorithmic density**.

Here are the highest-yield open sources and repositories to harvest, categorized
by format:

---

### 1. Curated Execution-Verified Benchmarks (Best for Automated Filtering)

These datasets come pre-packaged with unit tests and assertions, making them
turnkey for closed-loop execution filtering via `gcc`, `rustc`, or `pytest`.

* **HumanEval & MultiPL-E:**
* **What it is:** HumanEval contains 164 hand-written algorithmic problems with
  docstrings and unit tests. **MultiPL-E** translates these into 18+ languages,
  including C, C++, Rust, and Python.
* **Why it fits:** The prompts provide strict function signatures and
  docstrings; the test harnesses execute instantly with minimal boilerplate.
* **Source:** Hugging Face datasets (`nuprl/MultiPL-E`).


* **EvalPlus (HumanEval+ / MBPP+):**
* **What it is:** Significantly expands the test suites of standard HumanEval
  and MBPP with 80× more test cases generated via fuzzing and mutation testing.
* **Why it fits:** Prevents subtle edge-case bugs (e.g., off-by-one errors,
  buffer boundaries, empty inputs) from slipping into your training corpus.
* **Source:** Hugging Face datasets (`evalplus/humanevalplus`).


* **CodeContests (DeepMind):**
* **What it is:** Competitive programming problems gathered from Codeforces,
  Description/In-Out pairs, and massive private test suites. Available in C++,
  Python, and Java.
* **Why it fits:** Excellent for complex state machines, bitwise logic, dynamic
  programming, and low-level data structure manipulation.
* **Source:** Hugging Face datasets (`deepmind/code_contests`).



---

### 2. Systems-Specific Open Source Test Suites

Standard competitive programming benchmarks skew heavily toward math puzzles
rather than systems software (memory management, ring buffers, CRC/hashing,
bitwise manipulation). For systems-level data, extract directly from open-source
unit tests:

* **Kernel & Low-Level Tool Test Benches:**
* **Linux Kernel self-tests (`tools/testing/selftests/`):** Contains standalone
  C files testing memory mapping (`mmap`), ring buffers, signal handling, and
  hardware timer semantics.
* **musl-libc / newlib test suites:** Thousands of minimal, standalone C
  functions testing standard library implementations against strict edge cases.
* **FreeRTOS / Zephyr test cases:** Clean, freestanding C functions testing RTOS
  queue operations, bounded buffers, and hardware abstraction logic.


* **Exercism Open Source Tracks:**
* **What it is:** Exercism's GitHub repositories contain hundreds of progression
  exercises for **C**, **C++**, and **Rust**.
* **Why it fits:** Every exercise repository (e.g., `exercism/c`,
  `exercism/rust`) contains a markdown problem description, a stub file with
  signatures, and a dedicated test suite using standard runners (`Unity` for C,
  `cargo test` for Rust).
* **Source:** GitHub (`[github.com/exercism/c](https://github.com/exercism/c)`,
  `[github.com/exercism/rust](https://github.com/exercism/rust)`).


* **The Algorithms (Open-Source GitHub):**
* **What it is:** Community implementations of standard algorithms across
  Python, C, C++, and Rust (`TheAlgorithms/C`, `TheAlgorithms/Rust`).
* **Why it fits:** The code is explicitly written as standalone modules with
  embedded `main()` assertion tests or doc-tests, making extraction into prompt-
  solution pairs trivial via simple AST/regex slicing.



---

### 3. Machine-Readable Schema Archives

If you want structured datasets ready to parse into JSON manifests:

* **The Stack v2 / StarCoder Data (Filtered by Tests):**
* You can stream `bigcode/the-stack-v2` with filters targeting standalone test
  files (e.g., files matching `*_test.c`, `test_*.py`, `tests/*.rs`).
* Extract pairs where the file contains both the definition and an assertion
  block.


* **Rosetta Code Data:**
* Contains thousands of standard programming tasks implemented across dozens of
  languages, with many tasks focusing on low-level bit operations, endianness,
  and algorithms.
* **Source:** Hugging Face datasets (`huggingface/rosetta_code`) or the GitHub
  mirror (`acmeism/RosettaCodeData`).



---

### Summary Comparison

| Dataset / Source | Primary Languages | Focus Area | Harness Included? |
| --- | --- | --- | --- |
| **MultiPL-E** | C, C++, Rust, Python | Algorithms & Core Logic | Yes (Built-in assertions) |
| **EvalPlus** | Python, C++ | Strict Edge-Case Verification | Yes (Massive fuzzing suite) |
| **Exercism Tracks** | C, Rust, C++, Python | Idiomatic language design & stubs | Yes (Framework unit tests) |
| **The Algorithms** | C, Rust, C++, Python | Data structures, bitwise, math | Mixed (In-file `assert` loops) |
| **musl / selftests** | C | Low-level POSIX, memory, system calls | Yes (Executables returning 0/1) |

---

### Extraction Strategy for Your Pipeline

For a clean start without manual curation:

1. Pull the **MultiPL-E C and Rust subsets** from Hugging Face.
2. Clone **`exercism/c`** and **`exercism/rust`**, running a Python script to
parse the `README.md` into your prompt and the test `.c`/`.rs` file into your
compilation harness.
3. Feed those problem prompts to your input model, verify via your execution
runner, and accumulate your passed pairs into `workspace/till_corpus/`.

<!-- end of file: unplow/inbox/gemweb103.md -->
