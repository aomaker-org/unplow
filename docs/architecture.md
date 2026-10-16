# Path: unplow/docs/architecture.md
# unplow Architectural Specification
# Max Column: 80 Columns

## 1. Overview & Purpose

`unplow` is a dedicated research laboratory and runtime engine designed for
treating Large Language Models as compressed data archives and implementing
dual mathematical decompression paradigms:

1. **Lossy Static Weight Decompression (Paradigm 1)**:
   Extracts memorized pre-training data directly from static parameters $\theta$
   using temperature-annealed parameter basin exploration and Carlini et al.
   entropy calibration without requiring any external bitstream.
   - Enhanced via **Mechanistic Logit Lens Inspection** (Geva et al., 2021):
     extracts high-salience vocabulary clusters directly from static MLP
     down-projection value vectors without forward passes.
2. **Lossless Autoregressive Arithmetic Decompression (Paradigm 2)**:
   Pairs the LLM's next-token predictive probability distribution with a
   32-bit fixed-point Arithmetic Coder to achieve lossless text compression/
   decompression and deterministic generative decompression of arbitrary seeds.
   - Enhanced via **Framed `.unp` Binary Container** with 18-byte headers and
     deterministic sliding KV-cache horizons.

---

## 2. Component Topology

```
+-------------------------------------------------------------------------+
|                          CLI Engine (unplow)                            |
| (probe, inspect, compress, decompress, archive, unarchive, manifold)    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|                   Decompression Worker Supervisor                       |
|           - Additive File Handler (Directive 07)                        |
|           - Forensic Audit Logger (Directive 02)                        |
|           - Process Heartbeat & Telemetry (Directive 17)                |
+------------------+----------------------------------+-------------------+
                   |                                  |
                   v                                  v
+------------------+---------------+  +---------------+-------------------+
|        File-Backed Queue         |  |         Decompression Engines     |
|  - records/queue/incoming/       |  |  [Paradigm 1] WeightDecompressor  |
|  - records/queue/processing/     |  |  - Temperature Basin Exploration  |
|  - records/queue/done/           |  |  - Carlini Memorization Scoring   |
|  - records/queue/failed/         |  |  - Mechanistic Logit Lens Prober  |
|  - POSIX fcntl.flock concurrency |  |  [Paradigm 2] LLMArithmeticCodec  |
+----------------------------------+  |  - 32-bit Integer Interval Coding |
                                      |  - Framed .unp Binary Container   |
                                      |  - Sliding-Window KV Truncation   |
                                      +-----------------+-----------------+
                                                        |
                                                        v
                                      +-----------------+-----------------+
                                      |        Model Substrate            |
                                      |  - Local Safetensors Snapshots    |
                                      |  - Deterministic Mock Evaluator   |
                                      |  - Ollama & GGUF Local Endpoints  |
                                      +-----------------------------------+
```

---

## 3. Framed `.unp` Binary Container Layout

Paradigm 2 packages arithmetic bitstreams into self-verifying `.unp` archives
comprising an 18-byte big-endian header followed by raw interval bits:

| Offset | Field | Type | Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0x00` | `magic` | ASCII | 6 B | Magic bytes `UNPLOW` |
| `0x06` | `version` | uint8 | 1 B | Archive format version (`0x01`) |
| `0x07` | `flags` | uint8 | 1 B | `0x01` = Verified, `0x00` = Seed |
| `0x08` | `num_tokens`| uint32 | 4 B | Exact token count to decode |
| `0x0C` | `max_ctx_len`| uint16 | 2 B | KV-cache horizon stride |
| `0x0E` | `sha256_pre`| bytes | 4 B | First 32 bits of plaintext SHA-256 |
| `0x12` | `payload` | bytes | $N$ B | Raw 32-bit arithmetic bitstream |

---

## 4. Queue State Machine & Concurrency

The file-backed queue enforces atomic state transitions using POSIX advisory
file locking (`fcntl.flock(fd, LOCK_EX | LOCK_NB)`):

1. **`incoming/`**: Client or CLI enqueues task JSON payload atomically via
   dot-prefixed temporary file write and `os.replace`.
2. **`processing/`**: Worker acquires exclusive advisory lock, moves task file
   into `processing/`, records start timestamp and process PID.
3. **`done/`**: On successful execution, output result and metrics are stored,
   and file is moved atomically to `done/`.
4. **`failed/`**: On unhandled error, exception trace is recorded, and file is
   moved to `failed/`.

---

## 5. Air-Gap Safety & Forensic Audit Trail

- **Air-Gap Guardrails**: Directives 01 and 18 mandate 100% offline execution.
  Outbound network destinations are rejected at runtime; all operations are
  confined to localhost loopback interfaces and local model weights.
- **Cryptographic Audit Ledger**: Every task dispatch, prompt, compression
  payload, and extracted text is logged to
  `logs/forensics/audit_journal.jsonl` with ISO-8601 UTC microsecond timestamps,
  SHA-256 digests, and VmRSS memory metrics.

<!-- end of file: unplow/docs/architecture.md -->
