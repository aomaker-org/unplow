# Path: unplow/src/unplow/__init__.py
"""unplow - Autonomous LLM Decompression Engine & Arithmetic Codec.

Treats large language models as compressed data archives, providing:
1. Lossy Static Weight Decompression (unpacking pre-training training data from theta
   via temperature-decayed basin probing and Carlini et al. entropy calibration).
2. Lossless Autoregressive Arithmetic Compression & Decompression Engine
   (fixed-point integer CDF interval partitioning over next-token probabilities).
3. Generative Manifold Decompression (projecting arbitrary seeds/hashes into text).
4. Atomic file-backed queue scheduling and tamper-evident forensic ledgers.
"""

from .forensics import (
    ForensicAuditLogger,
    compute_file_sha256,
    compute_str_sha256,
    get_telemetry_snapshot,
)
from .models import (
    MockByteTokenizer,
    MockCausalLM,
    audit_system_models,
    load_causal_lm,
    resolve_model_path,
)
from .paradigm1 import (
    ExtractedRecord,
    WeightDecompressor,
)
from .paradigm2 import (
    CompressionStats,
    LLMArithmeticCodec,
    MAX_VAL,
    PRECISION_BITS,
    TOTAL_FREQ,
)
from .queue import (
    FileTaskQueue,
    Task,
    TaskStatus,
)
from .safety import (
    AdditiveFileHandler,
    AirGapViolationError,
    CANONICAL_ROOT_FILES,
    CANONICAL_SOURCE_DIRS,
    GitDestructiveOperationError,
    GitSafetyGuard,
    validate_additive_artifact_path,
    validate_local_only_url,
)
from .worker import (
    DecompressionWorker,
)

__version__ = "0.1.0"

__all__ = [
    "AdditiveFileHandler",
    "AirGapViolationError",
    "CANONICAL_ROOT_FILES",
    "CANONICAL_SOURCE_DIRS",
    "CompressionStats",
    "DecompressionWorker",
    "ExtractedRecord",
    "FileTaskQueue",
    "ForensicAuditLogger",
    "GitDestructiveOperationError",
    "GitSafetyGuard",
    "LLMArithmeticCodec",
    "MAX_VAL",
    "MockByteTokenizer",
    "MockCausalLM",
    "PRECISION_BITS",
    "TOTAL_FREQ",
    "Task",
    "TaskStatus",
    "WeightDecompressor",
    "audit_system_models",
    "compute_file_sha256",
    "compute_str_sha256",
    "get_telemetry_snapshot",
    "load_causal_lm",
    "resolve_model_path",
    "validate_additive_artifact_path",
    "validate_local_only_url",
]

# end of file: unplow/src/unplow/__init__.py
