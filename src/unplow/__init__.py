# Path: unplow/src/unplow/__init__.py
"""unplow - Autonomous LLM Decompression Engine & Arithmetic Codec.

Treats large language models as compressed data archives, providing:
1. Lossy Static Weight Decompression (unpacking pre-training training data from theta
   via temperature-decayed basin probing and Carlini et al. entropy calibration).
2. Lossless Autoregressive Arithmetic Compression & Decompression Engine
   (fixed-point integer CDF interval partitioning over next-token probabilities).
3. Generative Manifold Decompression (projecting arbitrary seeds/hashes into text).
4. Direct Mechanistic Weight Inspection via the Logit Lens (Geva et al., 2021).
5. Self-verifying framed .unp binary container format with bounded KV-cache horizon.
6. Atomic file-backed queue scheduling and tamper-evident forensic ledgers.
"""

from .circuits import (
    AssociativeCircuitTrace,
    SwiGLUCircuitDecompressor,
)
from .conditioning import (
    DOMAIN_NAME_TO_ID,
    DOMAIN_PRIORS,
    logits_to_integer_cdf_fast,
    pack_flags_with_domain,
    prewarm_kv_cache_for_domain,
    resolve_domain_id,
    select_optimal_domain_mdl,
    unpack_flags_with_domain,
)
from .container import (
    FLAG_LOSSLESS_VERIFIED,
    FLAG_MANIFOLD_SEED,
    UNP_HEADER_FORMAT,
    UNP_HEADER_SIZE,
    UNP_MAGIC,
    UNP_VERSION,
    UnpArchiveMetadata,
    pack_unp_container,
    slice_kv_cache,
    unpack_unp_container,
    verify_unp_checksum,
)
from .forensics import (
    ForensicAuditLogger,
    compute_file_sha256,
    compute_str_sha256,
    get_telemetry_snapshot,
)
from .mechanistic import (
    MechanisticWeightInspector,
    NeuronMemoryTrace,
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

__version__ = "0.3.0"

__all__ = [
    "AdditiveFileHandler",
    "AirGapViolationError",
    "AssociativeCircuitTrace",
    "CANONICAL_ROOT_FILES",
    "CANONICAL_SOURCE_DIRS",
    "CompressionStats",
    "DOMAIN_NAME_TO_ID",
    "DOMAIN_PRIORS",
    "DecompressionWorker",
    "ExtractedRecord",
    "FLAG_LOSSLESS_VERIFIED",
    "FLAG_MANIFOLD_SEED",
    "FileTaskQueue",
    "ForensicAuditLogger",
    "GitDestructiveOperationError",
    "GitSafetyGuard",
    "LLMArithmeticCodec",
    "MAX_VAL",
    "MechanisticWeightInspector",
    "MockByteTokenizer",
    "MockCausalLM",
    "NeuronMemoryTrace",
    "PRECISION_BITS",
    "SwiGLUCircuitDecompressor",
    "TOTAL_FREQ",
    "Task",
    "TaskStatus",
    "UNP_HEADER_FORMAT",
    "UNP_HEADER_SIZE",
    "UNP_MAGIC",
    "UNP_VERSION",
    "UnpArchiveMetadata",
    "WeightDecompressor",
    "audit_system_models",
    "compute_file_sha256",
    "compute_str_sha256",
    "get_telemetry_snapshot",
    "load_causal_lm",
    "logits_to_integer_cdf_fast",
    "pack_flags_with_domain",
    "pack_unp_container",
    "prewarm_kv_cache_for_domain",
    "resolve_domain_id",
    "resolve_model_path",
    "select_optimal_domain_mdl",
    "slice_kv_cache",
    "unpack_flags_with_domain",
    "unpack_unp_container",
    "validate_additive_artifact_path",
    "validate_local_only_url",
    "verify_unp_checksum",
]

# end of file: unplow/src/unplow/__init__.py
