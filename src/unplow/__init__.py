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

from .auditor import (
    LayerAuditReport,
    ModelInternalAuditor,
)
from .bundler import (
    BundleManifestReport,
    BundledFileEntry,
    WorkspaceBundler,
)
from .cache_coder import (
    FLAG_CACHE_MIXTURE,
    OnlinePrefixCacheCDF,
    SurprisalProfileReport,
    TokenSurprisalEntry,
    profile_sequence_surprisal,
)
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
    FLAG_BAYES_MIXER,
    FLAG_BIGRAM_CACHE,
    FLAG_CACHE_MIXTURE,
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
from .ctw_coder import (
    BayesTrajectoryStep,
    BayesianCTSMixerCDF,
)
from .forensics import (
    ForensicAuditLogger,
    compute_file_sha256,
    compute_str_sha256,
    get_telemetry_snapshot,
)
from .gguf_inspector import (
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    GGUFBinaryInspector,
    GGUFHeaderSummary,
    GGUFTensorInfo,
    SubstrateDeltaReport,
    build_synthetic_gguf_v3_bytes,
)
from .mechanistic import (
    MechanisticWeightInspector,
    NeuronMemoryTrace,
)
from .models import (
    DeterministicMockCausalLM,
    DeterministicMockTokenizer,
    MockByteTokenizer,
    MockCausalLM,
    audit_system_models,
    load_causal_lm,
    load_local_model,
    resolve_model_path,
)
from .paradigm1 import (
    ExtractedRecord,
    WeightDecompressor,
)
from .paradigm2 import (
    CompressionStats,
    ContainerResult,
    DecompressResult,
    LLMArithmeticCodec,
    MAX_VAL,
    PRECISION_BITS,
    TOTAL_FREQ,
)
from .eval import (
    FunctionalCorrectnessRunner,
    FunctionalTestResult,
    PerplexityReport,
    PerplexityTracker,
    ReplowEvaluationSummary,
    evaluate_artifact,
    format_summary_table,
)
from .queue import (
    FileTaskQueue,
    Task,
    TaskStatus,
)
from .replow import (
    ReplowArtifact,
    ReplowEngine,
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
from .steering import (
    FoldedCircuitTrace,
    MechanisticSteeringEngine,
    SteeredBasinRecord,
)
from .subspace_coder import (
    Order1BigramCacheCDF,
    SVDSubspaceSteeringEngine,
    SubspaceSteeringRecord,
)
from .task_vectors import (
    CrossSubstrateTaskVectorEngine,
    TaskVectorSpectrumReport,
    TaskVectorSteeredRecord,
)
from .till_guard import (
    DiskTelemetry,
    TILL_DOMAINS,
    TillDiskGuardian,
    TillTrainReport,
    init_till_staging,
    train_till_domain,
)
from .worker import (
    DecompressionWorker,
)

__version__ = "0.11.0"

__all__ = [
    "AdditiveFileHandler",
    "AirGapViolationError",
    "AssociativeCircuitTrace",
    "BayesTrajectoryStep",
    "BayesianCTSMixerCDF",
    "BundleManifestReport",
    "BundledFileEntry",
    "CANONICAL_ROOT_FILES",
    "CANONICAL_SOURCE_DIRS",
    "CompressionStats",
    "ContainerResult",
    "CrossSubstrateTaskVectorEngine",
    "DOMAIN_NAME_TO_ID",
    "DOMAIN_PRIORS",
    "DecompressResult",
    "DecompressionWorker",
    "DeterministicMockCausalLM",
    "DeterministicMockTokenizer",
    "DiskTelemetry",
    "ExtractedRecord",
    "FLAG_BAYES_MIXER",
    "FLAG_BIGRAM_CACHE",
    "FLAG_CACHE_MIXTURE",
    "FLAG_LOSSLESS_VERIFIED",
    "FLAG_MANIFOLD_SEED",
    "FileTaskQueue",
    "FoldedCircuitTrace",
    "ForensicAuditLogger",
    "FunctionalCorrectnessRunner",
    "FunctionalTestResult",
    "GGML_TYPE_F16",
    "GGML_TYPE_F32",
    "GGML_TYPE_Q4_0",
    "GGML_TYPE_Q8_0",
    "GGUFBinaryInspector",
    "GGUFHeaderSummary",
    "GGUFTensorInfo",
    "GitDestructiveOperationError",
    "GitSafetyGuard",
    "LLMArithmeticCodec",
    "LayerAuditReport",
    "MAX_VAL",
    "MechanisticSteeringEngine",
    "MechanisticWeightInspector",
    "MockByteTokenizer",
    "MockCausalLM",
    "ModelInternalAuditor",
    "NeuronMemoryTrace",
    "OnlinePrefixCacheCDF",
    "Order1BigramCacheCDF",
    "PRECISION_BITS",
    "PerplexityReport",
    "PerplexityTracker",
    "ReplowArtifact",
    "ReplowEngine",
    "ReplowEvaluationSummary",
    "SVDSubspaceSteeringEngine",
    "SteeredBasinRecord",
    "SubspaceSteeringRecord",
    "SubstrateDeltaReport",
    "SurprisalProfileReport",
    "SwiGLUCircuitDecompressor",
    "TILL_DOMAINS",
    "TOTAL_FREQ",
    "Task",
    "evaluate_artifact",
    "format_summary_table",
    "TaskStatus",
    "TaskVectorSpectrumReport",
    "TaskVectorSteeredRecord",
    "TillDiskGuardian",
    "TillTrainReport",
    "TokenSurprisalEntry",
    "UNP_HEADER_FORMAT",
    "UNP_HEADER_SIZE",
    "UNP_MAGIC",
    "UNP_VERSION",
    "UnpArchiveMetadata",
    "WeightDecompressor",
    "WorkspaceBundler",
    "audit_system_models",
    "build_synthetic_gguf_v3_bytes",
    "compute_file_sha256",
    "compute_str_sha256",
    "get_telemetry_snapshot",
    "init_till_staging",
    "load_causal_lm",
    "load_local_model",
    "logits_to_integer_cdf_fast",
    "pack_flags_with_domain",
    "pack_unp_container",
    "prewarm_kv_cache_for_domain",
    "profile_sequence_surprisal",
    "resolve_domain_id",
    "resolve_model_path",
    "select_optimal_domain_mdl",
    "slice_kv_cache",
    "train_till_domain",
    "unpack_flags_with_domain",
    "unpack_unp_container",
    "validate_additive_artifact_path",
    "validate_local_only_url",
    "verify_unp_checksum",
]

# end of file: unplow/src/unplow/__init__.py
