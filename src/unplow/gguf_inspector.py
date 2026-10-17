# Path: unplow/src/unplow/gguf_inspector.py
"""Pure-Python GGUF v2/v3 Binary Parser, Vectorized Block Dequantizer, and Task-Vector Forensics.

Implements:
1. Memory-mapped GGUF v2/v3 binary parsing and tensor indexing (Gerganov, 2023)
   operating 100% offline with zero external C++ dependencies.
2. Vectorized NumPy block dequantizers for F32, F16, Q8_0, and Q4_0 formats
   (Dettmers et al., 2022).
3. Cross-substrate Task-Vector weight divergence forensics (Ilharco et al., 2023)
   comparing dequantized GGUF tensor layers against PyTorch reference weights.
"""

from dataclasses import asdict, dataclass
import logging
import mmap
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

logger = logging.getLogger(__name__)

GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32

# GGML Tensor Type Constants (Gerganov, 2023)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q6_K = 14

GGML_TYPE_NAMES: Dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
}

# (block_elements, bytes_per_block) for directly dequantizable formats
GGML_BLOCK_SPECS: Dict[int, Tuple[int, int]] = {
    GGML_TYPE_F32: (1, 4),
    GGML_TYPE_F16: (1, 2),
    GGML_TYPE_Q4_0: (32, 18),  # 2-byte fp16 scale + 16 bytes (32 x 4-bit nibbles)
    GGML_TYPE_Q8_0: (32, 34),  # 2-byte fp16 scale + 32 bytes (32 x int8 quants)
    GGML_TYPE_Q6_K: (256, 210),  # 128B ql + 64B qh + 16B sc + 2B fp16 d
}


@dataclass
class GGUFTensorInfo:
    """Metadata describing a single tensor stored inside a GGUF container."""

    name: str
    shape: Tuple[int, ...]
    ggml_type: int
    ggml_type_name: str
    relative_offset: int
    num_elements: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GGUFHeaderSummary:
    """Summary of GGUF container header, metadata key-values, and tensor catalog."""

    file_path: str
    file_size_bytes: int
    version: int
    tensor_count: int
    metadata_kv_count: int
    alignment: int
    data_offset: int
    metadata: Dict[str, Any]
    tensors: Dict[str, GGUFTensorInfo]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tensors"] = {k: v.to_dict() for k, v in self.tensors.items()}
        return d


@dataclass
class SubstrateDeltaReport:
    """Task-Vector delta report comparing a dequantized GGUF tensor to a PyTorch matrix."""

    gguf_tensor_name: str
    ggml_type_name: str
    shape: Tuple[int, ...]
    gguf_l2_norm: float
    ref_l2_norm: float
    cosine_similarity: float
    relative_frobenius_delta: float
    top_divergent_columns: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GGUFBinaryInspector:
    """Memory-mapped GGUF v2/v3 parser and vectorized block dequantizer.

    Operates strictly offline via mmap with zero external C++ dependencies.
    """

    def __init__(self, gguf_path: Union[str, Path]):
        self.path = Path(gguf_path)
        if not self.path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")

    @staticmethod
    def _read_string(buf: Union[memoryview, mmap.mmap], offset: int) -> Tuple[str, int]:
        str_len = struct.unpack_from("<Q", buf, offset)[0]
        offset += 8
        raw = bytes(buf[offset : offset + str_len])
        return raw.decode("utf-8", errors="replace"), offset + str_len

    @classmethod
    def _read_value(
        cls, buf: Union[memoryview, mmap.mmap], offset: int, vtype: int
    ) -> Tuple[Any, int]:
        # GGUF metadata value types (0..12)
        scalar_fmt = {
            0: ("<B", 1),   # UINT8
            1: ("<b", 1),   # INT8
            2: ("<H", 2),   # UINT16
            3: ("<h", 2),   # INT16
            4: ("<I", 4),   # UINT32
            5: ("<i", 4),   # INT32
            6: ("<f", 4),   # FLOAT32
            7: ("<?", 1),   # BOOL
            10: ("<Q", 8),  # UINT64
            11: ("<q", 8),  # INT64
            12: ("<d", 8),  # FLOAT64
        }
        if vtype in scalar_fmt:
            fmt, sz = scalar_fmt[vtype]
            val = struct.unpack_from(fmt, buf, offset)[0]
            return val, offset + sz
        if vtype == 8:  # STRING
            return cls._read_string(buf, offset)
        if vtype == 9:  # ARRAY
            elem_type, count = struct.unpack_from("<IQ", buf, offset)
            offset += 12
            # Avoid materializing massive vocabulary string arrays in summary dict
            if count > 64:
                for _ in range(count):
                    _, offset = cls._read_value(buf, offset, elem_type)
                return f"<array type={elem_type} len={count}>", offset
            arr = []
            for _ in range(count):
                item, offset = cls._read_value(buf, offset, elem_type)
                arr.append(item)
            return arr, offset
        raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")

    def parse_header(self) -> GGUFHeaderSummary:
        """Parses the GGUF v2/v3 header, metadata table, and tensor index."""
        file_size = self.path.stat().st_size
        with self.path.open("rb") as f, mmap.mmap(
            f.fileno(), 0, access=mmap.ACCESS_READ
        ) as mm:
            magic = mm[0:4]
            if magic != GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF magic bytes: {magic!r}")

            version, tensor_count, kv_count = struct.unpack_from("<IQQ", mm, 4)
            if version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version: {version}")

            offset = 24
            metadata: Dict[str, Any] = {}
            for _ in range(kv_count):
                key, offset = self._read_string(mm, offset)
                vtype = struct.unpack_from("<I", mm, offset)[0]
                offset += 4
                val, offset = self._read_value(mm, offset, vtype)
                metadata[key] = val

            alignment = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
            tensors: Dict[str, GGUFTensorInfo] = {}

            for _ in range(tensor_count):
                t_name, offset = self._read_string(mm, offset)
                n_dims = struct.unpack_from("<I", mm, offset)[0]
                offset += 4
                dims = []
                n_elems = 1
                for _ in range(n_dims):
                    d = struct.unpack_from("<Q", mm, offset)[0]
                    offset += 8
                    dims.append(int(d))
                    n_elems *= int(d)
                ggml_type, rel_offset = struct.unpack_from("<IQ", mm, offset)
                offset += 12
                tensors[t_name] = GGUFTensorInfo(
                    name=t_name,
                    shape=tuple(dims),
                    ggml_type=int(ggml_type),
                    ggml_type_name=GGML_TYPE_NAMES.get(
                        int(ggml_type), f"TYPE_{ggml_type}"
                    ),
                    relative_offset=int(rel_offset),
                    num_elements=int(n_elems),
                )

            remainder = offset % alignment
            data_offset = offset + ((alignment - remainder) % alignment)

            return GGUFHeaderSummary(
                file_path=str(self.path),
                file_size_bytes=file_size,
                version=version,
                tensor_count=tensor_count,
                metadata_kv_count=kv_count,
                alignment=alignment,
                data_offset=data_offset,
                metadata=metadata,
                tensors=tensors,
            )

    @staticmethod
    def dequantize_buffer(
        raw_bytes: bytes, ggml_type: int, num_elements: int
    ) -> np.ndarray:
        """Vectorized NumPy dequantization for F32, F16, Q8_0, and Q4_0 buffers.

        Returns a 1D float32 array of length `num_elements`.
        """
        if ggml_type == GGML_TYPE_F32:
            return np.frombuffer(raw_bytes, dtype=np.float32, count=num_elements).copy()

        if ggml_type == GGML_TYPE_F16:
            f16 = np.frombuffer(raw_bytes, dtype=np.float16, count=num_elements)
            return f16.astype(np.float32)

        if ggml_type == GGML_TYPE_Q8_0:
            # Each 34-byte block has 1 float16 scale (2 bytes) + 32 int8 values
            num_blocks = num_elements // 32
            raw_u8 = np.frombuffer(
                raw_bytes, dtype=np.uint8, count=num_blocks * 34
            ).reshape(num_blocks, 34)
            scales = raw_u8[:, :2].copy().view(np.float16).astype(np.float32)
            quants = raw_u8[:, 2:].view(np.int8).astype(np.float32)
            return (scales * quants).reshape(-1)

        if ggml_type == GGML_TYPE_Q4_0:
            # Each 18-byte block has 1 float16 scale (2 bytes) + 16 uint8 packed nibbles
            num_blocks = num_elements // 32
            raw_u8 = np.frombuffer(
                raw_bytes, dtype=np.uint8, count=num_blocks * 18
            ).reshape(num_blocks, 18)
            scales = raw_u8[:, :2].copy().view(np.float16).astype(np.float32)
            packed = raw_u8[:, 2:]
            low_nibbles = (packed & 0x0F).astype(np.int8) - 8
            high_nibbles = (packed >> 4).astype(np.int8) - 8
            quants = np.concatenate([low_nibbles, high_nibbles], axis=1).astype(
                np.float32
            )
            return (scales * quants).reshape(-1)

        if ggml_type == GGML_TYPE_Q6_K:
            # Each 210-byte block has 128 bytes ql, 64 bytes qh, 16 bytes scales, 2 bytes fp16 d
            num_blocks = num_elements // 256
            raw_arr = np.frombuffer(
                raw_bytes, dtype=np.uint8, count=num_blocks * 210
            ).reshape(num_blocks, 210)

            ql = raw_arr[:, :128]
            qh = raw_arr[:, 128:192]
            sc = raw_arr[:, 192:208].view(np.int8).astype(np.float32)
            d = raw_arr[:, 208:210].copy().view(np.float16).astype(np.float32)

            idx_0 = [0] * 16 + [1] * 16
            idx_2 = [2] * 16 + [3] * 16
            idx_4 = [4] * 16 + [5] * 16
            idx_6 = [6] * 16 + [7] * 16

            halves = []
            for h in range(2):
                ql_sub = ql[:, h * 64 : (h + 1) * 64]
                qh_sub = qh[:, h * 32 : (h + 1) * 32]
                sc_sub = sc[:, h * 8 : (h + 1) * 8]

                ql_low = ql_sub[:, :32].astype(np.int32)
                ql_high = ql_sub[:, 32:64].astype(np.int32)
                qh_l = qh_sub.astype(np.int32)

                q1 = ((ql_low & 0x0F) | (((qh_l >> 0) & 3) << 4)) - 32
                q2 = ((ql_high & 0x0F) | (((qh_l >> 2) & 3) << 4)) - 32
                q3 = ((ql_low >> 4) | (((qh_l >> 4) & 3) << 4)) - 32
                q4 = ((ql_high >> 4) | (((qh_l >> 6) & 3) << 4)) - 32

                sc0 = sc_sub[:, idx_0]
                sc2 = sc_sub[:, idx_2]
                sc4 = sc_sub[:, idx_4]
                sc6 = sc_sub[:, idx_6]

                y1 = d * sc0 * q1
                y2 = d * sc2 * q2
                y3 = d * sc4 * q3
                y4 = d * sc6 * q4

                halves.append(np.concatenate([y1, y2, y3, y4], axis=1))

            return np.concatenate(halves, axis=1).reshape(-1).astype(np.float32)

        raise ValueError(
            f"Unsupported GGML quantization type for direct dequant: {ggml_type}"
        )

    def load_dequantized_tensor(
        self, tensor_name: str, max_elements: Optional[int] = None
    ) -> np.ndarray:
        """Reads and dequantizes a specific tensor from the GGUF file via mmap."""
        summary = self.parse_header()
        if tensor_name not in summary.tensors:
            raise KeyError(f"Tensor {tensor_name!r} not found in {self.path.name}")

        t_info = summary.tensors[tensor_name]
        if t_info.ggml_type not in GGML_BLOCK_SPECS:
            raise ValueError(
                f"Tensor {tensor_name} uses {t_info.ggml_type_name}; "
                f"supported types: F32, F16, Q4_0, Q8_0, Q6_K"
            )

        block_elems, block_bytes = GGML_BLOCK_SPECS[t_info.ggml_type]
        target_elems = (
            min(t_info.num_elements, max_elements)
            if max_elements is not None
            else t_info.num_elements
        )
        # Round target_elems up to block boundary
        num_blocks = max(1, (target_elems + block_elems - 1) // block_elems)
        actual_elems = num_blocks * block_elems
        byte_len = num_blocks * block_bytes
        abs_offset = summary.data_offset + t_info.relative_offset

        with self.path.open("rb") as f, mmap.mmap(
            f.fileno(), 0, access=mmap.ACCESS_READ
        ) as mm:
            raw_slice = bytes(mm[abs_offset : abs_offset + byte_len])

        dequant = self.dequantize_buffer(raw_slice, t_info.ggml_type, actual_elems)
        if max_elements is None and len(t_info.shape) == 2:
            # GGML stores 2D shapes as (ne0=cols, ne1=rows)
            return dequant.reshape(t_info.shape[1], t_info.shape[0])
        return dequant[:target_elems]

    def compare_with_reference_tensor(
        self,
        gguf_tensor_name: str,
        ref_tensor: torch.Tensor,
        top_k_cols: int = 5,
    ) -> SubstrateDeltaReport:
        """Computes Task-Vector / Fine-Tuning divergence (Ilharco et al., 2023).

        Compares a dequantized GGUF tensor against a reference PyTorch weight matrix.
        """
        summary = self.parse_header()
        t_info = summary.tensors[gguf_tensor_name]
        gguf_mat = self.load_dequantized_tensor(gguf_tensor_name)
        ref_mat = ref_tensor.detach().to(torch.float32).cpu().numpy()

        if gguf_mat.shape != ref_mat.shape:
            gguf_mat = gguf_mat.reshape(ref_mat.shape)

        diff = gguf_mat - ref_mat
        gguf_norm = float(np.linalg.norm(gguf_mat))
        ref_norm = float(np.linalg.norm(ref_mat))
        diff_norm = float(np.linalg.norm(diff))

        denom = max(gguf_norm * ref_norm, 1e-12)
        cos_sim = float(np.sum(gguf_mat * ref_mat) / denom)
        rel_frob = diff_norm / max(ref_norm, 1e-12)

        col_deltas = np.linalg.norm(diff, axis=0)
        top_cols = np.argsort(col_deltas)[::-1][:top_k_cols].tolist()

        return SubstrateDeltaReport(
            gguf_tensor_name=gguf_tensor_name,
            ggml_type_name=t_info.ggml_type_name,
            shape=tuple(ref_mat.shape),
            gguf_l2_norm=round(gguf_norm, 4),
            ref_l2_norm=round(ref_norm, 4),
            cosine_similarity=round(cos_sim, 6),
            relative_frobenius_delta=round(rel_frob, 6),
            top_divergent_columns=[int(c) for c in top_cols],
        )


def build_synthetic_gguf_v3_bytes(
    tensor_name: str, matrix_f32: np.ndarray, ggml_type: int = GGML_TYPE_Q8_0
) -> bytes:
    """Constructs a valid, self-contained GGUF v3 binary image in memory.

    Enables deterministic unit testing of F32 and Q8_0 dequantization.
    """
    rows, cols = matrix_f32.shape
    flat = matrix_f32.astype(np.float32).reshape(-1)
    if flat.size % 32 != 0:
        raise ValueError("Synthetic GGUF matrix element count must be divisible by 32.")

    if ggml_type == GGML_TYPE_Q8_0:
        blocks = flat.reshape(-1, 32)
        max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
        scales_f16 = (max_abs / 127.0).astype(np.float16)
        safe_scales = np.where(
            scales_f16.astype(np.float32) == 0, 1.0, scales_f16.astype(np.float32)
        )
        quants_i8 = np.clip(np.round(blocks / safe_scales), -127, 127).astype(np.int8)
        payload = bytearray()
        for b_idx in range(blocks.shape[0]):
            payload.extend(scales_f16[b_idx].tobytes())
            payload.extend(quants_i8[b_idx].tobytes())
        tensor_bytes = bytes(payload)
    elif ggml_type == GGML_TYPE_F32:
        tensor_bytes = flat.tobytes()
    else:
        raise ValueError(f"Unsupported synthetic GGML type: {ggml_type}")

    buf = bytearray()
    # Header: magic (4s), version=3 (I), tensor_count=1 (Q), kv_count=1 (Q)
    buf.extend(struct.pack("<4sIQQ", GGUF_MAGIC, 3, 1, 1))

    # 1 KV entry: "general.architecture" = "qwen2"
    k_bytes = b"general.architecture"
    v_bytes = b"qwen2"
    buf.extend(struct.pack("<Q", len(k_bytes)) + k_bytes)
    buf.extend(struct.pack("<I", 8))  # STRING type
    buf.extend(struct.pack("<Q", len(v_bytes)) + v_bytes)

    # 1 Tensor Info entry: name, n_dims=2, ne0=cols, ne1=rows, ggml_type, rel_offset=0
    t_bytes = tensor_name.encode("utf-8")
    buf.extend(struct.pack("<Q", len(t_bytes)) + t_bytes)
    buf.extend(struct.pack("<IQQIQ", 2, cols, rows, ggml_type, 0))

    # Pad to 32-byte alignment
    rem = len(buf) % GGUF_DEFAULT_ALIGNMENT
    if rem != 0:
        buf.extend(b"\x00" * (GGUF_DEFAULT_ALIGNMENT - rem))

    buf.extend(tensor_bytes)
    return bytes(buf)

# end of file: unplow/src/unplow/gguf_inspector.py
