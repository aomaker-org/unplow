# Path: unplow/src/unplow/visualizer.py
"""
Checkpoint Visualizer: Renders raw neural weights, parameter update deltas,
and multi-projection attention mosaics as self-contained PNG heatmaps.

Supports embedded standard PNG tEXt metadata chunks and pure-Python bitmap
font visual banners for air-gapped comparison traceability.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import zlib

import torch

# Standard 5x7 ASCII bitmap glyph definitions (printable ASCII characters)
FONT_5X7: Dict[str, List[int]] = {
    " ": [0x00, 0x00, 0x00, 0x00, 0x00],
    "!": [0x00, 0x00, 0x5F, 0x00, 0x00],
    '"': [0x00, 0x07, 0x00, 0x07, 0x00],
    "#": [0x14, 0x7F, 0x14, 0x7F, 0x14],
    "$": [0x24, 0x2A, 0x7F, 0x2A, 0x12],
    "%": [0x23, 0x13, 0x08, 0x64, 0x62],
    "&": [0x36, 0x49, 0x55, 0x22, 0x50],
    "'": [0x00, 0x05, 0x03, 0x00, 0x00],
    "(": [0x00, 0x1C, 0x22, 0x41, 0x00],
    ")": [0x00, 0x41, 0x22, 0x1C, 0x00],
    "*": [0x14, 0x08, 0x3E, 0x08, 0x14],
    "+": [0x08, 0x08, 0x3E, 0x08, 0x08],
    ",": [0x00, 0x80, 0x60, 0x00, 0x00],
    "-": [0x08, 0x08, 0x08, 0x08, 0x08],
    ".": [0x00, 0x60, 0x60, 0x00, 0x00],
    "/": [0x20, 0x10, 0x08, 0x04, 0x02],
    ":": [0x00, 0x36, 0x36, 0x00, 0x00],
    ";": [0x00, 0x56, 0x36, 0x00, 0x00],
    "<": [0x08, 0x14, 0x22, 0x41, 0x00],
    "=": [0x14, 0x14, 0x14, 0x14, 0x14],
    ">": [0x00, 0x41, 0x22, 0x14, 0x08],
    "?": [0x02, 0x01, 0x51, 0x09, 0x06],
    "@": [0x32, 0x49, 0x79, 0x41, 0x3E],
    "0": [0x3E, 0x51, 0x49, 0x45, 0x3E],
    "1": [0x00, 0x42, 0x7F, 0x40, 0x00],
    "2": [0x42, 0x61, 0x51, 0x49, 0x46],
    "3": [0x21, 0x41, 0x45, 0x4B, 0x31],
    "4": [0x18, 0x14, 0x12, 0x7F, 0x10],
    "5": [0x27, 0x45, 0x45, 0x45, 0x39],
    "6": [0x3C, 0x4A, 0x49, 0x49, 0x30],
    "7": [0x01, 0x71, 0x09, 0x05, 0x03],
    "8": [0x36, 0x49, 0x49, 0x49, 0x36],
    "9": [0x06, 0x49, 0x49, 0x29, 0x1E],
    "A": [0x7E, 0x11, 0x11, 0x11, 0x7E],
    "B": [0x7F, 0x49, 0x49, 0x49, 0x36],
    "C": [0x3E, 0x41, 0x41, 0x41, 0x22],
    "D": [0x7F, 0x41, 0x41, 0x22, 0x1C],
    "E": [0x7F, 0x49, 0x49, 0x49, 0x41],
    "F": [0x7F, 0x09, 0x09, 0x09, 0x01],
    "G": [0x3E, 0x41, 0x49, 0x49, 0x7A],
    "H": [0x7F, 0x08, 0x08, 0x08, 0x7F],
    "I": [0x00, 0x41, 0x7F, 0x41, 0x00],
    "J": [0x20, 0x40, 0x41, 0x3F, 0x01],
    "K": [0x7F, 0x08, 0x14, 0x22, 0x41],
    "L": [0x7F, 0x40, 0x40, 0x40, 0x40],
    "M": [0x7F, 0x02, 0x0C, 0x02, 0x7F],
    "N": [0x7F, 0x04, 0x08, 0x10, 0x7F],
    "O": [0x3E, 0x41, 0x41, 0x41, 0x3E],
    "P": [0x7F, 0x09, 0x09, 0x09, 0x06],
    "Q": [0x3E, 0x41, 0x51, 0x21, 0x5E],
    "R": [0x7F, 0x09, 0x19, 0x29, 0x46],
    "S": [0x46, 0x49, 0x49, 0x49, 0x31],
    "T": [0x01, 0x01, 0x7F, 0x01, 0x01],
    "U": [0x3F, 0x40, 0x40, 0x40, 0x3F],
    "V": [0x1F, 0x20, 0x40, 0x20, 0x1F],
    "W": [0x7F, 0x20, 0x18, 0x20, 0x7F],
    "X": [0x63, 0x14, 0x08, 0x14, 0x63],
    "Y": [0x07, 0x08, 0x70, 0x08, 0x07],
    "Z": [0x61, 0x51, 0x49, 0x45, 0x43],
    "[": [0x00, 0x7F, 0x41, 0x41, 0x00],
    "\\": [0x02, 0x04, 0x08, 0x10, 0x20],
    "]": [0x00, 0x41, 0x41, 0x7F, 0x00],
    "^": [0x04, 0x02, 0x01, 0x02, 0x04],
    "_": [0x40, 0x40, 0x40, 0x40, 0x40],
    "`": [0x00, 0x01, 0x02, 0x04, 0x00],
    "a": [0x20, 0x54, 0x54, 0x78, 0x40],
    "b": [0x7F, 0x48, 0x44, 0x44, 0x38],
    "c": [0x38, 0x44, 0x44, 0x44, 0x20],
    "d": [0x38, 0x44, 0x44, 0x48, 0x7F],
    "e": [0x38, 0x54, 0x54, 0x54, 0x18],
    "f": [0x08, 0x7E, 0x09, 0x01, 0x02],
    "g": [0x08, 0x14, 0x54, 0x54, 0x3C],
    "h": [0x7F, 0x08, 0x04, 0x04, 0x78],
    "i": [0x00, 0x44, 0x7D, 0x40, 0x00],
    "j": [0x20, 0x40, 0x44, 0x3D, 0x00],
    "k": [0x7F, 0x10, 0x28, 0x44, 0x00],
    "l": [0x00, 0x41, 0x7F, 0x40, 0x00],
    "m": [0x7C, 0x04, 0x18, 0x04, 0x78],
    "n": [0x7C, 0x08, 0x04, 0x04, 0x78],
    "o": [0x38, 0x44, 0x44, 0x44, 0x38],
    "p": [0x7C, 0x14, 0x14, 0x14, 0x08],
    "q": [0x08, 0x14, 0x14, 0x18, 0x7C],
    "r": [0x7C, 0x08, 0x04, 0x04, 0x08],
    "s": [0x48, 0x54, 0x54, 0x54, 0x20],
    "t": [0x04, 0x3F, 0x44, 0x40, 0x20],
    "u": [0x3C, 0x40, 0x40, 0x20, 0x7C],
    "v": [0x1C, 0x20, 0x40, 0x20, 0x1C],
    "w": [0x3C, 0x40, 0x30, 0x40, 0x3C],
    "x": [0x44, 0x28, 0x10, 0x28, 0x44],
    "y": [0x0C, 0x50, 0x50, 0x50, 0x3C],
    "z": [0x44, 0x64, 0x54, 0x4C, 0x44],
    "{": [0x00, 0x08, 0x36, 0x41, 0x00],
    "|": [0x00, 0x00, 0x7F, 0x00, 0x00],
    "}": [0x00, 0x41, 0x36, 0x08, 0x00],
    "~": [0x08, 0x04, 0x08, 0x10, 0x08],
}


def draw_text_rgb(
    pixels: bytearray,
    width: int,
    height: int,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = (220, 230, 245),
    scale: int = 1,
) -> int:
    """Draws ASCII text onto an RGB pixel buffer using the 5x7 bitmap font."""
    cur_x = x
    for ch in text:
        glyph = FONT_5X7.get(ch, FONT_5X7.get(ch.upper(), FONT_5X7[" "]))
        for col in range(5):
            col_bits = glyph[col]
            for row in range(7):
                if (col_bits >> row) & 1:
                    for dy in range(scale):
                        for dx in range(scale):
                            px = cur_x + col * scale + dx
                            py = y + row * scale + dy
                            if 0 <= px < width and 0 <= py < height:
                                idx = (py * width + px) * 3
                                pixels[idx] = color[0]
                                pixels[idx + 1] = color[1]
                                pixels[idx + 2] = color[2]
        cur_x += (5 + 1) * scale
    return cur_x


def draw_rect_rgb(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
) -> None:
    """Fills a rectangular region on an RGB pixel buffer with a solid color."""
    x_start = max(0, min(x0, x1))
    x_end = min(width, max(x0, x1))
    y_start = max(0, min(y0, y1))
    y_end = min(height, max(y0, y1))
    for y in range(y_start, y_end):
        row_offset = y * width * 3
        for x in range(x_start, x_end):
            idx = row_offset + x * 3
            pixels[idx] = color[0]
            pixels[idx + 1] = color[1]
            pixels[idx + 2] = color[2]


def make_png_chunk(tag: bytes, data: bytes) -> bytes:
    """Builds a standard PNG chunk with 4-byte length, tag, data, and 4-byte CRC."""
    length_bytes = struct.pack(">I", len(data))
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    crc_bytes = struct.pack(">I", crc)
    return length_bytes + tag + data + crc_bytes


def encode_png_rgb(
    width: int,
    height: int,
    rgb_bytes: bytes,
    text_metadata: Optional[Dict[str, str]] = None,
) -> bytes:
    """Encodes raw RGB byte arrays into valid PNG binary format via zlib.

    Supports embedding standard PNG tEXt chunks for provenance tracking
    without external dependencies (Directive 18 compliant).
    """
    header = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, 8 bit-depth, color type 2 (RGB), defl, filter, no interlace
    ihdr = make_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))

    # Standard tEXt metadata chunks
    text_chunks = b""
    if text_metadata:
        for k, v in text_metadata.items():
            k_bytes = str(k).encode("latin-1")[:79]
            v_bytes = str(v).encode("utf-8")
            chunk_data = k_bytes + b"\x00" + v_bytes
            text_chunks += make_png_chunk(b"tEXt", chunk_data)

    # Scanline: prepend filter byte 0x00 to each row
    raw_scanlines = b"".join(
        b"\x00" + rgb_bytes[y * width * 3 : (y + 1) * width * 3]
        for y in range(height)
    )
    idat = make_png_chunk(b"IDAT", zlib.compress(raw_scanlines, level=6))
    iend = make_png_chunk(b"IEND", b"")
    return header + ihdr + text_chunks + idat + iend


def read_png_metadata(png_source: Union[bytes, str, Path]) -> Dict[str, str]:
    """Extracts all embedded tEXt metadata chunks from a PNG file or buffer."""
    if isinstance(png_source, (str, Path)):
        p = Path(png_source)
        if not p.exists():
            return {}
        data = p.read_bytes()
    else:
        data = bytes(png_source)

    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return {}

    offset = 8
    meta: Dict[str, str] = {}
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        tag = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length  # 4 len + 4 tag + length + 4 crc

        if tag == b"tEXt":
            parts = chunk_data.split(b"\x00", 1)
            if len(parts) == 2:
                k = parts[0].decode("latin-1", errors="replace")
                v = parts[1].decode("utf-8", errors="replace")
                meta[k] = v
        elif tag == b"IEND":
            break
    return meta


def compute_tensor_metrics(tensor: torch.Tensor) -> Dict[str, Any]:
    """Computes numerical summary metrics for a tensor."""
    t = tensor.detach().cpu().float()
    abs_t = t.abs()
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "mean_abs": float(abs_t.mean()),
        "max_abs": float(abs_t.max()),
        "std": float(t.std()),
        "l2_norm": float(torch.norm(t)),
        "param_count": int(t.numel()),
    }


def colormap_diverging(val: float) -> Tuple[int, int, int]:
    """Maps normalized divergence val in [-1.0, 1.0] to a high-contrast palette.

    -1.0: Electric Cyan  (0, 210, 255)  -> Suppression / down-weighting
     0.0: Deep Slate      (20, 24, 34)   -> Stable / zero gradient delta
    +1.0: Vivid Amber/Red (255, 70, 20)  -> Amplification / up-weighting
    """
    clipped = max(-1.0, min(1.0, val))
    if clipped < 0:
        a = -clipped
        return (
            int(20 * (1.0 - a) + 0 * a),
            int(24 * (1.0 - a) + 210 * a),
            int(34 * (1.0 - a) + 255 * a),
        )
    a = clipped
    return (
        int(20 * (1.0 - a) + 255 * a),
        int(24 * (1.0 - a) + 70 * a),
        int(34 * (1.0 - a) + 20 * a),
    )


def colormap_inferno(val: float) -> Tuple[int, int, int]:
    """Maps normalized intensity val in [0.0, 1.0] to an inferno thermal palette.

    0.0: Charcoal (15, 15, 20)
    0.3: Deep Purple (80, 20, 110)
    0.7: Vivid Orange (230, 90, 30)
    1.0: Pale Yellow (255, 240, 150)
    """
    c = max(0.0, min(1.0, val))
    if c < 0.3:
        a = c / 0.3
        return (
            int(15 * (1.0 - a) + 80 * a),
            int(15 * (1.0 - a) + 20 * a),
            int(20 * (1.0 - a) + 110 * a),
        )
    if c < 0.7:
        a = (c - 0.3) / 0.4
        return (
            int(80 * (1.0 - a) + 230 * a),
            int(20 * (1.0 - a) + 90 * a),
            int(110 * (1.0 - a) + 30 * a),
        )
    a = (c - 0.7) / 0.3
    return (
        int(230 * (1.0 - a) + 255 * a),
        int(90 * (1.0 - a) + 240 * a),
        int(30 * (1.0 - a) + 150 * a),
    )


def render_tensor_heatmap(
    tensor: torch.Tensor,
    output_path: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
    colormap: str = "inferno",
    checkpoint_path: Optional[Union[str, Path]] = None,
    tensor_name: Optional[str] = None,
    include_banner: bool = True,
) -> Path:
    """Renders a 2D weight matrix as a standalone PNG heatmap with metadata."""
    t = tensor.detach().cpu().float()
    if t.ndim == 1:
        dim = int(t.numel() ** 0.5)
        if dim * dim == t.numel():
            t = t.view(dim, dim)
        else:
            t = t.unsqueeze(0).repeat(32, 1)

    pool = torch.nn.AdaptiveAvgPool2d(target_size)
    downsampled = pool(t.unsqueeze(0).unsqueeze(0)).squeeze().numpy()
    h, w = downsampled.shape

    min_v, max_v = float(downsampled.min()), float(downsampled.max())
    span = max(max_v - min_v, 1e-8)
    norm = (downsampled - min_v) / span

    banner_h = 36 if include_banner else 0
    total_h = h + banner_h
    pixels = bytearray(total_h * w * 3)

    # Render banner
    if include_banner:
        draw_rect_rgb(pixels, w, total_h, 0, 0, w, banner_h, (14, 17, 24))
        draw_rect_rgb(pixels, w, total_h, 0, banner_h - 1, w, banner_h, (40, 50, 70))
        target_name = Path(checkpoint_path).name if checkpoint_path else "Current Checkpoint"
        t_label = tensor_name or "Neural Weight Matrix"
        draw_text_rgb(pixels, w, total_h, f"CHECKPOINT: {target_name}", 8, 5, (0, 210, 255))
        draw_text_rgb(
            pixels, w, total_h, f"TENSOR: {t_label} | RANGE: [{min_v:.3f}, {max_v:.3f}]", 8, 19, (200, 210, 230)
        )

    # Render heatmap pixels
    for y in range(h):
        py = y + banner_h
        for x in range(w):
            rgb = colormap_inferno(norm[y, x])
            idx = (py * w + x) * 3
            pixels[idx] = rgb[0]
            pixels[idx + 1] = rgb[1]
            pixels[idx + 2] = rgb[2]

    metrics = compute_tensor_metrics(t)
    text_meta = {
        "Title": "Unplow Neural Weight Heatmap",
        "CheckpointTarget": str(checkpoint_path or "unspecified"),
        "TensorName": str(tensor_name or "unspecified"),
        "Metrics": json.dumps(metrics),
        "CreatedAt": datetime.now(timezone.utc).isoformat(),
        "Software": "unplow v0.17.0",
    }

    png_bytes = encode_png_rgb(w, total_h, bytes(pixels), text_metadata=text_meta)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)
    return out


def render_delta_heatmap(
    tensor_old: torch.Tensor,
    tensor_new: torch.Tensor,
    output_path: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
    checkpoint_target: Optional[Union[str, Path]] = None,
    checkpoint_compare: Optional[Union[str, Path]] = None,
    tensor_name: Optional[str] = None,
    include_banner: bool = True,
) -> Path:
    """Renders parameter updates (tensor_new - tensor_old) as a diverging heatmap."""
    t_old = tensor_old.detach().cpu().float()
    t_new = tensor_new.detach().cpu().float()
    diff = t_new - t_old

    if diff.ndim == 1:
        dim = int(diff.numel() ** 0.5)
        if dim * dim == diff.numel():
            diff = diff.view(dim, dim)
        else:
            diff = diff.unsqueeze(0).repeat(32, 1)

    pool = torch.nn.AdaptiveAvgPool2d(target_size)
    downsampled = pool(diff.unsqueeze(0).unsqueeze(0)).squeeze().numpy()
    h, w = downsampled.shape

    max_abs = max(abs(float(downsampled.min())), abs(float(downsampled.max())), 1e-8)
    norm = downsampled / max_abs

    banner_h = 38 if include_banner else 0
    total_h = h + banner_h
    pixels = bytearray(total_h * w * 3)

    metrics = compute_tensor_metrics(diff)
    t_name = Path(checkpoint_target).name if checkpoint_target else "Target"
    c_name = Path(checkpoint_compare).name if checkpoint_compare else "Compare"
    lbl = tensor_name or "Weight Tensor"

    # Render header banner with comparison input files and metrics
    if include_banner:
        draw_rect_rgb(pixels, w, total_h, 0, 0, w, banner_h, (14, 17, 24))
        draw_rect_rgb(pixels, w, total_h, 0, banner_h - 1, w, banner_h, (0, 180, 220))
        draw_text_rgb(pixels, w, total_h, f"TARGET: {t_name} | COMPARE: {c_name}", 8, 6, (0, 210, 255))
        draw_text_rgb(
            pixels,
            w,
            total_h,
            f"TENSOR: {lbl} | MEAN |dW|: {metrics['mean_abs']:.2e} | MAX: {metrics['max_abs']:.2e}",
            8,
            20,
            (210, 220, 240),
        )

    for y in range(h):
        py = y + banner_h
        for x in range(w):
            rgb = colormap_diverging(norm[y, x])
            idx = (py * w + x) * 3
            pixels[idx] = rgb[0]
            pixels[idx + 1] = rgb[1]
            pixels[idx + 2] = rgb[2]

    text_meta = {
        "Title": "Unplow Weight Differential Update",
        "CheckpointTarget": str(checkpoint_target or "unspecified"),
        "CheckpointCompare": str(checkpoint_compare or "unspecified"),
        "TensorName": str(tensor_name or "unspecified"),
        "Metrics": json.dumps(metrics),
        "CreatedAt": datetime.now(timezone.utc).isoformat(),
        "Software": "unplow v0.17.0",
    }

    png_bytes = encode_png_rgb(w, total_h, bytes(pixels), text_metadata=text_meta)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)
    return out


def render_checkpoint_dashboard(
    checkpoint_old: Union[str, Path],
    checkpoint_new: Union[str, Path],
    output_path: Union[str, Path],
    grid_size: Tuple[int, int] = (512, 512),
    include_banner: bool = True,
) -> Path:
    """Renders a 4-quadrant visualizer comparing two model checkpoints.

    Quadrant Layout:
    - Top-Left     : Current Attention Weights W_latest (Inferno Palette)
    - Top-Right    : Q-Projection Delta dW_Q (Diverging Palette)
    - Bottom-Left  : K-Projection Delta dW_K (Diverging Palette)
    - Bottom-Right : Output-Projection Delta dW_O (Diverging Palette)
    """
    w_old = torch.load(checkpoint_old, map_location="cpu", weights_only=False)
    w_new = torch.load(checkpoint_new, map_location="cpu", weights_only=False)

    def _find_key(keys: List[str], substring: str) -> str:
        for k in keys:
            if substring in k and "weight" in k:
                return k
        return keys[0]

    all_keys = list(w_new.keys())
    k_q = _find_key(all_keys, "q_proj")
    k_k = _find_key(all_keys, "k_proj")
    k_o = _find_key(all_keys, "o_proj")

    def _get_tensor(d: Dict[str, torch.Tensor], key: str, fallback_idx: int) -> torch.Tensor:
        if key in d:
            return d[key]
        vals = list(d.values())
        return vals[fallback_idx % len(vals)]

    t_q_old = _get_tensor(w_old, k_q, 0).float()
    t_q_new = _get_tensor(w_new, k_q, 0).float()
    t_k_old = _get_tensor(w_old, k_k, 1).float()
    t_k_new = _get_tensor(w_new, k_k, 1).float()
    t_o_old = _get_tensor(w_old, k_o, 2).float()
    t_o_new = _get_tensor(w_new, k_o, 2).float()

    total_w, quad_h = grid_size
    banner_h = 38 if include_banner else 0
    total_h = quad_h + banner_h

    panel_size = (total_w // 2, quad_h // 2)
    pool = torch.nn.AdaptiveAvgPool2d(panel_size)

    # 1. Top-Left: Raw Q weights
    p1 = pool(t_q_new.unsqueeze(0).unsqueeze(0)).squeeze().numpy()
    p1_norm = (p1 - p1.min()) / max(p1.max() - p1.min(), 1e-8)

    # 2. Top-Right: Q delta
    diff_q = pool((t_q_new - t_q_old).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
    p2_norm = diff_q / max(abs(float(diff_q.min())), abs(float(diff_q.max())), 1e-8)

    # 3. Bottom-Left: K delta
    diff_k = pool((t_k_new - t_k_old).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
    p3_norm = diff_k / max(abs(float(diff_k.min())), abs(float(diff_k.max())), 1e-8)

    # 4. Bottom-Right: O delta
    diff_o = pool((t_o_new - t_o_old).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
    p4_norm = diff_o / max(abs(float(diff_o.min())), abs(float(diff_o.max())), 1e-8)

    half_w, half_h = panel_size
    pixels = bytearray(total_w * total_h * 3)

    t_name = Path(checkpoint_new).name
    c_name = Path(checkpoint_old).name

    # Render header banner with comparison input files
    if include_banner:
        draw_rect_rgb(pixels, total_w, total_h, 0, 0, total_w, banner_h, (14, 17, 24))
        draw_rect_rgb(pixels, total_w, total_h, 0, banner_h - 1, total_w, banner_h, (0, 180, 220))
        draw_text_rgb(pixels, total_w, total_h, f"TARGET: {t_name} | COMPARE: {c_name}", 8, 6, (0, 210, 255))
        draw_text_rgb(
            pixels,
            total_w,
            total_h,
            "TL: W_Q (Inferno) | TR: dW_Q (Diverging) | BL: dW_K (Diverging) | BR: dW_O",
            8,
            20,
            (210, 220, 240),
        )

    for y in range(quad_h):
        py = y + banner_h
        for x in range(total_w):
            # Render subtle divider borders
            if x in (0, half_w - 1, half_w, total_w - 1) or y in (0, half_h - 1, half_h, quad_h - 1):
                rgb = (50, 55, 68)
            elif x < half_w and y < half_h:
                rgb = colormap_inferno(p1_norm[y, x])
            elif x >= half_w and y < half_h:
                rgb = colormap_diverging(p2_norm[y, x - half_w])
            elif x < half_w and y >= half_h:
                rgb = colormap_diverging(p3_norm[y - half_h, x])
            else:
                rgb = colormap_diverging(p4_norm[y - half_h, x - half_w])

            idx = (py * total_w + x) * 3
            pixels[idx] = rgb[0]
            pixels[idx + 1] = rgb[1]
            pixels[idx + 2] = rgb[2]

    metrics = {
        "q_proj": compute_tensor_metrics(t_q_new - t_q_old),
        "k_proj": compute_tensor_metrics(t_k_new - t_k_old),
        "o_proj": compute_tensor_metrics(t_o_new - t_o_old),
    }

    text_meta = {
        "Title": "Unplow Multi-Projection Attention Mosaic",
        "CheckpointTarget": str(checkpoint_new),
        "CheckpointCompare": str(checkpoint_old),
        "Mode": "dashboard",
        "Metrics": json.dumps(metrics),
        "CreatedAt": datetime.now(timezone.utc).isoformat(),
        "Software": "unplow v0.17.0",
    }

    png_bytes = encode_png_rgb(total_w, total_h, bytes(pixels), text_metadata=text_meta)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)
    return out


def encode_apng_rgb(
    frames: List[Tuple[int, int, bytes]],
    delay_ms: int = 500,
    loop: int = 0,
    text_metadata: Optional[Dict[str, str]] = None,
) -> bytes:
    """Encodes a sequence of RGB frames into an Animated PNG (APNG) stream.

    Zero-dependency standard-library implementation compliant with RFC 7689.
    Supported natively by all modern browsers and viewers without ffmpeg.
    """
    if not frames:
        raise ValueError("Frames list cannot be empty for APNG encoding.")

    w, h, _ = frames[0]
    num_frames = len(frames)

    header = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, 8 bit-depth, color type 2 (RGB), defl, filter, no interlace
    ihdr = make_png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))

    # acTL: num_frames (4 bytes), num_plays (4 bytes, 0 = infinite loop)
    actl = make_png_chunk(b"acTL", struct.pack(">II", num_frames, loop))

    # Standard tEXt metadata chunks
    text_chunks = b""
    if text_metadata:
        for k, v in text_metadata.items():
            k_bytes = str(k).encode("latin-1")[:79]
            v_bytes = str(v).encode("utf-8")
            chunk_data = k_bytes + b"\x00" + v_bytes
            text_chunks += make_png_chunk(b"tEXt", chunk_data)

    delay_num = max(1, int(delay_ms))
    delay_den = 1000

    chunks: List[bytes] = [header, ihdr, actl, text_chunks]

    seq_number = 0
    for idx, (frame_w, frame_h, frame_rgb) in enumerate(frames):
        scanlines = b"".join(
            b"\x00" + frame_rgb[y * frame_w * 3 : (y + 1) * frame_w * 3]
            for y in range(frame_h)
        )
        compressed = zlib.compress(scanlines, level=6)

        # fcTL chunk: >IIIIIHHBB (26 bytes)
        # sequence_number, width, height, x_offset, y_offset, delay_num, delay_den, dispose_op, blend_op
        fctl_data = struct.pack(
            ">IIIIIHHBB",
            seq_number,
            frame_w,
            frame_h,
            0,
            0,
            delay_num,
            delay_den,
            0,  # APNG_DISPOSE_OP_NONE
            0,  # APNG_BLEND_OP_SOURCE
        )
        chunks.append(make_png_chunk(b"fcTL", fctl_data))
        seq_number += 1

        if idx == 0:
            chunks.append(make_png_chunk(b"IDAT", compressed))
        else:
            fdat_payload = struct.pack(">I", seq_number) + compressed
            chunks.append(make_png_chunk(b"fdAT", fdat_payload))
            seq_number += 1

    chunks.append(make_png_chunk(b"IEND", b""))
    return b"".join(chunks)


def decode_png_rgb(png_bytes: bytes) -> Tuple[int, int, bytes]:
    """Decodes standard RGB or RGBA PNG binary bytes into raw RGB pixel buffer."""
    if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid PNG signature.")

    offset = 8
    width = height = color_type = bit_depth = 0
    idat_parts: List[bytes] = []

    while offset + 8 <= len(png_bytes):
        length = struct.unpack(">I", png_bytes[offset : offset + 4])[0]
        tag = png_bytes[offset + 4 : offset + 8]
        data = png_bytes[offset + 8 : offset + 8 + length]
        offset += 12 + length

        if tag == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", data)
        elif tag == b"IDAT":
            idat_parts.append(data)
        elif tag == b"IEND":
            break

    if not idat_parts or width == 0 or height == 0:
        raise ValueError("Incomplete PNG data or missing IDAT.")

    raw = zlib.decompress(b"".join(idat_parts))
    bpp = 3 if color_type == 2 else 4
    stride = 1 + width * bpp
    rgb = bytearray(width * height * 3)
    prev_row = bytearray(width * bpp)

    for y in range(height):
        filter_type = raw[y * stride]
        curr_raw = raw[y * stride + 1 : (y + 1) * stride]
        curr_row = bytearray(width * bpp)
        for i in range(len(curr_raw)):
            x = curr_raw[i]
            a = curr_row[i - bpp] if i >= bpp else 0
            b = prev_row[i]
            c = prev_row[i - bpp] if i >= bpp else 0
            if filter_type == 0:
                val = x
            elif filter_type == 1:
                val = (x + a) & 0xFF
            elif filter_type == 2:
                val = (x + b) & 0xFF
            elif filter_type == 3:
                val = (x + (a + b) // 2) & 0xFF
            elif filter_type == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                val = (x + pr) & 0xFF
            else:
                val = x
            curr_row[i] = val
        prev_row = curr_row

        if color_type == 2:
            rgb[y * width * 3 : (y + 1) * width * 3] = curr_row
        elif color_type == 6:
            for x_idx in range(width):
                rgb[(y * width + x_idx) * 3 : (y * width + x_idx) * 3 + 3] = curr_row[
                    x_idx * 4 : x_idx * 4 + 3
                ]

    return width, height, bytes(rgb)


def compile_images_to_apng(
    image_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    fps: float = 2.0,
    loop: int = 0,
    text_metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """Stitches an existing series of PNG image files into an animated APNG file."""
    if not image_paths:
        raise ValueError("Image paths list cannot be empty.")

    frames: List[Tuple[int, int, bytes]] = []
    base_w, base_h = 0, 0

    for idx, p in enumerate(image_paths):
        path = Path(p)
        if not path.exists():
            continue
        w, h, rgb = decode_png_rgb(path.read_bytes())
        if not frames:
            base_w, base_h = w, h
            frames.append((w, h, rgb))
        else:
            if (w, h) == (base_w, base_h):
                frames.append((w, h, rgb))
            else:
                t = torch.frombuffer(rgb, dtype=torch.uint8).float().view(h, w, 3)
                t = t.permute(2, 0, 1).unsqueeze(0)
                resampled = torch.nn.functional.interpolate(
                    t, size=(base_h, base_w), mode="bilinear", align_corners=False
                ).squeeze(0).permute(1, 2, 0).byte()
                frames.append((base_w, base_h, bytes(resampled.contiguous().view(-1).tolist())))

    if not frames:
        raise ValueError("No valid PNG images could be loaded from provided paths.")

    delay_ms = max(50, int(1000.0 / max(0.1, fps)))
    meta = text_metadata or {}
    meta.setdefault("Title", "Unplow Image Sequence Timelapse")
    meta.setdefault("FrameCount", str(len(frames)))
    meta.setdefault("Fps", f"{fps:.1f}")
    meta.setdefault("CreatedAt", datetime.now(timezone.utc).isoformat())
    meta.setdefault("Software", "unplow v0.17.0")

    apng_bytes = encode_apng_rgb(frames, delay_ms=delay_ms, loop=loop, text_metadata=meta)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(apng_bytes)
    return out


def select_model_tensor(
    state_dict: Dict[str, torch.Tensor],
    tensor_name: Optional[str] = None,
) -> Tuple[str, torch.Tensor]:
    """Selects target tensor by explicit match or heuristic fallback."""
    if tensor_name and tensor_name in state_dict:
        return tensor_name, state_dict[tensor_name]

    if tensor_name:
        for k in state_dict:
            if tensor_name.lower() in k.lower():
                return k, state_dict[k]

    for sub in ["q_proj", "gate_proj", "up_proj", "down_proj", "k_proj", "v_proj", "o_proj"]:
        for k, v in state_dict.items():
            if sub in k and "weight" in k and v.ndim >= 2:
                return k, v

    for k, v in state_dict.items():
        if "weight" in k and v.ndim >= 2:
            return k, v

    k = next(iter(state_dict.keys()))
    return k, state_dict[k]


def pool_tensor_2d(tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """Downsamples any 1D/2D/ND tensor to a target 2D spatial grid via adaptive pooling."""
    t = tensor.detach().cpu().float()
    if t.ndim == 1:
        dim = int(t.numel() ** 0.5)
        if dim * dim == t.numel():
            t = t.view(dim, dim)
        else:
            t = t.unsqueeze(0).repeat(32, 1)
    elif t.ndim > 2:
        t = t.view(t.shape[0], -1)
    pool = torch.nn.AdaptiveAvgPool2d(target_size)
    return pool(t.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)


def render_checkpoint_timelapse(
    checkpoint_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    domain: str = "python",
    tensor_name: Optional[str] = None,
    mode: str = "delta",
    fps: float = 2.0,
    target_size: Tuple[int, int] = (512, 512),
    include_banner: bool = True,
) -> Path:
    """Compiles a chronological sequence of model checkpoints into an APNG video.

    Modes:
      - 'delta'     : Differential update relative to initial checkpoint (rev001).
      - 'step'      : Differential update relative to immediate predecessor (rev[i-1]).
      - 'raw'       : Absolute weight intensity heatmap using inferno colormap.
      - 'dashboard' : 4-quadrant attention mosaic comparing current revision vs rev001.
    """
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths cannot be empty.")

    def _extract_rev(p: Union[str, Path]) -> int:
        m = re.search(r"rev(\d+)", Path(p).name)
        return int(m.group(1)) if m else 0

    sorted_paths = sorted([Path(p) for p in checkpoint_paths], key=_extract_rev)
    total_frames = len(sorted_paths)

    w_base = torch.load(sorted_paths[0], map_location="cpu", weights_only=False)
    resolved_key, _ = select_model_tensor(w_base, tensor_name)

    w_prev = w_base
    frames: List[Tuple[int, int, bytes]] = []

    banner_h = 38 if include_banner else 0
    w_dim, h_dim = target_size
    total_h = h_dim + banner_h

    for idx, p in enumerate(sorted_paths):
        curr_name = p.name
        rev_num = _extract_rev(p)
        w_curr = torch.load(p, map_location="cpu", weights_only=False)

        pixels = bytearray(total_h * w_dim * 3)

        if mode == "dashboard":
            panel_size = (w_dim // 2, h_dim // 2)
            _, t_q_base = select_model_tensor(w_base, "q_proj")
            _, t_q_curr = select_model_tensor(w_curr, "q_proj")
            _, t_k_base = select_model_tensor(w_base, "k_proj")
            _, t_k_curr = select_model_tensor(w_curr, "k_proj")
            _, t_o_base = select_model_tensor(w_base, "o_proj")
            _, t_o_curr = select_model_tensor(w_curr, "o_proj")

            p1 = pool_tensor_2d(t_q_curr, panel_size).numpy()
            p1_norm = (p1 - p1.min()) / max(p1.max() - p1.min(), 1e-8)

            diff_q = (pool_tensor_2d(t_q_curr, panel_size) - pool_tensor_2d(t_q_base, panel_size)).numpy()
            p2_norm = diff_q / max(abs(float(diff_q.min())), abs(float(diff_q.max())), 1e-8)

            diff_k = (pool_tensor_2d(t_k_curr, panel_size) - pool_tensor_2d(t_k_base, panel_size)).numpy()
            p3_norm = diff_k / max(abs(float(diff_k.min())), abs(float(diff_k.max())), 1e-8)

            diff_o = (pool_tensor_2d(t_o_curr, panel_size) - pool_tensor_2d(t_o_base, panel_size)).numpy()
            p4_norm = diff_o / max(abs(float(diff_o.min())), abs(float(diff_o.max())), 1e-8)

            half_w, half_h = panel_size
            for y in range(h_dim):
                py = y + banner_h
                for x in range(w_dim):
                    if x in (0, half_w - 1, half_w, w_dim - 1) or y in (0, half_h - 1, half_h, h_dim - 1):
                        rgb = (50, 55, 68)
                    elif x < half_w and y < half_h:
                        rgb = colormap_inferno(p1_norm[y, x])
                    elif x >= half_w and y < half_h:
                        rgb = colormap_diverging(p2_norm[y, x - half_w])
                    elif x < half_w and y >= half_h:
                        rgb = colormap_diverging(p3_norm[y - half_h, x])
                    else:
                        rgb = colormap_diverging(p4_norm[y - half_h, x - half_w])

                    p_idx = (py * w_dim + x) * 3
                    pixels[p_idx] = rgb[0]
                    pixels[p_idx + 1] = rgb[1]
                    pixels[p_idx + 2] = rgb[2]

            diff_metrics = compute_tensor_metrics(
                pool_tensor_2d(t_q_curr, panel_size) - pool_tensor_2d(t_q_base, panel_size)
            )

        elif mode in ("delta", "step"):
            ref_dict = w_base if mode == "delta" else w_prev
            _, t_ref = select_model_tensor(ref_dict, tensor_name)
            curr_tensor_key, t_curr = select_model_tensor(w_curr, tensor_name)

            p_ref = pool_tensor_2d(t_ref, target_size)
            p_curr = pool_tensor_2d(t_curr, target_size)
            diff = p_curr - p_ref

            downsampled = diff.numpy()
            max_abs = max(abs(float(downsampled.min())), abs(float(downsampled.max())), 1e-8)
            norm = downsampled / max_abs

            for y in range(h_dim):
                py = y + banner_h
                for x in range(w_dim):
                    rgb = colormap_diverging(norm[y, x])
                    p_idx = (py * w_dim + x) * 3
                    pixels[p_idx] = rgb[0]
                    pixels[p_idx + 1] = rgb[1]
                    pixels[p_idx + 2] = rgb[2]

            diff_metrics = compute_tensor_metrics(diff)
            resolved_key = curr_tensor_key

        else:  # raw mode
            curr_tensor_key, t_curr = select_model_tensor(w_curr, tensor_name)
            p_curr = pool_tensor_2d(t_curr, target_size)
            downsampled = p_curr.numpy()
            min_v, max_v = float(downsampled.min()), float(downsampled.max())
            span = max(max_v - min_v, 1e-8)
            norm = (downsampled - min_v) / span

            for y in range(h_dim):
                py = y + banner_h
                for x in range(w_dim):
                    rgb = colormap_inferno(norm[y, x])
                    p_idx = (py * w_dim + x) * 3
                    pixels[p_idx] = rgb[0]
                    pixels[p_idx + 1] = rgb[1]
                    pixels[p_idx + 2] = rgb[2]

            diff_metrics = compute_tensor_metrics(p_curr)
            resolved_key = curr_tensor_key

        # Header banner
        if include_banner:
            draw_rect_rgb(pixels, w_dim, total_h, 0, 0, w_dim, banner_h, (14, 17, 24))
            draw_rect_rgb(pixels, w_dim, total_h, 0, banner_h - 1, w_dim, banner_h, (0, 180, 220))
            line1 = f"TIMELAPSE: till-{domain} | FRAME {idx + 1}/{total_frames} (rev{rev_num:03d})"
            short_k = resolved_key.split(".")[-2] if "." in resolved_key else resolved_key
            if mode == "dashboard":
                line2 = f"MOSAIC: W_Q | dW_Q | dW_K | dW_O vs rev001 | MEAN |dW|: {diff_metrics['mean_abs']:.2e}"
            elif mode == "delta":
                line2 = f"DELTA vs rev001 | TENSOR: {short_k} | MEAN |dW|: {diff_metrics['mean_abs']:.2e}"
            elif mode == "step":
                line2 = f"STEP DELTA | TENSOR: {short_k} | MEAN |dW|: {diff_metrics['mean_abs']:.2e}"
            else:
                line2 = f"RAW WEIGHTS (Inferno) | TENSOR: {short_k} | L2: {diff_metrics['l2_norm']:.2e}"

            draw_text_rgb(pixels, w_dim, total_h, line1, 8, 6, (0, 210, 255))
            draw_text_rgb(pixels, w_dim, total_h, line2, 8, 20, (210, 220, 240))

        frames.append((w_dim, total_h, bytes(pixels)))
        w_prev = w_curr

    delay_ms = max(50, int(1000.0 / max(0.1, fps)))
    meta = {
        "Title": "Unplow Neural Checkpoint Evolution Timelapse",
        "Domain": f"till-{domain}",
        "Mode": mode,
        "TensorName": resolved_key,
        "TotalFrames": str(total_frames),
        "Fps": f"{fps:.1f}",
        "InitialCheckpoint": sorted_paths[0].name,
        "FinalCheckpoint": sorted_paths[-1].name,
        "CreatedAt": datetime.now(timezone.utc).isoformat(),
        "Software": "unplow v0.17.0",
    }

    apng_bytes = encode_apng_rgb(frames, delay_ms=delay_ms, loop=0, text_metadata=meta)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(apng_bytes)
    return out


# end of file: unplow/src/unplow/visualizer.py
