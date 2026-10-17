# Path: unplow/src/unplow/visualizer.py
"""
Checkpoint Visualizer: Renders raw neural weights, parameter update deltas,
and multi-projection attention mosaics as self-contained PNG heatmaps.
"""

from pathlib import Path
import struct
from typing import Dict, List, Optional, Tuple, Union
import zlib

import torch


def encode_png_rgb(width: int, height: int, rgb_bytes: bytes) -> bytes:
    """Encodes raw RGB byte arrays into valid PNG binary format via zlib.

    Eliminates external image library dependencies (Pillow/matplotlib) to
    preserve air-gapped lightweight runtime execution under Directive 18.
    """

    def _chunk(tag: bytes, data: bytes) -> bytes:
        length_bytes = struct.pack(">I", len(data))
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        crc_bytes = struct.pack(">I", crc)
        return length_bytes + tag + data + crc_bytes

    header = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, 8 bit-depth, color type 2 (RGB), defl, filter, no interlace
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    # Scanline: prepend filter byte 0x00 to each row
    raw_scanlines = b"".join(
        b"\x00" + rgb_bytes[y * width * 3 : (y + 1) * width * 3]
        for y in range(height)
    )
    idat = _chunk(b"IDAT", zlib.compress(raw_scanlines, level=6))
    iend = _chunk(b"IEND", b"")
    return header + ihdr + idat + iend


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
) -> Path:
    """Renders a 2D weight matrix as a standalone PNG heatmap."""
    t = tensor.detach().cpu().float()
    if t.ndim == 1:
        # Expand 1D bias or vector into square or rectangular matrix
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

    pixels = bytearray(h * w * 3)
    idx = 0
    for y in range(h):
        for x in range(w):
            rgb = colormap_inferno(norm[y, x])
            pixels[idx] = rgb[0]
            pixels[idx + 1] = rgb[1]
            pixels[idx + 2] = rgb[2]
            idx += 3

    png_bytes = encode_png_rgb(w, h, bytes(pixels))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)
    return out


def render_delta_heatmap(
    tensor_old: torch.Tensor,
    tensor_new: torch.Tensor,
    output_path: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
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

    pixels = bytearray(h * w * 3)
    idx = 0
    for y in range(h):
        for x in range(w):
            rgb = colormap_diverging(norm[y, x])
            pixels[idx] = rgb[0]
            pixels[idx + 1] = rgb[1]
            pixels[idx + 2] = rgb[2]
            idx += 3

    png_bytes = encode_png_rgb(w, h, bytes(pixels))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)
    return out


def render_checkpoint_dashboard(
    checkpoint_old: Union[str, Path],
    checkpoint_new: Union[str, Path],
    output_path: Union[str, Path],
    grid_size: Tuple[int, int] = (512, 512),
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

    panel_size = (grid_size[0] // 2, grid_size[1] // 2)
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

    total_w, total_h = grid_size
    half_w, half_h = panel_size
    pixels = bytearray(total_w * total_h * 3)

    for y in range(total_h):
        for x in range(total_w):
            # Render subtle divider borders
            if x in (0, half_w - 1, half_w, total_w - 1) or y in (0, half_h - 1, half_h, total_h - 1):
                rgb = (50, 55, 68)
            elif x < half_w and y < half_h:
                rgb = colormap_inferno(p1_norm[y, x])
            elif x >= half_w and y < half_h:
                rgb = colormap_diverging(p2_norm[y, x - half_w])
            elif x < half_w and y >= half_h:
                rgb = colormap_diverging(p3_norm[y - half_h, x])
            else:
                rgb = colormap_diverging(p4_norm[y - half_h, x - half_w])

            idx = (y * total_w + x) * 3
            pixels[idx] = rgb[0]
            pixels[idx + 1] = rgb[1]
            pixels[idx + 2] = rgb[2]

    png_bytes = encode_png_rgb(total_w, total_h, bytes(pixels))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png_bytes)
    return out


# end of file: unplow/src/unplow/visualizer.py
