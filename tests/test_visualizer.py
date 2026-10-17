# Path: unplow/tests/test_visualizer.py
"""Unit tests for the zero-dependency neural weight visualizer.

Validates PNG binary generation, colormaps, differential heatmap rendering,
and multi-projection attention mosaic dashboards.
"""

from pathlib import Path
import struct
import zlib

import pytest
import torch

from unplow.visualizer import (
    colormap_diverging,
    colormap_inferno,
    encode_png_rgb,
    read_png_metadata,
    render_checkpoint_dashboard,
    render_delta_heatmap,
    render_tensor_heatmap,
)


def test_encode_png_rgb():
    width = 4
    height = 4
    rgb_data = bytes([255, 0, 0] * (width * height))
    png_bytes = encode_png_rgb(width, height, rgb_data)

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in png_bytes
    assert b"IDAT" in png_bytes
    assert png_bytes.endswith(b"IEND\xaeB`\x82")


def test_colormaps():
    # Diverging: -1.0 (Cyan), 0.0 (Slate), +1.0 (Amber/Red)
    cyan = colormap_diverging(-1.0)
    assert cyan == (0, 210, 255)
    slate = colormap_diverging(0.0)
    assert slate == (20, 24, 34)
    amber = colormap_diverging(1.0)
    assert amber == (255, 70, 20)

    # Inferno: 0.0 (Charcoal), 1.0 (Pale Yellow)
    c0 = colormap_inferno(0.0)
    assert c0 == (15, 15, 20)
    c1 = colormap_inferno(1.0)
    assert c1 == (255, 240, 150)


def test_render_tensor_heatmap(tmp_path: Path):
    out_file = tmp_path / "tensor_heatmap.png"
    t = torch.randn(64, 64)
    res = render_tensor_heatmap(t, out_file, target_size=(32, 32))

    assert res.exists()
    data = res.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 64


def test_render_tensor_heatmap_1d(tmp_path: Path):
    out_file = tmp_path / "tensor_1d.png"
    t = torch.randn(64)
    res = render_tensor_heatmap(t, out_file, target_size=(16, 16))

    assert res.exists()
    assert res.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_render_delta_heatmap(tmp_path: Path):
    out_file = tmp_path / "delta_heatmap.png"
    t_old = torch.zeros(32, 32)
    t_new = torch.ones(32, 32) * 0.05
    res = render_delta_heatmap(t_old, t_new, out_file, target_size=(16, 16))

    assert res.exists()
    data = res.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 64


def test_render_checkpoint_dashboard(tmp_path: Path):
    chk1 = tmp_path / "chk1.pt"
    chk2 = tmp_path / "chk2.pt"
    out_file = tmp_path / "dashboard.png"

    state_dict_1 = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(32, 32),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(32, 32),
    }
    state_dict_2 = {
        "model.layers.0.self_attn.q_proj.weight": state_dict_1["model.layers.0.self_attn.q_proj.weight"] + 0.01,
        "model.layers.0.self_attn.k_proj.weight": state_dict_1["model.layers.0.self_attn.k_proj.weight"] - 0.01,
        "model.layers.0.self_attn.o_proj.weight": state_dict_1["model.layers.0.self_attn.o_proj.weight"] + 0.02,
    }

    torch.save(state_dict_1, chk1)
    torch.save(state_dict_2, chk2)

    res = render_checkpoint_dashboard(chk1, chk2, out_file, grid_size=(64, 64))

    assert res.exists()
    data = res.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(data) > 100


def test_png_text_metadata_roundtrip():
    width = 8
    height = 8
    rgb_data = bytes([100, 150, 200] * (width * height))
    meta_in = {
        "Title": "Test Comparison",
        "CheckpointTarget": "target_rev024.pt",
        "CheckpointCompare": "baseline_rev023.pt",
        "Software": "unplow v0.17.0",
    }
    png_bytes = encode_png_rgb(width, height, rgb_data, text_metadata=meta_in)
    meta_out = read_png_metadata(png_bytes)

    assert meta_out["Title"] == "Test Comparison"
    assert meta_out["CheckpointTarget"] == "target_rev024.pt"
    assert meta_out["CheckpointCompare"] == "baseline_rev023.pt"
    assert meta_out["Software"] == "unplow v0.17.0"


def test_draw_text_and_rect_rgb():
    from unplow.visualizer import draw_rect_rgb, draw_text_rgb

    w, h = 60, 20
    pixels = bytearray(w * h * 3)
    draw_rect_rgb(pixels, w, h, 0, 0, w, h, (10, 20, 30))
    # Check rect color
    assert (pixels[0], pixels[1], pixels[2]) == (10, 20, 30)

    # Draw text
    end_x = draw_text_rgb(pixels, w, h, "REV24", 2, 2, (255, 255, 255), scale=1)
    assert end_x > 2
    # Verify at least one pixel changed to white
    white_found = False
    for i in range(0, len(pixels), 3):
        if (pixels[i], pixels[i + 1], pixels[i + 2]) == (255, 255, 255):
            white_found = True
            break
    assert white_found is True


def test_render_delta_with_banner_and_metadata(tmp_path: Path):
    from unplow.visualizer import read_png_metadata

    out_file = tmp_path / "delta_banner.png"
    t1 = torch.zeros(16, 16)
    t2 = torch.ones(16, 16) * 0.12

    res = render_delta_heatmap(
        t1,
        t2,
        out_file,
        target_size=(16, 16),
        checkpoint_target="target_model.pt",
        checkpoint_compare="compare_model.pt",
        tensor_name="q_proj.weight",
        include_banner=True,
    )

    assert res.exists()
    meta = read_png_metadata(res)
    assert meta["CheckpointTarget"] == "target_model.pt"
    assert meta["CheckpointCompare"] == "compare_model.pt"
    assert meta["TensorName"] == "q_proj.weight"
    assert "Metrics" in meta


# end of file: unplow/tests/test_visualizer.py
