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
    compile_images_to_apng,
    decode_png_rgb,
    encode_apng_rgb,
    encode_png_rgb,
    read_png_metadata,
    render_checkpoint_dashboard,
    render_checkpoint_timelapse,
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


def test_encode_and_decode_apng_rgb():
    width = 8
    height = 8
    frame1 = (width, height, bytes([255, 0, 0] * (width * height)))
    frame2 = (width, height, bytes([0, 255, 0] * (width * height)))
    frame3 = (width, height, bytes([0, 0, 255] * (width * height)))

    meta = {"Title": "Test APNG", "Software": "unplow v0.17.0"}
    apng_data = encode_apng_rgb([frame1, frame2, frame3], delay_ms=250, loop=0, text_metadata=meta)

    assert apng_data.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"acTL" in apng_data
    assert b"fcTL" in apng_data
    assert b"fdAT" in apng_data
    assert apng_data.endswith(b"IEND\xaeB`\x82")

    # Metadata extraction
    parsed_meta = read_png_metadata(apng_data)
    assert parsed_meta["Title"] == "Test APNG"
    assert parsed_meta["Software"] == "unplow v0.17.0"

    # Frame 0 decoded by standard PNG decoder
    w_dec, h_dec, rgb_dec = decode_png_rgb(apng_data)
    assert (w_dec, h_dec) == (width, height)
    assert rgb_dec == frame1[2]


def test_compile_images_to_apng(tmp_path: Path):
    w, h = 8, 8
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    out_apng = tmp_path / "compiled.png"

    img1.write_bytes(encode_png_rgb(w, h, bytes([10, 20, 30] * (w * h))))
    img2.write_bytes(encode_png_rgb(w, h, bytes([40, 50, 60] * (w * h))))

    res = compile_images_to_apng([img1, img2], out_apng, fps=2.0)
    assert res.exists()
    data = res.read_bytes()
    assert b"acTL" in data
    assert b"fcTL" in data
    meta = read_png_metadata(res)
    assert meta["FrameCount"] == "2"


def test_render_checkpoint_timelapse(tmp_path: Path):
    chk1 = tmp_path / "adapter_model_rev001.pt"
    chk2 = tmp_path / "adapter_model_rev002.pt"
    chk3 = tmp_path / "adapter_model_rev003.pt"
    out_timelapse = tmp_path / "timelapse.png"

    base_state = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(16, 16),
        "model.layers.0.mlp.up_proj.weight": torch.randn(16, 16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(16, 16),
    }
    torch.save(base_state, chk1)

    state2 = {k: v + 0.05 for k, v in base_state.items()}
    torch.save(state2, chk2)

    state3 = {k: v + 0.10 for k, v in base_state.items()}
    torch.save(state3, chk3)

    # Test delta mode
    res_delta = render_checkpoint_timelapse(
        checkpoint_paths=[chk1, chk2, chk3],
        output_path=out_timelapse,
        domain="test",
        mode="delta",
        fps=2.0,
        target_size=(16, 16),
        include_banner=True,
    )
    assert res_delta.exists()
    meta_delta = read_png_metadata(res_delta)
    assert meta_delta["TotalFrames"] == "2"
    assert meta_delta["Domain"] == "till-test"
    assert meta_delta["Mode"] == "delta"

    # Test dashboard mode
    out_dash = tmp_path / "timelapse_dash.png"
    res_dash = render_checkpoint_timelapse(
        checkpoint_paths=[chk1, chk2, chk3],
        output_path=out_dash,
        domain="test",
        mode="dashboard",
        fps=2.0,
        target_size=(32, 32),
        include_banner=True,
    )
    assert res_dash.exists()
    meta_dash = read_png_metadata(res_dash)
    assert meta_dash["Mode"] == "dashboard"


def test_extract_apng_frames(tmp_path):
    from unplow.visualizer import encode_apng_rgb, extract_apng_frames

    w, h = 16, 16
    f1 = bytes([255, 0, 0] * (w * h))
    f2 = bytes([0, 255, 0] * (w * h))
    apng_data = encode_apng_rgb([(w, h, f1), (w, h, f2)], delay_ms=250)

    extracted = extract_apng_frames(apng_data)
    assert len(extracted) == 2
    for idx, (meta, png) in enumerate(extracted):
        assert meta["frame_index"] == idx
        assert meta["width"] == w
        assert meta["height"] == h
        assert meta["delay_ms"] == 250
        assert png.startswith(b"\x89PNG\r\n\x1a\n")
        assert png.endswith(b"IEND\xaeB`\x82")


def test_filter_homogeneous_checkpoints(tmp_path):
    from unplow.visualizer import filter_homogeneous_checkpoints

    # Create 3 normal-sized checkpoints (e.g. 1000 bytes) and 1 mock checkpoint (100 bytes)
    p1 = tmp_path / "adapter_model_rev001.pt"
    p2 = tmp_path / "adapter_model_rev002.pt"
    p3 = tmp_path / "adapter_model_rev003.pt"
    p_mock = tmp_path / "adapter_model_rev004.pt"

    p1.write_bytes(b"A" * 1000)
    p2.write_bytes(b"B" * 1000)
    p3.write_bytes(b"C" * 1000)
    p_mock.write_bytes(b"M" * 50)

    filtered = filter_homogeneous_checkpoints([p1, p2, p3, p_mock])
    assert len(filtered) == 3
    expected = [
        "adapter_model_rev001.pt",
        "adapter_model_rev002.pt",
        "adapter_model_rev003.pt",
    ]
    assert [p.name for p in filtered] == expected


# end of file: unplow/tests/test_visualizer.py
