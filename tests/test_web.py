# Path: unplow/tests/test_web.py
"""Unit and integration tests for the Unplow Checkpoint Visualizer Web Studio.

Validates checkpoint scanning, live HTTP server endpoints, SPA serving,
real-time generation APIs, and image metadata extraction.
"""

from pathlib import Path
import threading
import time
from typing import Generator
import urllib.error
import urllib.request
import json

import pytest
import torch

from unplow.web import (
    ThreadingHTTPServer,
    UnplowWebHandler,
    generate_visualization_file,
    get_available_images,
    get_checkpoint_domains,
)


@pytest.fixture
def mock_checkpoints(tmp_path: Path):
    """Creates a temporary mock models directory with multi-domain checkpoints."""
    models_dir = tmp_path / "models"
    domain_dir = models_dir / "till-python-lora"
    domain_dir.mkdir(parents=True)

    state1 = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(16, 16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(16, 16),
    }
    state2 = {
        "model.layers.0.self_attn.q_proj.weight": state1["model.layers.0.self_attn.q_proj.weight"] + 0.05,
        "model.layers.0.self_attn.k_proj.weight": state1["model.layers.0.self_attn.k_proj.weight"] - 0.05,
        "model.layers.0.self_attn.o_proj.weight": state1["model.layers.0.self_attn.o_proj.weight"] + 0.02,
    }

    chk1 = domain_dir / "adapter_model_rev001.pt"
    chk2 = domain_dir / "adapter_model_rev002.pt"
    torch.save(state1, chk1)
    torch.save(state2, chk2)
    return chk1, chk2, models_dir


def test_get_checkpoint_domains(mock_checkpoints):
    chk1, chk2, models_dir = mock_checkpoints
    domains = get_checkpoint_domains(models_dir)

    assert "till-python" in domains
    items = domains["till-python"]
    assert len(items) == 2
    # Should be sorted descending by revision
    assert items[0]["revision"] == 2
    assert items[1]["revision"] == 1
    assert items[0]["filename"] == "adapter_model_rev002.pt"


def test_get_available_images(tmp_path: Path):
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    sample_png = target_dir / "sample.png"
    sample_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    images = get_available_images([target_dir])
    assert len(images) == 1
    assert images[0]["filename"] == "sample.png"
    assert images[0]["url"] == "/images/sample.png"


def test_generate_visualization_file(mock_checkpoints, tmp_path: Path):
    chk1, chk2, _ = mock_checkpoints
    res = generate_visualization_file({
        "mode": "dashboard",
        "checkpoint": str(chk2),
        "compare": str(chk1),
        "size": 32,
        "include_banner": True,
        "output_filename": "test_dash.png",
    })

    assert res["status"] == "success"
    assert res["filename"] == "test_dash.png"
    p = Path(res["path"])
    assert p.exists()
    assert p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "CheckpointTarget" in res["metadata"]


@pytest.fixture
def running_server() -> Generator[str, None, None]:
    """Spins up ThreadingHTTPServer on an ephemeral port in a daemon thread."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), UnplowWebHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    time.sleep(0.05)
    yield base_url

    server.shutdown()
    server.server_close()


def test_web_server_endpoints(running_server: str, mock_checkpoints):
    base_url = running_server
    chk1, chk2, _ = mock_checkpoints

    # 1. GET / (SPA index)
    with urllib.request.urlopen(f"{base_url}/") as resp:
        assert resp.status == 200
        assert "text/html" in resp.headers.get("Content-Type", "")
        content = resp.read().decode("utf-8")
        assert "UNPLOW" in content
        assert "VISUALIZER STUDIO" in content

    # 2. GET /api/checkpoints
    with urllib.request.urlopen(f"{base_url}/api/checkpoints") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "domains" in data

    # 3. GET /api/status
    with urllib.request.urlopen(f"{base_url}/api/status") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "storage" in data

    # 4. GET /api/images
    with urllib.request.urlopen(f"{base_url}/api/images") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert isinstance(data, list)

    # 5. POST /api/generate
    payload = json.dumps({
        "mode": "dashboard",
        "checkpoint": str(chk2),
        "compare": str(chk1),
        "size": 32,
        "include_banner": True,
        "output_filename": "live_test_mosaic.png",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res_data = json.loads(resp.read().decode("utf-8"))
        assert res_data["status"] == "success"
        assert res_data["filename"] == "live_test_mosaic.png"

    # 6. GET /images/live_test_mosaic.png
    with urllib.request.urlopen(f"{base_url}/images/live_test_mosaic.png") as resp:
        assert resp.status == 200
        assert "image/png" in resp.headers.get("Content-Type", "")
        img_bytes = resp.read()
        assert img_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    # 7. GET /nonexistent (404)
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{base_url}/nonexistent_route")
    assert exc_info.value.code == 404


# end of file: unplow/tests/test_web.py
