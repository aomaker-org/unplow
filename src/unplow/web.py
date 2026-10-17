# Path: unplow/src/unplow/web.py
"""
Interactive Browser Front End for Checkpoint & Differential Visualizations.

Provides a lightweight, zero-dependency, air-gapped web server (Directive 18)
using standard-library http.server. Features interactive checkpoint selection,
real-time heatmap & mosaic dashboard generation, metadata inspection,
pan/zoom viewing, and visualization history.
"""

from datetime import datetime, timezone
import getpass
import glob
import http.server
import json
import os
from pathlib import Path
import re
import socket
import socketserver
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.parse
import webbrowser

import torch

from .till_guard import TillDiskGuardian
from .visualizer import (
    compile_images_to_apng,
    compute_tensor_metrics,
    encode_apng_rgb,
    read_png_metadata,
    render_checkpoint_dashboard,
    render_checkpoint_timelapse,
    render_delta_heatmap,
    render_tensor_heatmap,
)

DEFAULT_PORT = 8765
DEFAULT_HOST = os.environ.get("UNPLOW_HOST", "0.0.0.0")


def get_headless_status() -> bool:
    """Detects if running on a headless server without active X11/Wayland display."""
    return not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")


def get_lan_ip_addresses() -> List[str]:
    """Identifies primary local network IP addresses for remote browser access."""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        primary_ip = s.getsockname()[0]
        s.close()
        if primary_ip and not primary_ip.startswith("127.") and primary_ip != "0.0.0.0":
            ips.append(primary_ip)
    except Exception:
        pass
    try:
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
        if host_ip and not host_ip.startswith("127.") and host_ip not in ips and host_ip != "0.0.0.0":
            ips.append(host_ip)
    except Exception:
        pass
    return ips or ["127.0.0.1"]


def get_checkpoint_domains(base_dir: Path = Path("models")) -> Dict[str, List[Dict[str, Any]]]:
    """Scans and categorizes available model checkpoints by language domain."""
    domains: Dict[str, List[Dict[str, Any]]] = {}
    pattern = str(base_dir / "till-*-lora" / "adapter_model_rev*.pt")
    for filepath in sorted(glob.glob(pattern)):
        p = Path(filepath)
        domain_name = p.parent.name.replace("-lora", "")
        m = re.search(r"rev(\d+)", p.name)
        rev = int(m.group(1)) if m else 0
        stat = p.stat()
        item = {
            "filename": p.name,
            "path": str(p),
            "domain": domain_name,
            "revision": rev,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
        domains.setdefault(domain_name, []).append(item)

    # Sort each domain descending by revision (latest first)
    for d in domains:
        domains[d].sort(key=lambda x: x["revision"], reverse=True)
    return domains


def get_available_images(search_dirs: Optional[List[Path]] = None) -> List[Dict[str, Any]]:
    """Discovers all visualization PNGs and parses their embedded tEXt metadata."""
    if search_dirs is None:
        search_dirs = [Path("workspace/visualizations"), Path("workspace/targets")]

    found_images: List[Dict[str, Any]] = []
    seen_names = set()

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for p in sorted(s_dir.glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True):
            if p.name in seen_names:
                continue
            seen_names.add(p.name)
            stat = p.stat()
            meta = read_png_metadata(p)
            metrics_data = {}
            if "Metrics" in meta:
                try:
                    metrics_data = json.loads(meta["Metrics"])
                except Exception:
                    metrics_data = {}

            is_timelapse = (
                "TotalFrames" in meta
                or "FrameCount" in meta
                or "timelapse" in p.name.lower()
            )
            frames_count = int(meta.get("TotalFrames", meta.get("FrameCount", 1)))
            fps_val = float(meta.get("Fps", 2.0))

            found_images.append({
                "filename": p.name,
                "path": str(p),
                "url": f"/images/{p.name}",
                "size_bytes": stat.st_size,
                "size_kb": round(stat.st_size / 1024, 1),
                "mtime": stat.st_mtime,
                "mtime_iso": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "title": meta.get("Title", p.stem),
                "target": meta.get("CheckpointTarget", ""),
                "compare": meta.get("CheckpointCompare", ""),
                "tensor": meta.get("TensorName", ""),
                "mode": meta.get("Mode", "single"),
                "metrics": metrics_data,
                "raw_metadata": meta,
                "is_timelapse": is_timelapse,
                "frames_count": frames_count,
                "fps": fps_val,
            })

    found_images.sort(key=lambda x: x["mtime"], reverse=True)
    return found_images


def generate_visualization_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """Executes visualization rendering request and saves PNG with embedded metadata."""
    mode = params.get("mode", "dashboard")
    size_dim = int(params.get("size", 512))
    size = (size_dim, size_dim)
    include_banner = bool(params.get("include_banner", True))

    out_dir = Path("workspace/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    custom_name = params.get("output_filename")

    if mode == "timelapse":
        domain = params.get("domain", "python").replace("till-", "")
        if domain == "all" and params.get("checkpoint"):
            m = re.search(r"till-([a-zA-Z0-9_-]+)-lora", params.get("checkpoint", ""))
            domain = m.group(1) if m else "python"
        elif domain == "all":
            domain = "python"

        timelapse_mode = params.get("timelapse_mode", "delta")
        fps = float(params.get("fps", 2.0))
        tensor_req = params.get("tensor")
        max_revision = params.get("max_revision")

        domain_dir = Path("models") / f"till-{domain}-lora"
        if not domain_dir.exists():
            candidates = list(Path(".").glob(f"**/till-{domain}-lora"))
            if candidates:
                domain_dir = candidates[0]
            else:
                raise FileNotFoundError(f"Checkpoint domain directory 'till-{domain}-lora' not found.")

        all_ckpts = sorted(domain_dir.glob("adapter_model_rev*.pt"))
        if not all_ckpts:
            raise FileNotFoundError(f"No checkpoint files found in {domain_dir}")

        if max_revision:
            try:
                max_rev_int = int(max_revision)
                all_ckpts = [
                    p
                    for p in all_ckpts
                    if re.search(r"rev(\d+)", p.name)
                    and int(re.search(r"rev(\d+)", p.name).group(1)) <= max_rev_int
                ]
            except Exception:
                pass

        custom_name = params.get("output_filename")
        if custom_name:
            out_name = custom_name if custom_name.endswith(".png") else f"{custom_name}.png"
        else:
            out_name = f"timelapse_{domain}_{timelapse_mode}_{timestamp}.png"

        out_file = out_dir / out_name
        res = render_checkpoint_timelapse(
            checkpoint_paths=all_ckpts,
            output_path=out_file,
            domain=domain,
            tensor_name=tensor_req,
            mode=timelapse_mode,
            fps=fps,
            target_size=size,
            include_banner=include_banner,
        )
        meta = read_png_metadata(res)
        return {
            "status": "success",
            "filename": res.name,
            "path": str(res),
            "url": f"/images/{res.name}",
            "size_bytes": res.stat().st_size,
            "metadata": meta,
            "metrics": {},
            "is_timelapse": True,
            "frames_count": len(all_ckpts),
            "fps": fps,
        }

    target_path = Path(params.get("checkpoint", ""))
    if not target_path.exists():
        raise FileNotFoundError(f"Target checkpoint not found: {target_path}")

    compare_path_str = params.get("compare")
    compare_path = Path(compare_path_str) if compare_path_str else None
    if compare_path and not compare_path.exists():
        raise FileNotFoundError(f"Comparison checkpoint not found: {compare_path}")

    mode = params.get("mode", "dashboard" if compare_path else "raw")
    t_name = target_path.stem

    if custom_name:
        out_name = custom_name if custom_name.endswith(".png") else f"{custom_name}.png"
    elif mode == "dashboard" and compare_path:
        out_name = f"dashboard_{t_name}_vs_{compare_path.stem}_{timestamp}.png"
    elif compare_path:
        out_name = f"delta_{t_name}_vs_{compare_path.stem}_{timestamp}.png"
    else:
        out_name = f"raw_{t_name}_{timestamp}.png"

    out_file = out_dir / out_name

    if mode == "dashboard":
        if not compare_path:
            raise ValueError("Dashboard mode requires a baseline comparison checkpoint.")
        res = render_checkpoint_dashboard(
            checkpoint_old=compare_path,
            checkpoint_new=target_path,
            output_path=out_file,
            grid_size=size,
            include_banner=include_banner,
        )
    elif compare_path:
        w_old = torch.load(compare_path, map_location="cpu", weights_only=False)
        w_new = torch.load(target_path, map_location="cpu", weights_only=False)
        tensor_req = params.get("tensor")
        tensor_name = tensor_req or next((k for k in w_new if "q_proj" in k and "weight" in k), list(w_new.keys())[0])
        if tensor_name not in w_old or tensor_name not in w_new:
            raise KeyError(f"Tensor '{tensor_name}' not present in both checkpoints.")
        res = render_delta_heatmap(
            tensor_old=w_old[tensor_name],
            tensor_new=w_new[tensor_name],
            output_path=out_file,
            target_size=size,
            checkpoint_target=target_path,
            checkpoint_compare=compare_path,
            tensor_name=tensor_name,
            include_banner=include_banner,
        )
    else:
        w = torch.load(target_path, map_location="cpu", weights_only=False)
        tensor_req = params.get("tensor")
        tensor_name = tensor_req or next((k for k in w if "q_proj" in k and "weight" in k), list(w.keys())[0])
        if tensor_name not in w:
            raise KeyError(f"Tensor '{tensor_name}' not found in checkpoint.")
        res = render_tensor_heatmap(
            tensor=w[tensor_name],
            output_path=out_file,
            target_size=size,
            checkpoint_path=target_path,
            tensor_name=tensor_name,
            include_banner=include_banner,
        )

    meta = read_png_metadata(res)
    metrics_data = {}
    if "Metrics" in meta:
        try:
            metrics_data = json.loads(meta["Metrics"])
        except Exception:
            metrics_data = {}

    return {
        "status": "success",
        "filename": res.name,
        "path": str(res),
        "url": f"/images/{res.name}",
        "size_bytes": res.stat().st_size,
        "metadata": meta,
        "metrics": metrics_data,
    }


SPA_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unplow - Neural Checkpoint Visualizer</title>
<style>
  :root {
    --bg-main: #0b0e14;
    --bg-panel: #141a23;
    --bg-panel-alt: #1a2330;
    --border-color: #273142;
    --border-focus: #00d2ff;
    --text-main: #e6edf3;
    --text-muted: #8b949e;
    --cyan: #00d2ff;
    --amber: #ff5722;
    --emerald: #00e676;
    --purple: #b388ff;
  }
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace, sans-serif;
  }
  body {
    background: var(--bg-main);
    color: var(--text-main);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Navbar */
  header {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border-color);
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo { font-size: 1.25rem; font-weight: 700; color: var(--cyan); letter-spacing: -0.5px; }
  .badge {
    background: #004d61;
    color: var(--cyan);
    font-size: 0.75rem;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 600;
  }
  .header-telemetry { display: flex; gap: 16px; font-size: 0.85rem; color: var(--text-muted); }
  .status-pill { display: inline-flex; align-items: center; gap: 6px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--emerald); }

  /* Main Workspace */
  main {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  /* Left Panel: Controls */
  .control-panel {
    width: 380px;
    background: var(--bg-panel);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: 16px;
    gap: 16px;
    flex-shrink: 0;
  }
  .section-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 8px;
    font-weight: 700;
  }
  .form-group { display: flex; flex-direction: column; gap: 6px; }
  label { font-size: 0.85rem; font-weight: 500; color: var(--text-main); }
  select, input[type="text"], input[type="number"] {
    background: var(--bg-panel-alt);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    padding: 8px 10px;
    border-radius: 6px;
    font-size: 0.85rem;
    outline: none;
    transition: border 0.15s;
  }
  select:focus, input:focus { border-color: var(--border-focus); }
  .mode-tabs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
  .tab-btn {
    background: var(--bg-panel-alt);
    border: 1px solid var(--border-color);
    color: var(--text-muted);
    padding: 8px 4px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
  }
  .tab-btn.active {
    background: #003644;
    border-color: var(--cyan);
    color: var(--cyan);
  }
  .video-toolbar {
    display: none;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: #0f141c;
    border-bottom: 1px solid var(--border-color);
    font-size: 0.8rem;
    flex-shrink: 0;
  }
  .scrubber-slider {
    flex: 1;
    accent-color: var(--cyan);
    cursor: pointer;
  }
  .playback-btn {
    background: var(--bg-panel-alt);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    cursor: pointer;
  }
  .playback-btn:hover { border-color: var(--cyan); color: var(--cyan); }
  .btn-submit {
    background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
    color: #ffffff;
    border: none;
    padding: 12px;
    border-radius: 6px;
    font-size: 0.95rem;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(0, 180, 216, 0.25);
    transition: transform 0.1s, opacity 0.15s;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
  }
  .btn-submit:hover { opacity: 0.95; }
  .btn-submit:active { transform: scale(0.98); }
  .btn-submit:disabled { opacity: 0.5; cursor: not-allowed; }

  /* Right Workspace: Viewer & Inspector */
  .workspace-view {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .viewer-toolbar {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border-color);
    padding: 8px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
    flex-shrink: 0;
  }
  .zoom-controls { display: flex; gap: 6px; }
  .tool-btn {
    background: var(--bg-panel-alt);
    border: 1px solid var(--border-color);
    color: var(--text-main);
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.8rem;
    cursor: pointer;
  }
  .tool-btn:hover { border-color: var(--cyan); color: var(--cyan); }

  .canvas-stage {
    flex: 1;
    background: #06080b;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: auto;
    position: relative;
    padding: 24px;
  }
  .canvas-stage img {
    max-width: 95%;
    max-height: 95%;
    box-shadow: 0 10px 30px rgba(0,0,0,0.8);
    border-radius: 6px;
    border: 1px solid var(--border-color);
    transition: transform 0.15s ease-out;
  }

  /* Metadata Drawer */
  .metadata-card {
    background: var(--bg-panel);
    border-top: 1px solid var(--border-color);
    padding: 12px 20px;
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    font-size: 0.8rem;
    flex-shrink: 0;
  }
  .meta-col { display: flex; flex-direction: column; gap: 2px; }
  .meta-label { color: var(--text-muted); font-size: 0.7rem; text-transform: uppercase; }
  .meta-val { color: var(--cyan); font-weight: 600; font-family: monospace; }

  /* Bottom Gallery Bar */
  .gallery-bar {
    background: #0d121a;
    border-top: 1px solid var(--border-color);
    padding: 10px 16px;
    display: flex;
    gap: 12px;
    overflow-x: auto;
    align-items: center;
    flex-shrink: 0;
    height: 110px;
  }
  .gallery-thumb {
    width: 80px;
    height: 80px;
    border-radius: 6px;
    border: 2px solid var(--border-color);
    overflow: hidden;
    cursor: pointer;
    flex-shrink: 0;
    position: relative;
    transition: all 0.15s;
  }
  .gallery-thumb:hover { border-color: var(--cyan); transform: translateY(-2px); }
  .gallery-thumb.active { border-color: var(--cyan); box-shadow: 0 0 8px rgba(0,210,255,0.4); }
  .gallery-thumb img { width: 100%; height: 100%; object-fit: cover; }
  .thumb-label {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: rgba(11, 14, 20, 0.85);
    font-size: 0.65rem;
    padding: 2px 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: center;
  }

  /* Spinner */
  .spinner {
    border: 3px solid rgba(255,255,255,0.1);
    border-top: 3px solid #ffffff;
    border-radius: 50%;
    width: 16px;
    height: 16px;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

  /* Remote SSH Modal */
  .modal-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0, 0, 0, 0.75);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
  }
  .modal-box {
    background: var(--bg-panel);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    width: 520px;
    max-width: 90%;
    padding: 20px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
  }
  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 8px;
  }
  .modal-close {
    background: none;
    border: none;
    color: var(--text-muted);
    font-size: 1.2rem;
    cursor: pointer;
  }
  .modal-close:hover { color: var(--text-main); }
  .code-box {
    background: #06080b;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 8px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: monospace;
    font-size: 0.82rem;
    color: var(--cyan);
    margin: 8px 0;
  }
  .ssh-badge {
    background: #102a3a;
    border: 1px solid #00d2ff;
    color: #00d2ff;
    font-size: 0.72rem;
    padding: 3px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 600;
  }
  .ssh-badge:hover { background: #003644; }
</style>
</head>
<body>

<!-- Remote SSH Access Guide Modal -->
<div id="ssh-modal" class="modal-overlay" style="display: none;">
  <div class="modal-box">
    <div class="modal-header">
      <h3 style="font-size: 1rem; color: var(--text-main);">Remote Access from Windows 11 Host</h3>
      <button class="modal-close" onclick="closeSshModal()">&times;</button>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 12px;">
      Server <code style="color: var(--cyan);">fg26</code> is running headlessly.
      Connect from your Windows 11 host via:
    </div>

    <div class="section-title">Option 1: SSH Port Forwarding (Recommended & Encrypted)</div>
    <div class="code-box">
      <span id="ssh-cmd-text">ssh -L 8765:localhost:8765 fekerr@fg26</span>
      <button class="tool-btn" onclick="copySshCmd()">Copy</button>
    </div>
    <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 14px;">
      Run in Windows PowerShell or Windows Terminal, then open
      <a href="http://localhost:8765" target="_blank" style="color: var(--cyan);">http://localhost:8765</a>
    </div>

    <div class="section-title">Option 2: Direct LAN Connection</div>
    <div style="font-size: 0.82rem; margin-top: 4px; display: flex; flex-direction: column; gap: 4px;">
      <div>Direct Hostname:
        <a id="lan-link-hostname" href="http://fg26:8765" target="_blank" style="color: var(--cyan);">
          http://fg26:8765
        </a>
      </div>
      <div>Direct IP:
        <a id="lan-link-ip" href="http://10.0.0.31:8765" target="_blank" style="color: var(--cyan);">
          http://10.0.0.31:8765
        </a>
      </div>
    </div>
  </div>
</div>

<header>
  <div class="brand">
    <div class="logo">UNPLOW</div>
    <div class="badge">VISUALIZER STUDIO</div>
    <button class="ssh-badge" onclick="openSshModal()">SSH Remote Guide</button>
  </div>
  <div class="header-telemetry">
    <div class="status-pill"><div class="dot"></div> <span id="telemetry-status">Engine Active</span></div>
    <div id="storage-status">Disk: Checking...</div>
    <div id="server-host-badge" style="color: var(--cyan); font-family: monospace;">fg26:8765</div>
  </div>
</header>

<main>
  <!-- Left Control Panel -->
  <aside class="control-panel">
    <div class="form-group">
      <div class="section-title">Visualization Mode</div>
      <div class="mode-tabs">
        <button class="tab-btn active" data-mode="dashboard" onclick="setMode('dashboard')">Dashboard</button>
        <button class="tab-btn" data-mode="delta" onclick="setMode('delta')">Differential</button>
        <button class="tab-btn" data-mode="raw" onclick="setMode('raw')">Raw Weights</button>
        <button class="tab-btn" data-mode="timelapse" onclick="setMode('timelapse')">Timelapse</button>
      </div>
    </div>

    <div class="form-group" id="group-timelapse-type" style="display: none;">
      <label for="select-timelapse-type">Timelapse Mode</label>
      <select id="select-timelapse-type" onchange="onTimelapseTypeChange()">
        <option value="delta" selected>Cumulative Delta (vs rev001)</option>
        <option value="step">Step-by-Step Delta (vs rev[i-1])</option>
        <option value="raw">Raw Weight Intensity Heatmap</option>
        <option value="dashboard">Attention Mosaic Evolution</option>
      </select>
    </div>

    <div class="form-group" id="group-timelapse-fps" style="display: none;">
      <label for="select-timelapse-fps">Video Frame Rate</label>
      <select id="select-timelapse-fps">
        <option value="1">1.0 fps (Slow Inspection)</option>
        <option value="2" selected>2.0 fps (Standard)</option>
        <option value="5">5.0 fps (Fast Motion)</option>
        <option value="10">10.0 fps (High Speed)</option>
      </select>
    </div>

    <div class="form-group">
      <label for="select-domain">Domain Filter</label>
      <select id="select-domain" onchange="onDomainChange()">
        <option value="all">All Domains</option>
      </select>
    </div>

    <div class="form-group">
      <label for="select-target">Target Checkpoint</label>
      <select id="select-target" onchange="onTargetChange()"></select>
    </div>

    <div class="form-group" id="group-compare">
      <label for="select-compare">Baseline Checkpoint (Compare)</label>
      <select id="select-compare"></select>
    </div>

    <div class="form-group" id="group-tensor">
      <label for="select-tensor">Attention / MLP Tensor</label>
      <select id="select-tensor">
        <option value="model.layers.27.self_attn.q_proj.weight">Layer 27 Q-Projection (Query)</option>
        <option value="model.layers.27.self_attn.k_proj.weight">Layer 27 K-Projection (Key)</option>
        <option value="model.layers.27.self_attn.v_proj.weight">Layer 27 V-Projection (Value)</option>
        <option value="model.layers.27.self_attn.o_proj.weight">Layer 27 O-Projection (Output)</option>
        <option value="model.layers.27.mlp.gate_proj.weight">Layer 27 MLP Gate-Projection</option>
        <option value="model.layers.27.mlp.up_proj.weight">Layer 27 MLP Up-Projection</option>
        <option value="model.layers.27.mlp.down_proj.weight">Layer 27 MLP Down-Projection</option>
      </select>
    </div>

    <div class="form-group">
      <label for="select-size">Render Resolution</label>
      <select id="select-size">
        <option value="512" selected>512 x 512</option>
        <option value="256">256 x 256 (Fast)</option>
        <option value="1024">1024 x 1024 (HD)</option>
      </select>
    </div>

    <div class="form-group" style="flex-direction: row; align-items: center; gap: 8px;">
      <input type="checkbox" id="check-banner" checked>
      <label for="check-banner" style="font-size: 0.8rem; cursor: pointer;">Include Comparison Header Banner</label>
    </div>

    <button id="btn-generate" class="btn-submit" onclick="triggerGenerate()">
      <span id="btn-spinner" class="spinner" style="display: none;"></span>
      <span id="btn-text">Generate Visualization</span>
    </button>
  </aside>

  <!-- Right Viewer & Gallery -->
  <section class="workspace-view">
    <div class="viewer-toolbar">
      <div id="view-filename" style="font-weight: 600; color: var(--cyan);">No Image Selected</div>
      <div class="zoom-controls">
        <button class="tool-btn" onclick="zoomImage(0.2)">+ Zoom In</button>
        <button class="tool-btn" onclick="zoomImage(-0.2)">- Zoom Out</button>
        <button class="tool-btn" onclick="resetZoom()">100%</button>
        <button class="tool-btn" onclick="downloadActive()">Download File</button>
      </div>
    </div>

    <div class="video-toolbar" id="video-toolbar">
      <button class="playback-btn" id="btn-play-pause" onclick="togglePlayPause()">⏸ Pause</button>
      <button class="playback-btn" onclick="stepFrame(-1)">◀ Prev</button>
      <button class="playback-btn" onclick="stepFrame(1)">▶ Next</button>
      <input type="range" class="scrubber-slider" id="video-scrubber"
             min="1" max="1" value="1" oninput="onScrubVideo(this.value)">
      <span id="video-frame-indicator"
            style="font-family: monospace; min-width: 90px; color: var(--cyan);">Frame 1 / 1</span>
      <select id="video-speed-select" onchange="changePlaybackSpeed()" style="padding: 2px 6px; font-size: 0.75rem;">
        <option value="0.5">0.5x</option>
        <option value="1.0" selected>1.0x</option>
        <option value="2.0">2.0x</option>
        <option value="5.0">5.0x</option>
      </select>
    </div>

    <div class="canvas-stage" id="stage">
      <img id="main-image" src="" alt="Neural Heatmap" style="display: none;">
      <canvas id="canvas-freeze"
              style="display: none; max-width: 100%; max-height: 100%; object-fit: contain;"></canvas>
    </div>

    <!-- Metadata Pill Bar -->
    <div class="metadata-card" id="meta-card">
      <div class="meta-col">
        <div class="meta-label">Target Checkpoint</div>
        <div class="meta-val" id="meta-target">-</div>
      </div>
      <div class="meta-col">
        <div class="meta-label">Baseline Compare</div>
        <div class="meta-val" id="meta-compare">-</div>
      </div>
      <div class="meta-col">
        <div class="meta-label">Tensor / Projection</div>
        <div class="meta-val" id="meta-tensor">-</div>
      </div>
      <div class="meta-col">
        <div class="meta-label">Mean |&Delta;W|</div>
        <div class="meta-val" id="meta-mean">-</div>
      </div>
      <div class="meta-col">
        <div class="meta-label">Max |&Delta;W|</div>
        <div class="meta-val" id="meta-max">-</div>
      </div>
      <div class="meta-col">
        <div class="meta-label">Dimensions</div>
        <div class="meta-val" id="meta-size">-</div>
      </div>
      <div class="meta-col">
        <div class="meta-label">Embedded Provenance</div>
        <div class="meta-val" id="meta-software">-</div>
      </div>
    </div>

    <!-- Visualizer History Thumbnails -->
    <div class="gallery-bar" id="gallery">
      <!-- Thumbnails populated dynamically -->
    </div>
  </section>
</main>

<script>
  let currentMode = 'dashboard';
  let checkpointsData = {};
  let currentZoom = 1.0;
  let activeImage = null;

  async function init() {
    await fetchCheckpoints();
    await fetchGallery();
    await fetchStatus();
    setInterval(fetchStatus, 5000);
  }

  function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
    const isTl = (mode === 'timelapse');
    document.getElementById('group-compare').style.display = (mode === 'raw' || isTl) ? 'none' : 'flex';
    document.getElementById('group-timelapse-type').style.display = isTl ? 'flex' : 'none';
    document.getElementById('group-timelapse-fps').style.display = isTl ? 'flex' : 'none';
    const tType = document.getElementById('select-timelapse-type').value;
    const hideTensor = (mode === 'dashboard' || (isTl && tType === 'dashboard'));
    document.getElementById('group-tensor').style.display = hideTensor ? 'none' : 'flex';
    document.getElementById('btn-text').textContent = isTl ? 'Generate Timelapse Video' : 'Generate Visualization';
  }

  function onTimelapseTypeChange() {
    const tType = document.getElementById('select-timelapse-type').value;
    document.getElementById('group-tensor').style.display = (tType === 'dashboard') ? 'none' : 'flex';
  }

  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.storage) {
        document.getElementById('storage-status').textContent =
          `Free: ${data.storage.free_gb.toFixed(1)} GB (${data.storage.free_percent.toFixed(1)}%)`;
      }
      if (data.active_iteration) {
        document.getElementById('telemetry-status').textContent = `Pipeline Iteration #${data.active_iteration}`;
      }
      if (data.server) {
        const badge = document.getElementById('server-host-badge');
        badge.textContent = `${data.server.hostname}:${window.location.port || 8765}`;
        if (data.server.ssh_command) {
          document.getElementById('ssh-cmd-text').textContent = data.server.ssh_command;
        }
        if (data.server.lan_ips && data.server.lan_ips.length > 0) {
          const ip = data.server.lan_ips[0];
          const port = window.location.port || 8765;
          const ipEl = document.getElementById('lan-link-ip');
          ipEl.textContent = `http://${ip}:${port}`;
          ipEl.href = `http://${ip}:${port}`;
          const hostEl = document.getElementById('lan-link-hostname');
          hostEl.textContent = `http://${data.server.hostname}:${port}`;
          hostEl.href = `http://${data.server.hostname}:${port}`;
        }
      }
    } catch (e) {
      console.warn("Status fetch failed", e);
    }
  }

  function openSshModal() {
    document.getElementById('ssh-modal').style.display = 'flex';
  }

  function closeSshModal() {
    document.getElementById('ssh-modal').style.display = 'none';
  }

  function copySshCmd() {
    const txt = document.getElementById('ssh-cmd-text').textContent;
    navigator.clipboard.writeText(txt).then(() => {
      alert('Copied SSH tunnel command to clipboard: ' + txt);
    });
  }

  async function fetchCheckpoints() {
    try {
      const res = await fetch('/api/checkpoints');
      const data = await res.json();
      checkpointsData = data.domains || {};

      const domSelect = document.getElementById('select-domain');
      domSelect.innerHTML = '<option value="all">All Domains</option>';
      Object.keys(checkpointsData).forEach(d => {
        const opt = document.createElement('option');
        opt.value = d;
        opt.textContent = `${d} (${checkpointsData[d].length} revs)`;
        domSelect.appendChild(opt);
      });
      populateCheckpointDropdowns();
    } catch (e) {
      console.error("Failed to load checkpoints", e);
    }
  }

  function onDomainChange() {
    populateCheckpointDropdowns();
  }

  function populateCheckpointDropdowns() {
    const domainFilter = document.getElementById('select-domain').value;
    const targetSel = document.getElementById('select-target');
    const compareSel = document.getElementById('select-compare');

    targetSel.innerHTML = '';
    compareSel.innerHTML = '';

    let items = [];
    Object.keys(checkpointsData).forEach(d => {
      if (domainFilter === 'all' || domainFilter === d) {
        items.push(...checkpointsData[d]);
      }
    });

    items.sort((a, b) => b.mtime - a.mtime);

    items.forEach((item, idx) => {
      const optT = document.createElement('option');
      optT.value = item.path;
      optT.textContent = `[${item.domain}] rev${String(item.revision).padStart(3, '0')} (${item.filename})`;
      targetSel.appendChild(optT);

      const optC = document.createElement('option');
      optC.value = item.path;
      optC.textContent = `[${item.domain}] rev${String(item.revision).padStart(3, '0')} (${item.filename})`;
      compareSel.appendChild(optC);
    });

    if (items.length > 1) {
      compareSel.selectedIndex = 1; // Default to previous checkpoint
    }
  }

  function onTargetChange() {
    const targetPath = document.getElementById('select-target').value;
    const compareSel = document.getElementById('select-compare');
    const targetItem = findCheckpointByPath(targetPath);
    if (!targetItem) return;

    // Smart default: find the previous revision in the same domain
    const domainList = checkpointsData[targetItem.domain] || [];
    const prevItem = domainList.find(x => x.revision === targetItem.revision - 1);
    if (prevItem) {
      compareSel.value = prevItem.path;
    }
  }

  function findCheckpointByPath(p) {
    for (const d of Object.keys(checkpointsData)) {
      const match = checkpointsData[d].find(x => x.path === p);
      if (match) return match;
    }
    return null;
  }

  async function fetchGallery() {
    try {
      const res = await fetch('/api/images');
      const images = await res.json();
      const gallery = document.getElementById('gallery');
      gallery.innerHTML = '';

      if (images.length === 0) {
        gallery.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem; ' +
          'padding: 10px;">No images generated yet. Click Generate!</div>';
        return;
      }

      images.forEach((img, idx) => {
        const thumb = document.createElement('div');
        thumb.className = `gallery-thumb ${idx === 0 ? 'active' : ''}`;
        thumb.title = `${img.filename} (${img.size_kb} KB)`;
        thumb.onclick = () => selectActiveImage(img);

        const imgEl = document.createElement('img');
        imgEl.src = img.url;
        const lbl = document.createElement('div');
        lbl.className = 'thumb-label';
        lbl.textContent = img.filename;

        thumb.appendChild(imgEl);
        thumb.appendChild(lbl);
        gallery.appendChild(thumb);
      });

      if (images.length > 0 && !activeImage) {
        selectActiveImage(images[0]);
      }
    } catch (e) {
      console.error("Gallery fetch error", e);
    }
  }

  let isPaused = false;

  function togglePlayPause() {
    const img = document.getElementById('main-image');
    const cvs = document.getElementById('canvas-freeze');
    const btn = document.getElementById('btn-play-pause');
    if (!isPaused) {
      cvs.width = img.naturalWidth || 512;
      cvs.height = img.naturalHeight || 512;
      const ctx = cvs.getContext('2d');
      ctx.drawImage(img, 0, 0, cvs.width, cvs.height);
      img.style.display = 'none';
      cvs.style.display = 'block';
      btn.textContent = '▶ Play';
      isPaused = true;
    } else {
      cvs.style.display = 'none';
      img.style.display = 'block';
      btn.textContent = '⏸ Pause';
      isPaused = false;
    }
  }

  function stepFrame(delta) {
    const scrubber = document.getElementById('video-scrubber');
    const maxVal = parseInt(scrubber.max, 10) || 1;
    let nextVal = parseInt(scrubber.value, 10) + delta;
    if (nextVal < 1) nextVal = 1;
    if (nextVal > maxVal) nextVal = maxVal;
    scrubber.value = nextVal;
    onScrubVideo(nextVal);
  }

  function onScrubVideo(val) {
    const scrubber = document.getElementById('video-scrubber');
    const maxVal = parseInt(scrubber.max, 10) || 1;
    document.getElementById('video-frame-indicator').textContent = `Frame ${val} / ${maxVal}`;
  }

  function changePlaybackSpeed() {
    // APNG standard playback rate is baked into frame delay chunks
  }

  function selectActiveImage(imgData) {
    activeImage = imgData;
    const mainImg = document.getElementById('main-image');
    mainImg.src = imgData.url;
    mainImg.style.display = 'block';
    const cvs = document.getElementById('canvas-freeze');
    cvs.style.display = 'none';
    isPaused = false;
    document.getElementById('btn-play-pause').textContent = '⏸ Pause';

    document.querySelectorAll('.gallery-thumb').forEach(t => {
      t.classList.toggle('active', t.title.startsWith(imgData.filename));
    });

    const isTl = (
      imgData.is_timelapse
      || (imgData.raw_metadata && (imgData.raw_metadata.TotalFrames || imgData.raw_metadata.FrameCount))
    );
    const vToolbar = document.getElementById('video-toolbar');
    vToolbar.style.display = isTl ? 'flex' : 'none';

    if (isTl) {
      const fc = (
        imgData.frames_count
        || (imgData.raw_metadata && (imgData.raw_metadata.TotalFrames || imgData.raw_metadata.FrameCount))
        || 32
      );
      const scrubber = document.getElementById('video-scrubber');
      scrubber.max = fc;
      scrubber.value = 1;
      document.getElementById('video-frame-indicator').textContent = `Frame 1 / ${fc}`;
    }

    document.getElementById('view-filename').textContent = imgData.filename;
    const finalTarget = (
      imgData.target
      || (isTl && imgData.raw_metadata && imgData.raw_metadata.FinalCheckpoint)
      || imgData.filename
    );
    document.getElementById('meta-target').textContent = finalTarget;
    const initComp = (
      imgData.compare
      || (isTl && imgData.raw_metadata && imgData.raw_metadata.InitialCheckpoint)
      || (isTl ? 'rev001' : '(None)')
    );
    document.getElementById('meta-compare').textContent = initComp;
    const tensorLabel = (
      imgData.tensor
      || (isTl ? 'Timelapse Sequence' : (imgData.mode === 'dashboard' ? 'Multi-Projection Mosaic' : '-'))
    );
    document.getElementById('meta-tensor').textContent = tensorLabel;

    if (imgData.metrics && (imgData.metrics.mean_abs !== undefined || imgData.metrics.max_abs !== undefined)) {
      document.getElementById('meta-mean').textContent =
        imgData.metrics.mean_abs !== undefined ? imgData.metrics.mean_abs.toExponential(3) : '-';
      document.getElementById('meta-max').textContent =
        imgData.metrics.max_abs !== undefined ? imgData.metrics.max_abs.toExponential(3) : '-';
    } else {
      document.getElementById('meta-mean').textContent = isTl ? 'Animated Delta' : '-';
      document.getElementById('meta-max').textContent = isTl ? `${imgData.fps || 2.0} fps` : '-';
    }

    document.getElementById('meta-size').textContent = `${imgData.size_kb} KB`;
    const softLabel = (imgData.raw_metadata && imgData.raw_metadata.Software) || 'unplow v0.17.0';
    document.getElementById('meta-software').textContent = softLabel;
    resetZoom();
  }

  async function triggerGenerate() {
    const btn = document.getElementById('btn-generate');
    const spinner = document.getElementById('btn-spinner');
    const btnText = document.getElementById('btn-text');

    btn.disabled = true;
    spinner.style.display = 'inline-block';
    const isTl = (currentMode === 'timelapse');
    btnText.textContent = isTl ? 'Compiling Timelapse Video...' : 'Rendering Neural Heatmap...';

    const payload = {
      mode: currentMode,
      domain: document.getElementById('select-domain').value,
      checkpoint: document.getElementById('select-target').value,
      compare: (currentMode === 'raw' || isTl) ? null : document.getElementById('select-compare').value,
      timelapse_mode: document.getElementById('select-timelapse-type').value,
      fps: parseFloat(document.getElementById('select-timelapse-fps').value),
      tensor: document.getElementById('select-tensor').value,
      size: parseInt(document.getElementById('select-size').value, 10),
      include_banner: document.getElementById('check-banner').checked,
    };

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.status === 'success') {
        await fetchGallery();
      } else {
        alert("Generation failed: " + (data.error || "Unknown error"));
      }
    } catch (e) {
      alert("Error generating visualization: " + e.message);
    } finally {
      btn.disabled = false;
      spinner.style.display = 'none';
      btnText.textContent = isTl ? 'Generate Timelapse Video' : 'Generate Visualization';
    }
  }

  function zoomImage(delta) {
    currentZoom = Math.max(0.4, Math.min(4.0, currentZoom + delta));
    const img = document.getElementById('main-image');
    img.style.transform = `scale(${currentZoom})`;
  }

  function resetZoom() {
    currentZoom = 1.0;
    const img = document.getElementById('main-image');
    img.style.transform = 'scale(1.0)';
  }

  function downloadActive() {
    if (!activeImage) return;
    const a = document.createElement('a');
    a.href = activeImage.url;
    a.download = activeImage.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  window.onload = init;
</script>
</body>
</html>
"""


class UnplowWebHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request Handler for the Unplow Checkpoint Visualizer Web Studio."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP request logging clutter to keep terminal clean."""
        pass

    def send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            html_bytes = SPA_HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)
            return

        if path == "/api/checkpoints":
            domains = get_checkpoint_domains()
            self.send_json({"domains": domains})
            return

        if path == "/api/images":
            images = get_available_images()
            self.send_json(images)
            return

        if path == "/api/status":
            guardian = TillDiskGuardian(min_free_pct=15.0)
            telem = guardian.check_disk()
            active_iter = 0
            log_path = Path(".gemini/antigravity-cli/brain")
            for log_file in Path(".").glob("**/*.log"):
                if "task" in log_file.name:
                    try:
                        content = log_file.read_text(encoding="utf-8", errors="ignore")
                        m = re.findall(r"Iteration #(\d+)", content)
                        if m:
                            active_iter = max(active_iter, int(m[-1]))
                    except Exception:
                        pass
            hostname = socket.gethostname()
            username = getpass.getuser()
            lan_ips = get_lan_ip_addresses()
            port = self.server.server_address[1]
            self.send_json({
                "storage": telem.to_dict(),
                "active_iteration": active_iter or 8,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "server": {
                    "hostname": hostname,
                    "username": username,
                    "lan_ips": lan_ips,
                    "port": port,
                    "ssh_command": f"ssh -L {port}:localhost:{port} {username}@{hostname}",
                    "is_headless": get_headless_status(),
                },
            })
            return

        if path == "/api/timelapse":
            query = urllib.parse.parse_qs(parsed.query)
            domain = query.get("domain", ["python"])[0].replace("till-", "")
            mode = query.get("mode", ["delta"])[0]
            fps = float(query.get("fps", [2.0])[0])
            tensor_req = query.get("tensor", [None])[0]

            out_dir = Path("workspace/visualizations")
            matches = sorted(
                out_dir.glob(f"timelapse_{domain}_{mode}*.png"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if matches:
                latest = matches[0]
                meta = read_png_metadata(latest)
                self.send_json({
                    "status": "success",
                    "filename": latest.name,
                    "url": f"/images/{latest.name}",
                    "size_bytes": latest.stat().st_size,
                    "metadata": meta,
                })
                return

            domain_dir = Path("models") / f"till-{domain}-lora"
            ckpts = sorted(domain_dir.glob("adapter_model_rev*.pt"))
            if not ckpts:
                self.send_json({"status": "error", "error": f"No checkpoints found for domain {domain}"}, 404)
                return

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_file = out_dir / f"timelapse_{domain}_{mode}_{timestamp}.png"
            res = render_checkpoint_timelapse(
                checkpoint_paths=ckpts,
                output_path=out_file,
                domain=domain,
                tensor_name=tensor_req,
                mode=mode,
                fps=fps,
            )
            meta = read_png_metadata(res)
            self.send_json({
                "status": "success",
                "filename": res.name,
                "url": f"/images/{res.name}",
                "size_bytes": res.stat().st_size,
                "metadata": meta,
            })
            return

        if path.startswith("/images/"):
            img_name = Path(path).name
            allowed_dirs = [Path("workspace/visualizations"), Path("workspace/targets")]
            img_file = None
            for ad in allowed_dirs:
                candidate = ad / img_name
                if candidate.exists():
                    img_file = candidate
                    break
            if img_file and img_file.exists():
                data = img_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/api/generate", "/api/timelapse/render"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                params = json.loads(body.decode("utf-8"))
                result = generate_visualization_file(params)
                self.send_json(result, 200)
            except Exception as e:
                self.send_json({"status": "error", "error": str(e)}, 400)
            return

        self.send_response(404)
        self.end_headers()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_web_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = False,
) -> None:
    """Launches the Unplow Checkpoint Visualizer Studio Web Server."""
    server = ThreadingHTTPServer((host, port), UnplowWebHandler)
    hostname = socket.gethostname()
    username = getpass.getuser()
    lan_ips = get_lan_ip_addresses()
    primary_ip = lan_ips[0] if lan_ips else "127.0.0.1"
    is_headless = get_headless_status()

    print("=" * 80)
    print(" UNPLOW CHECKPOINT VISUALIZER STUDIO (HEADLESS / REMOTE ACCESS ENABLED)")
    print("=" * 80)
    print(f"[*] Server Listening Binding : http://{host}:{port}")
    print(f"[*] Headless Server Hostname : {hostname}")
    print(f"[*] Primary Local IP Address : {primary_ip}")
    print("-" * 80)
    print("HOW TO CONNECT FROM YOUR WINDOWS 11 HOST:")
    print()
    print("  Option 1 (SSH Port Forwarding - Recommended):")
    print("    Run in Windows Terminal / PowerShell:")
    print(f"      ssh -L {port}:localhost:{port} {username}@{hostname}")
    print("    Then open your Windows browser to:")
    print(f"      http://localhost:{port}")
    print()
    print("  Option 2 (Direct LAN Connection):")
    print("    Open your Windows browser directly to:")
    print(f"      http://{primary_ip}:{port}  or  http://{hostname}:{port}")
    print("=" * 80)
    print("[*] Press Ctrl+C to stop the web server.\n")

    if open_browser:
        if is_headless:
            print("[!] Headless environment detected: skipping local browser launch.")
            print("[!] Connect from Windows 11 using the SSH tunnel or direct LAN URL above.")
        else:
            local_url = f"http://127.0.0.1:{port}" if host == "0.0.0.0" else f"http://{host}:{port}"
            webbrowser.open(local_url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down visualizer server...")
    finally:
        server.server_close()


# end of file: unplow/src/unplow/web.py
