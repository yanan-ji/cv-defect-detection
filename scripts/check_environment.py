#!/usr/bin/env python3
"""Write a reproducible local environment snapshot."""
from __future__ import annotations
import json, platform, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
info = {"python": sys.version, "machine": platform.machine(), "platform": platform.platform(),
        "free_disk_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2)}
try:
    import torch
    info.update(torch=torch.__version__, mps_available=torch.backends.mps.is_available())
except Exception as e: info["torch_error"] = str(e)
try:
    info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip()
except Exception: info["git_commit"] = None
out = ROOT / "reports" / "environment.json"; out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(info, indent=2), encoding="utf-8")
print(json.dumps(info, indent=2))
