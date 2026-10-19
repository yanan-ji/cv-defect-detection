#!/usr/bin/env python3
"""Download NEU-DET only through an authenticated Kaggle client; never bypass access controls."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(); p.add_argument("--source", default="kaustubhdikshit/neu-surface-defect-database")
p.add_argument("--out", type=Path, default=ROOT / "data/raw"); args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", args.source, "-p", str(args.out)]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, text=True)
if result.returncode:
    raise SystemExit("Kaggle download failed. Configure ~/.kaggle/kaggle.json or manually download the detection dataset (images + XML annotations), place it under data/raw/, then run prepare_dataset.py.")
