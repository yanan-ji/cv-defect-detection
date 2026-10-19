from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np

CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def xyxy_to_yolo(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = map(float, box)
    return ((xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height,
            (xmax - xmin) / width, (ymax - ymin) / height)
