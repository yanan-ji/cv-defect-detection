#!/usr/bin/env python3
"""Single-image batch-1 end-to-end benchmark with shared preprocessing and NMS."""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "data/neu_det/images/val/crazing_109.jpg"
PT = ROOT / "experiments/yolo11n_320_conservative_aug/weights/best.pt"
ONNX = PT.with_suffix(".onnx")
WARMUP, RUNS = 10, 100
image = cv2.imread(str(IMAGE))
if image is None:
    raise FileNotFoundError(IMAGE)


def preprocess() -> np.ndarray:
    img = LetterBox(new_shape=(320, 320), auto=False, stride=32)(image=image)
    return np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0


def postprocess(raw: np.ndarray) -> None:
    det = non_max_suppression(torch.from_numpy(raw.copy()), conf_thres=0.25, iou_thres=0.7,
                             agnostic=False, max_det=300)[0]
    if len(det):
        scale_boxes((320, 320), det[:, :4], image.shape[:2])


def benchmark(label: str, infer) -> dict:
    for _ in range(WARMUP):
        postprocess(infer(preprocess()))
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        postprocess(infer(preprocess()))
        times.append((time.perf_counter()-start)*1000)
    return {"backend": label, "batch": 1, "warmup": WARMUP, "runs": RUNS,
            "mean_ms": statistics.mean(times), "p95_ms": float(np.percentile(times, 95)),
            "median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times)}


def main() -> None:
    results = []
    model = YOLO(PT).model.eval().to("mps")

    def pt_infer(arr: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            output = model(torch.from_numpy(arr).to("mps"))
            if isinstance(output, (tuple, list)):
                output = output[0]
            torch.mps.synchronize()
            return output.cpu().numpy()

    results.append(benchmark("PyTorch MPS", pt_infer))
    for provider in ("CPUExecutionProvider", "CoreMLExecutionProvider"):
        if provider not in ort.get_available_providers():
            continue
        try:
            session = ort.InferenceSession(str(ONNX), providers=[provider])
            actual = session.get_providers()
            def infer(arr: np.ndarray) -> np.ndarray:
                return session.run(None, {session.get_inputs()[0].name: arr})[0]
            row = benchmark("ONNX Runtime " + provider, infer)
            row["session_providers"] = actual
            row["provider_warning"] = "CoreML may partition unsupported ops to CPU; provider list does not prove full graph acceleration" if provider.startswith("CoreML") else None
            results.append(row)
        except Exception as exc:
            results.append({"backend": "ONNX Runtime " + provider, "error": str(exc)})
    report = {"host": platform.platform(), "processor": platform.processor(), "torch": torch.__version__,
              "onnxruntime": ort.__version__, "image": str(IMAGE.relative_to(ROOT)), "imgsz": 320,
              "measurement": "wall-clock preprocess + model + Ultralytics NMS + box scaling, batch=1, same image; MPS synchronized before CPU transfer",
              "results": results}
    path = ROOT / "reports/final_backend_benchmark.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
