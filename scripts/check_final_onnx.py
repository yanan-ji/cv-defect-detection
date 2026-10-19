#!/usr/bin/env python3
"""Compare final PyTorch and ONNX outputs under identical preprocessing/NMS."""
from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=ROOT / "data/neu_det/images/val/crazing_1.jpg")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/final_onnx_consistency.json")
    args = parser.parse_args()
    args.image = args.image.resolve()
    pt = ROOT / "experiments/yolo11n_320_conservative_aug/weights/best.pt"
    onnx = pt.with_suffix(".onnx")
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    letterboxed = LetterBox(new_shape=(320, 320), auto=False, stride=32)(image=image)
    tensor_np = np.ascontiguousarray(letterboxed[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    torch_model = YOLO(pt).model.eval().cpu()
    with torch.inference_mode():
        torch_output = torch_model(torch.from_numpy(tensor_np))
    if isinstance(torch_output, (tuple, list)):
        torch_output = torch_output[0]
    torch_raw = torch_output.detach().numpy()
    providers = ort.get_available_providers()
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    onnx_raw = session.run(None, {session.get_inputs()[0].name: tensor_np})[0]

    def decode(raw: np.ndarray) -> list[dict]:
        boxes = non_max_suppression(torch.from_numpy(raw.copy()), conf_thres=0.25, iou_thres=0.7,
                                   classes=None, agnostic=False, max_det=300)[0]
        if len(boxes):
            boxes[:, :4] = scale_boxes(letterboxed.shape[:2], boxes[:, :4], image.shape[:2]).round()
        return [{"xyxy": [float(x) for x in row[:4]], "confidence": float(row[4]),
                 "class_id": int(row[5])} for row in boxes.tolist()]

    pt_det, onnx_det = decode(torch_raw), decode(onnx_raw)
    paired = []
    for a, b in zip(pt_det, onnx_det):
        paired.append({"same_class": a["class_id"] == b["class_id"],
                       "max_box_abs_px": max(abs(x-y) for x, y in zip(a["xyxy"], b["xyxy"])),
                       "confidence_abs": abs(a["confidence"]-b["confidence"])})
    report = {
        "image": str(args.image.relative_to(ROOT)), "image_shape": list(image.shape),
        "input_shape": list(tensor_np.shape), "input_range": [float(tensor_np.min()), float(tensor_np.max())],
        "preprocess": "OpenCV BGR -> Ultralytics LetterBox 320x320 auto=False -> RGB CHW float32 /255",
        "raw_shape": list(torch_raw.shape), "raw_max_abs": float(np.max(np.abs(torch_raw-onnx_raw))),
        "raw_mean_abs": float(np.mean(np.abs(torch_raw-onnx_raw))),
        "nms": {"conf": 0.25, "iou": 0.7, "max_det": 300, "agnostic": False},
        "decode": "Ultralytics non_max_suppression: xywh-to-xyxy, class confidence, NMS; scale_boxes to original pixels",
        "pytorch_detections": pt_det, "onnx_detections": onnx_det, "paired_differences": paired,
        "same_count": len(pt_det) == len(onnx_det),
        "ort_available_providers": providers, "comparison_provider": "CPUExecutionProvider",
        "torch_device": "cpu", "onnx_file": str(onnx.relative_to(ROOT)),
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("raw_shape", "raw_max_abs", "raw_mean_abs", "same_count", "paired_differences")}, indent=2))


if __name__ == "__main__":
    main()
