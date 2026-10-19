# Execution status

Last updated: 2026-10-19 (Asia/Shanghai)

| Work item | Status | Evidence / command |
|---|---|---|
| Environment inspection | COMPLETED | `reports/environment.json`; `python scripts/check_environment.py` |
| Isolated Python 3.11 environment | COMPLETED | Conda env `cv_yolo` created |
| Project structure/configuration | COMPLETED | `configs/`, `scripts/`, `src/`, `tests/` |
| Dependency installation | COMPLETED | `cv_yolo` installed Ultralytics 8.3.253, PyTorch 2.14.1, ONNX 1.23.2 and ONNX Runtime 1.31.0 |
| NEU-DET download | COMPLETED | `data/raw/neu-surface-defect-database.zip`; 1,800 images and 1,800 VOC XML |
| Dataset conversion/validation | COMPLETED | `data/neu_det/split_manifest.json`, `prepare_summary.json`, `reports/dataset_validation.json` (0 issues) |
| Dataset integrity / leakage audit | COMPLETED | `reports/dataset_audit.json`: 0 invalid or missing files, 0 cross-split exact duplicates, 0 cross-split near duplicates at perceptual-hash Hamming <=4; split/config SHA-256 recorded. |
| COCO8 / NEU-DET smoke training | COMPLETED | `experiments/smoke_coco8/weights/best.pt`, `experiments/smoke_neu_det/weights/best.pt` |
| Formal baseline / ablations / metrics | COMPLETED | `baseline_yolo11n_320` trained 60 epochs from `yolo11n.pt`, MPS, 320, batch 8, seed 42; see `experiments/baseline_yolo11n_320/args.yaml`, `results.csv`, `weights/best.pt`, `weights/last.pt`. Ablations and repeats are listed below. |
| Error analysis | COMPLETED | `reports/error_diagnosis.md`, `reports/baseline_error_analysis/`：验证集固定阈值 TP/FP/FN、IoU 一对一匹配、127 个错误案例和改进假设。 |
| Validation confidence scan | COMPLETED | `reports/confidence_scan/metrics.csv` and curves; candidates predicted at conf=0.001, fixed NMS IoU=0.70 and match IoU=0.50. |
| YOLO11n 416 resolution control | COMPLETED | `experiments/yolo11n_416/`, validation at `runs/detect/val4`: mAP50-95 0.423, rolled-in-scale AP50 0.432; 100-run MPS end-to-end mean 12.96 ms. |
| Conservative texture augmentation | COMPLETED | `experiments/yolo11n_320_conservative_aug/`: 60/60 epochs, `best.pt` and `last.pt`; validation `reports/augmentation_val_metrics.log` shows mAP50-95 0.4410 and crazing AP50 0.4606. |
| Augmentation single-factor ablation | COMPLETED | HSV-only, Geometry-only, Flip-only all reached 60 epochs; validation metrics and six-class AP in `reports/augmentation_ablation_metrics.json`, chart in `reports/figures/augmentation_ablation.png`. |
| Augmentation multi-seed stability | COMPLETED | Baseline and Conservative seed 42/43/44 each completed 60 epochs; `reports/augmentation_repeat_metrics.json` records val mAP50-95 mean±sample SD 0.4230±0.0032 vs 0.4354±0.0050 on the same fixed split. |
| Final conservative model test | COMPLETED | Frozen seed-42 Conservative `best.pt` evaluated on test: P 0.7736, R 0.6758, mAP50 0.7601, mAP50-95 0.4402; evidence `reports/final_candidate_test_metrics.json` and `runs/detect/reports/final_candidate_test/ultralytics/`. Historical smoke and baseline test exposure disclosed in `reports/augmentation_ablation.md`. |
| Final model ONNX export | COMPLETED | `experiments/yolo11n_320_conservative_aug/weights/best.onnx` exists (10 MB), static 320; `onnx.checker` passed. Simplifier unavailable during export, so no claim of simplification. |
| PyTorch/ORT consistency | COMPLETED WITH SCOPE | Six validation images (one per class), identical preprocessing and NMS: counts/classes/integer boxes match on CPU; raw output and confidence differences in `reports/onnx_comparison/*.json`, script `scripts/check_final_onnx.py`. Not full-validation or cross-provider equivalence. |
| Final backend benchmark | COMPLETED WITH SCOPE | Batch 1, one validation image, 100 runs: PyTorch MPS 10.28/12.74 ms mean/P95; ORT CPU 5.87/7.30; ORT CoreML+CPU fallback 2.44/3.10. `reports/final_backend_benchmark.json`. Not production SLA. |
| Final engineering reports and cases | COMPLETED | `reports/final_project_summary.md`, `reports/interview_code_map.md`; 12 linked baseline validation error cases, full 127 in `reports/baseline_error_analysis/cases/`. |
| GitHub publication and artifact distribution | NOT COMPLETED | Git root is parent directory, `main` has no commit, project directory untracked; data and weights ignored. No remote publication or model artifact release performed. |
| Earlier smoke ONNX validation | COMPLETED | `experiments/smoke_neu_det/weights/best.onnx`; `onnx.checker` PASS recorded in `reports/onnx_export.json`. Final-model ORT comparison separately listed above. |
| End-to-end inference benchmark | COMPLETED | 10 runs, 3 warmups on MPS at 320: mean 14.09 ms, std 1.84 ms, 70.96 img/s; command in `scripts/benchmark.py` |
| Demo inference | PARTIAL | CLI implemented in `scripts/predict.py`; trained smoke weight is available |

## Recovery commands

```bash
conda activate cv_yolo
cd cv-defect-detection
pip install -r requirements.txt
python scripts/check_environment.py
python scripts/download_dataset.py
python scripts/prepare_dataset.py
python scripts/validate_dataset.py
python scripts/train.py --epochs 3 --name smoke_neu_det
```
