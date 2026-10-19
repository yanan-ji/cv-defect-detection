# 实际运行：YOLO11n NEU-DET 冒烟实验

本文件只记录已完成的真实运行，不把 3 epoch 冒烟结果当作正式 60 epoch baseline。

| 项目 | 实测值 |
|---|---:|
| 模型 | YOLO11n，预训练 `yolo11n.pt` |
| 设备 | Apple M4 / MPS |
| 输入 | 320 × 320 |
| batch / seed | 8 / 42 |
| epoch | 3 |
| 训练集 / 验证集 / 测试集 | 1440 / 180 / 180 images |
| 测试 Precision | 0.5881 |
| 测试 Recall | 0.5015 |
| 测试 mAP50 | 0.4697 |
| 测试 mAP50-95 | 0.2344 |

测试集的类别 mAP50：crazing 0.100、inclusion 0.498、patches 0.702、pitted_surface 0.587、rolled-in_scale 0.294、scratches 0.638。此早期模型对 crazing 的 Recall 为 0、rolled-in_scale 的 Recall 为 0.264；这些是后续错误分析和更长训练的候选问题，不能据此判断最终模型表现。

证据：`experiments/smoke_neu_det/results.csv`（训练验证）、`runs/detect/val/`（独立测试评估图表）、`experiments/smoke_neu_det/weights/best.pt`（权重）。命令：

```bash
python scripts/train.py --epochs 3 --batch 8 --name smoke_neu_det --device mps
python scripts/evaluate.py experiments/smoke_neu_det/weights/best.pt --imgsz 320 --split test --device mps
```

端到端推理（包含预处理、YOLO 推理与 NMS）在同一台 MPS 设备、320 输入、3 次预热后测 10 次：均值 **14.09 ms**，标准差 **1.84 ms**，约 **70.96 images/s**。这不是纯模型前向耗时，也不可与其他硬件直接比较。
