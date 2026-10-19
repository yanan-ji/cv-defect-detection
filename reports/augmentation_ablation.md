# 增强消融：配置、样本与执行状态

本报告只使用固定的 NEU-DET 划分和 validation split 选择模型。Baseline、组合增强权重及原日志保留不动。三组单因素实验以 `yolo11n.pt` 预训练权重各自重新开始，YOLO11n、320、60 epoch、batch 8、MPS、seed 42 和其余 Ultralytics 默认参数一致。输出目录彼此独立。

## 参数差异（读取实际 `args.yaml`）

| 参数 | Baseline | Conservative | HSV-only | Geometry-only | Flip-only |
|---|---:|---:|---:|---:|---:|
| `hsv_h` | 0.015 | 0.0 | 0.0 | 0.015 | 0.015 |
| `hsv_s` | 0.7 | 0.12 | 0.12 | 0.7 | 0.7 |
| `hsv_v` | 0.4 | 0.15 | 0.15 | 0.4 | 0.4 |
| `translate` | 0.1 | 0.03 | 0.1 | 0.03 | 0.1 |
| `scale` | 0.5 | 0.15 | 0.5 | 0.15 | 0.5 |
| `fliplr` | 0.5 | 0.0 | 0.5 | 0.5 | 0.0 |

未列出的增强参数相同，例如 `mosaic=1.0`、`degrees=0`、`flipud=0`、`mixup=0`。单因素指“一组增强参数”；HSV-only 同时改变 H/S/V 三个幅度，Geometry-only 同时改变 translate/scale，不可解释为各单参数独立贡献。

## 真实训练图片抽样

`scripts/sample_augmentations.py` 使用 Ultralytics 8.3.253 的 `RandomHSV`、`RandomPerspective`、`RandomFlip` 对 `data/neu_det/images/train/crazing_1.jpg` 及其真实标签操作；带框图片和像素统计保存在 `reports/figures/augmentation_samples/`。原图灰度像素比例 1.0，HSV Baseline 与 Conservative 后仍为 1.0，均值亮度分别由 160.64 变为 189.44 和 143.56，平均绝对像素变化分别为 28.80 和 17.07（各为一次随机抽样，不是总体均值）。在这张纯灰度图上 H/S 不产生彩色变化，实际可见变化主要是 V。原框 `[2,2,193,194]` 经强几何扰动后为 `[44.8,12.6,182.9,151.5]`，经保守几何扰动后为 `[3.6,2.4,194.9,194.7]`；强制水平翻转后为 `[7,2,198,194]`，关闭翻转则保持原框。此抽样单独调用算子，完整训练还包含 mosaic，不能把单张样本当成完整训练数据分布。

另已目视检查 Ultralytics 训练器实际输出的 `experiments/baseline_yolo11n_320/train_batch0.jpg` 与 `experiments/yolo11n_320_conservative_aug/train_batch0.jpg`：两者均含 mosaic 与随变换更新的目标框；组合增强图中同批纹理的裁切与位置有变化。这两张图可验证训练管线确实调用了增强，但不能据单批图像衡量总体增强强度。

## 验证集比较

| 实验 | 状态 | P | R | mAP50 | mAP50-95 | 六类 AP50/AP50-95 |
|---|---|---:|---:|---:|---:|---|
| Baseline seed 42 | COMPLETED | 0.585 | 0.727 | 0.730 | 0.419 | 见 `runs/detect/val3` |
| Conservative seed 42 | COMPLETED | 0.754 | 0.711 | 0.767 | 0.441 | 见 `reports/augmentation_val_metrics.log` |
| HSV-only seed 42 | COMPLETED | 0.638 | 0.736 | 0.729 | 0.417 | 见下表 |
| Geometry-only seed 42 | COMPLETED | 0.665 | 0.721 | 0.741 | 0.428 | 见下表 |
| Flip-only seed 42 | COMPLETED | 0.686 | 0.693 | 0.736 | 0.415 | 见下表 |

每组均在 60 epoch 后加载各自验证集最佳 `best.pt` 独立运行 `val`。原始浮点指标在 `reports/augmentation_ablation_metrics.json`，图表在 `reports/figures/augmentation_ablation.png`，运行日志在 `reports/augmentation_ablation_evaluation.log`。下面每格为 AP50 / AP50-95：

| 类别 | Baseline | Conservative | HSV-only | Geometry-only | Flip-only |
|---|---:|---:|---:|---:|---:|
| crazing | .332/.132 | .461/.194 | .384/.157 | .389/.156 | .403/.143 |
| inclusion | .835/.464 | .849/.477 | .878/.500 | .868/.491 | .850/.451 |
| patches | .941/.634 | .920/.611 | .924/.636 | .916/.623 | .914/.618 |
| pitted_surface | .896/.544 | .920/.592 | .826/.491 | .892/.574 | .893/.539 |
| rolled-in_scale | .444/.191 | .489/.236 | .418/.184 | .457/.206 | .433/.218 |
| scratches | .929/.551 | .962/.537 | .941/.536 | .926/.520 | .925/.518 |

组合增强在本次 seed 42 验证中 mAP50-95 比 Baseline 高 0.0216。三个单因素分别为 -0.0022、+0.0088、-0.0049，均未达到组合结果；这与参数交互相容，但不能仅凭 5 次训练证明交互机制，训练随机性也可能影响差值。HSV-only 提高 crazing AP50 但降低 pitted_surface AP；Geometry-only 总体改善较小；Flip-only 的总体 mAP50-95 低于 Baseline。按预先定义的主指标 mAP50-95，选择 Conservative 作为 seed 43/44 重复候选；两组重复实验均已完成。固定划分下的多 seed 衡量训练随机性；它不衡量不同数据划分、不同采集批次的泛化稳定性。

## 三种训练随机种子：固定同一划分

四次新增训练均达到 60 epoch；每次用本次验证集最佳权重重新在 val 评价。原始结果在 `reports/augmentation_repeat_metrics.json`，命令输出在 `reports/augmentation_repeat_evaluation.log`。

| Seed | Baseline mAP50-95 | Conservative mAP50-95 | 配对差值 |
|---:|---:|---:|---:|
| 42 | 0.4194 | 0.4410 | +0.0216 |
| 43 | 0.4241 | 0.4338 | +0.0096 |
| 44 | 0.4256 | 0.4315 | +0.0060 |
| 均值 ± 样本标准差 | 0.4230 ± 0.0032 | 0.4354 ± 0.0050 | +0.0124 ± 0.0082 |

三个种子差值同向，但样本量只有 3，不提供显著性结论。这里所有种子都使用同一个固定 train/val/test 划分，衡量的是训练随机性与 MPS 后端实际运行波动，不能代表跨划分、跨生产批次的泛化稳定性。MPS 训练日志包含 `index_put_with_accumulate_mps` 非确定性算子的提示，`deterministic=True` 不等于逐位复现。

## 冻结候选后的测试集评价

模型选择规则为 seed-42 消融的验证集 mAP50-95，结合 seed 43/44 重复方向确认；最终使用验证集最高的 Conservative seed-42 `best.pt`，`imgsz=320`、Ultralytics `val` 标准设置、MPS 对 test split 做一次评价。测试记录：`reports/final_candidate_test_metrics.json`、`reports/final_candidate_test.log`；PR 与混淆矩阵位于 `runs/detect/reports/final_candidate_test/ultralytics/`。

| Test 指标 | 既有 Baseline seed 42 | Conservative seed 42 | 差值 |
|---|---:|---:|---:|
| Precision | 0.6683 | 0.7736 | +0.1053 |
| Recall | 0.7216 | 0.6758 | -0.0458 |
| mAP50 | 0.7376 | 0.7601 | +0.0225 |
| mAP50-95 | 0.4255 | 0.4402 | +0.0147 |

Conservative 测试集各类 AP50/AP50-95：crazing 0.4657/0.1788、inclusion 0.8382/0.4887、patches 0.9075/0.5974、pitted_surface 0.7928/0.5425、rolled-in-scale 0.6954/0.2949、scratches 0.8609/0.5389。Baseline 的既有测试结果见 `reports/baseline_report.md`。结果显示精度改善与总体召回下降的权衡，不能仅报 mAP 提升。

MPS 训练日志包含 `index_put_with_accumulate_mps` 无确定性实现的 PyTorch 提示。虽已设置 `deterministic=True`，多 seed 结果应作为实际稳定性检查，不能声称同一 seed 在该后端逐位一致。

## 测试集接触史与恢复

早期 3-epoch smoke 模型和正式 Baseline 已在 test split 评价过，因此当前 test 不属于完全未接触的盲测集。后续消融和阈值选择仍只用 validation；最终候选冻结后再做一次独立 test 评价，并在最终报告披露这一历史。

串行入口：`python scripts/run_augmentation_ablation.py`。Flip-only 曾在第 8 轮中断，保留 `last.pt` 后已恢复并完成 60 轮；原始 `.train.log` 保留了中断痕迹。重复实验入口：`python scripts/run_augmentation_repeats.py --candidate conservative`。若进程中断，同一命令会跳过完整实验，并对存在 checkpoint 的未完成实验调用 `--resume`。不得把部分 epoch 指标当作完成结果。
