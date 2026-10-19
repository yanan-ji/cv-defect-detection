# Baseline 验证集错误诊断

**范围。** 本报告只使用 `data/neu_det/images/val` 和正式权重 `experiments/baseline_yolo11n_320/weights/best.pt`，不读取测试集，因此可用于提出调参假设。固定阈值分析的配置为置信度 `0.25`、IoU `0.50`；绿框是 GT、红框是预测，预测框旁写有类别及置信度。

## 匹配规则与口径

`scripts/error_analysis.py` 依次处理每个 GT，在尚未使用的预测中选择 IoU 最大且不低于 0.50 的框；该预测随即锁定，因此一个预测最多匹配一个 GT、每个 GT 也至多一个预测。匹配后类别相同是 TP，类别不同会同时形成分类混淆；未匹配 GT 为 FN，未匹配预测为 FP。该规则已用完全重合框 IoU=1 的断言复核。当前固定阈值矩阵没有 GT→错误类的非零项，错误主要是背景 FN/FP 而非类别互混。

这**不同于** Ultralytics AP：其 PR 曲线枚举置信度并按 IoU 阈值计算积分。Ultralytics 在 `patches_198.jpg` 自动去除了 1 个重复标签，因此 AP 使用 420 个有效实例；下表 TP/FP/FN 使用原始标签的 421 实例，不能直接拿两种数字相除或比较。

## 固定阈值统计

| 类别 | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| crazing | 10 | 7 | 51 | 0.588 | 0.164 |
| inclusion | 90 | 30 | 22 | 0.750 | 0.804 |
| patches | 85 | 34 | 8 | 0.714 | 0.914 |
| pitted_surface | 29 | 9 | 12 | 0.763 | 0.707 |
| rolled-in_scale | 28 | 33 | 34 | 0.459 | 0.452 |
| scratches | 51 | 39 | 1 | 0.567 | 0.981 |

来源：`reports/baseline_error_analysis/summary.json`。对应 AP 指标来自同一模型另一次验证运行 `runs/detect/val3`：总体 P/R/mAP50/mAP50-95 为 0.585/0.727/0.730/0.419；六类 AP50/AP50-95 依次为 crazing 0.332/0.132、inclusion 0.835/0.464、patches 0.941/0.634、pitted_surface 0.896/0.544、rolled-in_scale 0.444/0.191、scratches 0.929/0.551。

## 诊断与尺寸证据

- `crazing` 在固定阈值下仅匹配 10/61，问题以 FN 为主。其全部 GT 相对面积中位数 0.230，匹配 GT 中位数 0.256；不能把问题归因为“小目标”，因为漏检同样覆盖中等至大面积纹理。
- `rolled-in_scale` 的 P=0.459、R=0.452，FP 与 FN 同时偏高（33/34）；相对面积中位数为 0.152，匹配样本为 0.164，存在轻微尺寸差异但不足以单独证明尺度是原因。
- `scratches` R=0.981 但 FP=39，说明固定置信度 0.25 下检测较激进；后续应在验证集画阈值–P/R 曲线，而不能以测试集找阈值。
- `patches`、`inclusion` 的 AP 和固定阈值 Recall 较好，暂不是优先改动对象。没有证据支持添加注意力模块或结构改造。

## 代表性真实案例（12 张）

下列文件均含原图、GT、预测类别和置信度；完整逐框结构化明细见 `case_details.json`。

| 类别 | 案例 | 错误 | 预测置信度（主要） |
|---|---|---|---|
| crazing | `001_crazing_109.jpg` | FN | 无预测 |
| crazing | `004_crazing_119.jpg` | FN + FP | crazing 0.554 |
| inclusion | `029_inclusion_102.jpg` | FP | inclusion 0.733, 0.629 |
| inclusion | `031_inclusion_115.jpg` | FN + FP | inclusion 0.876, 0.797 |
| patches | `051_patches_127.jpg` | FN + FP | patches 0.793, 0.721 |
| patches | `053_patches_168.jpg` | FN + FP | patches 0.856, 0.845 |
| pitted_surface | `068_pitted_surface_125.jpg` | FN + FP | pitted_surface 0.488 |
| pitted_surface | `073_pitted_surface_233.jpg` | FN + FP | pitted_surface 0.364 |
| rolled-in_scale | `082_rolled-in_scale_109.jpg` | FN + FP | rolled-in_scale 0.421, 0.386 |
| rolled-in_scale | `086_rolled-in_scale_187.jpg` | FN + FP | rolled-in_scale 0.706, 0.594 |
| scratches | `110_scratches_111.jpg` | FP | scratches 0.762; inclusion 0.748 |
| scratches | `114_scratches_186.jpg` | FN + FP | scratches 0.894, 0.503 |

路径：`reports/baseline_error_analysis/cases/`。这些是证据样本，非人为挑选的“成功案例”。

## 可证伪改进假设（尚未训练）

1. **训练分辨率 416。** 假设：较高输入保留 rolled-in-scale 的局部形态，提升其验证 AP50/Recall；代价是更高 MPS 内存和端到端延迟。对照：固定 yolo11n、seed、60 epoch、batch（若内存不足记录调整）、默认增强和同一 val；比较 AP、尺寸分桶召回与 100 次延迟。
2. **保守纹理增强。** 假设：降低默认强饱和度扰动、使用有限亮度/对比度扰动，可改善 crazing 的纹理泛化而不显著损害 patches。改动：新增一个且仅一个 augmentation 配置。对照：与 baseline 相同训练预算、模型、输入及划分；验证集同时报告六类 AP，防止只优化单类。
3. **验证集阈值策略分析，而非立即改阈值。** 假设：提高 scratches 的推理置信度可降低 FP，但会牺牲 Recall。先在验证集扫描阈值并记录 P/R；若选择操作点，冻结后才在 test 上做一次确认。该项没有训练成本，但不是模型精度改进。

在完成上述受控对照前，不能声称任何假设成立。
