# 优化实验：运行状态与验证集阈值策略

本表只包含已真实执行的结果。所有模型共享 `split_manifest.json`、`yolo11n.pt` 来源、seed=42、60 epoch 和 MPS；训练串行进行。数值均为 validation split，尚未用 test 选择改进方案。

| 实验 | 训练状态 | 验证指标 | 关键类结果 | 延迟 | 说明 |
|---|---|---|---|---|---|
| YOLO11n-320 baseline | COMPLETED | P 0.585, R 0.727, mAP50 0.730, mAP50-95 0.419 | crazing AP50 0.332；rolled-in-scale AP50 0.444 | 见下方测速说明 | `runs/detect/val3` |
| YOLO11n-416 | COMPLETED | P 0.698, R 0.693, mAP50 0.747, mAP50-95 0.423 | rolled-in-scale AP50 0.432, Recall 0.516 | 12.96 ms mean, P95 16.21 ms (100 runs) | `runs/detect/val4` |
| YOLO11n-320 + 保守纹理增强 | COMPLETED | P 0.754, R 0.711, mAP50 0.767, mAP50-95 0.441 | crazing AP50 0.461；rolled-in-scale AP50 0.489 | 见下方测速说明 | `reports/augmentation_val_metrics.log`、`runs/detect/val5` |
| 320 + 阈值策略 | COMPLETED（无训练） | 固定匹配指标见下 | scratches F1 在 0.45 为 0.899 | 仅后处理改变 | `reports/confidence_scan` |

## 阈值扫描协议

只对 validation split 扫描。模型以 `conf=0.001` 推理以保留低分候选；NMS IoU 固定 **0.70**，这是预测框之间去重阈值；指标匹配 IoU 固定 **0.50**，这是预测与 GT 是否匹配的阈值，二者不能混淆。GT 以完全相同行的 YOLO 标签去重，与 Ultralytics 对 `patches_198.jpg` 的重复标签处理一致。完整每类 TP/FP/FN/P/R/F1 在 `confidence_scan/metrics.csv`，曲线为 `precision.png`、`recall.png`、`f1.png`。

## 验证集结论

宏平均固定阈值 F1 在 0.25 为 0.624、在 0.30 为 0.622，故不支持为所有类别把全局默认操作点改为 0.45。对 scratches，阈值从 0.25→0.45：Precision 0.567→0.860、Recall 0.981→0.942、F1 0.718→0.899；这支持将 0.45 作为**仅 scratches 的候选部署阈值**。其余类别可能受损，必须在冻结策略后独立 test 一次确认，不能将该扫描视为测试结果。

## 下一步

416 的 mAP50-95 比 baseline 高 0.004，但 rolled-in-scale AP50 低 0.012，且此前同协议延迟增加约 14%；它不支持作为该弱类的优先方案。保守纹理增强对照保持 320、seed、60 epoch、batch、模型及默认设置，只改变 `hsv_h/hsv_s/hsv_v`、平移/缩放和水平翻转；这些增强项被组合改变，因此本次不能识别其中哪一项单独起作用。相对 baseline，mAP50-95 提升 0.0216，crazing AP50 提升 0.1286，rolled-in-scale AP50 提升 0.0455；patches AP50 从 0.941 降至 0.920。仅单 seed，尚不能认定提升稳定。

### 延迟复核

100 次 MPS 端到端测量（含预处理、推理和 NMS，预热 15 次，单图 batch=1）在本次会话出现明显波动：同一张 `crazing_109.jpg`，增强模型先测均值 39.75 ms，随后 Baseline 18.85 ms、增强模型再测 23.00 ms。先前 320/416 对照使用另一张图，分别为 11.35/12.96 ms。当前无法将增强模型的延迟变化归因于模型本身；后续需要交替多轮、记录系统负载和逐阶段时间再比较。两个 320 模型架构、输入大小及参数量相同。
