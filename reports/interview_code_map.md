# 面试源码导图：技术问题 → 实现位置

以下行号按 2026-10-19 当前工作树记录；后续改代码需重新核对。项目没有改写 YOLO11 网络或损失。

| 面试知识点 | 源码位置 | 应能解释的内容 |
|---|---|---|
| VOC 标注读取与边界合法性 | `scripts/prepare_dataset.py:17` `parse_xml` | XML 的类别、宽高、xyxy 如何验证 |
| xyxy→YOLO 归一化 | `src/common.py:37` `xyxy_to_yolo` | 中心点/宽高与图像尺寸的关系 |
| 固定分层切分与 manifest | `scripts/prepare_dataset.py:30` `main` | seed 42、按主类别 80/10/10；重跑会重建处理目录 |
| 数据有效性/泄漏审计 | `scripts/validate_dataset.py:7` `main`；`scripts/audit_dataset.py:9,14,18` | 标签范围、SHA-256 完全重复和感知哈希近重复 |
| 训练起点与增强配置 | `scripts/train.py:7` `main` | `yolo11n.pt` 与 resume 的区别；不同增强模式、MPS、batch、seed |
| 顺序消融与断点续训 | `scripts/run_augmentation_ablation.py:8,12` | 如何判定完成、如何防止覆盖与 MPS 争用 |
| 同划分随机种子复核 | `scripts/run_augmentation_repeats.py:7,11`；`scripts/evaluate_augmentation_repeats.py:9,13` | 训练随机性不等于跨划分泛化 |
| 验证集 AP 对照 | `scripts/evaluate_augmentation_ablation.py:16,21` | Ultralytics PR 曲线 AP 与固定阈值 P/R 区别 |
| 固定阈值错误匹配 | `scripts/error_analysis.py:12,14,28-36` | IoU、匹配集合、FP/FN；需注意类别混淆计数口径 |
| 低置信度阈值扫描 | `scripts/scan_confidence.py:9,12` | 候选保留、NMS IoU 与 GT 匹配 IoU 分离 |
| 最终 ONNX 数值对照 | `scripts/check_final_onnx.py:21` `main` | LetterBox、BGR→RGB、CHW、/255、原始张量及同 NMS 比较 |
| 后端规范测速 | `scripts/benchmark_final_backends.py:30,35,42,55` | 同一输入、预热、同步、P95、CoreML CPU 回退 |
| CLI 推理与输出 | `scripts/predict.py:7` | 路径、阈值、设备、Ultralytics 保存结果 |

推荐学习顺序：先读数据处理与审计，再读训练配置和评价协议，然后看错误分析与消融，最后看 ONNX 导出、数值一致性和测速。重要边界：`scripts/error_analysis.py` 是项目的固定阈值诊断，不是 Ultralytics AP 计算；`scripts/benchmark_final_backends.py` 是单图端到端基准，不是生产压测。原始指标和证据索引见 `final_project_summary.md`。
