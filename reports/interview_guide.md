# 面试准备

## 两分钟项目介绍

我做的是钢材表面六类缺陷的目标检测，目标是在精度和产线部署延迟间寻找平衡。以预训练 YOLO11n 为 baseline，固定数据划分后比较输入尺度、保守纹理增强和 YOLO11s。工程上完成 VOC 到 YOLO 转换、数据校验、独立测试集评估、端到端测速和 ONNX 导出入口。实际指标必须从对应实验目录的日志引用；当前未训练完成时不报数字。

## 快问快答（理论与项目依据）

1. **为何 YOLO11 而非 Faster R-CNN？** 单阶段检测更适合延迟敏感部署；需用实际精度-延迟测量验证。依据：`scripts/benchmark.py`。
2. **为何选 YOLO11n？** 它是资源受限设备的合理 baseline；不是默认“最好”。依据：`configs/experiments.yaml`。
3. **为何还比较 YOLO11s？** 检验参数增加是否带来足以抵消延迟的收益。待完成真实实验。
4. **输入分辨率为何影响结果？** 缩小会损失小纹理细节，放大会提高计算量。依据：尺度消融配置。
5. **IoU 是什么？** 预测框与真值框交并比，用于匹配和 mAP 阈值。
6. **Precision/Recall 区别？** 前者关注预测为正的正确性，后者关注真目标被找回的比例。
7. **mAP50 与 mAP50-95？** 前者固定 IoU=.5，后者平均多个 IoU 阈值，更严格。
8. **NMS 做什么？** 对同一目标的重叠预测框保留高置信度框，避免重复计数。
9. **迁移学习为何有效？** 预训练特征提供边缘、纹理等通用起点，小数据更稳定。
10. **如何判断过拟合？** 同时查看训练损失和验证集 mAP，不使用测试集调参。
11. **如何保证公平？** 固定 seed、split、epoch、硬件与测速协议；逐项改动。
12. **为何不堆注意力模块？** 没有错误分析证据会扩大变量和维护成本。
13. **增强如何选？** 钢材灰度纹理可用适度亮度/对比度和轻微噪声；几何扰动需保守。待完成对照。
14. **类别混淆如何处理？** 先看混淆矩阵和样本标注，再考虑补数据或针对性增强。待真实结果。
15. **小目标怎么定义？** 按数据集中相对框面积分位数定义，不能凭直觉。
16. **参数量小为何不一定更快？** 算子、内存访问、后处理和硬件后端都会影响延迟。
17. **为什么要 ONNX？** 便于跨运行时部署，但必须做输出与延迟验证。
18. **如何处理标注异常？** 转换阶段检查类别、缺失、坐标范围，人工复核问题样本。
19. **为什么保留 checkpoint？** 支持中断恢复、复核最优权重和训练轨迹。
20. **如何优化实际速度？** 先测端到端瓶颈，再权衡尺寸、模型、阈值与运行时。

## 简历表述（待补真实数字）

中文：

- 构建 NEU-DET 工业表面缺陷检测流水线，包含 VOC→YOLO 转换、固定分层切分和标签校验。
- 基于 YOLO11n 设计分辨率、增强与模型规模的可复现实验框架，统一评估与端到端延迟协议。
- 实现单图/批量预测、ONNX 导出与结构校验；指标完成后从实验日志补充。

English:

- Built a reproducible NEU-DET object-detection pipeline with VOC-to-YOLO conversion, stratified splits, and label validation.
- Designed controlled YOLO11 resolution, augmentation, and model-scale experiments under a unified evaluation and latency protocol.
- Implemented image/batch inference and ONNX export validation; quantitative results are reported only after logged runs.
