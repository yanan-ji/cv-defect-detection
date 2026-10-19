# 从本项目学习 YOLO11 目标检测

本指南以本仓库已运行的 NEU-DET 项目为例。请先看 `reports/execution_status.md`，分清已经测得的数值与尚未完成的实验。

## 第一章：项目整体逻辑

目标是从钢材图片中同时回答“有什么缺陷”和“缺陷在哪里”。这就是目标检测，输出类别、置信度和边界框。Baseline 是一条可运行、可复核的起点，本项目是预训练 YOLO11n、320 输入和固定数据划分。完整路径为：XML 标注→YOLO 标签→训练/验证→独立测试→错误分析→只在验证集选择改进→部署。

## 第二章：数据与预处理

VOC XML 的框是 `(xmin,ymin,xmax,ymax)` 像素坐标；YOLO 标签为 `class cx cy w h`，且后四项相对图宽高归一化。转换在 `scripts/prepare_dataset.py` 的 `xyxy_to_yolo` 完成。`split_manifest.json` 固定划分；`scripts/audit_dataset.py` 检查跨 split 重复，因为同源图同时出现会夸大指标。数据增强只能改变训练集中的图像，不改变验证/测试标签含义。

## 第三章：CNN 基础

卷积核在局部滑动，学习边缘、纹理再到缺陷形态；步长决定下采样，padding 控制边缘尺寸。特征图是每层提取到的响应，叠加卷积会扩大感受野。池化或步长卷积压缩分辨率。BatchNorm 稳定特征分布，激活函数提供非线性，残差连接让深层网络更容易优化。网络实现由 Ultralytics 提供，而数据转换和审计是本项目代码。

## 第四章：YOLO11 结构

YOLO11 的 backbone 抽取不同层级纹理，neck 融合高分辨率定位信息与低分辨率语义信息，detect head 在多尺度特征上预测类别与边框。`scripts/train.py` 调用官方 `YOLO(...).train`，并非本项目手写网络。NMS 在推理末尾去除对同一对象的重复框。

## 第五章：训练流程

`scripts/train.py` 从 `yolo11n.pt` 加载预训练权重，batch 内图片先 forward 得预测，loss 衡量框定位、分类和分布误差；backward 计算梯度，优化器更新参数。一个 epoch 遍历一次 train。`configs/experiments.yaml` 和脚本固定 seed=42；`args.yaml` 保存实际超参数。看 `results.csv` 时，应主要用 val mAP 判断泛化，而非仅看 train loss。

## 第六章：模型评价

TP 是正确匹配真值的预测，FP 是多余/错误预测，FN 是漏掉的真值。Precision=TP/(TP+FP)，Recall=TP/(TP+FN)。IoU 是两个框交并比。AP 是 PR 曲线下的面积；mAP50 使用 IoU=.5，mAP50-95 平均 .5 到 .95 的多个阈值，因而更严格。正式测试只允许在验证权重选定后进行。

## 第七章：错误分析

运行 `scripts/error_analysis.py ... --split val`。绿色框为真值、红色框为预测；输出矩阵区分分类混淆、背景 FP 和背景 FN，并保存错误图片。再按 `size_records.json` 的相对面积分桶，比较各桶匹配率。结论必须引用具体样本和统计，不能先假定“小目标最难”。

## 第八章：算法改进

先写可证伪假设，例如“高分辨率可改善验证集中某尺寸桶的 FN，但延迟增加”。每次只改输入尺寸或一种增强，固定划分、seed、训练预算和评估协议。`scripts/run_experiments.py` 顺序运行，避免 MPS 资源争用。单次提升不是稳定结论；至少三 seed 才能报告均值和标准差。

## 第九章：模型部署

MPS 是 Apple 的 PyTorch 后端，与 NVIDIA CUDA 不同。`scripts/export_onnx.py` 导出 ONNX 并做图结构检查；ONNX Runtime 仍需要实际输出一致性验证。`scripts/benchmark.py` 测的是包含预处理与 NMS 的端到端时间，预热后报告统计量；它不等同于纯网络前向速度。

## 第十章：面试讲解

两分钟版本：说明钢材缺陷定位需求、NEU-DET 的固定划分、YOLO11n baseline、验证集错误诊断、精度-延迟对照和 ONNX 部署。被追问时拿出 `args.yaml`、`results.csv`、独立测试输出和审计报告。不要把计划中的 416/640、YOLO11s 或增强实验说成已完成；这是可复现项目最重要的诚信边界。
