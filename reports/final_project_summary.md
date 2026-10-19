# YOLO11 + NEU-DET：最终工程总结

## 问题—方法—结果

**问题。** 在 6 类钢材表面缺陷、1800 张图片的 NEU-DET 检测任务中，寻找可在 Apple Silicon 上训练、可导出的轻量模型，并识别低召回与误检来源。数据按固定种子 42 切成 train/val/test = 1440/180/180；模型/阈值选择使用 validation。早期 smoke 和 Baseline 已评价过 test，因此最后一次测试不属于完全未接触的盲测。

**方法。** 以 Ultralytics YOLO11n 预训练权重为起点训练 320×320、60 epoch、batch 8 的 Baseline；独立做 416 分辨率对照、保守组合增强、HSV/Geometry/Flip 三组单因素消融；Baseline 与保守增强在同一固定划分上做 seed 42/43/44 重复。错误分析使用验证集固定置信度 0.25、匹配 IoU 0.50；AP 与固定阈值 TP/FP/FN 分开解释。保守增强具体参数和训练样本见 `augmentation_ablation.md`。

**结果。** 以下训练结果均为 validation；320 组来自 `augmentation_ablation_metrics.json`，三随机种子来自 `augmentation_repeat_metrics.json`，416 来自既有 `optimization_experiments.md` 与 `runs/detect/val4` 记录。四舍五入到 4 位。

| 实验 | seed | P | R | mAP50 | mAP50-95 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| Baseline 320 | 42 | .5854 | .7273 | .7295 | .4194 | 60/60 |
| 416 分辨率 | 42 | .698 | .693 | .747 | .423 | 60/60，既有日志约 3 位精度 |
| 保守组合增强 320 | 42 | .7538 | .7114 | .7668 | .4410 | 60/60 |
| HSV-only 320 | 42 | .6376 | .7356 | .7286 | .4172 | 60/60 |
| Geometry-only 320 | 42 | .6652 | .7210 | .7412 | .4283 | 60/60 |
| Flip-only 320 | 42 | .6861 | .6932 | .7363 | .4145 | 60/60 |
| Baseline 320 | 43 | .6409 | .7396 | .7382 | .4241 | 60/60 |
| Baseline 320 | 44 | .6231 | .7366 | .7160 | .4256 | 60/60 |
| 保守组合增强 320 | 43 | 见原始 JSON | 见原始 JSON | 见原始 JSON | .4338 | 60/60 |
| 保守组合增强 320 | 44 | 见原始 JSON | 见原始 JSON | 见原始 JSON | .4315 | 60/60 |

同一固定划分的三种子 mAP50-95：Baseline 0.4230±0.0032、保守增强 0.4354±0.0050（样本标准差）；三个配对差值均为正，但 n=3，不能推出统计显著性或跨数据划分稳定性。单因素均未达到组合增强，不足以证明具体交互机制。最终冻结保守增强 seed 42；test 上 P=0.7736、R=0.6758、mAP50=0.7601、mAP50-95=0.4402；Baseline 既有 test mAP50-95=0.4255、R=0.7216。mAP 改善伴随 Recall 下降，不能只报单一指标。

## 最终模型 ONNX 与推理后端

权重：`experiments/yolo11n_320_conservative_aug/weights/best.pt`；导出：同目录 `best.onnx`，静态 batch 1、320×320，`onnx.checker` 通过。导出时环境缺少 `onnxslim`，简化步骤未执行，故不声称图已简化。比较脚本 `scripts/check_final_onnx.py` 将 BGR 图像经 Ultralytics LetterBox、RGB/CHW、float32/255 转换为同一输入，比较原始 `(1,10,2100)` 张量，再以 conf=0.25、NMS IoU=0.70、max_det=300 统一解码、NMS、还原原图坐标。CPU 上首张 `crazing_109.jpg` 的原始最大绝对差 0.001083、平均绝对差 0.0000194；检测数量、类别、整数像素框一致，置信度差 4.77e-7。六类各一张的逐图记录在 `onnx_comparison/`：全部检测数量/类别/整数框一致，六图最大原始绝对差 0.002014、最大对应置信度差 4.62e-6；这不是对全验证集或所有执行后端的等价证明。

同一张 `crazing_109.jpg`、batch 1、320 输入、10 次预热与 100 次计时，墙钟时间含预处理、模型、Ultralytics NMS、框还原；原始数据在 `final_backend_benchmark.json`：

| 软件执行后端 | 平均延迟 | P95 | 硬件解释 |
|---|---:|---:|---|
| PyTorch MPS | 10.28 ms | 12.74 ms | Apple Silicon GPU/MPS |
| ONNX Runtime CPU EP | 5.87 ms | 7.30 ms | CPU |
| ONNX Runtime CoreML EP | 2.44 ms | 3.10 ms | CoreML + CPU 回退；日志显示 320 节点中 315 个获 CoreML 支持 |

不可将三行说成相同硬件后端的加速比，也不能由单图计时推出生产吞吐量、功耗或多并发性能。ORT CoreML 与 MPS 是不同执行栈。

## 12 张验证集错误案例

下表来自 **Baseline seed 42** 的 `baseline_error_analysis/case_details.json`，不是最终保守增强模型的逐图误差。图片为真实原图叠加绿色 GT、红色预测框与置信度；固定 conf=0.25。原因列为可检查的表观归因，不是已证明的物理根因。

| 图片 | 类别 | 实际结果与问题 | 可能原因 |
|---|---|---|---|
| [001_crazing_109.jpg](baseline_error_analysis/cases/001_crazing_109.jpg) | crazing | 3 GT，0 预测，FN | 纹理弱/断续，全部低于操作阈值 |
| [004_crazing_119.jpg](baseline_error_analysis/cases/004_crazing_119.jpg) | crazing | 4 GT；最高 crazing 0.55；FN+FP | 局部裂纹与标框覆盖不一致 |
| [020_crazing_42.jpg](baseline_error_analysis/cases/020_crazing_42.jpg) | crazing | 1 GT，0 预测，FN | 低对比度候选丢失 |
| [028_crazing_96.jpg](baseline_error_analysis/cases/028_crazing_96.jpg) | crazing | 3 GT；预测 crazing 0.34；FN | 多处纹理仅局部识别 |
| [034_inclusion_155.jpg](baseline_error_analysis/cases/034_inclusion_155.jpg) | inclusion | 3 GT；预测最高 0.78；FP | 重复候选或定位偏差，见叠图 |
| [055_patches_180.jpg](baseline_error_analysis/cases/055_patches_180.jpg) | patches | 3 GT；预测最高 0.72；FP | 边界不规则导致多余框 |
| [058_patches_198.jpg](baseline_error_analysis/cases/058_patches_198.jpg) | patches | 5 个去重 GT；预测最高 0.88；FN | 重叠/重复标注需核对标签口径 |
| [073_pitted_surface_233.jpg](baseline_error_analysis/cases/073_pitted_surface_233.jpg) | pitted_surface | 2 GT；预测最高 0.36；FN+FP | 框位置偏差或相邻纹理干扰 |
| [082_rolled-in_scale_109.jpg](baseline_error_analysis/cases/082_rolled-in_scale_109.jpg) | rolled-in_scale | 2 GT；预测最高 0.42；FN+FP | 破碎形态使覆盖不足 |
| [095_rolled-in_scale_27.jpg](baseline_error_analysis/cases/095_rolled-in_scale_27.jpg) | rolled-in_scale | 1 GT，0 预测，FN | 弱纹理候选未过阈值 |
| [111_scratches_12.jpg](baseline_error_analysis/cases/111_scratches_12.jpg) | scratches | 2 GT；预测最高 0.52；FP | 细长纹理产生多余候选 |
| [119_scratches_221.jpg](baseline_error_analysis/cases/119_scratches_221.jpg) | scratches | 3 GT；预测最高 0.75；FP | 多余细长框；可结合验证阈值扫描 |

完整 127 张验证集错误图在 `baseline_error_analysis/cases/`。诊断匹配规则的限制和类别统计见 `error_diagnosis.md`；此处图片不证明最终候选在对应图上的错误已经消失。

## 实现边界、复现与未完成

Ultralytics 提供 YOLO11 网络、损失、训练器、基础数据增强、解码/NMS 与 AP 评价；PyTorch/ONNX Runtime 提供执行后端。本项目自主实现 VOC→YOLO 转换与固定分割、数据审计、实验编排、单因素/多种子评估、固定阈值错误分析、候选 ONNX 数值核对及统一测速脚本。没有宣称原创检测架构或注意力模块。

复现入口与注意事项在根目录 `README.md`。数据原图和权重被 `.gitignore` 排除；公开仓库需另行提供合法数据获取方法与模型权重分发方式。Git 根目录是项目父目录，目前 **main 无提交，项目目录整体未跟踪**，尚不能声称已正式发布到 GitHub。`requirements-lock.txt` 存在且记录本机环境；当前 manifest 与数据配置哈希和 `dataset_audit.json` 一致。`prepare_dataset.py` 会重建处理后的图像/标签，已有数据不要无故重复运行。尚未完成：生产部署、ORT 全验证集等价性、跨划分/跨采集批次泛化、ONNX 运行时在目标生产设备上的 SLA 验证。
