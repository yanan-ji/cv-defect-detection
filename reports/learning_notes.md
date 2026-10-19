# 学习笔记

## 脚本分工

`download_dataset.py` 只通过合法 Kaggle 凭据下载；`prepare_dataset.py` 将 VOC XML 变为 YOLO 的 `class cx cy w h`（归一化）格式并固定划分；`validate_dataset.py` 检查图像、标签和坐标。`train.py` 调用官方 Ultralytics 训练器，`evaluate.py` 仅在测试集计算最终指标，`predict.py` 是推理入口，`benchmark.py` 测端到端延迟，`export_onnx.py` 导出并检查 ONNX 图。

## 训练发生了什么

图像及边框经缩放和增强进入 YOLO11；backbone 从纹理中提取多尺度特征，neck 融合不同尺度信息，head 预测类别和框。训练时损失反向传播更新预训练权重；验证集的 Precision、Recall、mAP 用于观察泛化。`imgsz`、学习率、epoch、batch、增强强度和置信度阈值是最重要的参数。训练曲线若训练损失持续下降而验证 mAP 停滞或下降，需警惕过拟合。

正式 baseline 从公开预训练 `yolo11n.pt` 重新开始，绝不从 3-epoch smoke 权重继续；`train.py` 的 `seed=42`、确定性设置和固定 manifest 保证这次可复核。训练过程只让验证集参与模型选择。测试集只会在 `best.pt` 选定后运行一次最终报告。常见错误原因包括纹理相似造成类别混淆、边框尺度/形态差异、低对比度缺陷、训练不足，以及局部标注边界本身的不一致；必须用验证集中的可视化和混淆统计区分，不能仅凭直觉归因。

## 与 CT 重建的差异

CT 重建通常从投影恢复连续图像（强调物理模型与重建误差）；目标检测从已有图像输出离散的类别和边框（强调标注质量、定位 IoU、分类和 NMS）。训练、推理、评估、部署应分开：训练更新参数；推理固定参数产生预测；评估用有标签数据衡量预测；部署关注格式、延迟和一致性。

## 哪些是自己实现的

数据转换、固定切分、校验、实验调度、测速约定和报告是本项目代码；YOLO11 网络、损失、NMS、训练循环、绘制基础曲线来自 Ultralytics。ONNX Runtime 推理一致性验证须在权重导出后真实运行，不能只凭导出成功推断。

错误诊断要区分两种数字：`error_analysis.py` 在一个指定置信度和 IoU 下做一对一匹配，适合定位 FP、FN 和可视化；Ultralytics 的 AP 是遍历置信度构成 PR 曲线后得到的面积。前者不能替代后者，也不能用测试集去选择阈值。详见 `reports/error_diagnosis.md`。

`scan_confidence.py` 还区分 NMS IoU（预测间去重）和匹配 IoU（预测与标注是否算 TP）。为扫描低阈值，它以 0.001 请求候选框，再在本地按各 confidence 阈值过滤；如果直接用 0.25 推理，低阈值分析会是错误的，因为候选框已经丢失。

保守增强对照改变了多项训练图像变换（HSV 扰动、平移、缩放、翻转），验证集 mAP50-95 从 0.419 升至 0.441。这只能支持“这组配置在 seed=42 的一次实验中更好”，不能判断哪一个变换起作用，也不能证明统计稳定。详情见 `reports/optimization_experiments.md`。

三组单因素消融分别只改变 HSV 幅度、几何平移与缩放、水平翻转概率。对真实灰度训练图单独调用 Ultralytics 增强算子后，H/S 未给像素引入颜色，V 主要改变亮度；框会随几何与翻转同步变化。图像和逐像素统计在 `reports/figures/augmentation_samples/`。这些独立操作示例不包括完整训练中的 mosaic，实际 batch 图在各 `experiments/*/train_batch0.jpg`。

单因素 seed 42 验证集 mAP50-95：HSV-only 0.4172、Geometry-only 0.4283、Flip-only 0.4145，均未达到组合增强的 0.4410；这可能涉及参数交互，也可能受训练随机性影响，不能直接归因。固定同一划分的 seed 42/43/44 重复实验中，Baseline 均值 0.4230、Conservative 均值 0.4354，三个配对差值均为正，但只有三个种子，不作显著性结论，也不能代表跨数据划分或生产批次的泛化。冻结 Conservative seed 42 后，最终测试集 mAP50-95 为 0.4402，对应 Baseline 既有结果 0.4255；Precision 提高而 Recall 下降。早期 smoke 和 Baseline 已使用过测试集，因此不能称本次测试为完全盲测。原始数字见 `reports/augmentation_ablation.md`。
