# 最终创新点表述

## 创新点一：任务后果感知的运行时对象脆弱性评估方法

将运行时节点故障与遥感分类错误、目标漏检、误检、定位劣化和控制路径执行错误关联起来，形成可排序、可统计置信区间的任务级风险度量。

支撑证据：

- ResNet50 + EuroSAT 的运行时故障注入、风险排序与置信区间。
- YOLOv10n + DIOR 的检测任务故障模式统计，包括 task output failure 和 controlled execution error。

## 创新点二：风险损失受限的低内存保护配置优化方法

基于节点风险、保护收益、额外时延和额外内存开销，先求给定预算下最大风险降低方案，再允许可解释的风险降低损失上限，在该约束内最小化峰值额外保护内存。该方法避免为了极小的风险收益差异选择大缓冲 DMR，面向星载受限平台更强调低内存可部署性。

支撑证据：

- ResNet50 已有 multi-mode ILP 结果，可证明多机制预算优化优于简单单机制策略。
- ResNet50 bounded-loss memory ILP 结果显示，在 5% 风险降低损失容忍度下，部分预算点可以降低 50% 到 100% 峰值额外保护内存。
- YOLOv10n 当前主要证明 DMR 和 task output guard 保护有效，算法优势主要由 ResNet50 多机制和低内存优化实验支撑。

## 创新点三：面向遥感检测任务的任务效用一致性保护机制与可执行落地

针对 YOLO 检测输出定义目标数量、置信度总量、类别分布和框面积分布等任务效用一致性检查，并将保护配置编码为 ProtectionPlan，集成到 SPK 与 C Runtime，使保护方案可以被自动加载、执行、测量和复现实验。

支撑证据：

- YOLOv10n 10000 个 runtime fault event 中，Task-Utility Consistency Guard 将关键失效降低率从普通 task output guard 的 50.52% 提升到 97.21%，clean false positive 为 0。
- SPK ProtectionPlan section。
- C Runtime fault event、range guard、DMR 执行路径。
- 运行时同事件 protected/unprotected 对比结果。

## 不能作为创新单独表述的内容

- 中间结果故障注入。
- 敏感层或高风险层筛选。
- DMR/TMR/重复执行。
- 通用推理框架、ONNX 转译、静态内存规划。
