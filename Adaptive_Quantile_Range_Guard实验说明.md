# Adaptive Quantile Range Guard 实验说明

## 1. 实验目的

`Adaptive Quantile Range Guard` 的目标是改进普通 `range guard`。

普通 `range guard` 使用校准集上的全局最小值和最大值作为阈值：

```text
若节点输出中存在 NaN/Inf，或任意元素超出 [min, max]，则判定异常并触发重执行。
```

这个方法简单，但阈值可能过宽。`Adaptive Quantile Range Guard` 的想法是把阈值改成分位数范围，例如：

- `q99.9`
- `q99.5`
- `q99`

这样可以把极端正常值排除掉，使阈值更紧，理论上更容易检测故障。

## 2. 实验设置

本次实验是真实重放，不是根据已有 JSON 推断。

实验对象：

- 模型：ResNet50 + EuroSAT
- 节点：风险排序前 `8` 个节点
- 每节点故障事件：`32`
- 总故障事件：`256`
- 检查策略：`minmax`、`q99.9`、`q99.5`、`q99`
- clean holdout：每个节点 `64` 个样本

输出文件：

- `code/SPINNV2/artifacts/reports/paper_assets/mechanism_improvement_screening/adaptive_quantile_range_guard_measured.json`
- `code/SPINNV2/artifacts/reports/paper_assets/mechanism_improvement_screening/adaptive_quantile_range_guard_measured.csv`
- `code/SPINNV2/artifacts/reports/paper_assets/mechanism_improvement_screening/adaptive_quantile_range_guard.csv`

## 3. 结果

| 策略 | 关键故障数 | 检出的关键故障数 | 关键故障覆盖率 | clean false positive |
| --- | ---: | ---: | ---: | ---: |
| minmax | 7 | 0 | 0.0% | 0.0% |
| q99.9 | 7 | 0 | 0.0% | 0.0% |
| q99.5 | 7 | 0 | 0.0% | 0.0% |
| q99 | 7 | 0 | 0.0% | 0.0% |

结论：本次实验中，quantile 阈值没有比 minmax 检出更多关键故障，因此不采用为论文主方法。

## 4. 为什么效果不好

核心原因是：**导致分类错误的关键故障，并不一定表现为“大量元素越界”或“输出范围异常”。**

Range guard 检查的是数值范围异常。它擅长发现：

- NaN/Inf；
- 极大值；
- 极小值；
- 大量元素超出正常激活范围；
- 明显的爆炸式数值异常。

但本次 ResNet50 关键故障更像下面这种情况：

```text
少数元素发生变化
变化幅度没有超过 clean holdout 中允许的波动阈值
但这些变化经过后续层传播后，足以改变最终分类结果
```

因此，从最终任务看它是关键故障；但从当前节点输出的范围看，它仍然像一个“正常范围内的激活”。

一个典型现象是：

- `q99` 的阈值确实比 `minmax` 更紧；
- 它也检测到了一些非关键异常；
- 但 7 个真正关键故障没有越过告警阈值；
- 如果继续收紧阈值，可能会增加误报和无故障重执行开销。

也就是说，quantile range guard 的问题不是“完全不会报警”，而是：

> 它报警的位置和任务关键失效不匹配。

## 5. 这个负结果说明了什么

这个结果对论文反而有价值，因为它说明：

1. 单纯数值范围异常检测不足以覆盖所有任务级关键故障。
2. 更紧的阈值不一定带来更高的任务级保护收益。
3. 保护机制选择不能只看检测原语本身，必须结合任务后果和资源开销评估。

这正好支撑论文主线：

> 本文不是简单套用 range guard 或 DMR，而是通过任务级故障实验筛选有效保护机制，并将无效变体排除在主方法之外。

## 6. 论文中建议怎么写

建议把 `Adaptive Quantile Range Guard` 放在消融实验或负结果分析中，不作为创新点。

可以写成：

> 为验证更紧的激活范围阈值是否能够提升软件检测覆盖率，本文进一步评估了基于分位数的自适应范围保护策略。实验结果表明，在 ResNet50 + EuroSAT 的高风险节点上，`q99/q99.5/q99.9` 策略未能提高关键故障检测覆盖率。分析发现，部分任务级关键故障并未导致明显的激活范围越界，而是在正常数值范围内改变了后续判别结果。因此，本文最终未将该策略纳入主保护方法。

不要写成：

> 本文提出 Adaptive Quantile Range Guard 并取得明显效果。

因为当前数据不支持这个说法。

## 7. 当前结论

- `Adaptive Quantile Range Guard`：不采用。
- 普通 `range guard`：仍可作为已有保护 primitive 和对照机制。
- 论文主方法应继续聚焦：
  - `Task-Utility Consistency Guard`
  - `Confidence-Robust Bounded-Loss Memory ILP`
  - `ProtectionPlan` 自动落地执行闭环
