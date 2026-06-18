# Windows 正式完成实验摘要

| 模型 | 实验 | 事件数 | 未保护失效 | 方法 | 保护后失效 | 下降比例 | 说明 |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| resnet50_eurosat | formal_multi_seed_runtime_faults | 30000 | 555 | multi_mode_ilp_range_guard_dmr | 155 | 72.07% | Three real 10000-event runs, not bootstrap substitution. |
| yolov10n_dior_full_e30_b16 | formal_activation_prior_runtime_faults | 10000 | 287 | ilp_dmr_runtime_calibrated | 240 | 16.38% | Same seed and candidate sampling; activation-prior 10000 events. |
| yolov10n_dior_full_e30_b16 | formal_task_output_guard | 10000 | 287 | task_output_guard_rerun | 142 | 50.52% | Final-output semantic validity checks with rerun recovery under single-transient-fault model. |

## 结论边界

- ResNet50 结论来自 3 个真实 seed、共 30000 个运行时故障事件。
- YOLOv10n activation-prior 结论来自真实 10000 个运行时故障事件。
- YOLOv10n 的 ILP-DMR 在真实 activation-prior 下收益较保守；task_output_guard 对最终输出异常有更高覆盖，但属于检测输出语义检查机制。
- D2000/RISC-V 未验证，不能写成跨平台实机结论。
