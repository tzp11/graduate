# Generated C 保护路径状态

- codegen file: `E:\work\graduatepaper\code\SPINNV2\compiler\codegen\c_codegen.py`
- embedded SPK checksum verification: `True`
- generated wrapper calls `spkv2_run`: `True`
- generated wrapper exposes fault-event control: `False`
- generated wrapper exposes reliability stats: `False`
- protected generated C smoke status: `passed`

## 结论

Generated C 目前通过嵌入 SPK 并调用 Runtime 执行，因此可继承 SPK 内部的 ProtectionPlan 执行语义。

但当前生成接口未直接暴露 fault event 与 reliability stats，因此还不能作为批量注入实验入口。论文中应表述为：Generated C 可用于部署受保护模型，批量故障评估仍以 Runtime/ctypes/CLI 路径完成。

## 补充说明

- ResNet50 protected Generated C smoke 已通过：`build/generated/resnet50_protected_smoke/Release/resnet50_protected_main_test.exe`。
- YOLOv10n protected Generated C smoke 已通过：`build/generated/yolov10n_protected_smoke/Release/yolov10n_protected_main_test.exe`。
- 先前 ResNet50 的 C1060 来自旧版 codegen 将 94MB SPK 展开为约 603MB C 数组；当前默认外置 SPK 资产模式已修复该问题。
