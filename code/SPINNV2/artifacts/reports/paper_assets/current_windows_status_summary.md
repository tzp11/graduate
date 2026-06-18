# Windows 阶段当前结果摘要

## 正式完成实验

- `resnet50_eurosat` `formal_multi_seed_runtime_faults`: `555 -> 155`, reduction `72.07%`, events `30000`.
- `yolov10n_dior_full_e30_b16` `formal_activation_prior_runtime_faults`: `287 -> 240`, reduction `16.38%`, events `10000`.
- `yolov10n_dior_full_e30_b16` `formal_task_output_guard`: `287 -> 142`, reduction `50.52%`, events `10000`.

## ResNet50 + EuroSAT

- clean samples: `512`
- baseline/protected accuracy: `0.972656` / `0.972656`
- prediction agreement: `1.000000`
- false alarm rate: `0.003906`
- latency overhead: `19.534 ms` (`17.776%`)
- fault events: `1024`
- critical failures: `16 -> 4`
- observed reduction ratio: `75.000%`
- selected protected nodes: `30`
- used peak extra memory: `3211264` bytes

## YOLOv10n + DIOR activation-prior

- workload: YOLOv10n trained/evaluated on full DIOR (18000/2000/3463); injection screen: 125 eligible samples among first 128 test images
- fault sampling: activation-byte weighted prior, 125 samples x 16 injections
- critical failures: `56 -> 43`
- reduction ratio: `23.214%`
- latency overhead: `30.310 ms` (`18.065%`)
- extra memory: `3276800` bytes

## YOLOv10n + DIOR stratified

- fault sampling: stratified, 170 runtime nodes x 16 injections
- critical failures: `77 -> 13`
- reduction ratio: `83.117%`
- latency overhead: `30.310 ms` (`18.065%`)

## 当前不足

- ResNet50 仍需 10000+ fault events 和多 seed 排序稳定性。
- YOLOv10n 当前主要证明 DMR 保护有效，ILP 相对简单策略的优势还需要补强。
- SPK/权重边界实验、工程 P95/P99 指标和 Generated C smoke 仍需补齐。
