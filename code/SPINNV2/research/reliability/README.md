# Reliable Inference Research Workflow

This directory implements the thesis-specific reliability extension on top of
SPINNV2. Framework performance work outside required model compatibility is not
treated as a research contribution.

## Current Windows Baseline

- Environment: `E:\conda_envs\graduatepaper_reliable` with Python 3.11.
- GPU workload environment: `E:\conda_envs\graduatepaper_gpu_prevalidation`
  cloned from a locally verified CUDA 11.6/PyTorch 1.12.1 toolchain and
  extended only with Ultralytics 8.2.100.
- Toolchain: Visual Studio 2022/MSVC for the C runtime; the Release
  `cpu_generic` target uses AVX2/FMA on this Windows host.
- Tasks: `ResNet50 + EuroSAT` and `YOLOv10n + DIOR`.
- Fault scope: one transient single-bit flip in an FP32 runtime output tensor.
- YOLO export uses `simplify=True`; the fixed-shape graph requires the added
  `Gather` and `Slice` reference kernels after simplification.

## Commands

Run from the repository root with the isolated environment active, or replace
`python` with `E:\conda_envs\graduatepaper_reliable\python.exe`.

```powershell
python -m research.reliability.prepare_data --dataset eurosat
python -m research.reliability.prepare_data --dataset dior --dior-images <images> --dior-annotations <xml> --split train
python -m research.reliability.download_dior --source hf_parquet --splits train val test
python -m research.reliability.prepare_data --dataset dior --dior-parquet artifacts/data/dior/raw/hf_parquet/data/train-*.parquet --split train
python -m research.reliability.prepare_dior_hf_subset --train-samples 128 --val-samples 64 --test-samples 64 --workers 4
python -m research.reliability.models.train_resnet50 --train-mode head --device auto
python -m research.reliability.models.export_models --model resnet50 --checkpoint artifacts/experiments/windows_prevalidation/resnet50/best.pt --output artifacts/models/resnet50_eurosat.onnx
python -m research.reliability.models.export_models --model yolov10n --weights artifacts/experiments/windows_prevalidation/yolov10n/train/weights/best.pt --output artifacts/models/yolov10n_dior.onnx
python -m research.reliability.audit_spinn_compat artifacts/models/resnet50_eurosat.onnx
python -m research.reliability.inject_faults --checkpoint artifacts/experiments/windows_prevalidation/resnet50/best.pt --output artifacts/injections/resnet50_screen.jsonl
python -m research.reliability.build_risk_profile artifacts/injections/resnet50_screen.jsonl --output artifacts/reports/resnet50_risk.json
python -m research.reliability.map_runtime_candidates --candidate-map artifacts/injections/resnet50_screen.candidates.json --onnx artifacts/models/resnet50_eurosat.onnx --spk-debug artifacts/spk/resnet50_protected.spk.json --output artifacts/reports/resnet50_runtime_map.json
python -m research.reliability.export_risk_report --risk artifacts/reports/resnet50_risk.json --runtime-map artifacts/reports/resnet50_runtime_map.json --output-dir artifacts/reports/resnet50
python -m research.reliability.analyze_risk_concentration --ranked-csv artifacts/reports/resnet50/resnet50_risk_ranked.csv --output-dir artifacts/reports/resnet50
python -m research.reliability.profile_range_guard --checkpoint artifacts/experiments/windows_prevalidation/resnet50/best.pt --ranked-csv artifacts/reports/resnet50/resnet50_risk_ranked.csv --injections artifacts/injections/resnet50_screen.jsonl --output artifacts/reports/resnet50/resnet50_range_guard_top20.json --device cuda
python -m research.reliability.profile_protection_costs --ranked-csv artifacts/reports/resnet50/resnet50_risk_ranked.csv --range-report artifacts/reports/resnet50/resnet50_range_guard_top20.json --spk artifacts/spk/resnet50.spk --input artifacts/inputs/resnet50.bin --runtime-bench build/runtime/Release/spkv2_bench.exe --primitive-bench build/runtime/Release/spkv2_protection_bench.exe --runtime-warmup 3 --runtime-runs 10 --output-candidates artifacts/reports/resnet50/protection_candidates.json --output-profile artifacts/reports/resnet50/protection_cost_profile.json
python -m research.reliability.evaluate_budget_sweep --candidates artifacts/reports/resnet50/protection_candidates.json --cost-profile artifacts/reports/resnet50/protection_cost_profile.json --output-dir artifacts/reports/resnet50
python -m research.reliability.evaluate_runtime_dataset --library build/runtime/Release/spkv2_runtime_shared.dll --baseline-spk artifacts/spk/resnet50.spk --protected-spk artifacts/spk/resnet50_protected.spk --plan artifacts/plans/resnet50_plan.json --split val --max-samples 512 --output artifacts/reports/resnet50/runtime_val_calibration.json
python -m research.reliability.evaluate_runtime_faults --library build/runtime/Release/spkv2_runtime_shared.dll --baseline-spk artifacts/spk/resnet50.spk --protected-spk artifacts/spk/resnet50_protected.spk --plan artifacts/plans/resnet50_plan.json --ranked-csv artifacts/reports/resnet50/resnet50_risk_ranked.csv --events 1024 --output artifacts/reports/resnet50/runtime_faults.json
python -m research.reliability.optimize_plan artifacts/reports/protection_candidates.json --model-id resnet50_eurosat --latency-budget-ms 10 --memory-budget-bytes 8388608 --output artifacts/plans/resnet50_plan.json
python -m research.reliability.prepare_yolo_runtime_sample --onnx artifacts/models/yolov10n_dior.onnx --image artifacts/data/dior/images/test/example.jpg --label artifacts/data/dior/labels/test/example.txt --input-output artifacts/inputs/yolov10n_example.bin --reference-output artifacts/outputs/yolov10n_example_ort.bin --metadata artifacts/reports/yolov10n/example.json
python -m research.reliability.compare_onnx_runtime_spk --onnx artifacts/models/yolov10n_dior.onnx --spk artifacts/spk/yolov10n_dior_cpu_generic.spk --library build/runtime/Release/spkv2_runtime_shared.dll --input artifacts/inputs/yolov10n_example.bin --input-shape 1 3 640 640 --task detection --output artifacts/reports/yolov10n/runtime_alignment.json
python -m research.reliability.screen_yolo_runtime_faults --library build/runtime/Release/spkv2_runtime_shared.dll --spk artifacts/spk/yolov10n_dior_cpu_generic.spk --spk-debug artifacts/spk/yolov10n_dior_cpu_generic.spk.json --images artifacts/data/dior/images/test --labels artifacts/data/dior/labels/test --output artifacts/injections/yolov10n_runtime_screen.jsonl
python -m research.reliability.screen_yolo_runtime_faults --library build/runtime/Release/spkv2_runtime_shared.dll --spk artifacts/spk/yolov10n_dior_cpu_generic.spk --spk-debug artifacts/spk/yolov10n_dior_cpu_generic.spk.json --images artifacts/data/dior/images/test --labels artifacts/data/dior/labels/test --sampling stratified --injections-per-node 8 --output artifacts/injections/yolov10n_runtime_stratified.jsonl
python -m research.reliability.build_control_path_plan --injections artifacts/injections/yolov10n_runtime_stratified.jsonl --spk-debug artifacts/spk/yolov10n_dior_cpu_generic.spk.json --model-id yolov10n_dior --memory-budget-bytes 4194304 --output artifacts/plans/yolov10n_control_dmr.json
python -m research.reliability.export_detection_protection_report --baseline-injections artifacts/injections/yolov10n_runtime_stratified.jsonl --protected-injections artifacts/injections/yolov10n_runtime_control_dmr.jsonl --baseline-bench artifacts/reports/yolov10n/baseline_bench.json --protected-bench artifacts/reports/yolov10n/control_dmr_bench.json --extra-memory-bytes 48000 --output-dir artifacts/reports/yolov10n/control_dmr
```

CPU-only pipeline smoke runs must use a separate name and bounded sample set;
their checkpoints are not valid paper workloads:

```powershell
python -m research.reliability.models.train_resnet50 --run-name resnet50_smoke --train-mode head --device cpu --epochs 1 --max-train-samples 256 --max-val-samples 128 --max-test-samples 128
```

Compiler/runtime integration:

```powershell
python -m spinnv2.compiler compile artifacts/models/resnet50_eurosat.onnx -o artifacts/spk/resnet50_protected.spk --protection-plan artifacts/plans/resnet50_plan.json
cmake -S runtime -B build/runtime
cmake --build build/runtime --config Release
```

`spkv2_fault_run` accepts `model input output node tensor element bit invocation`
and prints structured injection/detection/recovery counters.

The budget optimizer minimizes expected critical task-failure probability under
an activation-byte-weighted runtime-object prior. DMR scratch is peak reused
storage, so the memory budget constrains the maximum selected scratch
requirement rather than summing buffers for sequential nodes. Range-guard
benefits use Wilson lower confidence bounds and false-positive rerun cost uses
the corresponding upper bound.

Range thresholds calibrated in PyTorch are screening thresholds only. Before a
deployable plan is reported, compile the plan, collect clean runtime range
observations with `evaluate_runtime_plan`, recalibrate with
`recalibrate_range_plan`, and evaluate on disjoint runtime test inputs.

## Current ResNet50 Windows Result

- Release AVX2 `cpu_generic`: 10-run mean baseline latency `131.65 ms`;
  maximum absolute difference from ONNX Runtime `5.53e-05`, top-1 unchanged.
- Final feedback-calibrated ILP plan: 30 protected modes, predicted critical
  risk reduction `66.14%`, peak extra memory `3,211,264` bytes.
- Clean disjoint runtime test: 512 samples, accuracy unchanged at `97.27%`,
  prediction agreement `100%`, false alarms `2/512`, measured overhead
  `19.53 ms` under the `20 ms` budget.
- Runtime fault Monte Carlo: 1024 activation-byte-weighted, uniformly sampled
  FP32 bit flips; critical failures reduced from `16` to `4` (`75.00%`
  observed reduction; paired bootstrap 95% interval `50.00%` to `94.74%`).

## Current YOLOv10n Windows Status

- A minimal real-DIOR subset (`128/64/64`) completes the pipeline but obtains
  only `mAP@0.5=0.0272`; task-aware screening correctly declines to make a
  reliability estimate because it has no baseline true positives.
- A larger prevalidation subset (`1000/1000/256`) obtains
  `mAP@0.5=0.1973` and `mAP@0.5:0.95=0.1383`. It is sufficient to exercise
  detection reliability code, but remains explicitly outside final paper
  detection metrics.
- Its compiled model has `308` ONNX nodes with no unsupported operator,
  activation planning `146,338,400 -> 11,468,800` bytes, mean Release AVX2
  isolated mean Release AVX2 runtime `172.10 ms`, and maximum absolute difference from ONNX Runtime
  `6.10e-05` on a real DIOR image.
- Deployment-domain screening finds `32/1024` task failures under the
  activation-byte prior. Equal-count screening of all `170` runtime nodes
  finds `36/1360` failures: `25` task-output failures and `11` controlled
  execution errors caused by corrupted indexing/control-path values.
- `Gather`/`GatherElements` now reject non-finite or out-of-range indices so a
  corrupted control-path value is recorded as a critical failure instead of
  crashing the experiment process.
- A four-node control-path DMR plan removes all `11` observed controlled
  execution errors in the same `1360` stratified events, reducing critical
  failures from `36` to `25` with `48,000` bytes peak extra memory and
  `0.815 ms` (`0.473%`) isolated mean latency overhead. It detects and
  recovers all `32` injections landing on its four protected nodes.
- These subset runs are retained as implementation prevalidation only.

## Current Full-DIOR YOLOv10n Result

- Full materialized DIOR workload: `18000/2000/3463` train/validation/test
  images. The fixed 30-epoch task-adapted detector obtains test
  `mAP@0.5=0.84279`, `mAP@0.5:0.95=0.62100`, and `mAP@0.75=0.67921`.
- The exported ONNX graph contains `308` nodes and compiles without unsupported
  operators. Activation planning is `146,338,400 -> 11,468,800` bytes; a real
  DIOR image differs from ONNX Runtime by at most `6.10e-05`.
- The formal injection screen uses 125 evaluable images among the first 128
  DIOR test images, not all 3463 test images. In `170 x 16 = 2720` stratified
  runtime-object events, the unprotected model has `77` critical failures
  (`65` task-output failures, `12` controlled execution errors). In `2000`
  activation-byte-prior events it has `56` critical failures.
- A low-cost control-path DMR plan protects five nodes, uses `48,000` bytes
  peak extra scratch, adds `1.2215 ms` (`0.728%`) average latency, and removes
  all 12 stratified controlled execution errors.
- Detection DMR candidates are now formed from task risk times activation-byte
  exposure and measured runtime/DMR costs. Multi-output nodes unsupported by
  the current DMR primitive are excluded before ILP compilation and recorded
  in the cost profile.
- The runtime-feedback-calibrated ILP DMR plan protects 40 single-output nodes,
  uses `3,276,800` bytes peak extra scratch, and increases isolated runtime
  from `167.7848 ms` to `198.0945 ms` (`18.065%`), within its 20% / 4 MiB
  representative budget.
- With identical injected events, this plan reduces stratified failures
  `77 -> 13` (`83.12%`, paired bootstrap 95% interval `74.32%` to `91.30%`)
  and activation-prior failures `56 -> 43` (`23.21%`, interval `12.70%` to
  `34.55%`). Report both outcomes; the stratified reduction is not an estimate
  of arbitrary in-orbit fault distributions.
- Because the detection experiment currently exposes DMR candidates only, ILP,
  risk Top-k, and benefit/latency greedy tie at `48.91%` predicted mitigation
  under the representative budget; random DMR averages `47.12%`. The
  multi-mode optimization advantage is established by the ResNet50
  Range-Guard-plus-DMR study rather than overstated for YOLO.
- `figures/yolov10n_fault_recovery_case.png` in the formal report directory is
  a reproducible three-panel visualization of a DIOR detection fault and its
  protected runtime recovery.

## Role Of Training

Training is not part of the proposed inference optimization method. It only
adapts public pretrained models into valid remote-sensing workloads so that a
fault can be evaluated as a classification error, missed detection, false
positive, or mAP degradation. ResNet50 defaults to frozen-backbone,
classification-head-only calibration and caches frozen backbone features once
per split; full fine-tuning is used only if the resulting baseline accuracy is
inadequate. YOLOv10n requires DIOR adaptation
because its pretrained COCO task does not define the DIOR evaluation target.

## Interpretation Boundary

The runtime injection experiment measures relative robustness under the stated
software fault model. It is not a radiation-rate estimate and is not evidence
of flight qualification.
