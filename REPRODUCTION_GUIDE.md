# 毕业论文项目复现指南

本文档说明如何在一台**全新的 Windows 机器**上从零复现本项目的全部实验。

---

## 0. 前提条件

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11（64-bit） |
| GPU | NVIDIA GPU + CUDA 11.6+（用于模型训练和故障注入实验） |
| 编译器 | Visual Studio 2022（含 MSVC C/C++ 工具链） |
| CMake | 3.20+ |
| Git | 2.30+，并安装 Git LFS |
| Conda | Miniconda 或 Anaconda |
| 磁盘空间 | 至少 20 GB 可用空间 |

---

## 1. 克隆仓库

```powershell
# 安装 Git LFS（如未安装）
git lfs install

# 克隆仓库（LFS 文件会自动拉取）
git clone git@github.com:tzp11/graduate.git
cd graduate
```

克隆完成后你会得到：
- 全部源代码（编译器 + 运行时 + 研究脚本 + 测试）
- 实验报告和论文图表（`artifacts/reports/`）
- YOLOv8n 和 YOLOv10n 预训练权重（通过 LFS）

---

## 2. 创建 Conda 环境

### 2.1 基础可靠性研究环境

```powershell
# 创建 Python 3.11 环境
conda create -n graduatepaper_reliable python=3.11 -y
conda activate graduatepaper_reliable

# 进入代码目录
cd code/SPINNV2

# 安装基础依赖
pip install -r requirements-dev.txt
# requirements-dev.txt 包含: pytest>=8.0, numpy>=2.0, onnx>=1.16, onnxruntime>=1.18

# 安装额外研究依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib scipy pandas scikit-learn pulp
```

### 2.2 GPU 训练环境（用于模型训练和 GPU 故障注入）

```powershell
# 创建 GPU 环境（CUDA 11.6 + PyTorch 1.12.1）
conda create -n graduatepaper_gpu python=3.11 -y
conda activate graduatepaper_gpu

# 安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 或者 CUDA 12.x:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装 Ultralytics（用于 YOLOv10n 训练）
pip install ultralytics==8.2.100

# 安装其他依赖
pip install -r requirements-dev.txt
pip install matplotlib scipy pandas scikit-learn pulp
```

---

## 3. 编译 C 运行时

C 运行时是推理引擎的核心，必须编译后才能运行故障注入实验。

```powershell
cd code/SPINNV2

# 配置 CMake（使用 Visual Studio 2022 生成器）
cmake -S runtime -B build/runtime

# 编译 Release 版本（启用 AVX2 优化）
cmake --build build/runtime --config Release

# 运行 C 运行时测试
ctest --test-dir build/runtime --build-config Release
```

编译成功后，以下关键可执行文件会生成在 `build/runtime/Release/` 目录：

| 文件 | 用途 |
|------|------|
| `spkv2_run.exe` | 运行 SPK 模型推理 |
| `spkv2_bench.exe` | 性能基准测试 |
| `spkv2_fault_run.exe` | 单次故障注入运行 |
| `spkv2_protection_bench.exe` | 保护机制性能测试 |

---

## 4. 验证编译器基础功能

```powershell
cd code/SPINNV2
conda activate graduatepaper_reliable

# 验证编译器 CLI
python -m spinnv2.compiler --help
python -m spinnv2.compiler --print-target cpu_ref

# 运行编译器单元测试
pytest tests/compiler
pytest tests/codegen

# 运行端到端测试（小模型）
pytest tests/e2e
python tests/e2e/run_e2e.py --all

# 运行基准测试（跳过大模型）
python benchmarks/run_all.py --skip-large-run
```

---

## 5. 复现 ResNet50 + EuroSAT 实验

### 5.1 准备 EuroSAT 数据集

```powershell
python -m research.reliability.prepare_data --dataset eurosat
```

这会自动下载 EuroSAT 遥感图像分类数据集到 `artifacts/data/eurosat/`。

### 5.2 训练 ResNet50（分类头微调）

```powershell
conda activate graduatepaper_gpu

# 使用 GPU，冻结骨干网络只训练分类头
python -m research.reliability.models.train_resnet50 \
    --train-mode head \
    --device auto

# 训练完成后，权重保存在：
# artifacts/experiments/windows_prevalidation/resnet50_gpu_prevalidation/best.pt
```

> **注意**：如果只有 CPU，可以使用 smoke 模式验证流程：
> ```powershell
> python -m research.reliability.models.train_resnet50 --run-name resnet50_smoke --train-mode head --device cpu --epochs 1 --max-train-samples 256 --max-val-samples 128 --max-test-samples 128
> ```

### 5.3 导出 ONNX 模型

```powershell
python -m research.reliability.models.export_models \
    --model resnet50 \
    --checkpoint artifacts/experiments/windows_prevalidation/resnet50_gpu_prevalidation/best.pt \
    --output artifacts/models/resnet50_eurosat_gpu_prevalidation.onnx
```

### 5.4 检查 SPINNV2 算子兼容性

```powershell
python -m research.reliability.audit_spinn_compat \
    artifacts/models/resnet50_eurosat_gpu_prevalidation.onnx
```

### 5.5 编译为 SPK 部署包

```powershell
# 基线编译（无保护）
python -m spinnv2.compiler compile \
    artifacts/models/resnet50_eurosat_gpu_prevalidation.onnx \
    -o artifacts/spk/resnet50_eurosat_gpu_prevalidation_cpu_generic.spk \
    --target cpu_generic

# 带保护计划的编译
python -m spinnv2.compiler compile \
    artifacts/models/resnet50_eurosat_gpu_prevalidation.onnx \
    -o artifacts/spk/resnet50_protected.spk \
    --target cpu_generic \
    --protection-plan artifacts/plans/resnet50_final_dmr.json
```

### 5.6 故障注入与风险分析

```powershell
# PyTorch 层面故障注入筛选
python -m research.reliability.inject_faults \
    --checkpoint artifacts/experiments/windows_prevalidation/resnet50_gpu_prevalidation/best.pt \
    --output artifacts/injections/resnet50_gpu_prevalidation_screen.jsonl

# 构建风险画像
python -m research.reliability.build_risk_profile \
    artifacts/injections/resnet50_gpu_prevalidation_screen.jsonl \
    --output artifacts/reports/resnet50_gpu_prevalidation/resnet50_risk.json

# 运行时故障注入蒙特卡洛实验（需要已编译的 C 运行时）
python -m research.reliability.evaluate_runtime_faults \
    --library build/runtime/Release/spkv2_runtime_shared.dll \
    --baseline-spk artifacts/spk/resnet50_eurosat_gpu_prevalidation_cpu_generic.spk \
    --protected-spk artifacts/spk/resnet50_protected.spk \
    --plan artifacts/plans/resnet50_final_dmr.json \
    --ranked-csv artifacts/reports/resnet50_gpu_prevalidation/resnet50_risk_ranked.csv \
    --events 1024 \
    --output artifacts/reports/resnet50_gpu_prevalidation/runtime_faults.json
```

### 5.7 预期结果（ResNet50）

| 指标 | 预期值 |
|------|--------|
| Release AVX2 基线延迟 | ~131.65 ms（10-run 均值） |
| 与 ONNX Runtime 最大绝对差 | ≤ 5.53e-05 |
| 保护计划保护模式数 | 30 |
| 预测关键风险降低 | 66.14% |
| 额外峰值内存 | 3,211,264 bytes |
| 运行时故障关键失败降低 | 75.00%（bootstrap 95% CI: 50%-94.74%） |

---

## 6. 复现 YOLOv10n + DIOR 实验

### 6.1 准备 DIOR 数据集

```powershell
# 从 HuggingFace 下载 DIOR 数据集
python -m research.reliability.download_dior --source hf_parquet --splits train val test

# 转换为 YOLO 格式
python -m research.reliability.prepare_data \
    --dataset dior \
    --dior-parquet "artifacts/data/dior/raw/hf_parquet/data/train-*.parquet" \
    --split train
python -m research.reliability.prepare_data \
    --dataset dior \
    --dior-parquet "artifacts/data/dior/raw/hf_parquet/data/validation-*.parquet" \
    --split val
python -m research.reliability.prepare_data \
    --dataset dior \
    --dior-parquet "artifacts/data/dior/raw/hf_parquet/data/test-*.parquet" \
    --split test
```

> DIOR 完整数据集约 2.6 GB，包含 18000/2000/3463 张训练/验证/测试图像。

### 6.2 训练 YOLOv10n

```powershell
conda activate graduatepaper_gpu

# 使用 Ultralytics 训练 YOLOv10n，30 epoch
# （需要 GPU，训练时间取决于 GPU 型号）
# 权重保存到: artifacts/experiments/windows_prevalidation/yolov10n/dior_full_e30_b16/weights/best.pt
```

### 6.3 导出 ONNX 并编译

```powershell
# 导出为简化 ONNX
python -m research.reliability.models.export_models \
    --model yolov10n \
    --weights artifacts/experiments/windows_prevalidation/yolov10n/dior_full_e30_b16/weights/best.pt \
    --output artifacts/models/yolov10n_dior_full_e30_b16.onnx

# 编译为 SPK
python -m spinnv2.compiler compile \
    artifacts/models/yolov10n_dior_full_e30_b16.onnx \
    -o artifacts/spk/yolov10n_dior_full_e30_b16_cpu_generic.spk \
    --target cpu_generic
```

### 6.4 运行时故障注入

```powershell
# 分层故障注入
python -m research.reliability.screen_yolo_runtime_faults \
    --library build/runtime/Release/spkv2_runtime_shared.dll \
    --spk artifacts/spk/yolov10n_dior_full_e30_b16_cpu_generic.spk \
    --spk-debug artifacts/spk/yolov10n_dior_full_e30_b16_cpu_generic.spk.json \
    --images artifacts/data/dior/images/test \
    --labels artifacts/data/dior/labels/test \
    --sampling stratified \
    --injections-per-node 16 \
    --output artifacts/reports/yolov10n_dior_full_e30_b16/runtime_fault_stratified_170x16.jsonl
```

### 6.5 预期结果（YOLOv10n）

| 指标 | 预期值 |
|------|--------|
| 测试集 mAP@0.5 | 0.84279 |
| 测试集 mAP@0.5:0.95 | 0.62100 |
| ONNX 节点数 | 308 |
| 激活内存规划 | 146,338,400 → 11,468,800 bytes |
| Release AVX2 延迟 | ~167.78 ms |
| 分层故障关键失败降低 | 83.12%（bootstrap 95% CI: 74.32%-91.30%） |

---

## 7. 生成论文图表

```powershell
# 导出论文用的表格
python scripts/export_paper_tables.py

# 检查可复现性（对照冻结阈值）
python scripts/check_reproducibility.py
```

论文图表资产位于：`artifacts/reports/paper_assets/`

---

## 8. 目录结构说明

```
graduatepaper/
├── code/SPINNV2/           # 核心代码
│   ├── compiler/           # Python 编译器（ONNX → SIR → SPK）
│   ├── runtime/            # C 运行时推理引擎
│   ├── research/           # 可靠性研究脚本
│   │   └── reliability/    # 故障注入、风险分析、保护计划
│   ├── tests/              # 测试套件
│   ├── benchmarks/         # 性能基准
│   ├── scripts/            # 论文表格导出
│   ├── docs/               # 架构文档
│   ├── format/             # SIR/SPK 格式规范
│   └── artifacts/          # 实验产物（数据集、模型、报告）
│       ├── reports/        # 实验报告和论文图表
│       ├── audits/         # 算子兼容性审计
│       ├── plans/          # 保护计划配置
│       └── profiles/       # 性能分析
├── examples/               # 师兄论文参考截图
├── *.md                    # 项目说明文档
└── REPRODUCTION_GUIDE.md   # 本文件
```

---

## 9. 常见问题

### Q: CMake 编译失败？
确保安装了 Visual Studio 2022 的「使用 C++ 的桌面开发」工作负载，并且 CMake 能找到 MSVC 编译器。

### Q: Git LFS 文件拉取失败？
```powershell
git lfs install
git lfs pull
```

### Q: CUDA 不可用？
ResNet50 的头部微调可以在 CPU 上运行（加 `--device cpu`），但速度很慢。YOLOv10n 训练强烈建议使用 GPU。

### Q: EuroSAT 下载失败？
检查网络连接。EuroSAT 从 torchvision 数据集自动下载。如果被墙，可手动下载 EuroSAT.zip 放到 `artifacts/data/eurosat/eurosat/` 目录。

### Q: DIOR 数据集下载失败？
DIOR 数据集从 HuggingFace 下载。如果网络不通，可以：
1. 设置 HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
2. 或手动下载后放到 `artifacts/data/dior/raw/` 目录

### Q: 实验结果数值差异？
由于浮点运算的不确定性和硬件差异，以下指标可能有微小偏差：
- 延迟数值取决于 CPU 型号和系统负载
- 训练结果取决于 GPU 型号（FP32 计算可能有微小差异）
- 故障注入蒙特卡洛结果在置信区间内波动是正常的

已有的实验报告（`artifacts/reports/`）保存了原始机器上的完整结果，可作为参考基准。
