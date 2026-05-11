# 📜 脚本使用指南

本目录包含 KernelBench 基准测试的所有脚本：生成内核、评估内核、收集基线时间、分析结果以及调试/检查工具。

## 概览

| 脚本 | 用途 |
|------|------|
| `generate_and_eval_single_sample.py` | 生成并评估单个问题（本地 GPU） |
| `generate_and_eval_single_sample_modal.py` | 生成并评估单个问题（Modal 云 GPU） |
| `generate_samples.py` | 批量生成某一级别的所有内核 |
| `eval_from_generations.py` | 批量评估已生成的内核（本地或 Modal） |
| `run_and_check.py` | 评估一对（参考架构, 解决方案）文件并计算加速比 |
| `benchmark_eval_analysis.py` | 计算基准指标（fast_p、加速比等） |
| `generate_baseline_time.py` | 记录 PyTorch 基线时间（本地 GPU） |
| `generate_baseline_time_modal.py` | 记录 PyTorch 基线时间（Modal 云 GPU） |
| `get_baseline_time_single_problem.py` | 单个问题的快速基线计时 |
| `verify_bench.py` | 验证基准问题的自一致性 |
| `verify_generation.py` | 测试推理和提示构造流水线 |
| `inspect_baseline.py` | 检查 torch.compile 内部细节（dynamo、融合、代码生成） |
| `inspect_triton.py` | 检查 torch.compile 生成的 Triton 代码和火焰图 |
| `inspect_kernel_pytorch_profiler.py` | 使用 PyTorch Profiler 分析生成的内核 |
| `debug_stddout.py` | 测试 nvcc 编译错误捕获 |

---

## 核心工作流脚本

### 1. `generate_and_eval_single_sample.py` — 单问题评估（本地）

最简单的入口。获取问题、通过 LLM 生成内核、在本地 GPU 上评估。

```bash
uv run python scripts/generate_and_eval_single_sample.py \
    dataset_src=huggingface level=2 problem_id=40 \
    server_type=google model_name=gemini/gemini-2.5-flash \
    max_tokens=8192 temperature=0.0
```

**主要参数：**
- `dataset_src` — `huggingface` 或 `local`
- `level` — 问题级别（1, 2, 3, 4）
- `problem_id` — 逻辑问题索引
- `server_type` / `model_name` — LLM 提供商和模型
- `gpu_arch` — 目标 GPU 架构，如 `["Ada"]`、`["Hopper"]`
- `precision` — `fp32`、`fp16` 或 `bf16`
- `backend` — `cuda`、`triton`、`cute`、`tilelang`、`thunderkittens`
- `prompt_option` — `zero_shot`、`one_shot`、`few_shot`
- `is_reasoning_model` — 对 o1/o3/Gemini 2.5 思考模型设为 `True`
- 添加 `.verbose_logging` 启用完整日志

### 2. `generate_and_eval_single_sample_modal.py` — 单问题评估（Modal）

与上述脚本功能相同，但在 Modal 云 GPU 上评估。适用于没有本地 GPU 的场景。

```bash
uv run python scripts/generate_and_eval_single_sample_modal.py \
    dataset_src=huggingface level=1 problem_id=1 \
    eval_mode=modal gpu=L40S \
    server_type=google model_name=gemini/gemini-2.5-flash
```

**额外参数：**
- `gpu` — Modal GPU 类型：`L40S`、`H100`、`A100`、`L4`、`T4`、`A10G`

### 3. `generate_samples.py` — 批量生成

为某一级别的所有（或部分）问题生成内核。结果存储在 `runs/{run_name}/` 中。

```bash
uv run python scripts/generate_samples.py \
    run_name=my_run dataset_src=huggingface level=1 \
    num_workers=50 server_type=deepseek model_name=deepseek-chat temperature=0
```

**主要参数：**
- `run_name` — 本次运行的唯一名称（结果保存至 `runs/{run_name}/`）
- `num_workers` — 并行 API 线程数
- `num_samples` — 每个问题的采样数（默认 1，增大用于 pass@k）
- `subset` — 元组 `(start_id, end_id)`，仅生成部分问题
- 自动跳过已生成的内核

### 4. `eval_from_generations.py` — 批量评估

评估 `runs/{run_name}/` 中预先生成的内核。支持本地多 GPU 和 Modal 评估。

```bash
# 本地评估
uv run python scripts/eval_from_generations.py \
    run_name=my_run dataset_src=local level=1 \
    num_gpu_devices=8 timeout=300

# Modal 评估
uv run python scripts/eval_from_generations.py \
    run_name=my_run dataset_src=huggingface level=1 \
    eval_mode=modal gpu=H100 num_gpu_devices=16
```

**主要参数：**
- `eval_mode` — `local` 或 `modal`
- `num_gpu_devices` — 并行评估的 GPU 数量
- `timeout` — 每批次超时时间（秒），默认 180
- `build_cache` — 设为 `True` 在 CPU 上预编译后再 GPU 评估
- `num_cpu_workers` — 并行预编译的 CPU 工作线程数
- `num_correct_trials` — 正确性检查试验次数（默认 5）
- `num_perf_trials` — 性能计时试验次数（默认 100）
- 自动跳过已评估的问题
- 结果保存至 `runs/{run_name}/eval_results.json`

### 5. `run_and_check.py` — 评估文件对

评估特定的（参考架构, 解决方案）文件对。输出正确性、运行时间以及相对 PyTorch eager 和 torch.compile 的加速比。

```bash
# 本地文件作为参考（本地评估）
uv run python scripts/run_and_check.py \
    ref_origin=local \
    ref_arch_src_path=src/kernelbench/prompts/model_ex_add.py \
    kernel_src_path=src/kernelbench/prompts/model_new_ex_add.py \
    eval_mode=local

# KernelBench 问题作为参考（Modal 评估）
uv run python scripts/run_and_check.py \
    ref_origin=kernelbench level=2 problem_id=40 \
    kernel_src_path=path/to/generated_kernel.py \
    eval_mode=modal gpu=H100
```

**主要参数：**
- `ref_origin` — `local`（文件路径）或 `kernelbench`（level + problem_id）
- `eval_mode` — `local` 或 `modal`
- `check_kernel` — 静态分析可疑模式（默认 `True`）

---

## 分析脚本

### 6. `benchmark_eval_analysis.py` — 计算基准指标

计算编译率、正确率、几何平均加速比和 `fast_p` 分数。

```bash
uv run python scripts/benchmark_eval_analysis.py \
    run_name=my_run level=1 \
    hardware=L40S_matx3 baseline=baseline_time_torch
```

**主要参数：**
- `hardware` / `baseline` — 对应 `results/timing/{hardware}/{baseline}.json`
- `baseline_file` — 覆盖：基线 JSON 的直接路径
- `eval_results_dir` — 覆盖：runs 目录路径
- `output_file` — 将 JSON 结果写入文件

**输出内容：**
- 编译率和正确率
- 几何平均加速比（仅正确样本）
- 不同加速阈值 p = 0.0, 0.5, 0.8, 1.0, 1.5, 2.0 的 `fast_p` 分数
- Pass@k 指标（如可用）

---

## 基线计时脚本

### 7. `generate_baseline_time.py` — 本地基线计时

在本地 GPU 上记录所有级别的 PyTorch 参考时间。为多种配置生成 JSON 文件（eager、各种 torch.compile 模式）。

```bash
# 在脚本中编辑 hardware_name，然后运行：
uv run python scripts/generate_baseline_time.py
```

生成文件位于 `results/timing/{hardware_name}/`：
- `baseline_time_torch.json` — Eager 执行
- `baseline_time_torch_compile_inductor_{mode}.json` — torch.compile 各模式
- `baseline_time_torch_compile_cudagraphs.json` — CUDA graphs 后端

### 8. `generate_baseline_time_modal.py` — Modal 基线计时

功能同上，但在 Modal 云 GPU 上并行运行。

```bash
uv run python scripts/generate_baseline_time_modal.py \
    level=1 gpu=H100 hardware_name=H100_Modal \
    num_gpu_devices=8 num_trials=100
```

### 9. `get_baseline_time_single_problem.py` — 快速单问题计时

一个最小化脚本，对单个硬编码问题（softmax）进行计时。适合快速健全性检查。

```bash
uv run python scripts/get_baseline_time_single_problem.py
```

---

## 验证与调试脚本

### 10. `verify_bench.py` — 验证基准完整性

将每个参考模型与自身进行对比运行，确保自一致性（无非确定性问题）。

```bash
uv run python scripts/verify_bench.py
```

### 11. `verify_generation.py` — 测试推理流水线

测试 LLM 推理流水线和提示构造。适合迭代优化提示。

```bash
uv run python scripts/verify_generation.py [可选的架构文件路径]
```

### 12. `inspect_baseline.py` — 检查 torch.compile 内部细节

检查某个问题的 dynamo 跟踪、追踪图、融合决策和生成代码。

```bash
uv run python scripts/inspect_baseline.py
```

### 13. `inspect_triton.py` — 检查 Triton 代码和火焰图

[开发中] 生成 torch.compile 的 Triton 代码和 Chrome trace 火焰图用于性能分析。

```bash
uv run python scripts/inspect_triton.py
```

### 14. `inspect_kernel_pytorch_profiler.py` — 分析生成的内核

使用 PyTorch Profiler（仅 CUDA 活动）分析生成的内核，展示算子分解。

```bash
uv run python scripts/inspect_kernel_pytorch_profiler.py
```

### 15. `debug_stddout.py` — 测试编译错误捕获

测试 nvcc 编译错误是否被正确捕获。用于开发诊断。

```bash
uv run python scripts/debug_stddout.py
```

---

## 典型端到端工作流

```bash
# 步骤 1：为 Level 1 生成内核
uv run python scripts/generate_samples.py \
    run_name=exp_v1 dataset_src=huggingface level=1 \
    num_workers=50 server_type=deepseek model_name=deepseek-chat temperature=0

# 步骤 2：评估所有生成的内核
uv run python scripts/eval_from_generations.py \
    run_name=exp_v1 dataset_src=local level=1 \
    num_gpu_devices=8 timeout=300

# 步骤 3：分析结果
uv run python scripts/benchmark_eval_analysis.py \
    run_name=exp_v1 level=1 \
    hardware=L40S_matx3 baseline=baseline_time_torch
```
