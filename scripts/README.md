# 📜 Scripts Guide

This directory contains all the scripts for running the KernelBench benchmark: generating kernels, evaluating them, collecting baseline timings, analyzing results, and debugging/inspection utilities.

## Overview

| Script | Purpose |
|--------|---------|
| `generate_and_eval_single_sample.py` | Generate & evaluate a single problem (local GPU) |
| `generate_and_eval_single_sample_modal.py` | Generate & evaluate a single problem (Modal cloud GPU) |
| `generate_samples.py` | Batch generate kernels for a full level |
| `eval_from_generations.py` | Batch evaluate pre-generated kernels (local or Modal) |
| `run_and_check.py` | Evaluate a (reference, solution) file pair with speedup |
| `benchmark_eval_analysis.py` | Compute benchmark metrics (fast_p, speedup, etc.) |
| `generate_baseline_time.py` | Record PyTorch baseline timings (local GPU) |
| `generate_baseline_time_modal.py` | Record PyTorch baseline timings (Modal cloud GPU) |
| `get_baseline_time_single_problem.py` | Quick baseline timing for a single hardcoded problem |
| `verify_bench.py` | Verify benchmark problems are self-consistent |
| `verify_generation.py` | Test inference & prompt construction pipeline |
| `inspect_baseline.py` | Inspect torch.compile internals (dynamo, fusion, codegen) |
| `inspect_triton.py` | Inspect torch.compile-generated Triton code & flamegraphs |
| `inspect_kernel_pytorch_profiler.py` | Profile a generated kernel with PyTorch Profiler |
| `debug_stddout.py` | Test nvcc error capturing for compilation diagnostics |

---

## Core Workflow Scripts

### 1. `generate_and_eval_single_sample.py` — Single Problem (Local)

The easiest entry point. Fetches a problem, generates a kernel via LLM, and evaluates it on a local GPU.

```bash
uv run python scripts/generate_and_eval_single_sample.py \
    dataset_src=huggingface level=2 problem_id=40 \
    server_type=google model_name=gemini/gemini-2.5-flash \
    max_tokens=8192 temperature=0.0
```

**Key parameters:**
- `dataset_src` — `huggingface` or `local`
- `level` — Problem level (1, 2, 3, 4)
- `problem_id` — Logical problem index
- `server_type` / `model_name` — LLM provider and model
- `gpu_arch` — Target GPU architecture, e.g. `["Ada"]`, `["Hopper"]`
- `precision` — `fp32`, `fp16`, or `bf16`
- `backend` — `cuda`, `triton`, `cute`, `tilelang`, `thunderkittens`
- `prompt_option` — `zero_shot`, `one_shot`, `few_shot`
- `is_reasoning_model` — Set `True` for o1/o3/Gemini 2.5 thinking models
- Add `.verbose_logging` to enable full logging

### 2. `generate_and_eval_single_sample_modal.py` — Single Problem (Modal)

Same as above but evaluates on a Modal cloud GPU. Useful when you don't have a local GPU.

```bash
uv run python scripts/generate_and_eval_single_sample_modal.py \
    dataset_src=huggingface level=1 problem_id=1 \
    eval_mode=modal gpu=L40S \
    server_type=google model_name=gemini/gemini-2.5-flash
```

**Additional parameter:**
- `gpu` — Modal GPU type: `L40S`, `H100`, `A100`, `L4`, `T4`, `A10G`

### 3. `generate_samples.py` — Batch Generation

Generates kernels for all (or a subset of) problems in a level. Stores results in `runs/{run_name}/`.

```bash
uv run python scripts/generate_samples.py \
    run_name=my_run dataset_src=huggingface level=1 \
    num_workers=50 server_type=deepseek model_name=deepseek-chat temperature=0
```

**Key parameters:**
- `run_name` — Unique name for this run (results go to `runs/{run_name}/`)
- `num_workers` — Number of parallel API threads
- `num_samples` — Samples per problem (default 1, increase for pass@k)
- `subset` — Tuple `(start_id, end_id)` to generate only a subset
- Skips already-generated kernels automatically

### 4. `eval_from_generations.py` — Batch Evaluation

Evaluates pre-generated kernels from `runs/{run_name}/`. Supports both local multi-GPU and Modal evaluation.

```bash
# Local evaluation
uv run python scripts/eval_from_generations.py \
    run_name=my_run dataset_src=local level=1 \
    num_gpu_devices=8 timeout=300

# Modal evaluation
uv run python scripts/eval_from_generations.py \
    run_name=my_run dataset_src=huggingface level=1 \
    eval_mode=modal gpu=H100 num_gpu_devices=16
```

**Key parameters:**
- `eval_mode` — `local` or `modal`
- `num_gpu_devices` — Number of GPUs for parallel evaluation
- `timeout` — Per-batch timeout in seconds (default 180)
- `build_cache` — Set `True` to pre-compile on CPU before GPU eval
- `num_cpu_workers` — CPU workers for parallel pre-compilation
- `num_correct_trials` — Correctness check trials (default 5)
- `num_perf_trials` — Performance timing trials (default 100)
- Skips already-evaluated problems automatically
- Results saved to `runs/{run_name}/eval_results.json`

### 5. `run_and_check.py` — Evaluate a File Pair

Evaluate a specific (reference, solution) pair. Prints correctness, runtime, and speedup over both PyTorch eager and torch.compile.

```bash
# Local file as reference (local eval)
uv run python scripts/run_and_check.py \
    ref_origin=local \
    ref_arch_src_path=src/kernelbench/prompts/model_ex_add.py \
    kernel_src_path=src/kernelbench/prompts/model_new_ex_add.py \
    eval_mode=local

# KernelBench problem as reference (Modal eval)
uv run python scripts/run_and_check.py \
    ref_origin=kernelbench level=2 problem_id=40 \
    kernel_src_path=path/to/generated_kernel.py \
    eval_mode=modal gpu=H100
```

**Key parameters:**
- `ref_origin` — `local` (file path) or `kernelbench` (level + problem_id)
- `eval_mode` — `local` or `modal`
- `check_kernel` — Static analysis for suspicious patterns (default `True`)

---

## Analysis Scripts

### 6. `benchmark_eval_analysis.py` — Compute Benchmark Metrics

Computes compilation rate, correctness rate, geometric mean speedup, and `fast_p` scores.

```bash
uv run python scripts/benchmark_eval_analysis.py \
    run_name=my_run level=1 \
    hardware=L40S_matx3 baseline=baseline_time_torch
```

**Key parameters:**
- `hardware` / `baseline` — Maps to `results/timing/{hardware}/{baseline}.json`
- `baseline_file` — Override: direct path to a baseline JSON
- `eval_results_dir` — Override: path to runs directory
- `output_file` — Write JSON output to a file

**Output includes:**
- Compilation rate & correctness rate
- Geometric mean speedup (correct samples)
- `fast_p` scores for thresholds p = 0.0, 0.5, 0.8, 1.0, 1.5, 2.0
- Pass@k metrics (if available)

---

## Baseline Timing Scripts

### 7. `generate_baseline_time.py` — Local Baseline Timing

Records PyTorch reference timings across all levels on a local GPU. Generates JSON files for multiple configurations (eager, torch.compile with various modes).

```bash
# Edit hardware_name in the script, then run:
uv run python scripts/generate_baseline_time.py
```

Produces files under `results/timing/{hardware_name}/`:
- `baseline_time_torch.json` — Eager execution
- `baseline_time_torch_compile_inductor_{mode}.json` — torch.compile variants
- `baseline_time_torch_compile_cudagraphs.json` — CUDA graphs backend

### 8. `generate_baseline_time_modal.py` — Modal Baseline Timing

Same purpose as above but runs on Modal cloud GPUs in parallel.

```bash
uv run python scripts/generate_baseline_time_modal.py \
    level=1 gpu=H100 hardware_name=H100_Modal \
    num_gpu_devices=8 num_trials=100
```

### 9. `get_baseline_time_single_problem.py` — Quick Single Timing

A minimal script to time a single hardcoded problem (softmax). Useful for quick sanity checks.

```bash
uv run python scripts/get_baseline_time_single_problem.py
```

---

## Verification & Debugging Scripts

### 10. `verify_bench.py` — Verify Benchmark Integrity

Runs every reference model against itself to ensure self-consistency (no non-determinism issues).

```bash
uv run python scripts/verify_bench.py
```

### 11. `verify_generation.py` — Test Inference Pipeline

Tests the LLM inference pipeline and prompt construction. Useful for iterating on prompts.

```bash
uv run python scripts/verify_generation.py [optional_arch_path]
```

### 12. `inspect_baseline.py` — Inspect torch.compile Internals

Inspects dynamo tracing, traced graph, fusion decisions, and generated code for a problem.

```bash
uv run python scripts/inspect_baseline.py
```

### 13. `inspect_triton.py` — Inspect Triton Code & Flamegraphs

[WIP] Generates torch.compile Triton code and Chrome trace flamegraphs for profiling.

```bash
uv run python scripts/inspect_triton.py
```

### 14. `inspect_kernel_pytorch_profiler.py` — Profile Generated Kernels

Profiles a generated kernel using PyTorch Profiler (CUDA activity only), showing operator breakdown.

```bash
uv run python scripts/inspect_kernel_pytorch_profiler.py
```

### 15. `debug_stddout.py` — Test Compilation Error Capture

Tests whether nvcc compilation errors are correctly captured. Used for development diagnostics.

```bash
uv run python scripts/debug_stddout.py
```

---

## Typical End-to-End Workflow

```bash
# Step 1: Generate kernels for Level 1
uv run python scripts/generate_samples.py \
    run_name=exp_v1 dataset_src=huggingface level=1 \
    num_workers=50 server_type=deepseek model_name=deepseek-chat temperature=0

# Step 2: Evaluate all generated kernels
uv run python scripts/eval_from_generations.py \
    run_name=exp_v1 dataset_src=local level=1 \
    num_gpu_devices=8 timeout=300

# Step 3: Analyze results
uv run python scripts/benchmark_eval_analysis.py \
    run_name=exp_v1 level=1 \
    hardware=L40S_matx3 baseline=baseline_time_torch
```
