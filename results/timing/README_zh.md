# ⏱️ 基线计时结果


本文件夹包含一组 KernelBench 问题的基线计时结果。

由于 KernelBench 测量的是参考架构运行时间（Runtime(reference architecture)）与 LLM 生成架构运行时间（Runtime(LLM-generated architecture)）之间的加速比，因此准确测量基线参考模块的运行时间非常重要。

## 这些基线 JSON 何时使用？

本文件夹中的基线计时 JSON 主要用于**离线分析/评分**，在这种场景下你需要一个一致的参考运行时间，而无需在每次运行时重新计时 PyTorch 参考实现。

相比之下，部分工作流使用**实时计时**来计算加速比：

- **`scripts/run_and_check.py`**
  - 通过调用 `kernelbench.timing.measure_ref_program_time(...)` 测量 PyTorch 参考运行时间（eager 模式以及可选的 `torch.compile`）。
  - 通过 `kernelbench.eval.eval_kernel_against_ref(...)` 测量候选内核的运行时间。
  - 输出加速比 = reference_time / kernel_time。

- **`kernelbench.eval.eval_kernel_against_ref(...)`**（库 API）
  - 直接测量候选内核的运行时间。
  - 也可能测量参考模型的运行时间以进行额外检查（例如异常加速比检测）。
  - **不会**读取本文件夹中的基线 JSON。

我们在多种硬件和多种 PyTorch 配置下提供了一组 KernelBench 问题的基线结果。
所有（当前）基线均在 PyTorch `2.5.0+cu124` 和 CUDA `12.4` 下运行。

注意：我们将很快更新为 PyTorch `2.9.0` 和 CUDA `12.8`

对于计时，我们测量的是墙钟时间（wall clock time）。预热 3 次，并收集 100 次试验的运行时间统计数据。

### 在你自己的集群上运行
由于你的集群可能与我们的不同（不同的 GPU、不同的功率设置等），你可以在自己的集群上创建基线结果。
请参阅 `uv run python scripts/generate_baseline_time.py` 了解如何设置和运行计时。

### 在 Modal 上运行
要在 Modal 上收集基线，请参阅 `uv run python scripts/generate_baseline_time_modal.py` 了解我们的做法。

### 硬件

我们在多种硬件上对 KernelBench 问题的基线时间进行了分析，涵盖不同代际的 GPU。

| 提供商 | GPU 型号 | 显存 | 功率 | GPU 架构 |
|----------|----------|---------|--------|--------------|
| Stanford matx3 | NVIDIA L40S | 48 GB | 300W | Ada |
| Together.ai | NVIDIA H100 | 80 GB | 700W | Hopper |
| Modal | NVIDIA L40S | 48 GB | 350W | Ada |
| Modal | NVIDIA H100 | 85 GB | 700W | Hopper |
| Modal | NVIDIA A100 | 42 GB | 400W | Ampere |
| Modal | NVIDIA L4 | 24 GB | 72W | Ada |
| Modal | NVIDIA T4 | 16 GB | 70W | Turing |
| Modal | NVIDIA A10G | 24 GB | 300W | Ampere |
| Lambda Labs | NVIDIA H100 PCIe | 80 GB | 350W | Hopper |

查看 `timing` 目录获取上述各硬件的计时结果。


### 计时配置

我们关注 Torch eager 执行以及使用各种后端和模式的 Torch Compile。

| 配置 | 后端 | 模式 | 描述 |
|--------------|---------|------|-------------|
| Torch (Eager) | - | - | 标准 PyTorch eager 执行 |
| Torch Compile | inductor | default | 默认的 torch.compile 行为 |
| Torch Compile | inductor | reduce-overhead | 针对减少开销进行优化 |
| Torch Compile | inductor | max-autotune | 启用最大自动调优 |
| Torch Compile | inductor | max-autotune-no-cudagraphs | 不使用 CUDA graphs 的最大自动调优 |
| Torch Compile | cudagraphs | - | 使用 AOT Autograd 的 CUDA graphs |


了解更多关于 Torch Compile 的[后端](https://pytorch.org/docs/stable/torch.compiler.html)和[模式](https://pytorch.org/docs/main/generated/torch.compile.html)。


### 致谢

感谢 PyTorch 团队的 [@PaliC](https://github.com/PaliC) 在各种 Torch 配置方面提供的专业知识。

感谢 [Modal](https://modal.com/) 赞助计算资源，使我们能够在多种 NVIDIA GPU 上收集运行时基线。
