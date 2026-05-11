# KernelBench: LLM 能写出高效的 GPU 内核吗？ [ICML '25]
一个用于评估 LLM 生成高效 GPU 内核能力的基准测试与环境

具体来说，我们让 LLM 在目标 GPU 上为 PyTorch 程序生成正确且高效的 CUDA / DSL 内核。

[arXiv](https://arxiv.org/html/2502.10517v1) | [博客文章](https://scalingintelligence.stanford.edu/blogs/kernelbench/) | [HuggingFace 数据集](https://huggingface.co/datasets/ScalingIntelligence/KernelBench)

<img src="./assets/figures/KernelBenchMascot.png" width="200">

## 版本
最新稳定版本位于 `main` 分支，我们持续更新和改进本仓库。
- [v0.1](https://github.com/ScalingIntelligence/KernelBench/tree/v0.1) - 参见[博客](https://scalingintelligence.stanford.edu/blogs/kernelbenchv01/)
- [v0](https://github.com/ScalingIntelligence/KernelBench/tree/v0) - 初始版本


HuggingFace [数据集](https://huggingface.co/datasets/ScalingIntelligence/KernelBench)已更新至 v0.1。

本仓库提供 KernelBench 的核心功能和一组易用的评估脚本。它并不旨在提供解决此任务的复杂智能体脚手架；我们建议克隆并修改本仓库用于你的实验，或将其作为 git submodule 使用。

## 👋 任务描述
我们将问题构建为让 LLM 将 PyTorch 描述的算子转译为 CUDA 内核，粒度由模型自行决定。
![KernelBenchMascot](./assets/figures/KernelBenchWorkFlow.png)

KernelBench 包含 4 个难度级别：
- **Level 1 🧱**：单内核算子（100 题）
    神经网络的基础构建模块（卷积、矩阵乘法、层归一化）
- **Level 2 🔗**：简单融合模式（100 题）
    融合内核比分离内核更快（Conv + Bias + ReLU、Matmul + Scale + Sigmoid）
- **Level 3 ⚛️**：完整模型架构（50 题）
    端到端优化整个模型架构（MobileNet、VGG、MiniGPT、Mamba）
- **Level 4 🤗**：HuggingFace 级别
    优化来自 HuggingFace 的完整模型架构

我们正在积极将 KernelBench 扩展到 `cuda` 之外的其他 DSL（见下文），同时也在增加 AMD GPU 支持。

## ⚖️ 评估
#### 方法论
评估模型生成的内核时，需要检查：
- **正确性 ✅**：使用随机输入对比参考 torch 算子，重复 `n_correctness` 次。
- **性能 ⏱️**：与参考 torch 算子对比，重复 `n_trial` 次以测量运行时间的加速比。

详见 `src/eval.py` 了解正确性检查和计时的实现细节，以及 `EVAL.md` 了解评估与基准测试指南 [WIP]。

我们提供了便捷脚本 `scripts/run_and_check.py`，用于评估单个生成代码与参考代码的对比，检查正确性并计算加速比。你可以通过设置 `eval_mode=local` 或 `eval_mode=modal` 在本地或远程评估内核。

#### 总体基准指标

为同时衡量**正确性**和**性能**，我们定义了指标 `fast_p`：既正确又具有大于阈值 `p` 的加速比的任务比例；加速比为 PyTorch 参考实际运行时间与生成内核运行时间之比。

一些示例来说明这个基于加速比过滤的指标：
* `fast_1` 表示 LLM 生成的内核既正确又**快于** PyTorch 基线的任务比例
* `fast_2` 表示 LLM 生成的内核既正确又**至少快 2 倍**于 PyTorch 基线的任务比例
* `fast_0` 表示 LLM 生成的内核**正确**的任务比例（等同于正确率）

你可以提高加速比阈值 `p` 来增加任务难度。


#### 计算总体基准性能

我们提供了脚本 `scripts/greedy_analysis.py` 来计算总体基准性能。
由于需要同时衡量**正确性**和**性能**，我们使用指标 `fast_p`：既正确又具有大于阈值 `p` 的加速比的任务比例；加速比为 PyTorch 参考实际运行时间与生成内核运行时间之比。

<!-- TODO: update to provide fast_p measurement script -->

## 🔍 目录结构
仓库组织如下：
```
KernelBench/
├── assets/
├── KernelBench/ # 基准测试数据集文件
├── src/kernelbench/ # KernelBench 逻辑代码
│   ├── unit_tests/  
│   ├── prompts/
│   ├── ....
├── scripts/ # 运行基准测试的辅助脚本
├── results/ # 各硬件的基线时间
├── runs/ # 你的运行结果存储位置
├── notebooks/ # 分析用的示例 notebook
├── pyproject.toml # 项目配置与依赖
```

## 🔧 环境配置

我们已迁移至使用 `pyproject.toml` 和 `uv` 进行依赖管理。如尚未安装，请先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
# 安装基础依赖（无需本地 GPU）
uv sync

# 安装 AMD ROCm 后端（需要 ROCm>=7.1）
uv add torch --index pytorch=https://download.pytorch.org/whl/rocm7.1

# 安装含 GPU 依赖（用于本地 GPU 评估）
uv sync --extra gpu

# 使用 uv 运行命令（自动使用正确的环境）
uv run python scripts/<script_name>.py ...
```

对于 AMD GPU 即 ROCm 后端（ROCm>=7.1），请执行 `uv remove torch && uv add torch --index pytorch=https://download.pytorch.org/whl/rocm7.1` 以配置 ROCm 兼容的 PyTorch 依赖。由于 ROCm 配置的复杂性，建议在 Docker 镜像中运行。

你仍然可以使用 `conda (python=3.10)` 创建环境，并通过 `requirements.txt` 安装依赖。

我们使用 `litellm` 进行 API 调用，请按照 `.env.example` 创建 `.env` 文件并设置你的密钥。

运行和分析内核需要 GPU。
如果本地没有 GPU，可以配置 [Modal](https://modal.com/) 进行云端无服务器 GPU 评估。创建账户后运行 `modal token new` 设置 modal token，然后使用 `generate_and_eval_single_sample_modal.py` 脚本。

你也可以通过我们的[教程 notebook](https://bit.ly/kernelbench-neurips-colab)（同时位于 notebooks/tutorial.ipynb）在 Google Colab 上试用。

## 🚀 使用方法
### 运行单个问题
从单个问题开始最为简便。这将获取问题、生成样本并评估样本。

```bash
# 例如：从 HuggingFace 获取 Level 2 第 40 题，使用 Google Gemini 2.5 Flash 生成

uv run python scripts/generate_and_eval_single_sample.py dataset_src=huggingface level=2 problem_id=40 server_type=google model_name=gemini/gemini-2.5-flash

# dataset_src 可选 "local" 或 "huggingface"
# 添加 .verbose_logging 以获取更多日志信息
```

**你可能需要修改的参数**
* **`gpu_arch`** - 根据你的 GPU，可能需要调整 `gpu_arch` 参数以匹配你的硬件。
* **`precision`** - 你可以通过 `precision=fp32` 指定张量精度。目前所有报告结果均为 `fp32`，但我们已添加 `fp16` 和 `bf16` 支持。
* **`backend`** - 我们也支持 `cuda` 之外的其他 GPU 编程语言。例如，直接指定 `backend=triton` 或 `backend=hip`。目前我们支持以下 NVIDIA GPU 编程框架和 DSL：`cuda`、`triton`、`cute`、`tilelang`、`thunderkittens`。

AMD GPU 注意事项：使用 `hip` 后端，当前支持的 `gpu_arch`：`gfx942`、`gfx950`。

ThunderKittens (TK) 本地配置说明：使用 `backend=thunderkittens` 时，你需要 git clone ThunderKittens 仓库并设置环境变量指向本地 ThunderKittens 目录：`export THUNDERKITTENS_ROOT=<ThunderKittens 文件夹路径>`。所有 ThunderKittens 程序如[示例](src/kernelbench/prompts/model_new_ex_add_thunderkittens.py)所示，应包含 `tk_root = os.environ.get("THUNDERKITTENS_ROOT", "/root/ThunderKittens")`，以便内核包含正确的 TK 原语。此外，TK 目前仅支持 BF16。

查看配置字段以获取完整的选项列表。注意，默认情况下我们会为模型提供一个 one-shot 示例以及最少的信息集；你可以查看其他提示词设置或在 `src/prompt_constructor_toml.py` 中构建自己的提示词。

### 运行所有问题

```bash
# 1. 生成回复并将内核存储到 runs/{run_name} 目录
uv run python scripts/generate_samples.py run_name=test_hf_level_1 dataset_src=huggingface level=1 num_workers=50 server_type=deepseek model_name=deepseek-chat temperature=0

# 2. 评估 runs/{run_name} 目录中所有已生成的内核
uv run python scripts/eval_from_generations.py run_name=test_hf_level_1 dataset_src=local level=1 num_gpu_devices=8 timeout=300

# 如果想加速评估，可以在 GPU 评估之前先在 CPU 上并行编译
# 在命令中添加 build_cache=True 和 num_cpu_workers=<cpu_worker数量>
```
### 分析评估结果以计算基准性能
我们提供了 `scripts/benchmark_eval_analysis.py` 来分析评估结果，计算成功率、计时指标和总体基准性能 `fast_p`。

```bash
uv run python scripts/benchmark_eval_analysis.py run_name=test_hf_level_1 level=1 hardware=L40S_matx3 baseline=baseline_time_torch
```
如果你使用不同的硬件，可以用 `scripts/generate_baseline_time.py` 脚本生成基线时间。
我们在 `results/timing` 中提供了跨代 NVIDIA GPU 的参考基线时间，但我们建议你生成自己的基线时间以获得更准确的结果（集群功率、软件版本等都会影响计时结果）。详见 `results/timing/README.md`。

## 🛣️ 路线图
查看我们的[路线图](https://github.com/ScalingIntelligence/KernelBench/issues/74)了解计划添加的功能。我们欢迎社区在这些方向上的贡献和讨论。

## 🔌 集成
你也可以将 KernelBench 作为库在你的项目中使用，例如：`from kernelbench import timing`、`from kernelbench import eval as kb_eval`、或 `from kernelbench.utils import set_gpu_arch`。

- **与 Harbor 集成** — 我们正在与 [Harbor](https://harborframework.com/docs) 集成，以支持更高吞吐量的评估和更丰富的智能体性能评估（超越模型 pass@1/k）。*（[进行中](https://github.com/harbor-framework/harbor/pull/999)）*

- **多轮 / 测试时扩展** — [Caesar](https://github.com/ScalingIntelligence/caesar) 是我们面向吞吐量的多轮推理引擎（ICML '25），用于论文中的迭代优化实验。它批量运行生成轨迹，跨轮次反馈正确性、运行时间和分析信号，实现顺序测试时扩展。

- **强化学习 (RLVR)** — [kernelbench-tinker](https://github.com/ScalingIntelligence/kernelbench-tinker) 是与 Thinking Machines Lab 的 [Tinker RL 库](https://github.com/thinking-machines-lab/tinker)的端到端集成。该流水线让策略模型生成内核，通过 Modal 在云 GPU 上评估，并将结果转换为 RL 奖励——一个用于在 GPU 内核优化上实验 RLVR 的最小化游乐场。

- **进化搜索** — 类似 AlphaEvolve 的进化搜索在优化问题中展现了发现创新解决方案的潜力。我们正在进行与 [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) 的集成，即将发布。

- **Roofline / 最大加速比分析** — *（实验性，WIP）* [simple-torchroofline](https://github.com/simonguozirui/simple-torchroofline) 为 PyTorch 程序提供分析性 Roofline 分析，估算目标 GPU 的光速（SoL）计算和内存上限——无需实际硬件。结合基于硬件计数器的经验性 Roofline 分析，这有助于验证报告的加速比是否在物理上合理。


## 🔍 已知用例
自发布以来，我们收到了来自研究人员、研究实验室和公司的广泛关注，他们使用 KernelBench 探索这一方向。我们记录了 KernelBench 的[已知用例](https://docs.google.com/document/d/e/2PACX-1vTjS-UMH1HB5n_PENq2k-3YRfXIXkqKIKeNC2zcWMyLPdl4Jrwvdk4dNDVSsM8ybKrCxZB7GJq1slZF/pub)以及相关的自动内核生成工作。如果你正在使用 KernelBench，我们很乐意了解更多！

免责声明：KernelBench 是一个**开源**评估框架和工具包。KernelBench 团队不审查、验证或认可单个内核或报告的结果。用户有责任独立验证使用该框架获得的任何结果。请查看 `EVAL.md` 获取更多关于基准测试和评估内核的指导。


## 🪪 许可证
MIT。详见 `LICENSE.md`。


## 引用
```bibtex
@misc{ouyang2025kernelbenchllmswriteefficient,
      title={KernelBench: Can LLMs Write Efficient GPU Kernels?}, 
      author={Anne Ouyang and Simon Guo and Simran Arora and Alex L. Zhang and William Hu and Christopher Ré and Azalia Mirhoseini},
      year={2025},
      eprint={2502.10517},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.10517}, 
}
```

感谢 [GPU Mode](https://gpu-mode.github.io/popcorn/)、[PyTorch](https://pytorch.org/)、[Modal Labs](https://modal.com/blog/accelerating-ai-research-case-study) 和广大开源社区的支持，使本项目成为可能。
