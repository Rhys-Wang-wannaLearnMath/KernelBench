# `optimized_cuda_code/` — 数据集说明

> English version: [README.md](./README.md)

本目录是 CUDA-L1 项目的优化代码发布数据集，按 GPU 设备分别给出。完整项目介绍请参见
[`../README.md`](../README.md)。

## 1. 目录结构

```
optimized_cuda_code/
├── 3090.json        # 完整记录（JSONL），对应 NVIDIA RTX 3090
├── a100.json        # 完整记录（JSONL），对应 NVIDIA A100
├── h100.json        # 完整记录（JSONL），对应 NVIDIA H100 (XSM)
├── h20.json         # 完整记录（JSONL），对应 NVIDIA H20
├── l40.json         # 完整记录（JSONL），对应 NVIDIA L40
└── codes/           # 同一数据的代码精简版本（详见 codes/README.zh.md）
```

每个顶层 `*.json` 文件均为 **JSON Lines** 格式（每行一个 JSON 对象），
**不是**一个完整的 JSON 文档。每个文件恰好包含 **250 条记录**，与
[KernelBench](https://github.com/ScalingIntelligence/KernelBench)
基准一致：

| 难度等级 | 任务数 |
|:-------:|:-----:|
| 1       | 100   |
| 2       | 100   |
| 3       | 50    |

## 2. 单条 JSONL 记录的字段说明

| 字段                                   | 类型             | 含义 |
|----------------------------------------|------------------|------|
| `level_id`                             | `int` (1/2/3)    | KernelBench 难度等级。 |
| `task_id`                              | `int`            | 该等级下的任务编号。 |
| `ref_code`                             | `str`            | KernelBench 提供的参考 PyTorch 实现。 |
| `custom_code`                          | `str` 或 `null`  | CUDA-L1 生成的优化实现。`null` 表示 RL 未能生成比参考更快的代码。 |
| `cuda_graph_code`                      | `str`            | 使用 **CUDA Graph** 包装的参考代码（基线之一）。 |
| `cudnn_code`                           | `str`            | 启用自定义 **torch cuDNN 后端 flag** 的参考代码（基线之一）。 |
| `score_default`                        | `float`          | 在**默认 PyTorch eager** 下，`custom_code` 相对 `ref_code` 的加速比。 |
| `score_torch_compile_default`          | `float`          | 当基线使用 **`torch.compile`**（默认模式）时的加速比。 |
| `score_torch_compile_reduce_overhead`  | `float`          | 同上，但基线为 `torch.compile(mode="reduce-overhead")`。 |
| `score_cuda_graph`                     | `float`          | `custom_code` 相对 `cuda_graph_code` 的加速比。 |
| `score_cudnn`                          | `float`          | `custom_code` 相对 `cudnn_code` 的加速比。 |

所有分数均为**执行时间比值**：`基线时间 / 优化代码时间`。`1.0` 表示持平，
`2.0` 表示优化代码相对该基线快两倍。当 `custom_code` 为 `null` 时，
对应的所有 score 也为 `null`。

## 3. 示例记录

```json
{
  "level_id": 1,
  "task_id": 1,
  "ref_code":         "import torch\nimport torch.nn as nn\n...",
  "custom_code":      "import torch\nimport torch.nn as nn\n...",
  "cuda_graph_code":  "import torch\nimport torch.nn as nn\n...",
  "cudnn_code":       "import torch\nimport torch.nn as nn\n...",
  "score_default":                       1.762,
  "score_torch_compile_default":         1.958,
  "score_torch_compile_reduce_overhead": 2.118,
  "score_cuda_graph":                    1.566,
  "score_cudnn":                         1.801
}
```

## 4. 数据读取方式

```python
import json

records = []
with open("h100.json", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(len(records))                      # 250
print(records[0]["score_default"])       # 例如 2.75
```

> ⚠️ **不要**对这些顶层文件使用 `json.load(f)`——它们是 JSONL，而不是单个 JSON 对象。
> 如果你想一次性 `json.load` 加载，请改用 `codes/*.json`。

## 5. 复现加速比

要在特定 GPU 上复现 CUDA-L1 的结果，请选择对应文件
（例如 H100 XSM 对应 `h100.json`），并在该设备上使用相同的基线测量
`custom_code` 的执行时间。评测细节请参阅上级目录
[`../README.md`](../README.md)；评测脚本位于 `eval/` 目录下。
