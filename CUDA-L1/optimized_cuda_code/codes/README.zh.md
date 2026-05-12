# `optimized_cuda_code/codes/` — 数据集说明

> English version: [README.md](./README.md)

本目录是 CUDA-L1 优化代码数据集的**精简（仅代码）版本**。如需包含全部基线和
5 种加速比的完整记录，请参见 [`../README.zh.md`](../README.zh.md) 以及
`optimized_cuda_code/` 下同名的 `*.json` 文件。

## 1. 目录结构

```
codes/
├── 3090.json     # NVIDIA RTX 3090
├── a100.json     # NVIDIA A100
├── h100.json     # NVIDIA H100 (XSM)
├── h20.json      # NVIDIA H20
└── l40.json      # NVIDIA L40
```

每个文件是一个**单一 JSON 对象**（可一次性 `json.load` 加载），其顶层
键为 KernelBench 的难度等级。相比 `../<gpu>.json`，本目录**移除了**
`cuda_graph_code`、`cudnn_code`，以及 `torch.compile` / `cuda_graph` /
`cudnn` 三类加速比——仅保留默认加速比与优化代码本身。

## 2. 顶层结构

```jsonc
{
  "1": [ <任务记录>, <任务记录>, ... ],   // 100 条（Level 1）
  "2": [ <任务记录>, <任务记录>, ... ],   // 100 条（Level 2）
  "3": [ <任务记录>, <任务记录>, ... ]    //  50 条（Level 3）
}
```

注意 `"1"`、`"2"`、`"3"` 是**字符串**形式的键（JSON 对象 key 均为字符串）。
每个 value 是按 `task_id` 排序的任务列表。每个文件总任务数 **250**，
与 KernelBench 一致。

## 3. 单条任务记录的字段说明

| 字段          | 类型              | 含义 |
|---------------|-------------------|------|
| `task_id`     | `int`             | 该等级下的任务编号。 |
| `ref_code`    | `str`             | KernelBench 提供的参考 PyTorch 实现。 |
| `custom_code` | `str` 或 `null`   | CUDA-L1 生成的优化实现；`null` 表示 RL 未能生成比参考更快的代码。 |
| `score`       | `str` 或 `null`   | 默认 PyTorch eager 下，`custom_code` 相对 `ref_code` 的加速比。**以字符串保存**（例如 `"1.762"`），等价于 `../<gpu>.json` 中的 `score_default`。当 `custom_code` 为 `null` 时该字段也为 `null`。 |

`score` 为执行时间比值 `ref_time / custom_time`：`1.0` 表示持平，`2.0`
表示快两倍。

## 4. 示例记录（`codes/h100.json`）

```jsonc
{
  "1": [
    {
      "task_id": 1,
      "ref_code":    "import torch\nimport torch.nn as nn\n...",
      "custom_code": "import torch\nimport torch.nn as nn\n...",
      "score": "2.750"
    },
    {
      "task_id": 2,
      "ref_code":    "import torch\nimport torch.nn as nn\n...",
      "custom_code": null,
      "score": null
    }
    /* ……其余 98 个 Level 1 任务…… */
  ],
  "2": [ /* 100 个 Level 2 任务 */ ],
  "3": [ /*  50 个 Level 3 任务 */ ]
}
```

## 5. 数据读取方式

```python
import json

with open("h100.json", "r") as f:
    data = json.load(f)

# 遍历每个等级下的每个任务
for level_id_str, tasks in data.items():
    for t in tasks:
        if t["custom_code"] is None:
            continue
        speedup = float(t["score"])      # 注意：score 以字符串形式保存
        print(level_id_str, t["task_id"], speedup)
```

## 6. 与上级目录如何选择

| 你的需求                                                   | 应使用                              |
|------------------------------------------------------------|-------------------------------------|
| 只要优化代码和一个默认加速比                               | `optimized_cuda_code/codes/*.json`  |
| 需要 `torch.compile` / CUDA Graph / cuDNN 等基线对比       | `optimized_cuda_code/*.json`（JSONL） |
| 想按等级分组，一次性 `json.load`                           | `optimized_cuda_code/codes/*.json`  |
| 想流式逐行处理                                             | `optimized_cuda_code/*.json`（JSONL） |
