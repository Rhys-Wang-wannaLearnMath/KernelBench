# `optimized_cuda_code/codes/` — Dataset Overview

> 中文版本: [README.zh.md](./README.zh.md)

Documentation for the *compact, code-only* view of the CUDA-L1
optimized-code release. For the full record (with all baselines and
all 5 score variants), see [`../README.md`](../README.md) and the
sibling `*.json` files in `optimized_cuda_code/`.

## 1. Directory layout

```
codes/
├── 3090.json     # NVIDIA RTX 3090
├── a100.json     # NVIDIA A100
├── h100.json     # NVIDIA H100 (XSM)
├── h20.json      # NVIDIA H20
└── l40.json      # NVIDIA L40
```

Each file is a **single JSON object** (loadable with one `json.load`)
keyed by KernelBench level. Compared with `../<gpu>.json`, the files
here **drop** the `cuda_graph_code`, `cudnn_code`, and the
`torch.compile` / `cuda_graph` / `cudnn` score variants — only the
default speedup and the optimized code itself are kept.

## 2. Top-level shape

```jsonc
{
  "1": [ <task record>, <task record>, ... ],   // 100 entries (Level 1)
  "2": [ <task record>, <task record>, ... ],   // 100 entries (Level 2)
  "3": [ <task record>, <task record>, ... ]    //  50 entries (Level 3)
}
```

Keys `"1"`, `"2"`, `"3"` are **strings** (JSON object keys). Each
value is a list of task records, ordered by `task_id`. Total task
count per file: **250**, matching KernelBench.

## 3. Schema of one task record

| Field         | Type            | Meaning |
|---------------|-----------------|---------|
| `task_id`     | `int`           | Task index within the level. |
| `ref_code`    | `str`           | Reference PyTorch implementation provided by KernelBench. |
| `custom_code` | `str` or `null` | CUDA-L1 generated, optimized implementation. `null` means RL failed to produce code faster than the reference. |
| `score`       | `str` or `null` | Speedup of `custom_code` over `ref_code` under default PyTorch eager. **Stored as a string** (e.g. `"1.762"`). Equivalent to `score_default` in `../<gpu>.json`. `null` when `custom_code` is `null`. |

The `score` is an execution-time ratio: `ref_time / custom_time`.
A value of `1.0` means parity; `2.0` means twice as fast.

## 4. Example entry (`codes/h100.json`)

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
    /* ...98 more Level-1 tasks... */
  ],
  "2": [ /* 100 Level-2 tasks */ ],
  "3": [ /*  50 Level-3 tasks */ ]
}
```

## 5. Reading the data

```python
import json

with open("h100.json", "r") as f:
    data = json.load(f)

# Iterate over every task in every level.
for level_id_str, tasks in data.items():
    for t in tasks:
        if t["custom_code"] is None:
            continue
        speedup = float(t["score"])      # remember: stored as a string
        print(level_id_str, t["task_id"], speedup)
```

## 6. When to use this folder vs. the parent

| Want…                                                       | Use                              |
|-------------------------------------------------------------|----------------------------------|
| Just the optimized code + a single default speedup          | `optimized_cuda_code/codes/*.json` |
| `torch.compile`, CUDA Graph, or cuDNN baseline comparisons  | `optimized_cuda_code/*.json` (JSONL) |
| Level-grouped, single-shot `json.load`                      | `optimized_cuda_code/codes/*.json` |
| Streaming, line-by-line consumption                         | `optimized_cuda_code/*.json` (JSONL) |
