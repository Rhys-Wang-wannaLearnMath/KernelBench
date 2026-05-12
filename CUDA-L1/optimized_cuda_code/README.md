# `optimized_cuda_code/` — Dataset Overview

> 中文版本: [README.zh.md](./README.zh.md)

Documentation for the CUDA-L1 optimized-code release under
`CUDA-L1/optimized_cuda_code/`. See the top-level
[`../README.md`](../README.md) for the full project description.

## 1. Directory layout

```
optimized_cuda_code/
├── 3090.json        # full record (JSONL), targeted at NVIDIA RTX 3090
├── a100.json        # full record (JSONL), targeted at NVIDIA A100
├── h100.json        # full record (JSONL), targeted at NVIDIA H100 (XSM)
├── h20.json         # full record (JSONL), targeted at NVIDIA H20
├── l40.json         # full record (JSONL), targeted at NVIDIA L40
└── codes/           # compact, code-only view of the same data (see codes/README.md)
```

Each top-level `*.json` file is a **JSON-Lines** file (one JSON object
per line), **not** a single JSON document. Every file contains exactly
**250 entries**, matching the
[KernelBench](https://github.com/ScalingIntelligence/KernelBench)
benchmark suite:

| Level | # tasks |
|:-----:|:-------:|
| 1     | 100     |
| 2     | 100     |
| 3     | 50      |

## 2. Schema of one JSONL record

| Field                                  | Type            | Meaning |
|----------------------------------------|-----------------|---------|
| `level_id`                             | `int` (1/2/3)   | KernelBench level index. |
| `task_id`                              | `int`           | Task index within the level. |
| `ref_code`                             | `str`           | Reference PyTorch implementation provided by KernelBench. |
| `custom_code`                          | `str` or `null` | CUDA-L1 generated, optimized implementation. `null` means RL failed to produce code faster than the reference. |
| `cuda_graph_code`                      | `str`           | Reference code wrapped with **CUDA Graph** (baseline). |
| `cudnn_code`                           | `str`           | Reference code with custom **torch cuDNN backend flags** enabled (baseline). |
| `score_default`                        | `float`         | Speedup of `custom_code` over `ref_code` under **default PyTorch eager** execution. |
| `score_torch_compile_default`          | `float`         | Speedup of `custom_code` over `ref_code` when the baseline uses **`torch.compile`** (default mode). |
| `score_torch_compile_reduce_overhead`  | `float`         | Same as above, with `torch.compile(mode="reduce-overhead")`. |
| `score_cuda_graph`                     | `float`         | Speedup of `custom_code` over `cuda_graph_code`. |
| `score_cudnn`                          | `float`         | Speedup of `custom_code` over `cudnn_code`. |

All scores are **execution-time ratios**: `baseline_time / custom_time`.
A value of `1.0` means parity; `2.0` means the optimized code is twice
as fast as the corresponding baseline. When `custom_code` is `null`,
every score is also `null`.

## 3. Example entry

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

## 4. Reading the data

```python
import json

records = []
with open("h100.json", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(len(records))                      # 250
print(records[0]["score_default"])       # e.g. 2.75
```

> ⚠️ Do **not** use `json.load(f)` on these top-level files — they are
> JSONL, not a single JSON object. Use `codes/*.json` if you want a
> single-shot `json.load`.

## 5. Reproducing the speedups

To reproduce CUDA-L1 results on a specific GPU, pick the matching file
(e.g. `h100.json` for H100 XSM) and benchmark `custom_code` against the
corresponding baseline on that exact device. See the parent
[`../README.md`](../README.md) for evaluation details and the `eval/`
folder for the harness.
