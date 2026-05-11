# Kernel 优化 Review 的 Preliminary Dataset

本目录把两个 SOTA LLM kernel-optimization 工作公开发布的"优化后 kernel"整理成统一格式的 **(参考 kernel, 优化后 kernel)** 对，作为 *semantic-review agent* 的 **preliminary test** 数据集：

- **CUDA-L1** — `CUDA-L1/optimized_cuda_code/{a100,3090,h100,h20,l40}.json`
- **KernelAgent** — `kernelagent-optimization-artifacts/<task>/`

每个 pair 都是一个独立目录，提供干净的 `ref.py` ↔ `optimized.py` 接口，方便做 differential testing、人工审查或下游工具消费。

> 英文版本：`README.md`。

---

## TL;DR

```bash
# 1. 从 upstream 仓库重建数据集
python scripts/build_dataset.py

# 2. 查看数据集内容
python scripts/list_pairs.py --stats
python scripts/list_pairs.py --source cuda_l1 --gpu a100 --level 1 --limit 10

# 3. 对单个 pair 跑差分正确性测试
python scripts/run_diff_test.py kernelagent/L1_T036_RMSNorm

# 4. 批量跑（带过滤），结果写到 reports/<run_name>/
python scripts/batch_diff_test.py --source kernelagent
python scripts/batch_diff_test.py --source cuda_l1 --gpu a100 --level 1 --limit 20
```

---

## 1. 数据集组成

`build_dataset.py` 跑完后：

| 来源 | GPU | 有优化 kernel 的 task 数 | 接口 |
|---|---|---|---|
| CUDA-L1 | a100 | 225 / 250 | `ModelNew`（`nn.Module` 直接替换） |
| CUDA-L1 | 3090 | 197 / 250 | `ModelNew` |
| CUDA-L1 | h100 | 210 / 250 | `ModelNew` |
| CUDA-L1 | h20  | 219 / 250 | `ModelNew` |
| CUDA-L1 | l40  | 213 / 250 | `ModelNew` |
| KernelAgent | h100 | 5 / 5 | `kernel_function`（自由函数） |
| **合计** | — | **1069 pairs** | — |

只有 upstream 真的产出了非空优化 kernel 的 task 才会被物化成 pair（否则跳过）。

### 两种接口、一套测试

两个 upstream 项目用了不同的约定，我们的 diff-test 框架统一处理：

| 约定 | 来源 | 形态 |
|---|---|---|
| `class ModelNew(nn.Module)` | CUDA-L1 | 与参考 `class Model` 接口完全一致；按 `ref_model(*inputs)` vs `opt_model(*inputs)` 比较。 |
| `def kernel_function(...)` | KernelAgent | 自由函数；我们自动把参考模型的参数（`weight`、`bias`、`kernel_size`、`stride`、`padding`、`dilation`、`output_padding`、`groups`、`eps`）作为 kwargs 绑定后再调用 `kernel_function(*inputs, **bound_params)`。 |

每个 pair 的接口类型记录在 `meta.json` 的 `optimized_interface` 字段。

---

## 2. 目录结构

```
preliminary_dataset/
├── README.md / README_zh.md
├── index.json                          ← 全部 pair 的机读索引
├── cuda_l1/
│   └── {a100,3090,h100,h20,l40}/
│       └── level{1,2,3}/
│           └── L{lvl}_T{tid:03d}_{name}/
│               ├── ref.py              ← CUDA-L1 自带的 ref_code（≠ 当前 KernelBench，见 §5）
│               ├── optimized.py        ← CUDA-L1 的 custom_code（ModelNew）
│               ├── baseline_cuda_graph.py   （可选）
│               ├── baseline_cudnn.py        （可选）
│               └── meta.json
├── kernelagent/
│   └── L1_T{tid:03d}_{name}/
│       ├── ref.py                      ← upstream 的 problem.py（= 当前 KernelBench）
│       ├── optimized.py                ← upstream 的 optimized_kernel_beam_search.py
│       ├── input_kernel.py             ← beam-search 的起点 kernel
│       └── meta.json
├── scripts/
│   ├── build_dataset.py                ← 从 upstream 重建数据集
│   ├── _runtime.py                     ← 共享的加载器 / 接口适配 / 比较器
│   ├── run_diff_test.py                ← 单个 pair 测试
│   ├── batch_diff_test.py              ← 子进程隔离的批量测试
│   └── list_pairs.py                   ← 索引查询 / 过滤
└── reports/                            ← batch_diff_test.py 的输出目录
```

### `meta.json` 字段（每个 pair 一份）

```jsonc
{
  "source": "cuda_l1",                  // 或 "kernelagent"
  "gpu": "a100",                        // upstream 声称的目标设备
  "level_id": 1,
  "task_id": 1,
  "task_name": "Square_matrix_multiplication",
  "kernelbench_file": "level1/1_Square_matrix_multiplication_.py",
  "optimized_interface": "ModelNew",    // 或 "kernel_function"
  "has_optimized": true,
  "has_baseline_cuda_graph": true,      // 仅 CUDA-L1
  "has_baseline_cudnn": true,           // 仅 CUDA-L1
  "scores": {                           // CUDA-L1 自报的加速比
    "score_default": 1.76,
    "score_torch_compile_default": 1.96,
    "score_torch_compile_reduce_overhead": 2.12,
    "score_cuda_graph": 1.57,
    "score_cudnn": 1.80
  },
  "ref_matches_kernelbench_current": false,
  "source_path": "CUDA-L1/optimized_cuda_code/a100.json"
}
```

### `index.json` 字段

```jsonc
{
  "num_pairs": 1069,
  "by_source": {"cuda_l1": 1064, "kernelagent": 5},
  "pairs": [
    {
      "pair_id": "cuda_l1__a100__L1T001",
      "source": "cuda_l1",
      "gpu": "a100",
      "level_id": 1,
      "task_id": 1,
      "task_name": "Square_matrix_multiplication",
      "dir": "cuda_l1/a100/level1/L1_T001_Square_matrix_multiplication",
      "optimized_interface": "ModelNew",
      "ref_matches_kernelbench_current": false,
      "scores": { ... }
    },
    ...
  ]
}
```

---

## 3. 辅助文件与 score 字段

`(ref.py, optimized.py)` 是你的主 review 目标，但每个 pair 目录里还有几个辅助文件，它们是额外的 **baseline** 或流水线中间产物。搞清楚它们的角色才能看懂 `meta.json` 里的 `scores`。

### 3.1 各文件的角色

| 文件 | 来源 | 语义 | 性能层级 | 角色 |
|------|------|------|---------|------|
| `ref.py` | 两者都有 | = 原 Task 定义 | 原生 PyTorch baseline | **ground-truth oracle** |
| `baseline_cuda_graph.py` | 仅 CUDA-L1 | 与 `ref.py` 等价 | baseline+（CUDA Graph 消除 launch overhead）| **更强 baseline**（非 LLM、规则化改写）|
| `baseline_cudnn.py` | 仅 CUDA-L1 | 与 `ref.py` 等价 | baseline+（`cudnn.benchmark=True` auto-tuner）| **更强 baseline**（非 LLM、规则化改写）|
| `input_kernel.py` | 仅 KernelAgent | 应与 `ref.py` 等价 | 一般（有时还比 `torch.compile` 慢）| beam-search **起点**（早期 KernelAgent 未经 beam-search 优化循环的输出）|
| `optimized.py` | 两者都有 | **应**与 `ref.py` 等价 ⚠️ | 快（声称的最终优化）| **LLM 流水线的最终产物**——你要 review 的对象 |

### 3.2 CUDA-L1 的 5 个 score 字段

五个 score 共用同一个公式：

```
score_X = runtime(baseline_X) / runtime(custom_code)
          ↑ 越高 = 声称的加速比越大
```

**分子的 baseline 在变；分母（`custom_code`，即 `optimized.py`）始终不变**：

| JSON 字段 | 论文/表格简写 | 分子 baseline | 测的是什么 |
|-----------|-----------|---------------|-----------|
| `score_default` | `score_default` | 原始 `ref.py` | 对比原生 PyTorch |
| `score_torch_compile_default` | `score_tc` | `torch.compile(Model)` 包裹的 `ref.py` | 对比 `torch.compile` 默认 mode |
| `score_torch_compile_reduce_overhead` | `score_tc_ro` | `torch.compile(Model, mode="reduce-overhead")` | 对比 `torch.compile` 的 reduce-overhead mode（它内部会用 CUDA Graph）|
| `score_cuda_graph` | `score_cg` | `baseline_cuda_graph.py` | 对比手写的 CUDA Graph 版 ref |
| `score_cudnn` | `score_cudnn` | `baseline_cudnn.py` | 对比 `cudnn.benchmark=True` 版 ref |

**实测例子 — L1/T1 (Square matmul) on A100:**

| Score | 值 | 解读 |
|-------|-----|------|
| `score_default` | **6.304×** | 对比原始 ref（最弱 baseline） |
| `score_tc` | 6.326× | `torch.compile` 对单一 matmul 几乎没帮助 |
| `score_tc_ro` | 6.245× | reduce-overhead 在这里也差不多 |
| `score_cg` | **5.093×** | ← CUDA Graph 是强得多的 baseline（分母”“变大“，比率下降） |
| `score_cudnn` | 5.190× | cuDNN auto-tuner 也更强 |

> **⚠️ 关于 CUDA-L1 宣传数字最需要警惕的一点。** 论文的 “≥3× 平均加速” 用的是 `score_default`（对比原生 PyTorch 裸跑）。切换到更硬的 `score_cuda_graph` / `score_tc_ro` baseline，很多 task 的加速比会大幅缩水。**哪个 baseline 才是“公平”的，完全取决于你下游实际部署时会用哪个 baseline。** 报告时不要混用。

### 3.3 对语义等价性 review 的启示

1. **LLM 的”真实“贡献** ≈ `runtime(optimized.py) / runtime(baseline_cuda_graph.py)`（而不是对比原始 ref）。如果 LLM 只是打赢裸跑的 ref，可能只是 “用户忘了套 CUDA Graph” 这个级别的生产力提升，不是真正的算法性贡献。
2. **KernelAgent beam-search 的增量贡献** ≈ `runtime(optimized.py) / runtime(input_kernel.py)`（仅 KernelAgent 有此数据）。
3. **跨 baseline 的语义等价性交叉验证。** 5 个版本（`ref`、`baseline_cuda_graph`、`baseline_cudnn`、`input_kernel`、`optimized`）的输出应该相互 allclose。如果 `optimized.py` 和前 4 个中任何一个对不上（尤其是语义上最“干净”的 `baseline_cuda_graph.py`），就是 LLM 引入偏离的强信号。
4. **Reward-hacking 的 “金线”。** 前 4 个文件都是非 LLM 产物。如果 review agent 把它们标注为语义错误，大概是在误报；只对 `optimized.py` 有时报错的 agent，打的才是真 LLM 问题。

---

## 4. 脚本

### `build_dataset.py`

幂等的提取器。读 upstream 仓库 → 写出标准目录结构 → 输出 `index.json`。

```bash
python scripts/build_dataset.py                 # 重建（保留已有文件）
python scripts/build_dataset.py --force         # 先清空 cuda_l1/ + kernelagent/ 再重建
python scripts/build_dataset.py --no-cuda-l1
python scripts/build_dataset.py --no-kernelagent
python scripts/build_dataset.py --dataset-root /some/other/path
```

源路径硬编码为：
- `<repo>/CUDA-L1/optimized_cuda_code/{gpu}.json`
- `<repo>/kernelagent-optimization-artifacts/<task_folder>/`

（`<repo>` = `preliminary_dataset/` 的父目录）

### `list_pairs.py`

无需 GPU 的索引查询。

```bash
python scripts/list_pairs.py --stats
python scripts/list_pairs.py --source cuda_l1 --gpu a100
python scripts/list_pairs.py --level 1 --task 1,2,3 --full
python scripts/list_pairs.py --pattern Matmul --paths-only
```

### `run_diff_test.py`

跑单个 pair 的差分测试。需要 CUDA + 对应 upstream 的运行时（KernelAgent 需要 PyTorch + Triton；CUDA-L1 的 `load_inline` 内核需要 PyTorch 的 cpp_extension）。

```bash
python scripts/run_diff_test.py <pair_dir>
python scripts/run_diff_test.py kernelagent/L1_T036_RMSNorm
python scripts/run_diff_test.py cuda_l1/a100/level1/L1_T001_Square_matrix_multiplication \
    --rtol 1e-3 --atol 1e-3 --dtype bf16 --json /tmp/result.json
```

退出码：`0` 通过，`1` 语义不符，`2` 运行时/导入错误。

### `batch_diff_test.py`

带过滤的批量执行器。每个 pair 在独立子进程中跑（一个崩溃不会污染其它）。

```bash
python scripts/batch_diff_test.py --source kernelagent
python scripts/batch_diff_test.py --source cuda_l1 --gpu a100 --level 1 --limit 50
python scripts/batch_diff_test.py --run-name baseline_a100_L1 --dtype bf16
```

输出（写到 `reports/<run_name>/`）：
- `results.jsonl` — 每个 pair 一行的完整 `DiffResult` JSON
- `summary.json`  — 聚合计数与分桶
- `failures.md`   — 人类可读的失败/错误列表

---

## 5. 重要警示：到底用哪个“参考 kernel”？

KernelBench 自己也在演进，两个 upstream **不是基于同一个 KernelBench snapshot** 跑的优化。**始终明确你用的 oracle 是什么**。

| pair 来源 | `ref.py` 实际是什么 | 与当前 `KernelBench/level{1,2,3}/*.py` 是否一致？ |
|---|---|---|
| **KernelAgent** | upstream 的 `problem.py` 原样拷贝 | ✅ 5 个全部 byte-identical |
| **CUDA-L1**     | upstream JSON 内嵌的 `ref_code` 字段 | ❌ 全部 1064 个都和当前主干不同 |

CUDA-L1 的差异分三层（以 `a100.json` 250 条为例）：

| 差异类型 | 数量 | 例子 |
|---|---|---|
| 仅输入分布：`torch.rand` → `torch.randn` | 191 | L1/T1 |
| 分布 + 形状常数（如 `N = 4096` → `N = 2048`） | 20 | L1/T1 |
| `class Model.forward` 函数体本身改了（真·语义变更） | **25** | L1/T12: `A.unsqueeze(1) * B`（当前）vs `torch.diag(A) @ B`（CUDA-L1） |

`build_dataset.py` 会把 `ref_matches_kernelbench_current: false` 写进每个 CUDA-L1 pair 的 `meta.json`，方便下游过滤。

> **怎么处理？** 评估 CUDA-L1 的 `optimized.py` 时，**用 pair 目录里自带的 `ref.py`**（即它自己的 oracle）。要做跨数据集对比 / 与当前 KernelBench 对齐时，把 CUDA-L1 看作"针对旧 snapshot 优化"，并选择：
>
> 1. 直接用旧 oracle 单独报分（最干净）；或
> 2. 在做统一对比前，把那 25 个语义不同的 task patch 掉（或直接剔除）。
>
> 25 个真·语义不同的 task 可以用 §7.4 的脚本一键列出来。

---

## 6. Diff-test 流程（`_runtime.py` 在做什么）

每个 pair：

1. **导入** `ref.py` 与 `optimized.py` 为相互隔离的模块（独立 `sys.modules` 入口）。
2. **构造输入**：调 `ref.get_init_inputs()` 与 `ref.get_inputs()`；按 `--device` 搬到 GPU；按 `--dtype` 可选地转换浮点类型。
3. **构造模型**：
   - `ModelNew` 接口：分别实例化 `Model(*init)` 和 `ModelNew(*init)`，把参考模型的 `state_dict` 复制到优化模型（`strict=False`），保证参数完全一致。
   - `kernel_function` 接口：实例化 `Model(*init)` 后绑定其参数到一个闭包，等价于 `kernel_function(*inputs, **bound_params)`。
4. **前向**：`torch.inference_mode()` 下两边都跑，CUDA 全 sync。
5. **比较**：把输出搬到 CPU + fp32，用 `torch.allclose(rtol, atol)`；记录 `max_abs_diff`、`max_rel_diff`，以及形状/dtype 不一致的 note。

默认容差 `rtol=atol=1e-3`（与 KernelAgent upstream 一致）。indexing/copy 这种应该 bit-wise 等价的算子，用更严的 `1e-5`。

> ⚠️ **Diff-test ≠ 语义等价。** 随机输入下的 allclose 能抓到大错误，但通常会漏掉：
> - shape-specialized 的硬编码（kernel 只对 upstream 那一组特定 shape 正确）；
> - 边界 bug（`BLOCK ± 1`、`K = 1`、`batch = 1`、prime 大小）；
> - 数值边界 bug（NaN 传播、±0、denormal、fp16 overflow）；
> - 越界读写**碰巧**落在无害内存上；
> - 通过 timing 测量作弊的 reward hacking（warmup-based dispatch trick），correctness check 完全看不到。
>
> 这套框架是 **第一道便宜的 filter**，不是完整 review。完整 review 流水线还应包括：tile 边界 shape mutation、NaN/Inf 注入、sentinel 输出 buffer、Compute Sanitizer（`memcheck`/`racecheck`/`initcheck`/`synccheck`），以及 KB 驱动的人工 / agent 分析。

---

## 7. 操作 recipes

### 7.1 验证最小子集（KernelAgent）

```bash
python scripts/batch_diff_test.py --source kernelagent --run-name ka_only
cat reports/ka_only/summary.json
```

### 7.2 抽查单 GPU、Level 1 的 CUDA-L1

```bash
python scripts/batch_diff_test.py --source cuda_l1 --gpu a100 --level 1 \
    --run-name cudal1_a100_L1
```

### 7.3 跨 GPU 对比同一个 task

```bash
python scripts/list_pairs.py --task 1 --pattern cuda_l1 --paths-only \
    | xargs -I{} python scripts/run_diff_test.py {}
```

### 7.4 找出真·语义不同的 ref（Model.forward 体不一致）

```python
import ast, json
from pathlib import Path

DATASET = Path("preliminary_dataset")
KB = Path("KernelBench")

def model_class_src(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            return ast.unparse(node)
    return ""

diverged = []
for p in json.loads((DATASET / "index.json").read_text())["pairs"]:
    if p["source"] != "cuda_l1": continue
    pair = DATASET / p["dir"]
    ref = (pair / "ref.py").read_text()
    kb_file = json.loads((pair / "meta.json").read_text())["kernelbench_file"]
    kb_src = (KB / kb_file).read_text()
    if model_class_src(ref).strip() != model_class_src(kb_src).strip():
        diverged.append(p["pair_id"])
print(len(diverged), "pairs have a different Model.forward body")
```

### 7.5 长批跑前先 dry-run 看清楚选了哪些

```bash
python scripts/batch_diff_test.py --source cuda_l1 --gpu h100 --level 2 --dry-run
```

---

## 8. 环境要求

- Python ≥ 3.11（本仓库 `.python-version` = `3.11`）
- 与 GPU 匹配的 PyTorch + CUDA
- KernelAgent kernel 需要：Triton（upstream 标注 ≥ 3.5.1）
- CUDA-L1 的 `load_inline` kernel 需要：`nvcc` 在 `PATH` 上、`TORCH_CUDA_ARCH_LIST` 与 GPU 匹配
- 一块支持 CUDA 的 GPU（diff harness 默认 `--device cuda`）

`scripts/` 中的"建数据集"部分纯 stdlib，不需要 torch；只有 `_runtime.py`（以及 diff 测试器）需要导入 torch。

---

## 9. 来源与许可

- CUDA-L1 artifacts：`<repo>/CUDA-L1/`，参见其 `README.md` 与 `LICENSE`。
- KernelAgent artifacts：`<repo>/kernelagent-optimization-artifacts/`，参见其 `README.md`。
- 本目录的脚本只是把上游内容**重新整理**成 pair 目录，不修改原始代码。每份 `meta.json` 都记录了 upstream 的 `source_path`，可追溯。
