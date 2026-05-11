# Preliminary Dataset for Kernel-Optimization Review

This folder is a curated, locally-rebuildable dataset of **(reference kernel, optimized kernel)** pairs harvested from two state-of-the-art LLM kernel-optimization artifact releases:

- **CUDA-L1** — `CUDA-L1/optimized_cuda_code/{a100,3090,h100,h20,l40}.json`
- **KernelAgent** — `kernelagent-optimization-artifacts/<task>/`

It is designed to support the *preliminary test* of a **semantic-review agent** for LLM-generated kernels: each pair is a self-contained directory exposing a clean `ref.py` ↔ `optimized.py` interface, ready for differential testing, manual inspection, or downstream tooling.

> Sister doc: `README_zh.md` (中文版).

---

## TL;DR

```bash
# 1. (Re)build the dataset from the upstream repos
python scripts/build_dataset.py

# 2. List what's in the dataset
python scripts/list_pairs.py --stats
python scripts/list_pairs.py --source cuda_l1 --gpu a100 --level 1 --limit 10

# 3. Run a differential-correctness test on one pair
python scripts/run_diff_test.py kernelagent/L1_T036_RMSNorm

# 4. Batch run with a filter, results saved under reports/<run_name>/
python scripts/batch_diff_test.py --source kernelagent
python scripts/batch_diff_test.py --source cuda_l1 --gpu a100 --level 1 --limit 20
```

---

## 1. Dataset composition

After `build_dataset.py`:

| Source | GPU | Tasks with optimized kernel | Interface |
|---|---|---|---|
| CUDA-L1 | a100 | 225 / 250 | `ModelNew` (drop-in `nn.Module`) |
| CUDA-L1 | 3090 | 197 / 250 | `ModelNew` |
| CUDA-L1 | h100 | 210 / 250 | `ModelNew` |
| CUDA-L1 | h20  | 219 / 250 | `ModelNew` |
| CUDA-L1 | l40  | 213 / 250 | `ModelNew` |
| KernelAgent | h100 | 5 / 5 | `kernel_function` (free function) |
| **Total** | — | **1069 pairs** | — |

Only tasks where the upstream emitted a non-null optimized kernel are materialized as pairs.

### Two interfaces, one harness

The two upstream projects use different conventions; our diff-test harness handles both transparently:

| Convention | Where | What it looks like |
|---|---|---|
| `class ModelNew(nn.Module)` | CUDA-L1 | Drop-in replacement for `class Model`. Compared as `ref_model(*inputs)` vs `opt_model(*inputs)`. |
| `def kernel_function(...)` | KernelAgent | Free function; we auto-bind the reference model's parameters (`weight`, `bias`, `kernel_size`, `stride`, `padding`, `dilation`, `output_padding`, `groups`, `eps`) as kwargs, then call as `kernel_function(*inputs, **bound_params)`. |

The interface used by each pair is recorded in `meta.json` → `optimized_interface`.

---

## 2. Directory layout

```
preliminary_dataset/
├── README.md / README_zh.md
├── index.json                          ← machine-readable index of every pair
├── cuda_l1/
│   └── {a100,3090,h100,h20,l40}/
│       └── level{1,2,3}/
│           └── L{lvl}_T{tid:03d}_{name}/
│               ├── ref.py              ← CUDA-L1's ref_code (≠ current KernelBench, see §5)
│               ├── optimized.py        ← CUDA-L1's custom_code (ModelNew)
│               ├── baseline_cuda_graph.py   (optional)
│               ├── baseline_cudnn.py        (optional)
│               └── meta.json
├── kernelagent/
│   └── L1_T{tid:03d}_{name}/
│       ├── ref.py                      ← upstream problem.py (= current KernelBench)
│       ├── optimized.py                ← upstream optimized_kernel_beam_search.py
│       ├── input_kernel.py             ← beam-search starting kernel
│       └── meta.json
├── scripts/
│   ├── build_dataset.py                ← rebuild from upstream sources
│   ├── _runtime.py                     ← shared loader / interface adapter / comparator
│   ├── run_diff_test.py                ← test one pair
│   ├── batch_diff_test.py              ← test many in subprocess
│   └── list_pairs.py                   ← query / filter the index
└── reports/                            ← outputs of batch_diff_test.py
```

### `meta.json` schema (per pair)

```jsonc
{
  "source": "cuda_l1",                  // or "kernelagent"
  "gpu": "a100",                        // upstream-declared device
  "level_id": 1,
  "task_id": 1,
  "task_name": "Square_matrix_multiplication",
  "kernelbench_file": "level1/1_Square_matrix_multiplication_.py",
  "optimized_interface": "ModelNew",    // or "kernel_function"
  "has_optimized": true,
  "has_baseline_cuda_graph": true,      // CUDA-L1 only
  "has_baseline_cudnn": true,           // CUDA-L1 only
  "scores": {                           // CUDA-L1's reported speedups
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

### `index.json` schema

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

## 3. Auxiliary files & score fields

The `(ref.py, optimized.py)` pair is the main review target, but each pair directory also contains **auxiliary files** that act as additional baselines or pipeline artifacts. Understanding them is essential to interpreting the `scores` dictionary in `meta.json`.

### 3.1 Per-pair files, by role

| File | Source | Semantics | Performance tier | Role |
|------|--------|-----------|------------------|------|
| `ref.py` | both | = original task definition | native-PyTorch baseline | **ground-truth oracle** |
| `baseline_cuda_graph.py` | CUDA-L1 only | equivalent to `ref.py` | baseline+ (CUDA Graph eliminates launch overhead) | **stronger baseline** (non-LLM, rule-based) |
| `baseline_cudnn.py` | CUDA-L1 only | equivalent to `ref.py` | baseline+ (`cudnn.benchmark=True`, auto-tuner) | **stronger baseline** (non-LLM, rule-based) |
| `input_kernel.py` | KernelAgent only | should be equivalent to `ref.py` | moderate (sometimes slower than `torch.compile`) | beam-search **starting point** (earlier KernelAgent version before the optimization loop) |
| `optimized.py` | both | **should** be equivalent to `ref.py` ⚠️ | fast (the claimed final optimization) | **LLM pipeline final output** — your review target |

### 3.2 CUDA-L1 score fields

All five scores share the same formula:

```
score_X = runtime(baseline_X) / runtime(custom_code)
          ↑ higher = larger claimed speedup
```

The **numerator baseline** differs; the **denominator (`custom_code`, i.e. `optimized.py`) is always the same**:

| JSON field | Short name (paper/tables) | Numerator baseline | What it measures |
|------------|---------------------------|---------------------|------------------|
| `score_default` | `score_default` | raw `ref.py` | vs native PyTorch |
| `score_torch_compile_default` | `score_tc` | `torch.compile(Model)` on `ref.py` | vs `torch.compile` default mode |
| `score_torch_compile_reduce_overhead` | `score_tc_ro` | `torch.compile(Model, mode="reduce-overhead")` | vs `torch.compile` with reduce-overhead (internally uses CUDA Graph) |
| `score_cuda_graph` | `score_cg` | `baseline_cuda_graph.py` | vs manually CUDA-Graph-wrapped ref |
| `score_cudnn` | `score_cudnn` | `baseline_cudnn.py` | vs `cudnn.benchmark=True` ref |

**Concrete example — L1/T1 (Square matmul) on A100:**

| Score | Value | Reading |
|-------|-------|---------|
| `score_default` | **6.304×** | vs raw ref (weakest baseline) |
| `score_tc` | 6.326× | `torch.compile` barely helps on a single matmul |
| `score_tc_ro` | 6.245× | reduce-overhead mode is almost identical here |
| `score_cg` | **5.093×** | ← CUDA Graph is a much stronger baseline (denominator ↑ ⇒ ratio ↓) |
| `score_cudnn` | 5.190× | cuDNN auto-tuner also stronger |

> **⚠️ The single most important caveat about CUDA-L1's headline numbers.** The paper's “≥3× average speedup” cites `score_default` (vs raw PyTorch). Under the harder `score_cuda_graph` / `score_tc_ro` baselines, a large fraction of the speedup evaporates on many tasks. **Which baseline is “fair” depends entirely on what you compare against in your own downstream deployment.** Do not mix baselines when reporting.

### 3.3 Why this matters for semantic-equivalence review

1. **The “real” LLM contribution** ≈ `runtime(optimized.py) / runtime(baseline_cuda_graph.py)` (not vs raw ref). If the LLM only beats raw ref, it might just be missing a trivial CUDA Graph wrap — not a true algorithmic improvement.
2. **KernelAgent's beam-search incremental gain** ≈ `runtime(optimized.py) / runtime(input_kernel.py)` (only available for KernelAgent pairs).
3. **Cross-baseline semantic-equivalence check.** All five variants (`ref`, `baseline_cuda_graph`, `baseline_cudnn`, `input_kernel`, `optimized`) should produce numerically close outputs. If `optimized.py` disagrees with any of the first four — especially `baseline_cuda_graph.py`, which is the cleanest semantic copy of `ref.py` — that is a strong signal of LLM-introduced divergence.
4. **Reward-hacking “gold line”.** The first four files are non-LLM artifacts. A review agent that flags any of them as semantically wrong is likely producing false positives; one that flags only `optimized.py` is hitting real LLM issues.

---

## 4. Scripts

### `build_dataset.py`

Idempotent extractor. Reads upstream repos, writes the canonical layout, and emits `index.json`.

```bash
python scripts/build_dataset.py                 # rebuild (skips existing files)
python scripts/build_dataset.py --force         # wipe cuda_l1/ + kernelagent/ then rebuild
python scripts/build_dataset.py --no-cuda-l1
python scripts/build_dataset.py --no-kernelagent
python scripts/build_dataset.py --dataset-root /some/other/path
```

Source paths are hard-coded to:
- `<repo>/CUDA-L1/optimized_cuda_code/{gpu}.json`
- `<repo>/kernelagent-optimization-artifacts/<task_folder>/`

(where `<repo>` is the parent directory of `preliminary_dataset/`).

### `list_pairs.py`

Query the index without GPU.

```bash
python scripts/list_pairs.py --stats
python scripts/list_pairs.py --source cuda_l1 --gpu a100
python scripts/list_pairs.py --level 1 --task 1,2,3 --full
python scripts/list_pairs.py --pattern Matmul --paths-only
```

### `run_diff_test.py`

Run a single pair's diff test. Requires CUDA + the runtime from the relevant upstream (PyTorch + Triton for KernelAgent, PyTorch with cpp_extension for CUDA-L1's `load_inline` kernels).

```bash
python scripts/run_diff_test.py <pair_dir>
python scripts/run_diff_test.py kernelagent/L1_T036_RMSNorm
python scripts/run_diff_test.py cuda_l1/a100/level1/L1_T001_Square_matrix_multiplication \
    --rtol 1e-3 --atol 1e-3 --dtype bf16 --json /tmp/result.json
```

Exit codes: `0` pass, `1` semantic fail, `2` runtime/import error.

### `batch_diff_test.py`

Filtered batch runner. Each pair runs in an isolated subprocess (one crash cannot poison others).

```bash
python scripts/batch_diff_test.py --source kernelagent
python scripts/batch_diff_test.py --source cuda_l1 --gpu a100 --level 1 --limit 50
python scripts/batch_diff_test.py --run-name baseline_a100_L1 --dtype bf16
```

Outputs (under `reports/<run_name>/`):
- `results.jsonl` — one line per pair, full `DiffResult` JSON
- `summary.json`  — aggregate counts + breakdown
- `failures.md`   — human-readable list of failing / erroring pairs

---

## 5. Important caveat: which "reference" do we use?

KernelBench has evolved across versions, and the two upstream projects do **not** share the same reference snapshot. Always be explicit about your oracle.

| Pair source | What `ref.py` actually is | Matches current `KernelBench/level{1,2,3}/*.py`? |
|---|---|---|
| **KernelAgent** | Upstream's `problem.py`, copied verbatim | ✅ All 5 are byte-identical |
| **CUDA-L1**     | The `ref_code` field shipped inside the upstream JSON | ❌ All 1064 differ from current main |

The CUDA-L1 differences fall into three layers (sample: `a100.json` 250 entries):

| Type of difference | Count | Example |
|---|---|---|
| Input distribution: `torch.rand` → `torch.randn` only | 191 | L1/T1 |
| Distribution + shape constants (e.g., `N = 4096` → `N = 2048`) | 20 | L1/T1 |
| `class Model.forward` body itself differs (true semantic divergence) | **25** | L1/T12: `A.unsqueeze(1) * B` (current) vs `torch.diag(A) @ B` (CUDA-L1) |

`build_dataset.py` records `ref_matches_kernelbench_current: false` in `meta.json` for every CUDA-L1 pair so this can be filtered downstream.

> **What to do.** When evaluating CUDA-L1's `optimized.py`, **use the `ref.py` shipped in the same pair directory** (i.e., its own oracle). When comparing across datasets or against the current KernelBench, treat CUDA-L1 as testing against an **older snapshot** and either:
>
> 1. report scores separately with the older oracle (cleanest), or
> 2. patch the 25 functionally-different tasks (or simply drop them) before doing a unified comparison against current KernelBench.

The 25 functionally-divergent tasks are listed in `meta.json` as `ref_matches_kernelbench_current: false` plus their `forward` body actually differing (we expose the boolean only; you can recompute the AST-level diff with the snippet in §7.4 if needed).

---

## 6. The diff-test procedure (what `_runtime.py` does)

For each pair:

1. **Import** `ref.py` and `optimized.py` as isolated modules (separate `sys.modules` entries).
2. **Build inputs** by calling `ref.get_init_inputs()` and `ref.get_inputs()`; move tensors to `--device`, optionally cast to `--dtype`.
3. **Build models**:
   - `ModelNew` interface: instantiate both `Model(*init)` and `ModelNew(*init)`, copy `state_dict` from ref to opt (`strict=False`) so parameters are identical.
   - `kernel_function` interface: instantiate `Model(*init)` and bind its parameters into a closure that calls `kernel_function(*inputs, **bound_params)`.
4. **Forward** both under `torch.inference_mode()` with full CUDA sync.
5. **Compare** output(s) on CPU-fp32 with `torch.allclose(rtol, atol)`. Record `max_abs_diff`, `max_rel_diff`, plus shape / dtype mismatch notes.

Default tolerance is `rtol=atol=1e-3` (matches KernelAgent's upstream); use stricter `1e-5` for index/copy-class operators where bitwise-or-near-bitwise equivalence is expected.

> ⚠️ **Diff-test ≠ semantic equivalence.** Allclose under random inputs catches gross errors but routinely misses:
> - shape-specialised hard-codes (e.g., a kernel that only works for the upstream's exact shape),
> - boundary bugs (`BLOCK ± 1`, `K = 1`, `batch = 1`, prime sizes),
> - numerical-edge bugs (NaN propagation, ±0, denormals, overflow at fp16),
> - OOB read/write that *happens* to land on benign memory,
> - reward hacking on timing measurement (warmup dispatch tricks) that doesn't show in correctness checks.
>
> The harness here is the **first cheap filter**, not the full review. The full review pipeline should add: shape mutation at tile boundaries, NaN/Inf injection, sentinel output buffers, Compute Sanitizer (`memcheck`/`racecheck`/`initcheck`/`synccheck`), and your knowledge-base-driven manual analysis.

---

## 7. Recipes

### 7.1 Verify the smallest possible subset (KernelAgent)

```bash
python scripts/batch_diff_test.py --source kernelagent --run-name ka_only
cat reports/ka_only/summary.json
```

### 7.2 Spot-check CUDA-L1 on one GPU, level 1

```bash
python scripts/batch_diff_test.py --source cuda_l1 --gpu a100 --level 1 \
    --run-name cudal1_a100_L1
```

### 7.3 Compare across GPUs for a single task

```bash
python scripts/list_pairs.py --task 1 --pattern cuda_l1 --paths-only \
    | xargs -I{} python scripts/run_diff_test.py {}
```

### 7.4 Find the truly-functional ref divergences (Model.forward differs)

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

### 7.5 Dry-run a filter before committing to a long batch

```bash
python scripts/batch_diff_test.py --source cuda_l1 --gpu h100 --level 2 --dry-run
```

---

## 8. Environment requirements

- Python ≥ 3.11 (this repo's `.python-version` is `3.11`)
- PyTorch with CUDA matching your GPU
- For KernelAgent kernels: Triton (≥ 3.5.1 per upstream)
- For CUDA-L1 `load_inline` kernels: `nvcc` available on `PATH`, with `TORCH_CUDA_ARCH_LIST` matching your GPU
- A CUDA-capable GPU (the diff harness sets `--device cuda` by default)

The `scripts/` are pure-stdlib for *building* the dataset; only `_runtime.py` (and therefore the diff testers) import torch.

---

## 9. Provenance & licensing

- CUDA-L1 artifacts: `<repo>/CUDA-L1/` — see its `README.md` and `LICENSE`.
- KernelAgent artifacts: `<repo>/kernelagent-optimization-artifacts/` — see its `README.md`.
- The reformatting / scripts in this folder do not modify the upstream code; they only reorganise it into pair directories. Each `meta.json` records the upstream `source_path` for traceability.
