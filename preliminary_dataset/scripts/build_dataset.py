#!/usr/bin/env python3
"""Build the preliminary_dataset layout from upstream CUDA-L1 and KernelAgent artifacts.

This script is idempotent: running it again regenerates the dataset from scratch.
It does NOT touch the source repositories; it only reads from them.

Sources:
  - /data/zrwang/KernelBench/CUDA-L1/optimized_cuda_code/{gpu}.json   (5 GPUs)
  - /data/zrwang/KernelBench/kernelagent-optimization-artifacts/{task_folder}/

Outputs (under <dataset_root>):
  - cuda_l1/{gpu}/level{1,2,3}/L{lvl}_T{tid:03d}_{name}/{ref.py, optimized.py, ...}
  - kernelagent/L1_T{tid:03d}_{name}/{ref.py, optimized.py, input_kernel.py, meta.json}
  - index.json  (machine-readable index of every pair)

Usage:
    python scripts/build_dataset.py                # default: rebuild
    python scripts/build_dataset.py --force        # wipe existing output first
    python scripts/build_dataset.py --no-cuda-l1   # skip CUDA-L1
    python scripts/build_dataset.py --no-kernelagent
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Defaults (paths relative to repo root)
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]  # /data/zrwang/KernelBench
DATASET_ROOT = (
    Path(__file__).resolve().parents[1]
)  # /data/zrwang/KernelBench/preliminary_dataset

CUDA_L1_ROOT = REPO_ROOT / "CUDA-L1" / "optimized_cuda_code"
KERNELAGENT_ROOT = REPO_ROOT / "kernelagent-optimization-artifacts"
KERNELBENCH_ROOT = REPO_ROOT / "KernelBench"

CUDA_L1_GPUS = ["a100", "3090", "h100", "h20", "l40"]

# --------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Sanitize a KernelBench filename into a folder-safe short name."""
    name = re.sub(r"^\d+_", "", name)  # drop leading id prefix
    name = re.sub(r"\.py$", "", name)  # drop .py
    name = re.sub(r"_+", "_", name)  # collapse underscores
    name = name.strip("_")
    if len(name) > 80:
        name = name[:80].rstrip("_")
    return name or "unnamed"


def _build_kernelbench_index() -> dict[tuple[int, int], tuple[str, str]]:
    """Map (level_id, task_id) -> (filename, source) for the current KernelBench."""
    idx: dict[tuple[int, int], tuple[str, str]] = {}
    for level in ("level1", "level2", "level3"):
        lid = int(level[-1])
        level_dir = KERNELBENCH_ROOT / level
        if not level_dir.is_dir():
            continue
        for p in sorted(level_dir.iterdir()):
            if not p.name.endswith(".py"):
                continue
            m = re.match(r"^(\d+)_", p.name)
            if not m:
                continue
            tid = int(m.group(1))
            idx[(lid, tid)] = (p.name, p.read_text())
    return idx


def _get_task_name(kb_filename: str) -> str:
    return _slugify(kb_filename)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# --------------------------------------------------------------------------
# CUDA-L1 extraction
# --------------------------------------------------------------------------


def _extract_cuda_l1(
    dataset_root: Path,
    kb_index: dict[tuple[int, int], tuple[str, str]],
) -> list[dict]:
    """Walk all GPU JSON files in CUDA-L1 and materialize per-task folders."""
    if not CUDA_L1_ROOT.is_dir():
        print(f"[!] CUDA-L1 root not found: {CUDA_L1_ROOT}", file=sys.stderr)
        return []

    pairs: list[dict] = []

    for gpu in CUDA_L1_GPUS:
        json_path = CUDA_L1_ROOT / f"{gpu}.json"
        if not json_path.is_file():
            print(f"[!] missing {json_path}", file=sys.stderr)
            continue

        print(f"[cuda_l1] processing {gpu}.json ...")
        with json_path.open() as f:
            entries = [json.loads(line) for line in f if line.strip()]

        kept = 0
        skipped_no_custom = 0
        for e in entries:
            lvl = e["level_id"]
            tid = e["task_id"]
            custom = e.get("custom_code")
            if not custom or custom == "None":
                skipped_no_custom += 1
                continue

            kb_file, kb_src = kb_index.get((lvl, tid), (f"L{lvl}_T{tid}.py", ""))
            task_name = _get_task_name(kb_file)
            task_dir = (
                dataset_root
                / "cuda_l1"
                / gpu
                / f"level{lvl}"
                / f"L{lvl}_T{tid:03d}_{task_name}"
            )

            _write_text(task_dir / "ref.py", e["ref_code"])
            _write_text(task_dir / "optimized.py", custom)

            if e.get("cuda_graph_code") and e["cuda_graph_code"] != "None":
                _write_text(task_dir / "baseline_cuda_graph.py", e["cuda_graph_code"])
            if e.get("cudnn_code") and e["cudnn_code"] != "None":
                _write_text(task_dir / "baseline_cudnn.py", e["cudnn_code"])

            # Detect reference mismatch vs current KernelBench
            ref_matches_kb = kb_src.strip() == e["ref_code"].strip()

            meta = {
                "source": "cuda_l1",
                "gpu": gpu,
                "level_id": lvl,
                "task_id": tid,
                "task_name": task_name,
                "kernelbench_file": f"level{lvl}/{kb_file}",
                "optimized_interface": "ModelNew",  # CUDA-L1 convention
                "has_optimized": True,
                "has_baseline_cuda_graph": (
                    task_dir / "baseline_cuda_graph.py"
                ).exists(),
                "has_baseline_cudnn": (task_dir / "baseline_cudnn.py").exists(),
                "scores": {
                    "score_default": e.get("score_default"),
                    "score_torch_compile_default": e.get("score_torch_compile_default"),
                    "score_torch_compile_reduce_overhead": e.get(
                        "score_torch_compile_reduce_overhead"
                    ),
                    "score_cuda_graph": e.get("score_cuda_graph"),
                    "score_cudnn": e.get("score_cudnn"),
                },
                "ref_matches_kernelbench_current": ref_matches_kb,
                "source_path": str(json_path.relative_to(REPO_ROOT)),
            }
            _write_text(task_dir / "meta.json", json.dumps(meta, indent=2) + "\n")

            pairs.append(
                {
                    "pair_id": f"cuda_l1__{gpu}__L{lvl}T{tid:03d}",
                    "source": "cuda_l1",
                    "gpu": gpu,
                    "level_id": lvl,
                    "task_id": tid,
                    "task_name": task_name,
                    "dir": str(task_dir.relative_to(dataset_root)),
                    "optimized_interface": "ModelNew",
                    "ref_matches_kernelbench_current": ref_matches_kb,
                    "scores": meta["scores"],
                }
            )
            kept += 1

        print(f"    kept={kept}, skipped(no custom_code)={skipped_no_custom}")

    return pairs


# --------------------------------------------------------------------------
# KernelAgent extraction
# --------------------------------------------------------------------------


def _extract_kernelagent(
    dataset_root: Path,
    kb_index: dict[tuple[int, int], tuple[str, str]],
) -> list[dict]:
    if not KERNELAGENT_ROOT.is_dir():
        print(f"[!] KernelAgent root not found: {KERNELAGENT_ROOT}", file=sys.stderr)
        return []

    pairs: list[dict] = []

    for task_dir_src in sorted(KERNELAGENT_ROOT.iterdir()):
        if not task_dir_src.is_dir():
            continue
        if not (task_dir_src / "problem.py").exists():
            continue

        # folder names start with a numeric prefix, e.g. "04_Matrix_vector_multiplication"
        m = re.match(r"^0*(\d+)_(.*)", task_dir_src.name)
        if not m:
            print(f"    [skip] unexpected folder name: {task_dir_src.name}")
            continue

        tid = int(m.group(1))
        # KernelAgent artifacts are all from KernelBench Level 1
        lvl = 1
        kb_file, _ = kb_index.get((lvl, tid), (f"L{lvl}_T{tid}.py", ""))
        task_name = (
            _get_task_name(kb_file) if kb_file.endswith(".py") else _slugify(m.group(2))
        )
        out_dir = dataset_root / "kernelagent" / f"L{lvl}_T{tid:03d}_{task_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(task_dir_src / "problem.py", out_dir / "ref.py")
        opt_src = task_dir_src / "optimized_kernel_beam_search.py"
        if opt_src.exists():
            shutil.copy(opt_src, out_dir / "optimized.py")
        input_src = task_dir_src / "input_kernel.py"
        if input_src.exists():
            shutil.copy(input_src, out_dir / "input_kernel.py")

        meta = {
            "source": "kernelagent",
            "gpu": "h100",  # per upstream README, experiments were run on H100
            "level_id": lvl,
            "task_id": tid,
            "task_name": task_name,
            "kernelbench_file": f"level{lvl}/{kb_file}",
            "optimized_interface": "kernel_function",  # KernelAgent convention
            "has_optimized": (out_dir / "optimized.py").exists(),
            "has_input_kernel": (out_dir / "input_kernel.py").exists(),
            "has_trace": (task_dir_src / "optimization_trace").is_dir(),
            "trace_path_upstream": (
                str((task_dir_src / "optimization_trace").relative_to(REPO_ROOT))
                if (task_dir_src / "optimization_trace").is_dir()
                else None
            ),
            "source_path": str(task_dir_src.relative_to(REPO_ROOT)),
        }
        _write_text(out_dir / "meta.json", json.dumps(meta, indent=2) + "\n")

        pairs.append(
            {
                "pair_id": f"kernelagent__L{lvl}T{tid:03d}",
                "source": "kernelagent",
                "gpu": "h100",
                "level_id": lvl,
                "task_id": tid,
                "task_name": task_name,
                "dir": str(out_dir.relative_to(dataset_root)),
                "optimized_interface": "kernel_function",
                "ref_matches_kernelbench_current": True,
                "scores": {},
            }
        )
        print(
            f"[kernelagent] copied L{lvl}/T{tid:03d} -> {out_dir.relative_to(dataset_root)}"
        )

    return pairs


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="wipe existing cuda_l1/ and kernelagent/ before rebuilding",
    )
    parser.add_argument("--no-cuda-l1", action="store_true")
    parser.add_argument("--no-kernelagent", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.force:
        for sub in ("cuda_l1", "kernelagent"):
            p = root / sub
            if p.exists():
                print(f"[force] removing {p}")
                shutil.rmtree(p)

    kb_index = _build_kernelbench_index()
    print(f"[kernelbench] indexed {len(kb_index)} reference tasks")

    all_pairs: list[dict] = []
    if not args.no_cuda_l1:
        all_pairs.extend(_extract_cuda_l1(root, kb_index))
    if not args.no_kernelagent:
        all_pairs.extend(_extract_kernelagent(root, kb_index))

    # Write master index.json
    index_path = root / "index.json"
    index_obj = {
        "num_pairs": len(all_pairs),
        "by_source": {},
        "pairs": all_pairs,
    }
    for p in all_pairs:
        src = p["source"]
        index_obj["by_source"].setdefault(src, 0)
        index_obj["by_source"][src] += 1
    index_path.write_text(json.dumps(index_obj, indent=2) + "\n")

    print("\n=== build summary ===")
    print(f"  total pairs: {len(all_pairs)}")
    for src, n in index_obj["by_source"].items():
        print(f"    {src}: {n}")
    print(f"  index written to: {index_path}")


if __name__ == "__main__":
    main()
