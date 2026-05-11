#!/usr/bin/env python3
"""Enumerate pairs in the dataset, with filtering.

Examples:
    python scripts/list_pairs.py                         # all pairs
    python scripts/list_pairs.py --source cuda_l1 --gpu a100
    python scripts/list_pairs.py --level 1 --limit 5 --full
    python scripts/list_pairs.py --source cuda_l1 --gpu a100 --level 1 --task 1,2,3
    python scripts/list_pairs.py --stats                 # summary only (no listing)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent.parent


def _load_index() -> list[dict]:
    index_path = DATASET_ROOT / "index.json"
    if not index_path.exists():
        print("[!] index.json not found. Run build_dataset.py first.", file=sys.stderr)
        sys.exit(2)
    return json.loads(index_path.read_text())["pairs"]


def _filter(pairs, args):
    def keep(p):
        if args.source and p["source"] != args.source:
            return False
        if args.gpu and p.get("gpu") != args.gpu:
            return False
        if args.level is not None and p["level_id"] != args.level:
            return False
        if args.task and p["task_id"] not in args.task:
            return False
        if args.pattern and args.pattern not in p["dir"]:
            return False
        return True
    return [p for p in pairs if keep(p)]


def _stats(pairs):
    by_source = Counter(p["source"] for p in pairs)
    by_gpu = Counter(p.get("gpu") for p in pairs)
    by_level = Counter(p["level_id"] for p in pairs)
    ref_mismatch = sum(1 for p in pairs if not p.get("ref_matches_kernelbench_current", True))
    return {
        "total": len(pairs),
        "by_source": dict(by_source),
        "by_gpu": dict(by_gpu),
        "by_level": dict(by_level),
        "ref_differs_from_kernelbench_current": ref_mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["cuda_l1", "kernelagent"])
    parser.add_argument("--gpu", choices=["a100", "3090", "h100", "h20", "l40"])
    parser.add_argument("--level", type=int, choices=[1, 2, 3])
    parser.add_argument("--task", type=lambda s: [int(x) for x in s.split(",") if x])
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="print full JSON per pair")
    parser.add_argument("--stats", action="store_true", help="print summary counts only")
    parser.add_argument("--paths-only", action="store_true", help="print only dir paths, one per line")
    args = parser.parse_args()

    pairs = _filter(_load_index(), args)
    if args.limit:
        pairs = pairs[: args.limit]

    if args.stats:
        print(json.dumps(_stats(pairs), indent=2))
        return 0

    if args.paths_only:
        for p in pairs:
            print(p["dir"])
        return 0

    if args.full:
        for p in pairs:
            print(json.dumps(p, indent=2))
        return 0

    # Default compact columnar output
    print(f"{'pair_id':<40} {'interface':<16} {'level':<5} {'task':<4} {'dir'}")
    print("-" * 120)
    for p in pairs:
        print(
            f"{p['pair_id']:<40} "
            f"{p.get('optimized_interface', '?'):<16} "
            f"{p['level_id']:<5} "
            f"{p['task_id']:<4} "
            f"{p['dir']}"
        )
    print(f"\n{len(pairs)} pair(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
