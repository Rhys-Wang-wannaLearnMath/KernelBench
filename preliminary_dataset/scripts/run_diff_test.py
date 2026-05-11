#!/usr/bin/env python3
"""Run a differential-correctness test on a single (ref.py, optimized.py) pair.

Usage:
    # Point at a pair directory (must contain ref.py and optimized.py)
    python scripts/run_diff_test.py cuda_l1/a100/level1/L1_T001_Square_matrix_multiplication

    # Control tolerance / dtype / device
    python scripts/run_diff_test.py <pair_dir> --rtol 1e-3 --atol 1e-3
    python scripts/run_diff_test.py <pair_dir> --dtype bf16 --device cuda:0

    # Emit JSON result
    python scripts/run_diff_test.py <pair_dir> --json out.json

Exit codes:
    0  pair passed
    1  pair failed (semantic mismatch)
    2  pair errored (import / runtime error)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _runtime import run_diff_test, result_to_dict  # noqa: E402


def _print_result(r):
    tag = "PASS" if r.passed else ("FAIL" if r.error is None else "ERROR")
    color = {"PASS": "\033[32m", "FAIL": "\033[31m", "ERROR": "\033[33m"}.get(tag, "")
    reset = "\033[0m" if color else ""
    print(f"{color}[{tag}]{reset} {r.pair_dir}")
    print(f"  interface     : {r.interface}")
    if r.elapsed_sec is not None:
        print(f"  elapsed       : {r.elapsed_sec:.3f}s")
    if r.ref_shape is not None:
        print(f"  shape (ref)   : {r.ref_shape}")
        print(f"  shape (opt)   : {r.opt_shape}")
        print(f"  dtype (ref)   : {r.ref_dtype}")
        print(f"  dtype (opt)   : {r.opt_dtype}")
    if r.max_abs_diff is not None:
        print(f"  max abs diff  : {r.max_abs_diff:.3e}")
        print(f"  max rel diff  : {r.max_rel_diff:.3e}")
    if r.error:
        print(f"  error         : {r.error}")
    for n in r.notes:
        first = n.splitlines()[0] if n else ""
        print(f"  note          : {first}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pair_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None,
                        choices=[None, "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
                        help="force dtype for all floating inputs + parameters (default: keep original)")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=Path, default=None, help="write machine-readable result JSON here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.pair_dir.is_dir():
        print(f"[!] not a directory: {args.pair_dir}", file=sys.stderr)
        return 2

    r = run_diff_test(
        args.pair_dir,
        device=args.device,
        dtype=args.dtype,
        rtol=args.rtol,
        atol=args.atol,
        seed=args.seed,
    )

    if not args.quiet:
        _print_result(r)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result_to_dict(r), indent=2) + "\n")

    if r.error:
        return 2
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
