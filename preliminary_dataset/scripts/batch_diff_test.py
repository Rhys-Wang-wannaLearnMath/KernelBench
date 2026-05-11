#!/usr/bin/env python3
"""Run diff tests across many pairs and produce a summary report.

Selection (all optional; combine with AND):
    --source {cuda_l1,kernelagent}
    --gpu    {a100,3090,h100,h20,l40}
    --level  {1,2,3}
    --task   1,2,3      (comma-separated task ids; may repeat)
    --limit  N          (cap total pairs)
    --pattern STR       (substring match on pair_dir)

Each pair is tested in an isolated subprocess so one crash cannot take down the rest.

Outputs (default under reports/):
    reports/<run_name>/results.jsonl    one line per pair (DiffResult serialised)
    reports/<run_name>/summary.json     aggregate counts + breakdowns
    reports/<run_name>/failures.md      human-readable list of FAIL / ERROR pairs

Usage:
    python scripts/batch_diff_test.py --source kernelagent
    python scripts/batch_diff_test.py --source cuda_l1 --gpu a100 --level 1 --limit 10
    python scripts/batch_diff_test.py --run-name my_run_01 --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent


def _load_index() -> list[dict]:
    index_path = DATASET_ROOT / "index.json"
    if not index_path.exists():
        print(f"[!] index.json not found. Run build_dataset.py first.", file=sys.stderr)
        sys.exit(2)
    return json.loads(index_path.read_text())["pairs"]


def _filter_pairs(pairs, args) -> list[dict]:
    def keep(p):
        if args.source and p["source"] != args.source:
            return False
        if args.gpu and p.get("gpu") != args.gpu:
            return False
        if args.level is not None and p["level_id"] != args.level:
            return False
        if args.task:
            if p["task_id"] not in args.task:
                return False
        if args.pattern and args.pattern not in p["dir"]:
            return False
        return True

    out = [p for p in pairs if keep(p)]
    if args.limit:
        out = out[: args.limit]
    return out


def _run_one(pair_dir: Path, args) -> dict:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_diff_test.py"),
        str(pair_dir),
        "--rtol", str(args.rtol),
        "--atol", str(args.atol),
        "--device", args.device,
        "--seed", str(args.seed),
        "--quiet",
        "--json", "/dev/stdout",
    ]
    if args.dtype:
        cmd += ["--dtype", args.dtype]

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        try:
            # run_diff_test writes JSON to the --json target file; we used /dev/stdout
            payload = json.loads(r.stdout)
        except json.JSONDecodeError:
            payload = {
                "pair_dir": str(pair_dir),
                "interface": "?",
                "passed": False,
                "error": f"could not parse child output (rc={r.returncode})",
                "notes": [r.stdout[-2000:], r.stderr[-2000:]],
            }
        return payload
    except subprocess.TimeoutExpired:
        return {
            "pair_dir": str(pair_dir),
            "interface": "?",
            "passed": False,
            "error": f"timeout after {args.timeout}s",
            "notes": [],
        }
    except Exception as e:
        return {
            "pair_dir": str(pair_dir),
            "interface": "?",
            "passed": False,
            "error": f"launcher error: {e}",
            "notes": [],
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=["cuda_l1", "kernelagent"])
    parser.add_argument("--gpu", choices=["a100", "3090", "h100", "h20", "l40"])
    parser.add_argument("--level", type=int, choices=[1, 2, 3])
    parser.add_argument(
        "--task",
        type=lambda s: [int(x) for x in s.split(",") if x],
        help="comma-separated task ids",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pattern", default=None)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None,
                        choices=[None, "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"])
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=300)

    parser.add_argument("--run-name", default=None, help="name of the output folder under reports/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pairs = _filter_pairs(_load_index(), args)
    print(f"[batch] will test {len(pairs)} pairs")
    if args.dry_run:
        for p in pairs[:30]:
            print(f"  {p['pair_id']:<40} {p['dir']}")
        if len(pairs) > 30:
            print(f"  ... and {len(pairs) - 30} more")
        return 0

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = DATASET_ROOT / "reports" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    failures_path = out_dir / "failures.md"

    counts = {"pass": 0, "fail": 0, "error": 0}
    by_source: dict[str, dict[str, int]] = {}
    fail_lines: list[str] = []

    with results_path.open("w") as rf:
        for i, p in enumerate(pairs, 1):
            pair_dir = DATASET_ROOT / p["dir"]
            payload = _run_one(pair_dir, args)
            payload["pair_id"] = p["pair_id"]
            payload["source"] = p["source"]
            payload["gpu"] = p.get("gpu")
            payload["level_id"] = p["level_id"]
            payload["task_id"] = p["task_id"]
            rf.write(json.dumps(payload) + "\n")
            rf.flush()

            if payload.get("passed"):
                counts["pass"] += 1
                status = "PASS"
            elif payload.get("error"):
                counts["error"] += 1
                status = "ERROR"
            else:
                counts["fail"] += 1
                status = "FAIL"

            src_counts = by_source.setdefault(p["source"], {"pass": 0, "fail": 0, "error": 0})
            src_counts[status.lower()] += 1

            if status != "PASS":
                fail_lines.append(
                    f"- **{status}** `{p['pair_id']}` "
                    f"(max_abs={payload.get('max_abs_diff')}, "
                    f"err={payload.get('error')})"
                )

            print(f"  [{i}/{len(pairs)}] {status:5s} {p['pair_id']}")

    summary = {
        "run_name": run_name,
        "num_pairs": len(pairs),
        "counts": counts,
        "by_source": by_source,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    failures_path.write_text(
        "# Batch Diff Test — Failures & Errors\n\n"
        f"Run: `{run_name}`\n\n"
        f"Total: {len(pairs)} — "
        f"pass={counts['pass']} fail={counts['fail']} error={counts['error']}\n\n"
        + ("\n".join(fail_lines) if fail_lines else "_no failures_\n")
    )

    print("\n=== summary ===")
    print(f"  total: {len(pairs)}")
    print(f"  pass : {counts['pass']}")
    print(f"  fail : {counts['fail']}")
    print(f"  error: {counts['error']}")
    print(f"  -> {out_dir}")

    return 0 if counts["fail"] == 0 and counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
