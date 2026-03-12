"""
generate_queries.py – Generate TPC-DS query instances for experiments.

Usage
-----
    python experiments/generate_queries.py \\
        --templates 1 3 7 19 42 \\
        --instances 50 \\
        --seed 42 \\
        --output data/queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the package is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from provenance_qs.data import TPCDS_TEMPLATES, generate_query_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TPC-DS query instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        type=int,
        default=sorted(TPCDS_TEMPLATES),
        help="TPC-DS template IDs to include.",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=50,
        help="Number of instances per template.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/queries.jsonl"),
        help="Output JSONL file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import random

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for tid in args.templates:
            try:
                instances = generate_query_instances(
                    tid, n_instances=args.instances, rng=rng
                )
            except ValueError as exc:
                print(f"[WARN] Skipping template {tid}: {exc}", file=sys.stderr)
                continue
            for i, (sql, label) in enumerate(instances):
                record = {
                    "query_id": f"q{label:03d}_{i:04d}",
                    "template_id": label,
                    "sql": sql,
                }
                fh.write(json.dumps(record) + "\n")
                total += 1

    print(f"Generated {total} query instances → {args.output}")


if __name__ == "__main__":
    main()
