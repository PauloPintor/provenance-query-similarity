"""
extract_provenance.py – Simulate provenance extraction for generated queries.

In a real deployment this script would connect to a provenance-aware database
engine (e.g. one that supports semiring provenance annotations) and execute
each query to obtain its lineage and Why-provenance.

For the purposes of this research repository we use a *deterministic synthetic
provenance generator* that assigns plausible provenance records based on the
SQL structure (table mentions, WHERE conditions, etc.) of each query.  The
generator ensures that:

- Queries from the same template share a large fraction of their provenance.
- Queries from different templates have mostly disjoint provenance.

This allows us to evaluate the effectiveness of provenance-based features for
query template identification.

Usage
-----
    python experiments/extract_provenance.py \\
        --input  data/queries.jsonl \\
        --output data/provenance.jsonl \\
        --seed   42
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from provenance_qs.data.loader import ProvenanceRecord, ProvenanceDataset, save_provenance_jsonl


# ---------------------------------------------------------------------------
# Synthetic provenance generation
# ---------------------------------------------------------------------------

def _extract_tables(sql: str) -> List[str]:
    """Heuristically extract table names mentioned in a SQL string."""
    pattern = re.compile(
        r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z_0-9]*)',
        re.IGNORECASE,
    )
    return list(dict.fromkeys(m.group(1).lower() for m in pattern.finditer(sql)))


def _synthetic_lineage(
    sql: str,
    template_id: int,
    instance_idx: int,
    rng: random.Random,
    n_tuples_per_table: int = 20,
    overlap_fraction: float = 0.7,
) -> Dict[str, List[int]]:
    """Generate a synthetic lineage dict for *sql*.

    Templates share ``overlap_fraction`` of tuple IDs; the remaining fraction
    is unique to each instance.
    """
    tables = _extract_tables(sql)
    lineage: Dict[str, List[int]] = {}
    for table in tables:
        shared_base = template_id * 1000 + hash(table) % 500
        n_shared = max(1, int(n_tuples_per_table * overlap_fraction))
        n_unique = n_tuples_per_table - n_shared
        shared_ids = list(range(shared_base, shared_base + n_shared))
        unique_base = shared_base + n_shared + instance_idx * 100
        unique_ids = list(range(unique_base, unique_base + n_unique))
        selected = sorted(rng.sample(shared_ids, min(n_shared, len(shared_ids))) + unique_ids)
        if selected:
            lineage[table] = selected
    return lineage


def _synthetic_witnesses(
    lineage: Dict[str, List[int]],
    n_witnesses: int,
    witness_size: int,
    rng: random.Random,
) -> List[List[List]]:
    """Build synthetic atomic witnesses from a lineage dict."""
    all_refs = [
        [rel, tid] for rel, tids in lineage.items() for tid in tids
    ]
    if not all_refs:
        return []
    witnesses = []
    for _ in range(n_witnesses):
        size = min(witness_size, len(all_refs))
        w = [list(ref) for ref in rng.sample(all_refs, size)]
        witnesses.append(w)
    return witnesses


def _synthetic_blocked_witnesses(
    lineage: Dict[str, List[int]],
    n_witnesses: int,
    rng: random.Random,
) -> List[Dict[str, List[int]]]:
    """Build synthetic blocked witnesses from a lineage dict."""
    tables = list(lineage.keys())
    if not tables:
        return []
    witnesses = []
    for _ in range(n_witnesses):
        n_rels = rng.randint(1, max(1, len(tables)))
        chosen_rels = rng.sample(tables, n_rels)
        w = {}
        for rel in chosen_rels:
            available = lineage[rel]
            k = rng.randint(1, max(1, len(available)))
            w[rel] = rng.sample(available, k)
        witnesses.append(w)
    return witnesses


def generate_synthetic_provenance(
    query_id: str,
    template_id: int,
    sql: str,
    instance_idx: int,
    rng: random.Random,
) -> ProvenanceRecord:
    lineage_data = _synthetic_lineage(sql, template_id, instance_idx, rng)
    n_witnesses = rng.randint(3, 8)
    witness_size = rng.randint(2, 5)
    atomic_why = _synthetic_witnesses(lineage_data, n_witnesses, witness_size, rng)
    blocked_why = _synthetic_blocked_witnesses(lineage_data, n_witnesses, rng)
    return ProvenanceRecord(
        query_id=query_id,
        template_id=template_id,
        sql=sql,
        lineage_data=lineage_data,
        atomic_why_data=atomic_why,
        blocked_why_data=blocked_why,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract (synthetic) provenance for queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  type=Path, default=Path("data/queries.jsonl"),
                   help="JSONL file of query records (output of generate_queries.py).")
    p.add_argument("--output", type=Path, default=Path("data/provenance.jsonl"),
                   help="Output JSONL file with provenance annotations.")
    p.add_argument("--seed",   type=int,  default=42,
                   help="Random seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    with args.input.open("r", encoding="utf-8") as fh:
        raw_records = [json.loads(line) for line in fh if line.strip()]

    instance_counts: Dict[int, int] = {}
    prov_records = []
    for raw in raw_records:
        tid = raw["template_id"]
        idx = instance_counts.get(tid, 0)
        instance_counts[tid] = idx + 1
        prov = generate_synthetic_provenance(
            query_id=raw["query_id"],
            template_id=tid,
            sql=raw["sql"],
            instance_idx=idx,
            rng=rng,
        )
        prov_records.append(prov)

    dataset = ProvenanceDataset(prov_records)
    save_provenance_jsonl(dataset, args.output)
    print(f"Extracted provenance for {len(prov_records)} queries → {args.output}")


if __name__ == "__main__":
    main()
