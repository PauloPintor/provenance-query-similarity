#!/usr/bin/env python3
# generate_and_split_all_provenance_datasets.py

from __future__ import annotations

import os
import glob
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from collections import Counter, defaultdict

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = os.environ.get("ROOT_DIR", ".")
INPUT_SUFFIX = os.environ.get("INPUT_SUFFIX", "_versions")
OUTPUT_SUFFIX = os.environ.get("OUTPUT_SUFFIX", "_split")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "250"))
TEST_PER_TEMPLATE = int(os.environ.get("TEST_PER_TEMPLATE", "250"))
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "13"))

DUCKDB_PATH = os.environ.get("DUCKDB_PATH", "tpcds.duckdb")
DUCKDB_SCHEMA = os.environ.get("DUCKDB_SCHEMA", "tpcds")

MAX_WSIZE_BIN = 8


# ============================================================
# HELPERS
# ============================================================


def find_template_dirs(root_dir: str, input_suffix: str) -> List[str]:
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if dirpath.endswith(input_suffix) and any(
            f.endswith(".sql") for f in filenames
        ):
            out.append(dirpath)
    return sorted(out)


def load_templates(dir_path: str) -> List[Tuple[str, str]]:
    files = sorted(glob.glob(os.path.join(dir_path, "*.sql")))
    if not files:
        raise FileNotFoundError(f"No .sql files found in {dir_path}")

    out: List[Tuple[str, str]] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            name = os.path.splitext(os.path.basename(fp))[0]
            out.append((name, f.read()))
    return out


def parse_template_id(name: str) -> int:
    s = name.strip()
    if s.startswith("q") and len(s) > 1 and s[1].isdigit():
        s = s[1:]
    head = s.split("_", 1)[0]
    return int(head)


def find_witnesses_col(cols: List[str]) -> int:
    if "witnesses" in cols:
        return cols.index("witnesses")
    if "prov" in cols:
        return cols.index("prov")
    if "token" in cols:
        return cols.index("token")
    raise RuntimeError(
        f"Could not find column 'witnesses' (or 'prov'/'token'). Available columns: {cols}"
    )


def stable_hash_u64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


def row_sort_key(row: Dict[str, Any], seed: int) -> int:
    qname = str(row.get("query_name", ""))
    qid = str(row.get("query_id", ""))
    return stable_hash_u64(f"{qname}|qid={qid}|seed={seed}")


def token_to_hash_i64(token: str) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=True)


def witness_tuple_to_hash_i64(w: Tuple[int, ...]) -> int:
    b = b"".join(int(x).to_bytes(8, "big", signed=True) for x in w)
    h = hashlib.blake2b(b, digest_size=8).digest()
    return int.from_bytes(h, "big", signed=True)


def parse_old_witness_string_to_tuple(witness_str: str) -> Tuple[int, ...]:
    s = str(witness_str).strip().strip("{}").strip()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return tuple(sorted(token_to_hash_i64(p) for p in parts))


def witness_tokens_to_tuple(witness_tokens: Any) -> Tuple[int, ...]:
    if witness_tokens is None:
        return ()

    if isinstance(witness_tokens, (list, tuple)):
        parts: List[str] = []
        for t in witness_tokens:
            if t is None:
                continue
            s = str(t).strip()
            if s:
                parts.append(s)
        if not parts:
            return ()
        return tuple(sorted(token_to_hash_i64(p) for p in parts))

    return parse_old_witness_string_to_tuple(str(witness_tokens))


def wsize_hist_from_counter(wsize_ctr: Counter[int], max_bin: int) -> List[int]:
    hist = [0] * (max_bin + 1)
    for size, c in wsize_ctr.items():
        if size <= 0:
            continue
        if size <= max_bin:
            hist[size - 1] += int(c)
        else:
            hist[max_bin] += int(c)
    return hist


def derive_prefix_and_outdir(in_dir: str) -> Tuple[str, str]:
    folder_name = os.path.basename(in_dir)  # q98_versions
    prefix = folder_name.split("_", 1)[0]  # q98
    base_name = folder_name[: -len(INPUT_SUFFIX)]  # q98
    out_folder_name = base_name + OUTPUT_SUFFIX  # q98_split
    parent = os.path.dirname(in_dir)
    out_dir = os.path.join(parent, out_folder_name)
    return prefix, out_dir


# ============================================================
# DUCKDB CLIENT
# ============================================================


class DuckClient:
    def __init__(self, db_path: str = ":memory:") -> None:
        self.con = duckdb.connect(db_path)

    def close(self) -> None:
        self.con.close()

    def execute(self, sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
        cur = self.con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        return cols, rows


# ============================================================
# FEATURE EXTRACTION
# ============================================================


def build_row(
    query_id: int,
    query_name: str,
    template_id: int,
    query_sql: str,
    cols: List[str],
    result_rows: List[Tuple[Any, ...]],
) -> Dict[str, Any]:
    nrows = len(result_rows)

    why_set: Set[Tuple[int, ...]] = set()
    hb_set: Set[int] = set()
    ln_set: Set[int] = set()

    token_ctr: Counter[int] = Counter()
    wit_ctr: Counter[int] = Counter()
    wsize_ctr: Counter[int] = Counter()

    nwitness_total = 0
    ntokens_total = 0

    if nrows > 0:
        wcol = find_witnesses_col(cols)

        for row in result_rows:
            cell = row[wcol]
            if cell is None:
                continue

            outer = cell if isinstance(cell, (list, tuple)) else [cell]

            for wit in outer:
                if wit is None:
                    continue

                w = witness_tokens_to_tuple(wit)
                if not w:
                    continue

                nwitness_total += 1
                wsize_ctr[len(w)] += 1
                ntokens_total += len(w)

                wh = int(witness_tuple_to_hash_i64(w))
                wit_ctr[wh] += 1

                for th in w:
                    token_ctr[int(th)] += 1

                why_set.add(w)
                hb_set.add(wh)
                for th in w:
                    ln_set.add(int(th))

    fullwhy_witnesses = [list(w) for w in sorted(why_set)]
    nwitness_unique = len(fullwhy_witnesses)
    ntokens_unique = len(ln_set)
    avg_w = (ntokens_total / nwitness_total) if nwitness_total > 0 else 0.0

    token_tf = [{"token": int(t), "tf": int(c)} for t, c in token_ctr.items()]
    witness_tf = [{"witness_hash": int(h), "tf": int(c)} for h, c in wit_ctr.items()]
    wsize_hist = wsize_hist_from_counter(wsize_ctr, MAX_WSIZE_BIN)

    return {
        "query_id": int(query_id),
        "template_id": int(template_id),
        "query_name": query_name,
        "query_sql": query_sql,
        "monoids_hash": sorted(hb_set),
        "lineage_hash": sorted(ln_set),
        "witnesses": fullwhy_witnesses,
        "nrows": int(nrows),
        "nwitness_unique": int(nwitness_unique),
        "nwitness_total": int(nwitness_total),
        "ntokens_unique": int(ntokens_unique),
        "ntokens_total": int(ntokens_total),
        "avg_w": float(avg_w),
        "token_tf": token_tf,
        "witness_tf": witness_tf,
        "wsize_hist": [int(x) for x in wsize_hist],
    }


# ============================================================
# SPLIT / WRITE
# ============================================================


def split_rows_by_template(
    rows: List[Dict[str, Any]],
    test_per_template: int,
    split_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_template: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_template[int(row["template_id"])].append(row)

    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for tid, group in sorted(by_template.items()):
        ordered = sorted(group, key=lambda r: row_sort_key(r, split_seed))
        n_test = min(test_per_template, len(ordered))
        test_rows.extend(ordered[:n_test])
        train_rows.extend(ordered[n_test:])
        print(
            f"[SPLIT] template={tid} total={len(group)} "
            f"test={n_test} train={len(group) - n_test}"
        )

    return train_rows, test_rows


def write_batches(
    rows: List[Dict[str, Any]],
    schema: pa.Schema,
    out_dir: str,
    batch_size: int,
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[WRITE] no rows to write into {out_dir}")
        return

    batch_idx = 1
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        out_file = os.path.join(out_dir, f"batch_{batch_idx:04d}.parquet")
        table = pa.Table.from_pylist(chunk, schema=schema)
        pq.write_table(table, out_file, compression="zstd")
        print(f"[WRITE] {out_file} ({len(chunk)} rows)")
        batch_idx += 1


# ============================================================
# PROCESS ONE QUERY FOLDER
# ============================================================


def process_one_template_dir(
    duck: DuckClient, template_dir: str, schema: pa.Schema
) -> None:
    print("\n" + "=" * 100)
    print(f"[DIR] {template_dir}")

    templates = load_templates(template_dir)
    prefix, out_dir = derive_prefix_and_outdir(template_dir)

    all_rows: List[Dict[str, Any]] = []

    for qid, (name, template_sql) in enumerate(templates):
        template_id = parse_template_id(name)
        print(f"[QUERY] {name} -> template {template_id}")

        cols, result_rows = duck.execute(template_sql)

        row = build_row(
            query_id=qid,
            query_name=name,
            template_id=template_id,
            query_sql=template_sql,
            cols=cols,
            result_rows=result_rows,
        )
        all_rows.append(row)

    print(f"[ROWS] total={len(all_rows)}")

    train_rows, test_rows = split_rows_by_template(
        all_rows,
        test_per_template=TEST_PER_TEMPLATE,
        split_seed=SPLIT_SEED,
    )

    train_dir = os.path.join(out_dir, f"{prefix}_train")
    test_dir = os.path.join(out_dir, f"{prefix}_test")

    write_batches(train_rows, schema, train_dir, BATCH_SIZE)
    write_batches(test_rows, schema, test_dir, BATCH_SIZE)

    print(f"[DONE] train -> {train_dir}")
    print(f"[DONE] test  -> {test_dir}")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    schema = pa.schema(
        [
            ("query_id", pa.int64()),
            ("template_id", pa.int64()),
            ("query_name", pa.string()),
            ("query_sql", pa.string()),
            ("monoids_hash", pa.list_(pa.int64())),
            ("lineage_hash", pa.list_(pa.int64())),
            ("witnesses", pa.list_(pa.list_(pa.int64()))),
            ("nrows", pa.int64()),
            ("nwitness_unique", pa.int64()),
            ("nwitness_total", pa.int64()),
            ("ntokens_unique", pa.int64()),
            ("ntokens_total", pa.int64()),
            ("avg_w", pa.float32()),
            (
                "token_tf",
                pa.list_(pa.struct([("token", pa.int64()), ("tf", pa.int32())])),
            ),
            (
                "witness_tf",
                pa.list_(pa.struct([("witness_hash", pa.int64()), ("tf", pa.int32())])),
            ),
            ("wsize_hist", pa.list_(pa.int32())),
        ]
    )

    template_dirs = find_template_dirs(ROOT_DIR, INPUT_SUFFIX)
    if not template_dirs:
        raise FileNotFoundError(
            f"No folders ending with '{INPUT_SUFFIX}' containing .sql files were found under {ROOT_DIR}"
        )

    print(f"[FOUND] {len(template_dirs)} template folder(s)")
    for d in template_dirs:
        print(f"  - {d}")

    duck = DuckClient(db_path=DUCKDB_PATH)
    duck.con.execute(f"SET schema '{DUCKDB_SCHEMA}';")

    try:
        for template_dir in template_dirs:
            process_one_template_dir(duck, template_dir, schema)
    finally:
        duck.close()

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
