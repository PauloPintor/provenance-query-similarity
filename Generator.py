#!/usr/bin/env python3
"""Generate the full provenance fields used in the template-identification article.

Based on the original generate_provenance_dataset.py and its Q84 correction.
No CountSketch, KMV, sampling, model features, or train/test splitting.
The annotated database is opened read-only; output must be a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_TEMPLATES = (3, 12, 13, 18, 20, 48, 66, 84, 89, 98)
FORMATS = ("witnesses", "single", "product", "sum-products")
MAX_WSIZE_BIN = 8


def token_to_hash_i64(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def witness_tuple_to_hash_i64(witness: tuple[int, ...]) -> int:
    payload = b"".join(int(t).to_bytes(8, "big", signed=True) for t in witness)
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def is_sequence(value: Any) -> bool:
    # DuckDB fetchmany returns LIST values as Python lists.
    return isinstance(value, (list, tuple))


def legacy_witness(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}")


def flat_tokens(value: Any) -> list[str]:
    """Read ONE concrete witness; never stringify nested lists as tokens."""
    if value is None:
        return []
    if is_sequence(value):
        tokens = []
        for token in value:
            if token is None:
                continue
            if is_sequence(token) or isinstance(token, dict):
                raise ValueError("Nested value inside a concrete witness; select an explicit provenance format")
            if legacy_witness(token):
                raise ValueError("A legacy witness string cannot be used as one token in a flat witness")
            text = str(token).strip()
            if text:
                tokens.append(text)
        return tokens
    if isinstance(value, dict):
        raise ValueError("Dictionary provenance is not supported")
    text = str(value).strip()
    if legacy_witness(text):
        return [part.strip() for part in text[1:-1].split(",") if part.strip()]
    return [text] if text else []


def witness_tokens_to_tuple(value: Any) -> tuple[int, ...]:
    tokens = flat_tokens(value)
    if len(tokens) != len(set(tokens)):
        raise ValueError(
            "Repeated token inside a witness. The article uses set-valued witnesses; "
            "refusing to silently remove repetitions or change the original counters."
        )
    witness = tuple(sorted(token_to_hash_i64(t) for t in tokens))
    if len(witness) != len(set(witness)):
        raise ValueError("Different tokens collide under the original 64-bit hash")
    return witness


def factor_choices(factor: Any) -> list[list[str]]:
    """A factor is a list of alternative concrete witnesses (or scalar tokens)."""
    if factor is None:
        return []
    values = list(factor) if is_sequence(factor) else [factor]
    return [tokens for value in values if (tokens := flat_tokens(value))]


def expanded_product(factors: Any, limit: int) -> Iterable[list[str]]:
    if not is_sequence(factors):
        raise ValueError("A product must be an explicit list of factors")
    if not factors:
        return
    choices = [factor_choices(factor) for factor in factors]
    count = math.prod(len(c) for c in choices)
    if count > limit:
        raise ValueError(f"Exact product contains {count} witnesses; limit is {limit}. No sampling is performed")
    # Do not deduplicate alternatives: occurrence frequencies must survive expansion.
    for selected in itertools.product(*choices):
        yield [token for alternative in selected for token in alternative]


def result_cell_to_witnesses(cell: Any, mode: str, max_expansion: int) -> Iterable[list[str]]:
    if mode not in FORMATS:
        raise ValueError(f"Unknown provenance format: {mode}")
    if cell is None:
        return
    if mode == "single":
        # Q84 has one flat witness per output row. Accept an explicit outer
        # wrapper only when it contains exactly that one concrete witness.
        value = cell
        if is_sequence(cell) and any(is_sequence(v) for v in cell if v is not None):
            non_null = [v for v in cell if v is not None]
            if len(non_null) != 1 or not is_sequence(non_null[0]):
                raise ValueError("Single-witness format received more than one witness or a mixed structure")
            value = non_null[0]
        tokens = flat_tokens(value)
        if tokens:
            yield tokens
        return
    if mode == "witnesses":
        if not is_sequence(cell):
            tokens = flat_tokens(cell)
            if tokens:
                yield tokens
            return
        for witness in cell:
            if witness is None:
                continue
            if not is_sequence(witness) and not legacy_witness(witness):
                raise ValueError("Flat token list in collection format; use 'single' for one witness per row")
            tokens = flat_tokens(witness)
            if tokens:
                yield tokens
        return
    if mode == "product":
        yield from expanded_product(cell, max_expansion)
        return
    if not is_sequence(cell):
        raise ValueError("A sum of products must be a list of products")
    emitted = 0
    for product in cell:
        if product is None:
            continue
        for witness in expanded_product(product, max_expansion):
            emitted += 1
            if emitted > max_expansion:
                raise ValueError("Sum of products exceeds the exact expansion limit; no sampling is performed")
            yield witness


def find_witnesses_col(columns: list[str], requested: str | None = None) -> int:
    names = [requested.lower()] if requested else ["witnesses", "prov", "token"]
    matches = [i for i, name in enumerate(columns) if name.lower() in names]
    if len(matches) != 1:
        raise ValueError(f"Expected one provenance column, found {len(matches)}: {columns}")
    return matches[0]


def wsize_hist_from_counter(counter: Counter, max_bin: int = MAX_WSIZE_BIN) -> list[int]:
    hist = [0] * (max_bin + 1)
    for size, count in counter.items():
        if size > 0:
            hist[min(size, max_bin + 1) - 1] += count
    return hist


def summarise_cells(cells: Iterable[Any], mode: str, max_expansion: int = 1_000_000) -> dict[str, Any]:
    """Retain the original occurrence-counting policy before query-level dedup."""
    token_counter: Counter = Counter()
    witness_counter: Counter = Counter()
    size_counter: Counter = Counter()
    witnesses: dict[int, tuple[int, ...]] = {}
    token_identities: dict[int, str] = {}
    nrows = 0
    for row_number, cell in enumerate(cells, start=1):
        nrows += 1
        row_witnesses = 0
        try:
            for tokens in result_cell_to_witnesses(cell, mode, max_expansion):
                witness = witness_tokens_to_tuple(tokens)
                if not witness:
                    continue
                for token in tokens:
                    identity = token_to_hash_i64(token)
                    previous = token_identities.setdefault(identity, token)
                    if previous != token:
                        raise ValueError("Different tokens collide under the original 64-bit hash")
                identity = witness_tuple_to_hash_i64(witness)
                previous_witness = witnesses.setdefault(identity, witness)
                if previous_witness != witness:
                    raise ValueError("Different witnesses collide under the original 64-bit hash")
                row_witnesses += 1
                witness_counter[identity] += 1
                size_counter[len(witness)] += 1
                token_counter.update(witness)
            if mode == "single" and row_witnesses != 1:
                raise ValueError(f"Expected one witness for this output row, found {row_witnesses}")
        except ValueError as exc:
            raise ValueError(f"Result row {row_number}: {exc}") from exc

    total_witnesses = sum(witness_counter.values())
    total_tokens = sum(token_counter.values())
    if max([0, *token_counter.values(), *witness_counter.values(), *size_counter.values()]) > (1 << 31) - 1:
        raise OverflowError("Frequency exceeds the original Parquet int32 schema")
    record = {
        "monoids_hash": sorted(witnesses),
        "lineage_hash": sorted(token_counter),
        "witnesses": [list(w) for w in sorted(witnesses.values())],
        "nrows": nrows,
        "nwitness_unique": len(witnesses),
        "nwitness_total": total_witnesses,
        "ntokens_unique": len(token_counter),
        "ntokens_total": total_tokens,
        "avg_w": total_tokens / total_witnesses if total_witnesses else 0.0,
        "token_tf": [{"token": t, "tf": n} for t, n in sorted(token_counter.items())],
        "witness_tf": [{"witness_hash": w, "tf": n} for w, n in sorted(witness_counter.items())],
        "wsize_hist": wsize_hist_from_counter(size_counter),
    }
    if sum(record["wsize_hist"]) != total_witnesses:
        raise ValueError("Witness-size histogram disagrees with witness occurrences")
    if sum(size * n for size, n in size_counter.items()) != total_tokens:
        raise ValueError("Token and witness occurrence counters disagree")
    return record


def output_schema(pa: Any) -> Any:
    # Preserve the original names, order and Arrow types, including string template_id.
    return pa.schema([
        ("query_id", pa.int64()), ("template_id", pa.string()),
        ("query_name", pa.string()), ("query_sql", pa.string()),
        ("monoids_hash", pa.list_(pa.int64())), ("lineage_hash", pa.list_(pa.int64())),
        ("witnesses", pa.list_(pa.list_(pa.int64()))),
        ("nrows", pa.int64()), ("nwitness_unique", pa.int64()),
        ("nwitness_total", pa.int64()), ("ntokens_unique", pa.int64()),
        ("ntokens_total", pa.int64()), ("avg_w", pa.float32()),
        ("token_tf", pa.list_(pa.struct([("token", pa.int64()), ("tf", pa.int32())]))),
        ("witness_tf", pa.list_(pa.struct([("witness_hash", pa.int64()), ("tf", pa.int32())]))),
        ("wsize_hist", pa.list_(pa.int32())),
    ])


def process_query(connection: Any, query_id: int, template_id: int, name: str,
                  sql: str, mode: str, max_expansion: int,
                  provenance_column: str | None = None) -> dict[str, Any]:
    cursor = connection.execute(sql)
    column = find_witnesses_col([c[0] for c in cursor.description], provenance_column)

    def cells() -> Iterable[Any]:
        while rows := cursor.fetchmany(64):
            for row in rows:
                yield row[column]

    try:
        fields = summarise_cells(cells(), mode, max_expansion)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Q{template_id}, {name}: {exc}") from exc
    return {"query_id": query_id, "template_id": str(template_id),
            "query_name": name, "query_sql": sql, **fields}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Directory containing qN_versions")
    parser.add_argument("--db", type=Path, default=Path(os.environ.get("DUCKDB_PATH", "tpcds_prov.duckdb")))
    parser.add_argument("--schema", default="tpcds")
    parser.add_argument("--output-root", type=Path, default=None, help="New directory; default: DATA_ROOT/generated_provenance")
    parser.add_argument("--templates", nargs="+", type=positive_int, default=list(DEFAULT_TEMPLATES))
    parser.add_argument("--batch-size", type=positive_int, default=25)
    parser.add_argument("--format", action="append", default=[], metavar="N=FORMAT", help="Explicit override: witnesses, single, product, sum-products")
    parser.add_argument("--max-expansion", type=positive_int, default=1_000_000, help="Maximum exact factorised witness occurrences per cell; never sampled")
    parser.add_argument("--provenance-column", default=None, help="Otherwise require exactly one of witnesses/prov/token")
    args = parser.parse_args(argv)
    if len(set(args.templates)) != len(args.templates):
        parser.error("Duplicate template IDs")
    args.formats = {n: ("single" if n == 84 else "witnesses") for n in args.templates}
    seen = set()
    for spec in args.format:
        try:
            template, mode = spec.split("=", 1)
            template = int(template.removeprefix("q"))
        except ValueError:
            parser.error(f"Invalid format override: {spec}")
        if template not in args.formats or mode not in FORMATS or template in seen:
            parser.error(f"Invalid, repeated, or unselected format override: {spec}")
        args.formats[template] = mode
        seen.add(template)
    return args


def generate(args: argparse.Namespace) -> Path:
    root = args.data_root.resolve()
    database = args.db.resolve()
    output = (args.output_root or root / "generated_provenance").resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. No files replaced; select a new output directory")
    if not database.is_file():
        raise FileNotFoundError(f"Annotated DuckDB database not found: {database}")
    planned = []
    for template in args.templates:
        source = root / f"q{template}_versions"
        files = sorted(source.glob("*.sql"))
        if not files:
            raise FileNotFoundError(f"No SQL files in {source}")
        if output == source or source in output.parents:
            raise ValueError("Output must be separate from every SQL input directory")
        planned.append((template, files))


    schema = output_schema(pa)
    manifest: dict[str, Any] = {
        "generator": "journal-full-provenance-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "database": str(database), "database_schema": args.schema,
        "duckdb_version": duckdb.__version__, "pyarrow_version": pa.__version__,
        "formats": args.formats, "max_expansion": args.max_expansion,
        "hashing": "original blake2b-64 signed big-endian; sorted witness token hashes",
        "query_ids": "zero-based sorted SQL order within each template, as in the original generator",
        "partitioning": "none; existing train/test partitions are not modified",
        "templates": [],
    }
    connection = duckdb.connect(str(database), read_only=True)
    try:
        connection.execute('SET schema = ' + "'" + args.schema.replace("'", "''") + "'")
        output.mkdir(parents=True, exist_ok=False)
        for template, files in planned:
            target = output / f"q{template}_versions_full"
            target.mkdir()
            buffer = []
            batch_index = 1
            queries = []
            for query_id, path in enumerate(files):
                sql = path.read_text(encoding="utf-8")
                print(f"[Q{template}] {query_id + 1}/{len(files)} {path.name}", flush=True)
                record = process_query(connection, query_id, template, path.stem, sql,
                                       args.formats[template], args.max_expansion, args.provenance_column)
                buffer.append(record)
                queries.append({"query_name": path.stem, "query_id": query_id,
                                "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest()})
                if len(buffer) == args.batch_size or query_id == len(files) - 1:
                    destination = target / f"batch_{batch_index:04d}.parquet"
                    with destination.open("xb") as handle:
                        pq.write_table(pa.Table.from_pylist(buffer, schema=schema), handle, compression="zstd")
                    print(f"[WRITE] {destination} ({len(buffer)} queries)", flush=True)
                    buffer.clear()
                    batch_index += 1
            manifest["templates"].append({"template_id": template, "queries": queries,
                                           "rows": len(files), "batches": batch_index - 1})
        with (output / "_SUCCESS.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    finally:
        connection.close()
    print(f"[DONE] {output}. Full datasets generated; no train/test partition created.", flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        generate(args)
    except (ValueError, RuntimeError, OverflowError, OSError, ImportError) as exc:
        raise SystemExit(f"[STOP] {exc}\nA run without _SUCCESS.json is incomplete. Existing datasets were not replaced.") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
