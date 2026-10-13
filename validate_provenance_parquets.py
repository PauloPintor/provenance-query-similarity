#!/usr/bin/env python3
"""Validate materialised query-level provenance Parquets.

Check the final article dataset, including Q13 and the corrected Q84 format.
Recompute Lineage and Atomic Why from stored witnesses, and cross-check Token
TF, counts and histograms using the stored Witness TF multiplicities. Q84 must
have one witness occurrence per output row. SQL-file comparison is optional.

SQL-shape heuristics are advisory, not proof that materialisation is wrong.
This audit does not execute SQL or independently recover witness frequencies
from the database. Internal consistency alone does not prove SQL correctness.

The script is read-only with respect to the input Parquets. It writes a
per-template summary and a bounded issue report to the selected output folder.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import pyarrow.parquet as pq


DEFAULT_TEMPLATE_IDS = (3, 12, 13, 18, 20, 48, 66, 84, 89, 98)
REQUIRED_COLUMNS = (
    "query_id",
    "template_id",
    "query_name",
    "query_sql",
    "lineage_hash",
    "monoids_hash",
    "witnesses",
    "token_tf",
    "witness_tf",
    "nrows",
    "nwitness_unique",
    "nwitness_total",
    "ntokens_unique",
    "ntokens_total",
    "avg_w",
    "wsize_hist",
)
SUMMARY_FIELDS = (
    "template_id",
    "files",
    "sql_files",
    "rows",
    "internal_errors",
    "warnings",
    "standard_provenance_rows",
    "factorised_provenance_rows",
    "direct_witness_rows",
    "ambiguous_prov_rows",
    "missing_query_sql_rows",
    "missing_parquet_queries",
    "parquet_queries_without_sql",
    "duplicate_query_names",
    "sql_text_mismatches",
    "witnesses_with_duplicate_tokens",
    "mean_lineage_size",
    "mean_unique_witnesses",
    "mean_total_witnesses",
    "min_witness_size",
    "max_witness_size",
)
ISSUE_FIELDS = (
    "severity",
    "code",
    "template_id",
    "file",
    "row_index",
    "query_id",
    "query_name",
    "details",
)


@dataclass
class TemplateStats:
    files: set[str]
    sql_files: int = 0
    rows: int = 0
    internal_errors: int = 0
    warnings: int = 0
    standard_rows: int = 0
    factorised_rows: int = 0
    direct_witness_rows: int = 0
    ambiguous_rows: int = 0
    missing_sql_rows: int = 0
    missing_parquet_queries: int = 0
    parquet_queries_without_sql: int = 0
    duplicate_query_names: int = 0
    sql_text_mismatches: int = 0
    witnesses_with_duplicate_tokens: int = 0
    lineage_total: int = 0
    unique_witness_total: int = 0
    witness_occurrence_total: int = 0
    min_witness_size: int | None = None
    max_witness_size: int | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=".",
        help="Root containing q*_train and q*_test directories.",
    )
    parser.add_argument(
        "--template-ids",
        default=",".join(str(value) for value in DEFAULT_TEMPLATE_IDS),
        help="Comma-separated template identifiers to audit.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help=(
            "Input glob relative to --data-root. May be repeated. Defaults to "
            "q*_train/*.parquet and q*_test/*.parquet (non-recursive)."
        ),
    )
    parser.add_argument(
        "--check-sql-files",
        action="store_true",
        help="Also check query coverage and text against q*_versions SQL files.",
    )
    parser.add_argument(
        "--versions-pattern",
        default="**/q*_versions/*.sql",
        help="SQL glob relative to --data-root; used with --check-sql-files.",
    )
    parser.add_argument(
        "--output-dir",
        default="provenance_validation",
        help="New directory for validation_summary.csv and validation_issues.csv.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of rows decoded from Parquet at a time.",
    )
    parser.add_argument(
        "--max-issues-per-code",
        type=int,
        default=25,
        help="Maximum detailed rows retained for each issue code.",
    )
    parser.add_argument(
        "--max-rows-per-template",
        type=int,
        default=0,
        help="Optional quick-audit limit per template; zero checks every row.",
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.max_issues_per_code <= 0:
        parser.error("--batch-size and --max-issues-per-code must be positive")
    if args.max_rows_per_template < 0:
        parser.error("--max-rows-per-template must be non-negative")
    return args


def parse_template_ids(value: str) -> set[int]:
    result = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not result:
        raise ValueError("At least one template identifier is required")
    return result


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    return [int(item) for item in value]


def as_witnesses(value: Any) -> List[Tuple[int, ...]]:
    if value is None:
        return []
    result: List[Tuple[int, ...]] = []
    for witness in value:
        if witness is None:
            continue
        result.append(tuple(int(token) for token in witness))
    return result


def as_tf_pairs(value: Any, key: str) -> List[Tuple[int, int]]:
    if value is None:
        return []
    result: List[Tuple[int, int]] = []
    for item in value:
        if item is None:
            continue
        result.append((int(item[key]), int(item["tf"])))
    return result


def pairs_to_counter(pairs: Sequence[Tuple[int, int]]) -> Counter[int]:
    result: Counter[int] = Counter()
    for identifier, frequency in pairs:
        result[identifier] += frequency
    return result


def witness_hash_i64(witness: Iterable[int]) -> int:
    canonical = tuple(sorted(int(token) for token in witness))
    payload = b"".join(
        token.to_bytes(8, "big", signed=True) for token in canonical
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def expected_wsize_hist(
    witnesses_by_hash: Mapping[int, Tuple[int, ...]],
    witness_tf: Mapping[int, int],
    max_bin: int = 8,
) -> List[int]:
    histogram = [0] * (max_bin + 1)
    for identifier, witness in witnesses_by_hash.items():
        frequency = int(witness_tf.get(identifier, 0))
        size = len(witness)
        if frequency <= 0 or size <= 0:
            continue
        index = size - 1 if size <= max_bin else max_bin
        histogram[index] += frequency
    return histogram


def sql_shape(sql: str) -> Tuple[bool, bool, bool, bool]:
    """Return (standard, factorised, direct_witness, ambiguous) flags."""
    compact = " ".join(sql.lower().split())
    standard_nested = bool(
        re.search(r"array_agg\s*\(\s*list_value\s*\(", compact)
    )
    standard_singleton = bool(
        re.search(
            r"array_agg\s*\(\s*[a-z_][a-z0-9_]*\.prov\s*\)", compact
        )
    )
    factor_arrays = re.findall(
        r"array_agg\s*\(\s*[a-z_][a-z0-9_]*\.prov\s*\)", compact
    )
    combines_factor_columns = bool(
        re.search(r"list_value\s*\([^)]*\b[a-z][a-z0-9_]*\.prov", compact)
    )
    factorised = combines_factor_columns and len(factor_arrays) >= 2
    list_value_arguments = re.findall(r"list_value\s*\(([^()]*)\)", compact)
    direct_witness = (
        not standard_nested
        and not factorised
        and any(arguments.count(".prov") >= 2 for arguments in list_value_arguments)
    )
    standard = standard_nested or (standard_singleton and not factorised)
    ambiguous = factorised and bool(re.search(r"\bselect\s+\*", compact))
    return standard, factorised, direct_witness, ambiguous


def template_from_path(path: str) -> int | None:
    match = re.search(r"(?:^|/)q(\d+)_(?:train|test|versions_full)(?:_2)?(?:/|$)", path)
    return int(match.group(1)) if match else None


def discover_files(root: str, patterns: Sequence[str]) -> List[str]:
    files: set[str] = set()
    for pattern in patterns:
        files.update(glob.glob(os.path.join(root, pattern), recursive=True))
    return sorted(
        path
        for path in files
        if path.endswith(".parquet")
        and not re.search(r"_sk\d+\.parquet$", os.path.basename(path))
    )


def normalise_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split())


def discover_sql_catalog(
    root: str,
    pattern: str,
    template_ids: set[int],
) -> Dict[Tuple[int, str], Tuple[str, str]]:
    catalog: Dict[Tuple[int, str], Tuple[str, str]] = {}
    paths = sorted(set(glob.glob(os.path.join(root, pattern), recursive=True)))
    for path in paths:
        match = re.search(r"(?:^|/)q(\d+)_versions(?:/|$)", path)
        if not match:
            continue
        template_id = int(match.group(1))
        if template_id not in template_ids:
            continue
        query_name = Path(path).stem
        key = (template_id, query_name)
        if key in catalog:
            raise RuntimeError(
                f"Duplicate SQL file for Q{template_id}/{query_name}: "
                f"{catalog[key][0]} and {path}"
            )
        with open(path, "r", encoding="utf-8") as handle:
            catalog[key] = (path, handle.read())
    return catalog


def iter_rows(
    path: str, columns: Sequence[str], batch_size: int
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    row_index = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        columns=list(columns), batch_size=batch_size, use_threads=False
    ):
        for row in batch.to_pylist():
            yield row_index, row
            row_index += 1


class Reporter:
    def __init__(self, max_per_code: int) -> None:
        self.max_per_code = max_per_code
        self.counts: Counter[str] = Counter()
        self.rows: List[Dict[str, Any]] = []

    def add(
        self,
        severity: str,
        code: str,
        template_id: int,
        path: str,
        row_index: int,
        row: Mapping[str, Any],
        details: str,
    ) -> None:
        self.counts[code] += 1
        if self.counts[code] > self.max_per_code:
            return
        self.rows.append(
            {
                "severity": severity,
                "code": code,
                "template_id": template_id,
                "file": path,
                "row_index": row_index,
                "query_id": row.get("query_id", ""),
                "query_name": row.get("query_name", ""),
                "details": details,
            }
        )


def validate_row(
    row: Mapping[str, Any],
    path: str,
    row_index: int,
    template_id: int,
    stats: TemplateStats,
    reporter: Reporter,
    authoritative_sql: str | None,
    authoritative_sql_path: str | None,
    check_sql_files: bool = False,
) -> None:
    def issue(severity: str, code: str, details: str) -> None:
        reporter.add(
            severity, code, template_id, path, row_index, row, details
        )
        if severity == "ERROR":
            stats.internal_errors += 1
        else:
            stats.warnings += 1

    lineage_list = as_int_list(row.get("lineage_hash"))
    monoid_list = as_int_list(row.get("monoids_hash"))
    witnesses = as_witnesses(row.get("witnesses"))
    token_pairs = as_tf_pairs(row.get("token_tf"), "token")
    witness_pairs = as_tf_pairs(row.get("witness_tf"), "witness_hash")

    lineage = set(lineage_list)
    monoids = set(monoid_list)
    token_tf = pairs_to_counter(token_pairs)
    witness_tf = pairs_to_counter(witness_pairs)

    if len(lineage_list) != len(lineage):
        issue("ERROR", "DUPLICATE_LINEAGE", "lineage_hash contains duplicates")
    if len(monoid_list) != len(monoids):
        issue("ERROR", "DUPLICATE_MONOIDS", "monoids_hash contains duplicates")
    if len(token_pairs) != len({key for key, _ in token_pairs}):
        issue("ERROR", "DUPLICATE_TOKEN_TF", "token_tf repeats token identifiers")
    if len(witness_pairs) != len({key for key, _ in witness_pairs}):
        issue(
            "ERROR",
            "DUPLICATE_WITNESS_TF",
            "witness_tf repeats witness identifiers",
        )
    if any(frequency <= 0 for _, frequency in token_pairs):
        issue("ERROR", "INVALID_TOKEN_TF", "token_tf contains a non-positive value")
    if any(frequency <= 0 for _, frequency in witness_pairs):
        issue(
            "ERROR",
            "INVALID_WITNESS_TF",
            "witness_tf contains a non-positive value",
        )

    canonical_witnesses = [tuple(sorted(witness)) for witness in witnesses]
    if any(not witness for witness in canonical_witnesses):
        issue("ERROR", "EMPTY_WITNESS", "stored witness contains no tokens")
    witnesses_with_duplicate_tokens = sum(
        len(witness) != len(set(witness)) for witness in canonical_witnesses
    )
    if witnesses_with_duplicate_tokens:
        stats.witnesses_with_duplicate_tokens += witnesses_with_duplicate_tokens
        issue(
            "ERROR",
            "DUPLICATE_TOKEN_WITHIN_WITNESS",
            f"{witnesses_with_duplicate_tokens} witnesses contain repeated tokens",
        )
    if witnesses != canonical_witnesses:
        issue(
            "ERROR",
            "NON_CANONICAL_WITNESS",
            "at least one witness is not sorted canonically",
        )
    if len(canonical_witnesses) != len(set(canonical_witnesses)):
        issue("ERROR", "DUPLICATE_WITNESS", "witnesses contains duplicates")

    lineage_expected = {
        token for witness in canonical_witnesses for token in witness
    }
    if lineage != lineage_expected:
        issue(
            "ERROR",
            "LINEAGE_MISMATCH",
            f"stored={len(lineage)} recomputed={len(lineage_expected)}",
        )

    witnesses_by_hash: Dict[int, Tuple[int, ...]] = {}
    hash_collision = False
    for witness in canonical_witnesses:
        identifier = witness_hash_i64(witness)
        previous = witnesses_by_hash.get(identifier)
        if previous is not None and previous != witness:
            hash_collision = True
        witnesses_by_hash[identifier] = witness
    if hash_collision:
        issue(
            "ERROR",
            "WITNESS_HASH_COLLISION",
            "different stored witnesses share a 64-bit identifier",
        )

    monoids_expected = set(witnesses_by_hash)
    if monoids != monoids_expected:
        issue(
            "ERROR",
            "ATOMIC_WHY_MISMATCH",
            f"stored={len(monoids)} recomputed={len(monoids_expected)}",
        )
    if set(witness_tf) != monoids_expected:
        issue(
            "ERROR",
            "WITNESS_TF_KEYS_MISMATCH",
            f"stored={len(witness_tf)} expected={len(monoids_expected)}",
        )
    if set(token_tf) != lineage_expected:
        issue(
            "ERROR",
            "TOKEN_TF_KEYS_MISMATCH",
            f"stored={len(token_tf)} expected={len(lineage_expected)}",
        )

    token_tf_expected: Counter[int] = Counter()
    for identifier, witness in witnesses_by_hash.items():
        multiplicity = witness_tf.get(identifier, 0)
        for token in witness:
            token_tf_expected[token] += multiplicity
    if token_tf != token_tf_expected:
        different_keys = sum(
            token_tf[key] != token_tf_expected[key]
            for key in set(token_tf) | set(token_tf_expected)
        )
        issue(
            "ERROR",
            "TOKEN_TF_MISMATCH",
            f"{different_keys} token frequencies differ",
        )

    nwitness_unique = len(canonical_witnesses)
    nwitness_total = sum(witness_tf.values())
    ntokens_unique = len(lineage_expected)
    ntokens_total = sum(token_tf_expected.values())
    nrows = safe_int(row.get("nrows"))
    if nrows < 0:
        issue("ERROR", "INVALID_RESULT_ROW_COUNT", "nrows must be a non-negative integer")
    elif nrows == 0 and nwitness_total != 0:
        issue("ERROR", "PROVENANCE_WITHOUT_RESULT_ROWS", "nrows is zero but witnesses are present")
    if template_id == 84 and nrows >= 0 and nwitness_total != nrows:
        issue(
            "ERROR",
            "Q84_WITNESS_COUNT_MISMATCH",
            f"expected one witness occurrence per result row: nrows={nrows}, "
            f"witness occurrences={nwitness_total}; distinct witnesses may repeat across rows",
        )
    expected_scalars = {
        "nwitness_unique": nwitness_unique,
        "nwitness_total": nwitness_total,
        "ntokens_unique": ntokens_unique,
        "ntokens_total": ntokens_total,
    }
    for column, expected in expected_scalars.items():
        stored = safe_int(row.get(column))
        if stored != expected:
            issue(
                "ERROR",
                "SCALAR_MISMATCH",
                f"{column}: stored={stored} recomputed={expected}",
            )

    expected_avg = ntokens_total / nwitness_total if nwitness_total else 0.0
    try:
        stored_avg = float(row.get("avg_w"))
    except (TypeError, ValueError):
        stored_avg = float("nan")
    if not math.isfinite(stored_avg) or not math.isclose(
        stored_avg, expected_avg, rel_tol=1e-6, abs_tol=1e-5
    ):
        issue(
            "ERROR",
            "AVG_W_MISMATCH",
            f"stored={stored_avg:.8g} recomputed={expected_avg:.8g}",
        )

    stored_hist = as_int_list(row.get("wsize_hist"))
    expected_hist = expected_wsize_hist(witnesses_by_hash, witness_tf)
    if stored_hist != expected_hist:
        issue(
            "ERROR",
            "WSIZE_HIST_MISMATCH",
            f"stored={stored_hist} recomputed={expected_hist}",
        )

    path_template = template_from_path(path)
    if path_template is not None and path_template != template_id:
        issue(
            "ERROR",
            "PATH_TEMPLATE_MISMATCH",
            f"directory=Q{path_template} row=Q{template_id}",
        )

    stored_sql = str(row.get("query_sql") or "").strip()
    if not stored_sql:
        stats.missing_sql_rows += 1
        issue("WARNING", "MISSING_QUERY_SQL", "query_sql is empty")

    sql = stored_sql
    if check_sql_files and authoritative_sql is None:
        stats.parquet_queries_without_sql += 1
        issue(
            "ERROR",
            "SQL_FILE_NOT_FOUND",
            "no matching SQL file exists in the q*_versions directory",
        )
    elif check_sql_files and authoritative_sql is not None:
        sql = authoritative_sql
        if stored_sql and normalise_sql(stored_sql) != normalise_sql(authoritative_sql):
            stats.sql_text_mismatches += 1
            issue(
                "ERROR",
                "QUERY_SQL_MISMATCH",
                f"stored query_sql differs from {authoritative_sql_path}",
            )

    if sql:
        standard, factorised, direct_witness, ambiguous = sql_shape(sql)
        if standard:
            stats.standard_rows += 1
        if factorised:
            stats.factorised_rows += 1
            issue(
                "WARNING",
                "FACTORISED_PROVENANCE_SQL",
                "SQL-shape heuristic suggests aggregated factors. Check the generator's "
                "explicit provenance format; this is not proof of a materialisation error",
            )
        if direct_witness:
            stats.direct_witness_rows += 1
            if template_id != 84:
                issue(
                    "WARNING",
                    "DIRECT_WITNESS_SQL",
                    "SQL-shape heuristic suggests one flat witness per result row; "
                    "check the explicit generator format for this template",
                )
        if ambiguous:
            stats.ambiguous_rows += 1
            issue(
                "WARNING",
                "AMBIGUOUS_PROV_COLUMNS",
                "SQL-shape heuristic suggests multiple provenance columns; check "
                "which column the generator selected",
            )

    stats.rows += 1
    stats.lineage_total += len(lineage)
    stats.unique_witness_total += nwitness_unique
    stats.witness_occurrence_total += nwitness_total
    sizes = [len(witness) for witness in canonical_witnesses]
    if sizes:
        row_min = min(sizes)
        row_max = max(sizes)
        stats.min_witness_size = (
            row_min
            if stats.min_witness_size is None
            else min(stats.min_witness_size, row_min)
        )
        stats.max_witness_size = (
            row_max
            if stats.max_witness_size is None
            else max(stats.max_witness_size, row_max)
        )


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    template_ids = parse_template_ids(args.template_ids)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; nothing overwritten: {output_dir}")
    patterns = args.patterns or (
        "q*_train/*.parquet",
        "q*_test/*.parquet",
    )
    files = discover_files(args.data_root, patterns)
    if not files:
        raise RuntimeError(f"No input Parquets found under {args.data_root!r}")
    sql_catalog = discover_sql_catalog(
        args.data_root, args.versions_pattern, template_ids
    ) if args.check_sql_files else {}

    stats_by_template: Dict[int, TemplateStats] = {
        template_id: TemplateStats(files=set()) for template_id in template_ids
    }
    for template_id in template_ids:
        stats_by_template[template_id].sql_files = sum(
            key[0] == template_id for key in sql_catalog
        )
    reporter = Reporter(args.max_issues_per_code)
    seen_queries: Counter[Tuple[int, str]] = Counter()

    for path in files:
        path_template = template_from_path(path)
        if path_template is not None and path_template not in template_ids:
            continue
        schema_names = set(pq.read_schema(path).names)
        missing = sorted(set(REQUIRED_COLUMNS) - schema_names)
        if missing:
            path_template = template_from_path(path)
            if path_template in template_ids:
                stats = stats_by_template[path_template]
                stats.files.add(path)
                stats.internal_errors += 1
                reporter.add(
                    "ERROR",
                    "MISSING_COLUMNS",
                    path_template,
                    path,
                    -1,
                    {},
                    ",".join(missing),
                )
            continue

        for row_index, row in iter_rows(path, REQUIRED_COLUMNS, args.batch_size):
            template_id = safe_int(row.get("template_id"))
            if template_id not in template_ids:
                if path_template in template_ids:
                    stats = stats_by_template[path_template]
                    stats.files.add(path)
                    stats.internal_errors += 1
                    reporter.add(
                        "ERROR", "INVALID_TEMPLATE_ID", path_template, path,
                        row_index, row, f"directory=Q{path_template}, row template_id={row.get('template_id')!r}",
                    )
                continue
            stats = stats_by_template[template_id]
            if (
                args.max_rows_per_template > 0
                and stats.rows >= args.max_rows_per_template
            ):
                continue
            stats.files.add(path)
            query_name = str(row.get("query_name") or "")
            if not query_name.strip():
                stats.internal_errors += 1
                reporter.add(
                    "ERROR", "EMPTY_QUERY_NAME", template_id, path, row_index, row,
                    "query_name must identify the query within its template",
                )
            query_key = (template_id, query_name)
            seen_queries[query_key] += 1
            if seen_queries[query_key] > 1:
                stats.duplicate_query_names += 1
                stats.internal_errors += 1
                reporter.add(
                    "ERROR",
                    "DUPLICATE_QUERY_NAME",
                    template_id,
                    path,
                    row_index,
                    row,
                    "query_name appears more than once across train/test Parquets",
                )
            sql_record = sql_catalog.get(query_key)
            validate_row(
                row,
                path,
                row_index,
                template_id,
                stats,
                reporter,
                None if sql_record is None else sql_record[1],
                None if sql_record is None else sql_record[0],
                check_sql_files=args.check_sql_files,
            )

    if args.check_sql_files and args.max_rows_per_template == 0:
        for template_id in sorted(template_ids):
            expected_names = {
                query_name
                for current_template, query_name in sql_catalog
                if current_template == template_id
            }
            observed_names = {
                query_name
                for (current_template, query_name), count in seen_queries.items()
                if current_template == template_id and count > 0
            }
            missing_names = sorted(expected_names - observed_names)
            if missing_names:
                stats = stats_by_template[template_id]
                stats.missing_parquet_queries = len(missing_names)
                stats.internal_errors += 1
                reporter.add(
                    "ERROR",
                    "SQL_QUERY_MISSING_FROM_PARQUETS",
                    template_id,
                    os.path.join(args.data_root, f"q{template_id}_versions"),
                    -1,
                    {},
                    f"{len(missing_names)} missing; first={missing_names[:10]}",
                )

    summary_rows: List[Dict[str, Any]] = []
    for template_id in sorted(template_ids):
        stats = stats_by_template[template_id]
        divisor = stats.rows or 1
        summary_rows.append(
            {
                "template_id": template_id,
                "files": len(stats.files),
                "sql_files": stats.sql_files,
                "rows": stats.rows,
                "internal_errors": stats.internal_errors,
                "warnings": stats.warnings,
                "standard_provenance_rows": stats.standard_rows,
                "factorised_provenance_rows": stats.factorised_rows,
                "direct_witness_rows": stats.direct_witness_rows,
                "ambiguous_prov_rows": stats.ambiguous_rows,
                "missing_query_sql_rows": stats.missing_sql_rows,
                "missing_parquet_queries": stats.missing_parquet_queries,
                "parquet_queries_without_sql": stats.parquet_queries_without_sql,
                "duplicate_query_names": stats.duplicate_query_names,
                "sql_text_mismatches": stats.sql_text_mismatches,
                "witnesses_with_duplicate_tokens": (
                    stats.witnesses_with_duplicate_tokens
                ),
                "mean_lineage_size": f"{stats.lineage_total / divisor:.6f}",
                "mean_unique_witnesses": (
                    f"{stats.unique_witness_total / divisor:.6f}"
                ),
                "mean_total_witnesses": (
                    f"{stats.witness_occurrence_total / divisor:.6f}"
                ),
                "min_witness_size": (
                    "" if stats.min_witness_size is None else stats.min_witness_size
                ),
                "max_witness_size": (
                    "" if stats.max_witness_size is None else stats.max_witness_size
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "validation_summary.csv"
    issues_path = output_dir / "validation_issues.csv"
    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_csv(issues_path, ISSUE_FIELDS, reporter.rows)

    print(
        f"[INPUT] parquet_files={len(files)} sql_files={len(sql_catalog)} "
        f"templates={sorted(template_ids)}"
    )
    print(f"[SQL FILE COMPARISON] {'enabled' if args.check_sql_files else 'disabled'}")
    if args.max_rows_per_template:
        print("[PARTIAL AUDIT] Coverage and duplicates outside the selected rows are not checked")
    print(f"[OUTPUT] summary={summary_path}")
    print(f"[OUTPUT] issues={issues_path}")
    print()
    print(
        "template rows internal_errors warnings "
        "factorised ambiguous"
    )
    for row in summary_rows:
        print(
            f"Q{int(row['template_id']):<3} "
            f"{int(row['rows']):>5} "
            f"{int(row['internal_errors']):>15} "
            f"{int(row['warnings']):>8} "
            f"{int(row['factorised_provenance_rows']):>10} "
            f"{int(row['ambiguous_prov_rows']):>9}"
        )

    total_errors = sum(stats.internal_errors for stats in stats_by_template.values())
    missing_templates = [
        template_id
        for template_id, stats in sorted(stats_by_template.items())
        if stats.rows == 0
    ]
    if missing_templates:
        print(f"[FAIL] no rows found for templates: {missing_templates}")
    if total_errors or missing_templates:
        print(f"[FAIL] internal_errors={total_errors}")
        return 1
    total_warnings = sum(stats.warnings for stats in stats_by_template.values())
    print(f"[OK] selected rows passed consistency checks; warnings={total_warnings}")
    print("[SCOPE] Stored witness frequencies were cross-checked, not verified against SQL execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
