#!/usr/bin/env python3

import os
import duckdb

DUCKDB_PATH = os.environ.get("DUCKDB_PATH", "tpcds.duckdb")
SCHEMA = os.environ.get("DUCKDB_SCHEMA", "tpcds")

# Short prefixes for TPC-DS base tables
PREFIXES = {
    "call_center": "CC",
    "catalog_page": "CP",
    "catalog_returns": "CR",
    "catalog_sales": "CS",
    "customer": "C",
    "customer_address": "CA",
    "customer_demographics": "CD",
    "date_dim": "D",
    "household_demographics": "HD",
    "income_band": "IB",
    "inventory": "INV",
    "item": "I",
    "promotion": "P",
    "reason": "R",
    "ship_mode": "SM",
    "store": "S",
    "store_returns": "SR",
    "store_sales": "SS",
    "time_dim": "T",
    "warehouse": "W",
    "web_page": "WP",
    "web_returns": "WR",
    "web_sales": "WS",
    "web_site": "WSI",
}


def table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    q = """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = ? AND table_name = ?
        LIMIT 1
    """
    return con.execute(q, [schema, table]).fetchone() is not None


def column_exists(
    con: duckdb.DuckDBPyConnection, schema: str, table: str, column: str
) -> bool:
    q = """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ? AND column_name = ?
        LIMIT 1
    """
    return con.execute(q, [schema, table, column]).fetchone() is not None


def add_and_fill_prov(
    con: duckdb.DuckDBPyConnection, schema: str, table: str, prefix: str
) -> None:
    fqtn = f"{schema}.{table}"

    if not column_exists(con, schema, table, "prov"):
        con.execute(f"ALTER TABLE {fqtn} ADD COLUMN prov VARCHAR")

    # Overwrites prov with short deterministic tokens such as SS1, SS2, ...
    con.execute(
        f"""
        UPDATE {fqtn} AS t
        SET prov = s.prov
        FROM (
            SELECT
                rowid,
                '{prefix}' || CAST(row_number() OVER (ORDER BY rowid) AS VARCHAR) AS prov
            FROM {fqtn}
        ) AS s
        WHERE t.rowid = s.rowid
        """
    )


def main() -> None:
    con = duckdb.connect(DUCKDB_PATH)

    print(f"[DB] {DUCKDB_PATH}")
    print(f"[SCHEMA] {SCHEMA}")

    for table, prefix in PREFIXES.items():
        if not table_exists(con, SCHEMA, table):
            print(f"[SKIP] {table} (table not found)")
            continue

        print(f"[OK] {table} -> {prefix}")
        add_and_fill_prov(con, SCHEMA, table, prefix)

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
