"""
TPC-DS query template definitions and instance generation.

This module provides:

- ``TPCDS_TEMPLATES`` – a catalogue of parameterised SQL query templates
  derived from the TPC-DS benchmark decision-support workload.  Each template
  is identified by its official TPC-DS query number and includes one or more
  named parameter slots.

- :func:`generate_query_instances` – fills parameter slots in a template to
  produce concrete query strings.

- :func:`load_query_dataset` – returns a dataset of (query_sql, template_id)
  pairs suitable for provenance extraction and ML experiments.

Note
----
The templates below are simplified, self-contained SQL statements that capture
the *structure* of TPC-DS queries while remaining executable against small
in-memory databases.  They are not verbatim copies of the official TPC-DS
benchmark text.
"""

from __future__ import annotations

import random
import string
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Template catalogue
# ---------------------------------------------------------------------------

#: Each entry: template_id → {"sql": parameterised SQL, "params": {name: [values]}}
TPCDS_TEMPLATES: Dict[int, Dict[str, Any]] = {
    1: {
        "description": "Customer with above-average returns",
        "sql": (
            "SELECT c_customer_id, c_first_name, c_last_name, "
            "       ca_city, sum(cr_return_amount) AS total_returns "
            "FROM customer "
            "JOIN customer_address ON c_current_addr_sk = ca_address_sk "
            "JOIN customer_returns  ON cr_customer_sk = c_customer_sk "
            "JOIN date_dim          ON cr_returned_date_sk = d_date_sk "
            "WHERE d_year = {d_year} "
            "  AND ca_state = '{ca_state}' "
            "GROUP BY c_customer_id, c_first_name, c_last_name, ca_city "
            "HAVING sum(cr_return_amount) > "
            "       (SELECT avg(total) FROM ("
            "           SELECT cr_customer_sk, sum(cr_return_amount) AS total "
            "           FROM customer_returns "
            "           JOIN date_dim ON cr_returned_date_sk = d_date_sk "
            "           WHERE d_year = {d_year} AND ca_state = '{ca_state}'"
            "           GROUP BY cr_customer_sk"
            "       ) sub) "
            "ORDER BY c_customer_id "
            "LIMIT {limit}"
        ),
        "params": {
            "d_year": list(range(1998, 2003)),
            "ca_state": list(string.ascii_uppercase[:20]),  # A–T: representative US state codes
            "limit": [100, 200, 500],
        },
    },
    3: {
        "description": "Brand and category promotion analysis",
        "sql": (
            "SELECT dt.d_year, item.i_brand_id, item.i_brand, "
            "       sum(ss_ext_sales_price) AS sum_agg "
            "FROM date_dim dt "
            "JOIN store_sales   ON dt.d_date_sk = store_sales.ss_sold_date_sk "
            "JOIN item          ON store_sales.ss_item_sk = item.i_item_sk "
            "WHERE item.i_manufact_id = {i_manufact_id} "
            "  AND dt.d_moy = {d_moy} "
            "GROUP BY dt.d_year, item.i_brand, item.i_brand_id "
            "ORDER BY dt.d_year, sum_agg DESC, item.i_brand_id "
            "LIMIT {limit}"
        ),
        "params": {
            "i_manufact_id": list(range(128, 145)),
            "d_moy": list(range(1, 13)),
            "limit": [100, 200],
        },
    },
    7: {
        "description": "Average quantity, list price, discount, and sales price",
        "sql": (
            "SELECT i_item_id, "
            "       avg(ss_quantity)        AS agg1, "
            "       avg(ss_list_price)      AS agg2, "
            "       avg(ss_coupon_amt)      AS agg3, "
            "       avg(ss_sales_price)     AS agg4 "
            "FROM store_sales "
            "JOIN customer_demographics ON ss_cdemo_sk = cd_demo_sk "
            "JOIN date_dim              ON ss_sold_date_sk = d_date_sk "
            "JOIN item                  ON ss_item_sk = i_item_sk "
            "JOIN promotion             ON ss_promo_sk = p_promo_sk "
            "WHERE cd_gender = '{cd_gender}' "
            "  AND cd_marital_status = '{cd_marital_status}' "
            "  AND cd_education_status = '{cd_education_status}' "
            "  AND (p_channel_email = 'N' OR p_channel_event = 'N') "
            "  AND d_year = {d_year} "
            "GROUP BY i_item_id "
            "ORDER BY i_item_id "
            "LIMIT {limit}"
        ),
        "params": {
            "cd_gender": ["M", "F"],
            "cd_marital_status": ["S", "M", "D", "W", "U"],
            "cd_education_status": [
                "Primary", "Secondary", "College", "Advanced Degree"
            ],
            "d_year": list(range(1998, 2003)),
            "limit": [100],
        },
    },
    19: {
        "description": "Top-10 items by ext_sales_price in category",
        "sql": (
            "SELECT i_brand_id, i_brand, i_manufact_id, i_manufact, "
            "       sum(ss_ext_sales_price) AS ext_price "
            "FROM date_dim "
            "JOIN store_sales ON d_date_sk = ss_sold_date_sk "
            "JOIN item        ON ss_item_sk = i_item_sk "
            "JOIN customer    ON ss_customer_sk = c_customer_sk "
            "JOIN customer_address ON c_current_addr_sk = ca_address_sk "
            "JOIN store            ON ss_store_sk = s_store_sk "
            "WHERE i_manager_id = {i_manager_id} "
            "  AND d_moy = {d_moy} "
            "  AND d_year = {d_year} "
            "  AND substr(ca_zip, 1, 5) <> substr(s_zip, 1, 5) "
            "GROUP BY i_brand_id, i_brand, i_manufact_id, i_manufact "
            "ORDER BY ext_price DESC, i_brand, i_brand_id, i_manufact_id, i_manufact "
            "LIMIT {limit}"
        ),
        "params": {
            "i_manager_id": list(range(1, 100)),
            "d_moy": list(range(11, 13)),
            "d_year": [1998, 1999, 2000, 2001, 2002],
            "limit": [100],
        },
    },
    42: {
        "description": "Items with above-average profit margin by year",
        "sql": (
            "SELECT dt.d_year, item.i_category_id, item.i_category, "
            "       sum(ss_ext_sales_price) "
            "FROM date_dim dt "
            "JOIN store_sales ON dt.d_date_sk = ss_sold_date_sk "
            "JOIN item        ON ss_item_sk = i_item_sk "
            "WHERE item.i_manager_id = {i_manager_id} "
            "  AND dt.d_moy = {d_moy} "
            "  AND dt.d_year = {d_year} "
            "GROUP BY dt.d_year, item.i_category_id, item.i_category "
            "ORDER BY sum(ss_ext_sales_price) DESC, dt.d_year, "
            "         item.i_category_id, item.i_category "
            "LIMIT {limit}"
        ),
        "params": {
            "i_manager_id": list(range(1, 100)),
            "d_moy": list(range(11, 13)),
            "d_year": [1998, 1999, 2000, 2001, 2002],
            "limit": [100],
        },
    },
}


# ---------------------------------------------------------------------------
# Query instance generation
# ---------------------------------------------------------------------------


def generate_query_instances(
    template_id: int,
    n_instances: int = 10,
    *,
    rng: Optional[random.Random] = None,
) -> List[Tuple[str, int]]:
    """Generate *n_instances* concrete SQL strings from a TPC-DS template.

    Parameters
    ----------
    template_id:
        Key into :data:`TPCDS_TEMPLATES`.
    n_instances:
        Number of query instances to generate.
    rng:
        Optional :class:`random.Random` instance for reproducibility.

    Returns
    -------
    list of (sql_string, template_id) tuples
    """
    if template_id not in TPCDS_TEMPLATES:
        raise ValueError(
            f"Template {template_id} not found. "
            f"Available: {sorted(TPCDS_TEMPLATES)}"
        )
    template = TPCDS_TEMPLATES[template_id]
    sql_template: str = template["sql"]
    param_ranges: Dict[str, list] = template["params"]

    if rng is None:
        rng = random.Random(42)

    instances = []
    for _ in range(n_instances):
        params = {k: rng.choice(v) for k, v in param_ranges.items()}
        instances.append((sql_template.format(**params), template_id))
    return instances


def load_query_dataset(
    template_ids: Optional[List[int]] = None,
    n_instances_per_template: int = 20,
    *,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """Build a synthetic query dataset from TPC-DS templates.

    Parameters
    ----------
    template_ids:
        Which templates to include.  Defaults to all templates in
        :data:`TPCDS_TEMPLATES`.
    n_instances_per_template:
        Number of concrete query instances generated per template.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (queries, labels)
        ``queries`` is a list of SQL strings;
        ``labels`` is a list of integer template IDs.
    """
    if template_ids is None:
        template_ids = sorted(TPCDS_TEMPLATES)

    rng = random.Random(seed)
    queries: List[str] = []
    labels: List[int] = []
    for tid in template_ids:
        instances = generate_query_instances(
            tid, n_instances=n_instances_per_template, rng=rng
        )
        for sql, label in instances:
            queries.append(sql)
            labels.append(label)
    return queries, labels
