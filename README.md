# Data Provenance-based Query Similarity

This repository contains the code used in the experiments for the paper  
**“Data Provenance-based Query Similarity”**.

The experiments evaluate whether **data provenance representations** can be used to measure similarity between SQL queries and classify them according to their **query template**.

The experiments use **TPC-DS queries executed on DuckDB**.

---

# Requirements

Python ≥ 3.10

Install dependencies:

pip install duckdb pyarrow numpy scikit-learn xgboost matplotlib seaborn

---

# Database

A **TPC-DS database in DuckDB** is required.

Example database file:

tpcds.duckdb

Set the database path:

export DUCKDB_PATH=tpcds.duckdb

---

# Query Templates

SQL query templates must be organized in folders named:

q1_versions  
q2_versions  
q3_versions  
...

Each folder contains multiple `.sql` files corresponding to different query instances.

Each query must return a column named:

witnesses

Each element of `witnesses` represents a **why-provenance witness** encoded as a list of provenance tokens.

---

# Running the pipeline

Run the following scripts in order.

## 1. Add provenance tokens to the database tables

python add_prov_cols.py

This adds a column `prov` to all TPC-DS tables and fills it with short identifiers such as:

SS1, SS2, SS3 ...  
C1, C2, C3 ...

---

## 2. Generate provenance features and create train/test datasets

python gen_parquet.py

This script:

- finds all folders ending with `_versions`
- executes the SQL queries in DuckDB
- extracts provenance features
- splits queries by template
- writes train/test parquet datasets

For each query folder:

250 queries per template → test set  
remaining queries → train set

Example output:

q1_split/  
&nbsp;&nbsp;&nbsp;&nbsp;q1_train/  
&nbsp;&nbsp;&nbsp;&nbsp;q1_test/

q2_split/  
&nbsp;&nbsp;&nbsp;&nbsp;q2_train/  
&nbsp;&nbsp;&nbsp;&nbsp;q2_test/

---

## 3. Run the experiments

python query_similarity.py

The script loads all parquet datasets and evaluates several provenance representations and machine learning models for query classification.
