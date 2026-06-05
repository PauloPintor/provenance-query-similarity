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

# Machine Learning Methods and Configurations

All models are implemented using standard, well-established libraries (`scikit-learn` and `XGBoost`). 

To ensure a fair baseline and straightforward reproducibility, **most classifiers rely on their default hyperparameter configurations**. A global random seed (`RANDOM_SEED = 13`) is strictly enforced across all models and data sampling functions to guarantee deterministic results across different execution runs.

## 1. Model Specifications

* **k-Nearest Neighbours (k-NN):** * The models were evaluated using both **k = 1 and k = 5** neighbours (adjustable via `SKLEARN_KNN_K` and `CUSTOM_KNN_K` in the provided script).
    * For vector-based feature representations, we utilise `scikit-learn`'s `NearestNeighbors` configured with the `cosine` metric.
    * For set-based and blocked representations (where direct set comparisons like Jaccard or soft-matching are required), we use a custom implementation that retrieves the top-k neighbours and applies similarity-weighted voting to determine the final class.
* **Logistic Regression:** 
    * Implemented via `scikit-learn`'s `LogisticRegression`. 
    * Runs with default parameters (L2 penalty, L-BFGS solver) and is strictly governed by the global random seed.
* **Multinomial Naive Bayes:** 
    * Implemented using `MultinomialNB()`. 
    * Kept entirely at its default settings, making it a fast, lightweight baseline for the frequency-based (TF) and hashed representations.
* **Random Forest Classifier:** 
    * Uses `scikit-learn`'s `RandomForestClassifier`. 
    * Configured with the default parameters (e.g., 100 trees, Gini impurity) and tied to the global random seed. It operates on dense matrices internally.
* **XGBoost:** 
    * Implemented using the `XGBClassifier` from the `xgboost` Python package. 
    * Relies entirely on the default gradient boosting parameters (e.g., learning rate of 0.3, max depth of 6) whilst enforcing the global random seed.

## 2. Feature Preprocessing & Scaling

To ensure the classifiers perform optimally, specific preprocessing pipelines are applied dynamically based on the model and feature type:

* **Standardisation:** For distance-based and linear models (k-NN and Logistic Regression), the feature vectors are scaled using `scikit-learn`'s `StandardScaler`. It automatically adapts to sparse matrices by skipping the mean centring (`with_mean=False`) to preserve sparsity.
* **TF-IDF Transformation:** For representations evaluating token or witness term frequencies (`token_tfidf` and `witness_tfidf`), raw frequency counts are transformed using `TfidfTransformer()` before being passed to the classifiers.
* **Feature Hashing:** High-dimensional provenance sets are mapped into fixed-dimensional vectors using a custom, deterministic 64-bit hashing function. The target dimensions are `32,768` (for lineage and token TF) and `16,384` (for atomic why and witness TF).

--

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
