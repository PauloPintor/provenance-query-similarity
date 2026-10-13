# Query-Level Provenance for Query-Template Identification

This repository contains the experimental code for studying **query-template identification from query-level data provenance**. Tuple-level Lineage and Why-provenance are aggregated across each query result to characterise the contributing tuples, their witness combinations, occurrence frequencies and structural properties.

The classifiers use these provenance representations, rather than SQL text, execution plans or result values, to predict the template associated with a query. The evaluation compares classification accuracy, macro-F1 and mean test-processing time across different representations, classifiers and training-history sizes.

## Scripts

| Script | Purpose |
| --- | --- |
| `evaluate_provenance.py` | Main classification experiments, timing measurements, selected PSD attributions and individual ablations. |
| `validate_provenance_parquets.py` | Read-only consistency checks on the materialised provenance data. |
| `psd_feature_experiments.py` | Comparisons between PSD descriptor groups, individual feature removals and single-descriptor models. |
| `psd_logreg_convergence.py` | Additional Logistic Regression convergence checks for three PSD descriptor groups. |
| `add_prov_cols.py` | Adds and populates provenance-token columns in the TPC-DS base tables. |
| `generate_provenance_dataset.py` | Optional provenance materialisation from the supplied SQL queries and an annotated DuckDB database. |

The classification and PSD analyses read materialised train/test Parquets. Once these files are available locally, running the analyses does not require access to DuckDB or SQL execution. The Parquets are not included in this Git repository because of their total size; see [Data Availability](#data-availability) and [Database and Provenance Preparation](#database-and-provenance-preparation).

## Installation

Use Python 3.10 or newer. From the repository root, create an environment and install the evaluation dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The evaluation uses NumPy, PyArrow, SciPy, scikit-learn, XGBoost, Matplotlib and Seaborn. DuckDB is needed only for database annotation and provenance generation.

For exact experimental reproduction, use the original library versions. An unpinned `requirements.txt` installs the dependencies but does not recreate the original environment. PSD runs record their library versions and input fingerprints in `manifest.json`; the convergence analysis checks these against its reference run.

## Dataset

The TPC-DS database used in the experiments was generated using Python at **scale factor 1 (SF1)**. The workload contains **12,500 distinct query instances** from ten TPC-DS templates: **Q3, Q12, Q13, Q18, Q20, Q48, Q66, Q84, Q89 and Q98**. Each Parquet record represents one query execution and contains its template label and materialised provenance.

| Partition | Queries per template | Total queries |
| --- | ---: | ---: |
| Training | 1,000 | 10,000 |
| Test | 250 | 2,500 |

The main experiments use nested training subsets of 50, 100, 250, 500, 750 and 1,000 queries per template, corresponding to 500, 1,000, 2,500, 5,000, 7,500 and 10,000 queries in total. The test set remains fixed. Subset selection is deterministic, with seed 13.

### Data Availability

The materialised provenance Parquets are excluded from this Git repository because of their total size. The directory layout below describes the local inputs expected by the scripts, not data bundled with the repository. The provenance generator can produce Parquets from the SQL workload and an annotated database, but does not recreate the final train/test partitions automatically.

The data root should contain the final partitions in this layout:

```text
q3_train/
    *.parquet
q3_test/
    *.parquet
...
q98_train/
    *.parquet
q98_test/
    *.parquet
```

Records include query identifiers, stored SQL, `lineage_hash`, `monoids_hash`, `witnesses`, `token_tf`, `witness_tf`, and provenance counts. The four counts used to construct PSD are `nwitness_unique`, `nwitness_total`, `ntokens_unique` and `ntokens_total`.

Use the existing partitions when reproducing the experiments. Neither the evaluator nor the PSD scripts creates a new train/test split.

### Configure Paths

The commands below assume the `qN_train` and `qN_test` directories are in the repository root. If they are under `data/`, change the first value to `./data`.

```bash
export PROV_DATA_ROOT="."
export PROV_TRAIN_GLOB="$PROV_DATA_ROOT/q*_train/*.parquet"
export PROV_TEST_GLOB="$PROV_DATA_ROOT/q*_test/*.parquet"
export PROV_OUTPUT_DIR="./results/evaluation"
```

These non-recursive patterns exclude directories such as `nn_q13_train` and nested backups. Keep only the final article partitions under matching directory names. The evaluator's broader recursive defaults remain active if the glob overrides are omitted.

The validator and PSD scripts receive `--data-root` explicitly. Their default input patterns also select `q*_train` and `q*_test` directly under that root.

## 1. Validate the Data

Run the validator before starting the experiments:

```bash
python validate_provenance_parquets.py \
  --data-root "$PROV_DATA_ROOT" \
  --output-dir ./results/validation
```

It checks agreement between the stored witnesses, Lineage, witness identifiers, term frequencies, counts and witness-size histograms. For Q84, it additionally checks that the total number of witness occurrences equals the number of output rows, consistent with its single-witness-per-result-row format.

The validator writes `validation_summary.csv` and `validation_issues.csv`. It does not modify the Parquets, and its output directory must not already exist. Exit status 1 indicates validation errors or missing selected templates; status 0 indicates that the checked records passed, possibly with advisory warnings.

If the original `qN_versions/*.sql` files are also available, add `--check-sql-files` to check query coverage and compare stored SQL text with those files. Without that option, external SQL files are not required.

These are consistency checks, not a re-execution of the SQL queries. Witness TF multiplicities are cross-checked against the other stored fields, not independently recovered from the database. The validator also does not enforce the exact train/test partition sizes.

## 2. Run the Main Experiments

### One Configuration

To check the pipeline with PSD and XGBoost using 50 training queries per template:

```bash
python evaluate_provenance.py \
  --method scalars --model xgboost --train-k 50 --skip-plots
```

### Complete Evaluation

To run all supported representation/classifier combinations at all six training sizes:

```bash
python evaluate_provenance.py --skip-plots
```

Configurations run sequentially in separate processes. This includes the more expensive direct-comparison methods, particularly Blocked Why-provenance.

For example, to run only Blocked Why-provenance with 5-NN at the largest training size:

```bash
python evaluate_provenance.py \
  --method why_blocked --model knn --knn-k 5 --train-k 1000 --skip-plots
```

**`--train-k` always denotes queries per template**, not the total training-set size.

### Representations and Classifiers

| `--method` | Representation | Supported classifiers |
| --- | --- | --- |
| `lineage_jaccard` | Direct Lineage | 1-NN, 5-NN |
| `atomic_why_jaccard` | Direct Atomic Why-provenance | 1-NN, 5-NN |
| `why_blocked` | Blocked Why-provenance | 1-NN, 5-NN |
| `lineage_vec` | Binary Lineage | Supervised models |
| `atomic_why_vec` | Binary Atomic Why-provenance | Supervised models |
| `token_tfidf` | Token TF with TF-IDF weighting | 1-NN, 5-NN, supervised models |
| `witness_tfidf` | Witness TF with TF-IDF weighting | 1-NN, 5-NN, supervised models |
| `scalars` | Seven-descriptor PSD | 1-NN, 5-NN, supervised models |

The supervised model identifiers are `logreg`, `naive_bayes`, `random_forest` and `xgboost`. For nearest-neighbour methods, use `--model knn` and select `--knn-k 1` or `--knn-k 5`.

The main evaluation uses library-default model settings, with seed 13 where applicable. Token-based vectors use 32,768 hash coordinates; witness-based vectors use 16,384. Direct set comparisons do not use vector feature hashing. TF-IDF and standardisation are fitted on training data only.

### Outputs

Outputs are stored under `PROV_OUTPUT_DIR`:

| File or directory | Contents |
| --- | --- |
| `final_template_results.csv` | Accuracy, macro-F1 and timing components for each configuration. |
| `final_template_per_class_metrics.csv` | Precision, recall, F1 and support for each template. |
| `Predictions/` | Query identifiers and predictions for each configuration. |
| `XAI/` | Selected PSD attribution arrays, importance summaries and individual-ablation results. |
| `Figures/` | Built-in plots, when requested. |
| `final_template_errors.csv` | Failed configurations, when applicable. |

For PSD with Logistic Regression and XGBoost at `--train-k 1000`, attribution and individual-ablation artifacts are generated automatically. `--skip-plots` suppresses figures, not these analyses.

### Resume or Rerun

Repeating a command skips configurations with the results and artifacts checked by the evaluator. Add `--force` to rerun matching configurations and replace their outputs.

Use a **new output directory** when changing the data, split, evaluator or software environment. Main-evaluation resume keys identify the training size, representation and classifier, but do not fingerprint the data or library versions. Do not run multiple controllers against the same output directory.

### Timing Fields

| Field | Meaning | Unit |
| --- | --- | --- |
| `fit_s` | Time in the model-fitting stage, including standardisation fitted within that stage. | Seconds |
| `build_test_s` | Construction of the test representation, including TF-IDF transformation where applicable. | Seconds |
| `pred_s` | Prediction or direct-comparison scoring; includes fitted scaling for vector k-NN and Logistic Regression. | Seconds |
| `latency_ms` | Mean amortised test-processing time per query. | Milliseconds |

The CSV field `latency_ms` is computed as:

```text
latency_ms = 1000 * (build_test_s + pred_s) / number_of_test_queries
```

It excludes SQL execution, provenance generation, Parquet loading, training-representation construction and model fitting. It is neither end-to-end runtime nor isolated online prediction latency. `fit_s` does not include every training preprocessing operation; for example, training TF-IDF construction happens before this stage.

For Blocked Why-provenance, test records are streamed and scoring time is accumulated separately from loading. The preliminary per-query witness-cache preparation is outside the scoring timer. Construction of inverted indexes during scoring remains inside it.

## 3. Analyse PSD Descriptors

### Descriptor Experiments

`psd_feature_experiments.py` reuses the evaluator's PSD construction, training selection, preprocessing and model functions. By default, it compares descriptor groups using XGBoost, Logistic Regression, Random Forest and 5-NN; performs individual feature removals; and trains seven single-descriptor XGBoost models.

The principal groups are:

| Identifier | Descriptors |
| --- | --- |
| `counts_4` | Distinct witnesses, total witness occurrences, distinct tokens and total token occurrences. |
| `full_7` | The four counts plus distinct/total token ratio, distinct/total witness ratio and distinct-witness/distinct-token ratio. |
| `reduced_5` | `full_7` without total witness occurrences and the distinct/total witness ratio. |

The script also evaluates ratio-only, token-only, witness-only and no-witness-count groups. These additional configurations are recorded separately in its results.

After completing the main evaluation, run:

```bash
python psd_feature_experiments.py \
  --evaluation-script ./evaluate_provenance.py \
  --data-root "$PROV_DATA_ROOT" \
  --reference-results "$PROV_OUTPUT_DIR/final_template_results.csv" \
  --output-dir ./results/psd_features
```

The reference check compares full-PSD scores against the main evaluation before proceeding with reduced-feature experiments. Use a fresh output directory for every run.

Outputs include `psd_group_results.csv`, `psd_single_results.csv`, `psd_leave_one_out_results.csv`, per-template metrics, predictions and a `manifest.json` recording the configuration, code fingerprints, data fingerprints and software versions. PSD experiment timings are diagnostic and should not replace the main evaluator's test-processing measurements.

### Logistic Regression Convergence

`psd_logreg_convergence.py` is a separate verification for `full_7`, `counts_4` and `reduced_5`. It increases only `max_iter`, using training convergence to select a common iteration cap. Test scores are not used to choose that cap, and earlier results are not overwritten.

Keep both PSD scripts in the same directory, then run:

```bash
python psd_logreg_convergence.py \
  --evaluation-script ./evaluate_provenance.py \
  --data-root "$PROV_DATA_ROOT" \
  --reference-run ./results/psd_features \
  --output-dir ./results/psd_convergence
```

The reference run must be complete and must have used the **same evaluator file, PSD experiment script and library versions**. The script also checks the data fingerprints and reproduces the reference predictions before assessing convergence. Even comment-only changes to a reference script change its fingerprint; use the original files when checking an existing run.

Outputs include convergence attempts, original/converged score comparisons, predictions and a manifest. The script also produces `full_psd_result.csv` and attribution artifacts for the converged full-PSD model. These are separate from the main evaluation's default-parameter outputs.

## Plotting

The evaluator can regenerate its built-in plots from saved CSV and NPZ artifacts without loading Parquets or fitting models:

```bash
python evaluate_provenance.py --plots-only
```

These plots include selected confusion matrices, an accuracy curve, global and class-specific PSD importance, contribution scatter plots and individual-ablation plots. The built-in accuracy curve uses four configurations and a per-template training-size axis. The training-time figure and the final five-series accuracy figure require their separate plotting scripts; they are not produced by this command.

## Database and Provenance Preparation

These steps are needed when generating provenance from SQL. They can be skipped when the final train/test Parquets are already available locally.

### 1. Prepare the TPC-DS Database

The experiments used a TPC-DS database generated using Python at SF1. The commands below assume that the TPC-DS base tables already exist in a DuckDB database. `add_prov_cols.py` annotates these tables; it does not generate the TPC-DS data.

```bash
python -m pip install duckdb
export DUCKDB_PATH="./tpcds.duckdb"
export DUCKDB_SCHEMA="tpcds"
```

Adjust the database path and schema to the existing database. These environment variables configure `add_prov_cols.py`; their defaults are `tpcds.duckdb` and `tpcds`, respectively.

### 2. Add Provenance Tokens

```bash
python add_prov_cols.py
```

For each TPC-DS base table present in the selected schema, the script adds a `prov VARCHAR` column if it does not already exist, then assigns a token to every row. Tokens combine a table-specific prefix with a sequential number, for example `C1` for a customer tuple and `SS1` for a store-sales tuple. Numbering follows the table's `rowid` order. Missing tables are reported and skipped.

**This script modifies the database in place and overwrites existing `prov` values.** Run it before materialising provenance, and use the same annotated database for all training and test queries. Token identities depend on the database's row ordering and should not be assumed to remain identical across independently regenerated databases.

### 3. Generate Provenance Parquets

`generate_provenance_dataset.py` requires the `qN_versions/*.sql` files and the annotated database. Pass the same database path and schema used above:

```bash
python generate_provenance_dataset.py \
  --data-root . \
  --db "$DUCKDB_PATH" \
  --schema "$DUCKDB_SCHEMA" \
  --output-root ./generated_provenance
```

Unlike the annotation script, the generator opens the database read-only. Its output root must be new. It writes `qN_versions_full` directories; it does **not** create the train/test split. Q84 uses the single-witness-per-result-row format; the other selected templates use witness collections.

For reproduction of the published classification results, use the final train/test partitions rather than repartitioning newly generated data.
