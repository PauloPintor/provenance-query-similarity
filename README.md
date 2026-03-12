# provenance-query-similarity

Code and experiments for the paper **"Data Provenance-based Query Similarity"**.

This repository implements provenance-based query representations (Lineage and
Why-provenance) and evaluates their effectiveness for query template
classification and similarity analysis on TPC-DS workloads.

---

## Overview

The project investigates the use of **data provenance** as a signal for
measuring semantic similarity between database queries.  We lift tuple-level
provenance notions to the query level and evaluate them as features for
identifying query templates.

### Provenance representations

| Representation | Class | Description |
|---|---|---|
| Lineage set | `Lineage` | Union of tuple-level lineage across all result tuples |
| Atomic Why-provenance | `AtomicWhyProvenance` | Set of minimal witnesses (proof trees) |
| Blocked Why-provenance | `BlockedWhyProvenance` | Witnesses grouped by source relation |
| Token-frequency features | `TokenFrequencyExtractor` | Relation-level bag-of-words |
| Witness-frequency features | `WitnessFrequencyExtractor` | Frequency vector over complete witnesses |
| Compact structural descriptors | `CompactStructuralDescriptor` | Aggregate statistics of provenance structure |
| Blocked witness features | `BlockedWitnessFrequencyExtractor` | Relation-set fingerprints of blocked witnesses |

### ML models evaluated

kNN · Logistic Regression · Random Forest · Naïve Bayes · XGBoost

---

## Installation

```bash
# Clone the repository
git clone https://github.com/PauloPintor/provenance-query-similarity.git
cd provenance-query-similarity

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install the package and its dependencies
pip install -e .

# For development (includes pytest)
pip install -e ".[dev]"
```

**Python ≥ 3.9** is required.

---

## Quick start

```python
from provenance_qs.provenance import Lineage, AtomicWhyProvenance, ProvenanceFeatureExtractor
from provenance_qs.models import QueryTemplateClassifier

# Build lineage objects
lineages = [
    Lineage([("orders", 1), ("customer", 5)]),
    Lineage([("orders", 2), ("customer", 5)]),
    Lineage([("inventory", 10), ("supplier", 3)]),
]
labels = [1, 1, 2]

# Extract features
extractor = ProvenanceFeatureExtractor(use_lineage=True, use_token=True, use_structural=True)
X = extractor.fit_transform(lineages=lineages)

# Train a classifier
clf = QueryTemplateClassifier("rf")
clf.fit(X, labels)
print(clf.predict(X))   # [1, 1, 2]
```

---

## Reproducing the experiments

The experiments directory contains scripts that can be run in sequence:

### Step 1 – Generate query instances

```bash
python experiments/generate_queries.py \
    --templates 1 3 7 19 42 \
    --instances 50 \
    --seed 42 \
    --output data/queries.jsonl
```

### Step 2 – Extract provenance

```bash
python experiments/extract_provenance.py \
    --input  data/queries.jsonl \
    --output data/provenance.jsonl \
    --seed   42
```

### Step 3 – Train and cross-validate classifiers

```bash
python experiments/train_models.py \
    --input   data/provenance.jsonl \
    --models  knn lr rf nb xgb \
    --folds   5 \
    --feature lineage token structural \
    --output  results/cv_results.json
```

### Step 4 – Full evaluation report

```bash
python experiments/evaluate.py \
    --input  data/provenance.jsonl \
    --output results/evaluation_report.json \
    --folds  5
```

---

## Repository structure

```
provenance-query-similarity/
├── src/
│   └── provenance_qs/
│       ├── provenance/            # Provenance representations & feature extraction
│       │   ├── lineage.py         #   Lineage sets
│       │   ├── why_provenance.py  #   Atomic & blocked Why-provenance
│       │   └── features.py        #   Feature extractors (all 5 families)
│       ├── similarity/            # Similarity & distance metrics
│       │   └── metrics.py
│       ├── models/                # ML classifiers & evaluation
│       │   ├── classifiers.py
│       │   └── evaluation.py
│       └── data/                  # TPC-DS query generation & I/O
│           ├── tpcds.py           #   Query templates and instance generator
│           └── loader.py          #   JSONL dataset persistence
├── experiments/                   # End-to-end experiment scripts
│   ├── generate_queries.py
│   ├── extract_provenance.py
│   ├── train_models.py
│   └── evaluate.py
├── tests/                         # pytest test suite (132 tests)
│   ├── test_lineage.py
│   ├── test_why_provenance.py
│   ├── test_features.py
│   ├── test_similarity.py
│   ├── test_classifiers.py
│   └── test_data.py
└── pyproject.toml
```

---

## Running the tests

```bash
pytest tests/ -v
```

---

## License

MIT
