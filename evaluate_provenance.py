#!/usr/bin/env python3
"""Final provenance-based template-classification experiment.

The program evaluates the representations reported in the article:

* exact Lineage and Atomic Why-provenance with Jaccard similarity;
* Blocked Why-provenance with the selected two-stage candidate reduction;
* PSD, Token TF-IDF, and Witness TF-IDF vector representations;
* hashed binary Lineage and Atomic Why-provenance for supervised models.

All 1-NN and 5-NN configurations use majority voting after selecting the
nearest neighbours with the representation-specific comparison measure.
Feature hashing is used only to construct vector features; the direct set
comparisons always use the complete sets.

Every completed configuration stores aggregate, per-template, and
per-execution results. The selected PSD configurations additionally store
class-specific SHAP attributions and leave-one-feature-out ablations. These
artifacts support the built-in plotting helpers via --plots-only. The final
article's timing figures and additional PSD feature-group experiments use
separate scripts.
"""

import argparse
from array import array
import csv
import gc
import glob
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Dict, Iterator, List, Tuple, Optional, Sequence, Set, Union

import numpy as np
import pyarrow.parquet as pq

from scipy import sparse
from scipy.sparse import csr_matrix

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from xgboost import DMatrix, XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

"""
    Global configuration parameters for training and evaluation.
    Modify these values to adapt the experiments to your environment.
"""

# ------------------------------------------------------------------
# Data paths
# ------------------------------------------------------------------
DATA_ROOT = os.environ.get("PROV_DATA_ROOT", ".")

# Glob patterns to locate training and testing parquet files
TRAIN_GLOB = os.environ.get(
    "PROV_TRAIN_GLOB", os.path.join(DATA_ROOT, "**", "*_train", "*.parquet")
)
TEST_GLOB = os.environ.get(
    "PROV_TEST_GLOB", os.path.join(DATA_ROOT, "**", "*_test", "*.parquet")
)
OUTPUT_DIR = os.environ.get("PROV_OUTPUT_DIR", os.path.join("results", "evaluation"))
RESULTS_CSV = os.path.join(OUTPUT_DIR, "final_template_results.csv")
ERRORS_CSV = os.path.join(OUTPUT_DIR, "final_template_errors.csv")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "Predictions")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "Figures")
XAI_DIR = os.path.join(OUTPUT_DIR, "XAI")
PER_CLASS_CSV = os.path.join(OUTPUT_DIR, "final_template_per_class_metrics.csv")
PSD_ABLATION_CSV = os.path.join(XAI_DIR, "psd_ablation.csv")


# ------------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------------
RANDOM_SEED = 13  # Seed for reproducibility

# Different training sizes used in experiments
TRAIN_K_LIST = (50, 100, 250, 500, 750, 1000)


# ------------------------------------------------------------------
# Feature vector representations
# ------------------------------------------------------------------
VECTOR_METHODS = [
    "scalars",  # Basic scalar features
    "token_tfidf",  # TF-IDF over token features
    "witness_tfidf",  # TF-IDF over witness features
    "lineage_vec",  # Vector representation of lineage
    "atomic_why_vec",  # Vector representation of atomic why-provenance
]


# ------------------------------------------------------------------
# Set-based similarity methods
# ------------------------------------------------------------------
SET_METHODS = [
    "lineage_jaccard",  # Jaccard similarity on full lineage sets
    "atomic_why_jaccard",  # Jaccard similarity on atomic why sets
    "why_blocked",  # Blocked comparison using why-provenance
]


# ------------------------------------------------------------------
# Classifiers evaluated
# ------------------------------------------------------------------
KNN_K_VALUES = (1, 5)

# Vector k-NN is reported for these representations. Lineage and Atomic
# Why-provenance use their exact set-based counterparts in the k-NN rows.
KNN_VECTOR_METHODS = {
    "scalars",
    "token_tfidf",
    "witness_tfidf",
}

SUPERVISED_MODELS = [
    "logreg",  # logistic regression
    "naive_bayes",  # multinomial Naive Bayes
    "random_forest",  # random forest classifier
    "xgboost",  # gradient boosted trees
]


# ------------------------------------------------------------------
# Hash dimensions for feature hashing
# ------------------------------------------------------------------
HASH_D_LINEAGE = 1 << 15  # Hash dimension for lineage vectors
HASH_D_ATOMIC_WHY = 1 << 14  # Hash dimension for atomic why vectors
HASH_D_TOKEN_TF = 1 << 15  # Hash dimension for token TF-IDF features
HASH_D_WITNESS_TF = 1 << 14  # Hash dimension for witness TF-IDF features


# ------------------------------------------------------------------
# Provenance Structural Descriptor (PSD) features
# ------------------------------------------------------------------
PSD_FEATURE_NAMES = (
    "witness_unique",
    "witness_total",
    "token_unique",
    "token_total",
    "token_unique_ratio",
    "witness_unique_ratio",
    "witness_token_ratio",
)

PSD_FEATURE_LABELS = (
    r"$|W|$",
    r"$|W|_{\mathrm{tot}}$",
    r"$|T|$",
    r"$|T|_{\mathrm{tot}}$",
    r"$|T|/|T|_{\mathrm{tot}}$",
    r"$|W|/|W|_{\mathrm{tot}}$",
    r"$|W|/|T|$",
)

XAI_MODELS = {"logreg", "xgboost"}


# ------------------------------------------------------------------
# Blocked why-provenance matching parameters
# ------------------------------------------------------------------
WHY_LN_TOPR = 50  # Number of candidates retained by Lineage ranking
WHY_QUERY_CANDIDATES = 25  # Maximum query candidates after structural filtering
WHY_WITNESS_CANDIDATES = 25  # Maximum candidate witnesses per source witness
WHY_AVG_W_TOL = 0.75  # Average weight tolerance threshold
WHY_HIST_COS_MIN = 0.90  # Minimum cosine similarity for histogram filtering
WHY_TRAIN_INDEX_CACHE = 32  # Maximum resident training inverted indexes

"""
    DATA MODEL
"""


@dataclass
class WhyCollection(Sequence[Sequence[int]]):
    """Compact ragged storage preserving every witness boundary and token."""

    tokens: array
    offsets: array

    @classmethod
    def from_nested(cls, value: Any) -> "WhyCollection":
        tokens = array("Q")
        offsets = array("Q", [0])
        if value is not None:
            for witness in value:
                if witness is None:
                    continue
                normalized = sorted(
                    {int(token) & 0xFFFFFFFFFFFFFFFF for token in witness}
                )
                tokens.extend(normalized)
                offsets.append(len(tokens))
        return cls(tokens=tokens, offsets=offsets)

    def __len__(self) -> int:
        return max(0, len(self.offsets) - 1)

    def __getitem__(
        self,
        index: Union[int, slice],
    ) -> Union[Sequence[int], List[Sequence[int]]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = self.offsets[index]
        end = self.offsets[index + 1]
        return memoryview(self.tokens)[start:end]


@dataclass
class Item:
    """
    In-memory representation of a single query example.

    This structure stores:
    - query metadata
    - lineage and why-provenance representations
    - token/witness term-frequency features
    - scalar summary statistics
    - optional blocked why-provenance auxiliary structures
    """

    query_id: int
    template_id: int
    query_name: str

    lineage: Set[int]
    atomic_why: Set[int]
    witnesses: Sequence[Sequence[int]]

    token_tf: List[Tuple[int, int]]
    witness_tf: List[Tuple[int, int]]

    nwitness_unique: int
    nwitness_total: int
    ntokens_unique: int
    ntokens_total: int

    # Auxiliary structures used by blocked why-provenance matching
    why_list: Sequence[Sequence[int]] = field(default_factory=list)
    why_inv: Dict[int, Sequence[int]] = field(default_factory=dict)
    avg_w: float = 0.0
    hist_w: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float32))


@dataclass(frozen=True)
class ItemRef:
    """Location and metadata of one Parquet row."""

    path: str
    row_index: int
    global_index: int
    query_id: int
    template_id: int
    query_name: str


"""
    INPUT / OUTPUT
"""


def list_files(pattern: str) -> List[str]:
    """
    Return all files matching a glob pattern, sorted for deterministic loading.
    """
    return sorted(glob.glob(pattern, recursive=True))


def _safe_int(x: Any, default: int = 0) -> int:
    """
    Convert a value to int, returning a default value on failure.
    """
    try:
        return int(x)
    except Exception:
        return default


def _as_set_int(x: Any) -> Set[int]:
    """
    Convert an input value into a set of integers.

    Accepts scalars, lists, tuples, NumPy arrays, or sets.
    Returns an empty set for None.
    """
    if x is None:
        return set()
    if isinstance(x, set):
        return {int(v) for v in x}
    if isinstance(x, (list, tuple, np.ndarray)):
        return {int(v) for v in x}
    return {int(x)}


def _as_why_list(x: Any) -> WhyCollection:
    """
    Convert a witness collection into sorted, duplicate-free integer lists.

    Each witness is normalised as a sorted, duplicate-free integer list.
    """
    return WhyCollection.from_nested(x)


def _as_tf_pairs(x: Any, key: str) -> List[Tuple[int, int]]:
    """
    Convert a list of term-frequency records into (id, tf) pairs.

    Parameters
    ----------
    x : Any
        Iterable of dictionary-like rows.
    key : str
        Field containing the term identifier.

    Only positive term frequencies are kept.
    """
    if not x:
        return []
    out: List[Tuple[int, int]] = []
    for row in x:
        if row is None:
            continue
        k = int(row[key])
        tf = int(row["tf"])
        if tf > 0:
            out.append((k, tf))
    return out


def required_columns(vector_methods: List[str], set_methods: List[str]) -> List[str]:
    """
    Determine the minimal set of parquet columns required for the
    configured vector and set-based methods.

    This avoids loading unnecessary columns from disk.
    """
    cols = {"query_id", "template_id", "query_name"}

    if "scalars" in vector_methods:
        cols.update(
            {
                "nwitness_unique",
                "nwitness_total",
                "ntokens_unique",
                "ntokens_total",
            }
        )

    if "token_tfidf" in vector_methods:
        cols.add("token_tf")

    if "witness_tfidf" in vector_methods:
        cols.add("witness_tf")

    if (
        "lineage_vec" in vector_methods
        or "lineage_jaccard" in set_methods
        or "why_blocked" in set_methods
    ):
        cols.add("lineage_hash")

    if (
        "atomic_why_vec" in vector_methods
        or "atomic_why_jaccard" in set_methods
    ):
        cols.add("monoids_hash")

    if "why_blocked" in set_methods:
        cols.add("witnesses")

    return sorted(cols)


def _item_from_row(r: Dict[str, Any]) -> Item:
    """Convert one decoded Parquet row into an Item."""
    qid = _safe_int(r.get("query_id"), -1)
    tid = _safe_int(r.get("template_id"), -1)
    raw_qname = r.get("query_name")
    qname = str(qid if raw_qname is None else raw_qname)

    return Item(
        query_id=qid,
        template_id=tid,
        query_name=qname,
        lineage=_as_set_int(r.get("lineage_hash")),
        atomic_why=_as_set_int(r.get("monoids_hash")),
        witnesses=_as_why_list(r.get("witnesses")),
        token_tf=_as_tf_pairs(r.get("token_tf"), "token"),
        witness_tf=_as_tf_pairs(r.get("witness_tf"), "witness_hash"),
        nwitness_unique=_safe_int(r.get("nwitness_unique")),
        nwitness_total=_safe_int(r.get("nwitness_total")),
        ntokens_unique=_safe_int(r.get("ntokens_unique")),
        ntokens_total=_safe_int(r.get("ntokens_total")),
    )


def iter_parquet_rows(
    path: str,
    columns: List[str],
    batch_size: int = 256,
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield decoded Parquet rows with stable file-relative row indices."""
    schema_names = set(pq.read_schema(path).names)
    missing = sorted(set(columns) - schema_names)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    row_index = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(
        columns=columns,
        batch_size=batch_size,
        use_threads=False,
    ):
        for row in batch.to_pylist():
            yield row_index, row
            row_index += 1


def iter_items_one_file(
    path: str,
    columns: List[str],
    batch_size: int = 256,
) -> Iterator[Item]:
    """Yield Item objects from one Parquet file without retaining the file."""
    for _, row in iter_parquet_rows(path, columns, batch_size=batch_size):
        yield _item_from_row(row)


def load_items_one_file(path: str, columns: List[str]) -> List[Item]:
    """
    Load one parquet file and convert its rows into Item objects.

    Only the requested columns are read when they exist in the schema.
    """
    return list(iter_items_one_file(path, columns, batch_size=256))


def load_all_splits(
    vector_methods: List[str],
    set_methods: List[str],
) -> Tuple[List[Item], List[Item]]:
    """
    Load all training and testing items from the configured parquet splits.

    Returns
    -------
    pool : list of Item
        Training items.
    test : list of Item
        Test items.
    """
    train_files = list_files(TRAIN_GLOB)
    test_files = list_files(TEST_GLOB)

    if not train_files:
        raise FileNotFoundError(f"No train files found for pattern {TRAIN_GLOB}")
    if not test_files:
        raise FileNotFoundError(f"No test files found for pattern {TEST_GLOB}")

    print(f"[INPUT] train_glob={TRAIN_GLOB}", flush=True)
    print(f"[INPUT] test_glob={TEST_GLOB}", flush=True)
    print(
        f"[INPUT] train_files={len(train_files)} test_files={len(test_files)}",
        flush=True,
    )
    print(f"[INPUT] first_train={train_files[0]}", flush=True)
    print(f"[INPUT] first_test={test_files[0]}", flush=True)

    cols = required_columns(vector_methods, set_methods)

    t0 = time.perf_counter()
    pool: List[Item] = []
    test: List[Item] = []

    for p in train_files:
        pool.extend(load_items_one_file(p, cols))
    for p in test_files:
        test.extend(load_items_one_file(p, cols))

    dt = time.perf_counter() - t0
    print(f"[LOAD] pool={len(pool)} test={len(test)} cols={cols} time={dt:.3f}s")
    return pool, test


def scan_item_refs(files: List[str]) -> List[ItemRef]:
    """Read only row locations and identifiers from a collection of files."""
    columns = ["query_id", "template_id", "query_name"]
    refs: List[ItemRef] = []
    global_index = 0
    for path in files:
        for row_index, row in iter_parquet_rows(path, columns, batch_size=4096):
            query_id = _safe_int(row.get("query_id"), -1)
            template_id = _safe_int(row.get("template_id"), -1)
            raw_query_name = row.get("query_name")
            query_name = str(query_id if raw_query_name is None else raw_query_name)
            refs.append(
                ItemRef(
                    path=path,
                    row_index=row_index,
                    global_index=global_index,
                    query_id=query_id,
                    template_id=template_id,
                    query_name=query_name,
                )
            )
            global_index += 1
    return refs


def select_ref_k_per_template(
    refs: List[ItemRef],
    k: int,
    seed: int,
) -> List[ItemRef]:
    """Apply the normal deterministic training selection to lightweight refs."""
    by_template: Dict[int, List[ItemRef]] = defaultdict(list)
    for ref in refs:
        by_template[ref.template_id].append(ref)

    insufficient = {
        template_id: len(rows)
        for template_id, rows in sorted(by_template.items())
        if len(rows) < k
    }
    if insufficient:
        raise ValueError(
            f"At least {k} training executions are required per template; "
            f"found {insufficient}"
        )

    salt = f"_seed_{seed}"
    selected: List[ItemRef] = []
    for _, rows in sorted(by_template.items()):
        rows = sorted(
            rows,
            key=lambda ref: (
                stable_hash_u64(ref.query_name + salt),
                ref.global_index,
            ),
        )
        selected.extend(rows[:k])
    return selected


def load_referenced_why_items(refs: List[ItemRef]) -> List[Item]:
    """Load full provenance only for the selected training row locations."""
    columns = required_columns([], ["why_blocked"])
    wanted_by_path: Dict[str, Set[int]] = defaultdict(set)
    for ref in refs:
        wanted_by_path[ref.path].add(ref.row_index)

    loaded: Dict[Tuple[str, int], Item] = {}
    for path in sorted(wanted_by_path):
        wanted = wanted_by_path[path]
        for row_index, row in iter_parquet_rows(path, columns, batch_size=1):
            if row_index in wanted:
                loaded[(path, row_index)] = _item_from_row(row)

    missing = [
        (ref.path, ref.row_index)
        for ref in refs
        if (ref.path, ref.row_index) not in loaded
    ]
    if missing:
        raise RuntimeError(f"Selected Parquet rows could not be loaded: {missing[:5]}")
    return [loaded[(ref.path, ref.row_index)] for ref in refs]


def prepare_streamed_why_splits(
    train_k: int,
) -> Tuple[List[Item], List[str], int]:
    """Select Why training rows by metadata and leave test rows on disk."""
    train_files = list_files(TRAIN_GLOB)
    test_files = list_files(TEST_GLOB)
    if not train_files:
        raise FileNotFoundError(f"No train files found for pattern {TRAIN_GLOB}")
    if not test_files:
        raise FileNotFoundError(f"No test files found for pattern {TEST_GLOB}")

    print(f"[INPUT] train_glob={TRAIN_GLOB}", flush=True)
    print(f"[INPUT] test_glob={TEST_GLOB}", flush=True)
    print(
        f"[INPUT] train_files={len(train_files)} test_files={len(test_files)}",
        flush=True,
    )
    print(f"[INPUT] first_train={train_files[0]}", flush=True)
    print(f"[INPUT] first_test={test_files[0]}", flush=True)

    t0 = time.perf_counter()
    train_refs = scan_item_refs(train_files)
    test_refs = scan_item_refs(test_files)
    train_templates = Counter(ref.template_id for ref in train_refs)
    test_templates = Counter(ref.template_id for ref in test_refs)
    if set(train_templates) != set(test_templates):
        raise ValueError(
            "Training and test templates differ: "
            f"train={sorted(train_templates)}, test={sorted(test_templates)}"
        )

    selected_refs = select_ref_k_per_template(train_refs, train_k, RANDOM_SEED)
    del train_refs
    gc.collect()
    print(
        f"[META] selected_train={len(selected_refs)} test={len(test_refs)} "
        f"time={time.perf_counter() - t0:.3f}s",
        flush=True,
    )

    train = load_referenced_why_items(selected_refs)
    print(
        f"[LOAD] selected Why training rows={len(train)}; "
        "test provenance will be streamed",
        flush=True,
    )
    return train, test_files, len(test_refs)


def iter_why_test_items(test_files: List[str]) -> Iterator[Item]:
    """Yield complete test executions one at a time for Why evaluation."""
    columns = required_columns([], ["why_blocked"])
    for path in test_files:
        yield from iter_items_one_file(path, columns, batch_size=1)


def validate_splits(
    pool: List[Item],
    test: List[Item],
    required_train_k: Optional[int] = None,
) -> None:
    """Validate the class coverage required by every reported experiment."""
    pool_counts = Counter(it.template_id for it in pool)
    test_counts = Counter(it.template_id for it in test)

    if not pool_counts or not test_counts:
        raise ValueError("The training pool and test split must both be non-empty")

    if set(pool_counts) != set(test_counts):
        missing_from_test = sorted(set(pool_counts) - set(test_counts))
        missing_from_pool = sorted(set(test_counts) - set(pool_counts))
        raise ValueError(
            "Training and test templates differ: "
            f"missing_from_test={missing_from_test}, "
            f"missing_from_pool={missing_from_pool}"
        )

    required = max(TRAIN_K_LIST) if required_train_k is None else required_train_k
    insufficient = {
        tid: count for tid, count in sorted(pool_counts.items()) if count < required
    }
    if insufficient:
        raise ValueError(
            f"At least {required} training executions are required per template; "
            f"found {insufficient}"
        )


"""
    DETERMINISTIC TRAIN SELECTION
"""


def stable_hash_u64(s: str) -> int:
    """
    Compute a deterministic 64-bit hash for a string.

    Uses a variant of the FNV-1a hash algorithm to produce a stable
    hash value across Python runs. This avoids Python's built-in hash
    randomization and ensures reproducible dataset sampling.
    """
    h = 1469598103934665603
    prime = 1099511628211
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * prime) & 0xFFFFFFFFFFFFFFFF
    return h


def select_k_per_template(pool: List[Item], k: int, seed: int) -> List[Item]:
    """
    Select up to k items per template in a deterministic way.

    Items are grouped by template_id and ordered using a stable hash
    derived from the query name and a seed. This guarantees that the
    same subset is selected across runs, which is important for
    reproducible experiments.
    """
    by_t: Dict[int, List[Item]] = defaultdict(list)

    # Group items by template
    for it in pool:
        by_t[it.template_id].append(it)

    out: List[Item] = []
    salt = f"_seed_{seed}"

    # Deterministically select k items per template
    for tid, lst in sorted(by_t.items()):
        keyed = [(stable_hash_u64(it.query_name + salt), it) for it in lst]
        keyed.sort(key=lambda kv: kv[0])
        out.extend([it for _, it in keyed[:k]])

    return out


"""
    LABEL ENCODING
"""


def build_label_mapping(train_items: List[Item]) -> Dict[int, int]:
    """
    Build a contiguous label mapping from template_id values.

    The mapping assigns each distinct template_id to an integer index,
    sorted by template_id, so labels are stable across runs.
    """
    classes = sorted({it.template_id for it in train_items})
    return {c: i for i, c in enumerate(classes)}


def encode_labels(items: List[Item], label_map: Dict[int, int]) -> np.ndarray:
    """
    Encode item template identifiers as integer class labels.
    """
    return np.array([label_map[it.template_id] for it in items], dtype=np.int32)


"""
    BLOCKED WHY CACHE
"""


def wsize_hist(why_list: Sequence[Sequence[int]], max_bin: int = 8) -> np.ndarray:
    """
    Build a normalized histogram of witness sizes.

    Witnesses of size 1..max_bin are counted in separate bins.
    Larger witnesses are accumulated in the final overflow bin.
    The resulting histogram is L2-normalized.
    """
    hist = np.zeros(max_bin + 1, dtype=np.float32)
    for w in why_list:
        s = len(w)
        if s <= 0:
            continue
        if s <= max_bin:
            hist[s - 1] += 1.0
        else:
            hist[max_bin] += 1.0

    n = float(np.linalg.norm(hist))
    if n > 0:
        hist /= n
    return hist


def build_why_inverted_index(
    why_list: Sequence[Sequence[int]],
) -> Dict[int, Sequence[int]]:
    """Build a compact token-to-witness index for blocked comparison."""
    inverted: Dict[int, array] = {}
    for witness_index, witness in enumerate(why_list):
        for token in witness:
            positions = inverted.get(token)
            if positions is None:
                positions = array("I")
                inverted[token] = positions
            positions.append(witness_index)
    return inverted


def build_why_cache(items: List[Item], build_inverted: bool = True) -> None:
    """
    Precompute auxiliary structures for blocked why-provenance similarity.

    For each item, this function builds:
    - why_list: sorted witnesses as lists of integers
    - why_inv: optional inverted index from tokens to witness indices
    - avg_w: average witness size
    - hist_w: normalized witness-size histogram

    The loaded witness lists are reused rather than copied. For test items,
    build_inverted can be disabled so that their indexes are built and released
    one execution at a time during evaluation.
    """
    for it in items:
        wl = it.witnesses if it.witnesses else it.why_list
        it.why_list = wl
        it.witnesses = []
        it.why_inv = build_why_inverted_index(wl) if build_inverted else {}
        it.avg_w = (
            float(sum(len(witness) for witness in wl) / len(wl)) if wl else 0.0
        )
        it.hist_w = wsize_hist(wl, 8)


"""
    SIMILARITY FUNCTIONS
"""


def jaccard_set(a: Set[int], b: Set[int]) -> float:
    """
    Compute the Jaccard similarity between two sets.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    uni = len(a) + len(b) - inter
    return float(inter / uni) if uni else 0.0


def jaccard_witness_list(a: Sequence[int], b: Sequence[int]) -> float:
    """
    Compute the Jaccard similarity between two sorted witness lists.

    The intersection is computed using a two-pointer merge scan,
    which is more efficient than converting the lists back to sets.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    i = j = 0
    inter = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            inter += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    if inter == 0:
        return 0.0
    uni = len(a) + len(b) - inter
    return float(inter / uni) if uni else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity assuming the input vectors are already normalized.
    """
    return float(np.dot(a, b))


def majority_vote_ordered(neighbour_labels: np.ndarray) -> int:
    """Return the majority class, breaking ties by the closest neighbour."""
    labels = [int(label) for label in neighbour_labels]
    if not labels:
        return -1

    counts = Counter(labels)
    max_votes = max(counts.values())
    tied = {label for label, count in counts.items() if count == max_votes}

    # Labels are ordered from the closest to the furthest neighbour.
    return next(label for label in labels if label in tied)


def predict_knn_vote(scores: np.ndarray, train_tids: np.ndarray, k: int) -> int:
    """
    Predict a class label using unweighted k-nearest-neighbour voting.

    The top-k highest-scoring neighbours are selected using the relevant
    provenance similarity. Their labels then receive one vote each, making
    the decision rule identical to that used by vector k-NN. Ties are broken
    in favour of the class represented by the closest tied neighbour.
    """
    if scores.size == 0:
        return -1

    k = min(k, len(scores))
    idx = np.argpartition(scores, len(scores) - k)[-k:]
    idx = idx[np.lexsort((idx, -scores[idx]))]

    return majority_vote_ordered(train_tids[idx])


def why_similarity_soft_blocked_cached(
    q_wl: Sequence[Sequence[int]],
    q_inv: Dict[int, Sequence[int]],
    cand_wl: Sequence[Sequence[int]],
    cand_inv: Dict[int, Sequence[int]],
    max_candidates: int,
) -> float:
    """
    Compute a symmetric blocked similarity between two why-provenance witness sets.

    For each witness in one side, only a limited number of candidate witnesses
    from the other side are explored using the inverted index. The score is the
    average of the best witness matches in both directions.
    """
    if not q_wl and not cand_wl:
        return 1.0
    if not q_wl or not cand_wl:
        return 0.0

    def directed(
        X_wl: Sequence[Sequence[int]],
        Y_wl: Sequence[Sequence[int]],
        Y_inv: Dict[int, Sequence[int]],
    ) -> float:
        """
        Compute the directed blocked similarity from X to Y.

        Each witness in X is matched against a restricted set of candidate
        witnesses in Y obtained through token overlap.
        """
        s = 0.0
        for x in X_wl:
            cand_idx: List[int] = []
            seen = set()

            for t in x:
                for j in Y_inv.get(t, []):
                    if j not in seen:
                        seen.add(j)
                        cand_idx.append(j)
                        if len(cand_idx) >= max_candidates:
                            break
                if len(cand_idx) >= max_candidates:
                    break

            best = 0.0
            for j in cand_idx:
                y = Y_wl[j]
                upper_bound = min(len(x), len(y)) / max(len(x), len(y))
                if upper_bound <= best:
                    continue
                v = jaccard_witness_list(x, y)
                if v > best:
                    best = v
                    if best == 1.0:
                        break
            s += best

        return float(s / len(X_wl)) if X_wl else 0.0

    s1 = directed(q_wl, cand_wl, cand_inv)
    s2 = directed(cand_wl, q_wl, q_inv)
    return float((s1 + s2) / 2.0)


"""
    HASH HELPERS
"""


def _hash_to_index(x: int, dim: int) -> int:
    """
    Hash an integer deterministically into the range [0, dim).

    This function applies a 64-bit mixing procedure before taking the
    modulo, helping spread values more uniformly across hash buckets.
    """
    z = int(x) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 33
    z = (z * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 33
    z = (z * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 33
    return int(z % dim)


"""
    HASHED VECTOR FEATURES
"""


def ratio(a: float, b: float) -> float:
    """
    Safely compute a / b.

    Returns 0.0 when the denominator is zero.
    """
    return float(a / b) if b else 0.0


def hashed_binary_matrix(list_of_sets: List[Set[int]], dim: int) -> csr_matrix:
    """
    Build a binary CSR matrix from a list of integer sets.

    Each input set is hashed into a fixed-dimensional feature space,
    and each active hashed position receives value 1.0.
    """
    indptr = [0]
    indices: List[int] = []
    data: List[float] = []

    for s in list_of_sets:
        cols = sorted({_hash_to_index(v, dim) for v in s})
        indices.extend(cols)
        data.extend([1.0] * len(cols))
        indptr.append(len(indices))

    return csr_matrix(
        (
            np.array(data, dtype=np.float32),
            np.array(indices, dtype=np.int32),
            np.array(indptr, dtype=np.int32),
        ),
        shape=(len(list_of_sets), dim),
        dtype=np.float32,
    )


def hashed_count_matrix(
    list_of_items: List[List[Tuple[int, int]]],
    dim: int,
) -> csr_matrix:
    """
    Build a count-based CSR matrix from hashed (token, frequency) pairs.

    Term identifiers are hashed into a fixed-dimensional space and
    frequencies that collide in the same bucket are accumulated.
    """
    indptr = [0]
    indices: List[int] = []
    data: List[float] = []

    for pairs in list_of_items:
        acc: Dict[int, float] = defaultdict(float)
        for tok, tf in pairs:
            j = _hash_to_index(tok, dim)
            acc[j] += float(tf)

        cols = sorted(acc.keys())
        indices.extend(cols)
        data.extend([acc[c] for c in cols])
        indptr.append(len(indices))

    return csr_matrix(
        (
            np.array(data, dtype=np.float32),
            np.array(indices, dtype=np.int32),
            np.array(indptr, dtype=np.int32),
        ),
        shape=(len(list_of_items), dim),
        dtype=np.float32,
    )


def build_scalars_matrix(items: List[Item]) -> np.ndarray:
    """
    Build a dense feature matrix from scalar summary statistics.

    The resulting feature vector includes raw counts and simple ratios
    derived from token and witness statistics.
    """
    X = []
    for it in items:
        X.append(
            [
                float(it.nwitness_unique),
                float(it.nwitness_total),
                float(it.ntokens_unique),
                float(it.ntokens_total),
                ratio(it.ntokens_unique, max(1, it.ntokens_total)),
                ratio(it.nwitness_unique, max(1, it.nwitness_total)),
                ratio(it.nwitness_unique, max(1, it.ntokens_unique)),
            ]
        )
    return np.asarray(X, dtype=np.float32)


"""
    VECTOR MODELS
"""


def fit_vector_model(
    model_name: str,
    Xtr,
    ytr: np.ndarray,
    knn_k: Optional[int] = None,
):
    """
    Fit a vector-based classifier on the training data.

    Depending on the selected model, this may include feature scaling
    and the use of sparse or dense representations.
    """
    if model_name == "knn":
        if knn_k not in KNN_K_VALUES:
            raise ValueError(f"Unsupported k-NN neighbourhood size: {knn_k}")
        scaler = (
            StandardScaler(with_mean=False)
            if sparse.issparse(Xtr)
            else StandardScaler()
        )
        Xtr_s = scaler.fit_transform(Xtr)
        nn = NearestNeighbors(n_neighbors=knn_k, metric="cosine")
        nn.fit(Xtr_s)
        return ("knn", scaler, nn)

    if model_name == "logreg":
        return make_pipeline(
            (
                StandardScaler(with_mean=False)
                if sparse.issparse(Xtr)
                else StandardScaler()
            ),
            LogisticRegression(random_state=RANDOM_SEED),
        ).fit(Xtr, ytr)

    if model_name == "naive_bayes":
        return MultinomialNB().fit(Xtr, ytr)

    if model_name == "random_forest":
        # scikit-learn accepts sparse input. Keeping it sparse avoids allocating
        # a potentially multi-gigabyte dense matrix.
        Xtr_fit = Xtr.tocsc() if sparse.issparse(Xtr) else Xtr
        return RandomForestClassifier(random_state=RANDOM_SEED).fit(Xtr_fit, ytr)

    if model_name == "xgboost":
        return XGBClassifier(random_state=RANDOM_SEED).fit(Xtr, ytr)

    raise ValueError(model_name)


def predict_vector_model(
    model_name: str, clf, Xte, ytr: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate predictions for a fitted vector-based model.
    """
    if model_name == "knn":
        _, scaler, nn = clf
        Xte_s = scaler.transform(Xte)
        _, idx = nn.kneighbors(Xte_s, return_distance=True)
        neigh_labels = ytr[idx]

        out = np.empty(Xte_s.shape[0], dtype=np.int32)
        for i in range(len(out)):
            out[i] = majority_vote_ordered(neigh_labels[i])
        return out

    if model_name == "random_forest":
        Xte = Xte.tocsr() if sparse.issparse(Xte) else Xte
        return clf.predict(Xte)

    return clf.predict(Xte)


"""
    RESULT RECORD
"""


@dataclass
class ResultRow:
    """
    Container for the evaluation results of one method/model combination.
    """

    train_k: int
    method: str
    model: str
    acc: float
    f1: float
    fit_s: float
    build_test_s: float
    pred_s: float
    latency_ms: float


@dataclass
class VectorEvaluationArtifacts:
    """Fitted vector state retained only when a later analysis requires it."""

    Xtr: Any
    Xte: Any
    ytr: np.ndarray
    yte: np.ndarray
    classifier: Any
    label_map: Dict[int, int]


"""
    EVALUATION: VECTOR METHODS
"""


def eval_vector_method(
    train: List[Item],
    test: List[Item],
    method: str,
    model_name: str,
    knn_k: Optional[int] = None,
    collect_artifacts: bool = False,
) -> Tuple[ResultRow, np.ndarray, Optional[VectorEvaluationArtifacts]]:
    """
    Evaluate one vector representation with one classifier.

    This function:
    - encodes labels
    - builds train and test feature matrices
    - fits the selected model
    - generates predictions
    - computes evaluation metrics and timing statistics
    """
    label_map = build_label_mapping(train)
    ytr = encode_labels(train, label_map)
    yte = encode_labels(test, label_map)

    if method == "scalars":
        Xtr = build_scalars_matrix(train)
        t0 = time.perf_counter()
        Xte = build_scalars_matrix(test)
        build_test_s = time.perf_counter() - t0

    elif method == "token_tfidf":
        Xtr_counts = hashed_count_matrix([it.token_tf for it in train], HASH_D_TOKEN_TF)
        tfidf = TfidfTransformer()
        Xtr = tfidf.fit_transform(Xtr_counts)

        t0 = time.perf_counter()
        Xte_counts = hashed_count_matrix([it.token_tf for it in test], HASH_D_TOKEN_TF)
        Xte = tfidf.transform(Xte_counts)
        build_test_s = time.perf_counter() - t0

    elif method == "witness_tfidf":
        Xtr_counts = hashed_count_matrix(
            [it.witness_tf for it in train], HASH_D_WITNESS_TF
        )
        tfidf = TfidfTransformer()
        Xtr = tfidf.fit_transform(Xtr_counts)

        t0 = time.perf_counter()
        Xte_counts = hashed_count_matrix(
            [it.witness_tf for it in test], HASH_D_WITNESS_TF
        )
        Xte = tfidf.transform(Xte_counts)
        build_test_s = time.perf_counter() - t0

    elif method == "lineage_vec":
        Xtr = hashed_binary_matrix([it.lineage for it in train], HASH_D_LINEAGE)
        t0 = time.perf_counter()
        Xte = hashed_binary_matrix([it.lineage for it in test], HASH_D_LINEAGE)
        build_test_s = time.perf_counter() - t0

    elif method == "atomic_why_vec":
        Xtr = hashed_binary_matrix([it.atomic_why for it in train], HASH_D_ATOMIC_WHY)
        t0 = time.perf_counter()
        Xte = hashed_binary_matrix([it.atomic_why for it in test], HASH_D_ATOMIC_WHY)
        build_test_s = time.perf_counter() - t0

    else:
        raise ValueError(method)

    t1 = time.perf_counter()
    clf = fit_vector_model(model_name, Xtr, ytr, knn_k=knn_k)
    fit_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    pred = predict_vector_model(model_name, clf, Xte, ytr=ytr)
    pred_s = time.perf_counter() - t2

    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
    latency_ms = ((build_test_s + pred_s) / max(1, len(test))) * 1000.0

    result = ResultRow(
            train_k=0,
            method=method,
            model=f"{knn_k}-nn" if model_name == "knn" else model_name,
            acc=acc,
            f1=f1,
            fit_s=fit_s,
            build_test_s=build_test_s,
            pred_s=pred_s,
            latency_ms=latency_ms,
        )
    artifacts = None
    if collect_artifacts:
        artifacts = VectorEvaluationArtifacts(
            Xtr=Xtr,
            Xte=Xte,
            ytr=ytr,
            yte=yte,
            classifier=clf,
            label_map=label_map,
        )

    return (
        result,
        pred,
        artifacts,
    )


"""
    EVALUATION: CUSTOM PROVENANCE NEAREST NEIGHBOR
"""


def eval_jaccard_custom(
    train: List[Item],
    test: List[Item],
    method: str,
    knn_k: int,
) -> Tuple[ResultRow, np.ndarray]:
    """
    Evaluate exact set-based k-NN using Jaccard similarity.

    Lineage and Atomic Why-provenance are compared directly as complete
    sets. Feature hashing is deliberately not applied in this path.
    """
    if knn_k not in KNN_K_VALUES:
        raise ValueError(f"Unsupported k-NN neighbourhood size: {knn_k}")

    if method == "lineage":
        get_rep = lambda it: it.lineage
    elif method == "atomic_why":
        get_rep = lambda it: it.atomic_why
    else:
        raise ValueError(method)

    label_map = build_label_mapping(train)

    # Retain references to the exact training sets used during search.
    t0 = time.perf_counter()
    train_reps = [get_rep(it) for it in train]
    train_tids = encode_labels(train, label_map)
    fit_s = time.perf_counter() - t0

    y_true = encode_labels(test, label_map)
    y_pred = np.full_like(y_true, -1)

    # Compare each test instance against all training representations.
    t1 = time.perf_counter()
    for i, q in enumerate(test):
        qrep = get_rep(q)
        scores = np.empty(len(train_reps), dtype=np.float32)
        for j, tr in enumerate(train_reps):
            scores[j] = jaccard_set(qrep, tr)
        y_pred[i] = predict_knn_vote(scores, train_tids, knn_k)
    pred_s = time.perf_counter() - t1

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    build_test_s = 0.0
    latency_ms = ((build_test_s + pred_s) / max(1, len(test))) * 1000.0

    return (
        ResultRow(
            train_k=0,
            method=f"{method}_jaccard",
            model=f"{knn_k}-nn",
            acc=acc,
            f1=f1,
            fit_s=fit_s,
            build_test_s=build_test_s,
            pred_s=pred_s,
            latency_ms=latency_ms,
        ),
        y_pred,
    )


def predict_why_blocked_item(
    q: Item,
    train: List[Item],
    train_tids: np.ndarray,
    train_ln: List[Set[int]],
    knn_k: int,
    train_inv_cache: Optional[
        OrderedDict[int, Dict[int, Sequence[int]]]
    ] = None,
) -> int:
    """Classify one execution using the article's blocked Why procedure."""
    q_inv = build_why_inverted_index(q.why_list)

    ln_scores = np.array(
        [jaccard_set(q.lineage, tr) for tr in train_ln], dtype=np.float32
    )
    candidate_count = min(WHY_LN_TOPR, len(train))
    top_idx = np.argpartition(ln_scores, -candidate_count)[-candidate_count:]
    top_idx = top_idx[np.argsort(ln_scores[top_idx])[::-1]]

    gated: List[int] = []
    for train_pos in top_idx.tolist():
        candidate_avg = train[train_pos].avg_w
        if max(q.avg_w, candidate_avg) > 0.0:
            relative_difference = (
                abs(q.avg_w - candidate_avg) / max(q.avg_w, candidate_avg)
            )
            if relative_difference > WHY_AVG_W_TOL:
                continue
        if cosine(q.hist_w, train[train_pos].hist_w) < WHY_HIST_COS_MIN:
            continue
        gated.append(train_pos)
        if len(gated) >= WHY_QUERY_CANDIDATES:
            break

    if not gated:
        gated = top_idx[: min(len(top_idx), WHY_QUERY_CANDIDATES)].tolist()

    scores = np.empty(len(gated), dtype=np.float32)
    for candidate_pos, train_pos in enumerate(gated):
        if train_inv_cache is None:
            candidate_inv = train[train_pos].why_inv
            if not candidate_inv:
                candidate_inv = build_why_inverted_index(train[train_pos].why_list)
                train[train_pos].why_inv = candidate_inv
        else:
            candidate_inv = train_inv_cache.get(train_pos)
            if candidate_inv is None:
                candidate_inv = build_why_inverted_index(train[train_pos].why_list)
                train_inv_cache[train_pos] = candidate_inv
                if len(train_inv_cache) > WHY_TRAIN_INDEX_CACHE:
                    train_inv_cache.popitem(last=False)
            else:
                train_inv_cache.move_to_end(train_pos)
        scores[candidate_pos] = float(
            why_similarity_soft_blocked_cached(
                q.why_list,
                q_inv,
                train[train_pos].why_list,
                candidate_inv,
                WHY_WITNESS_CANDIDATES,
            )
        )

    candidate_labels = train_tids[np.asarray(gated, dtype=np.int32)]
    return predict_knn_vote(scores, candidate_labels, knn_k)


def eval_why_blocked(
    train: List[Item],
    test: List[Item],
    knn_k: int,
) -> Tuple[ResultRow, np.ndarray]:
    """Evaluate blocked Why-provenance over in-memory test executions."""
    if knn_k not in KNN_K_VALUES:
        raise ValueError(f"Unsupported k-NN neighbourhood size: {knn_k}")

    label_map = build_label_mapping(train)
    t0 = time.perf_counter()
    train_tids = encode_labels(train, label_map)
    train_ln = [it.lineage for it in train]
    fit_s = time.perf_counter() - t0

    y_true = encode_labels(test, label_map)
    y_pred = np.full_like(y_true, -1)

    t1 = time.perf_counter()
    for qi, q in enumerate(test):
        y_pred[qi] = predict_why_blocked_item(
            q, train, train_tids, train_ln, knn_k
        )
    pred_s = time.perf_counter() - t1

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    build_test_s = 0.0
    latency_ms = ((build_test_s + pred_s) / max(1, len(test))) * 1000.0

    return (
        ResultRow(
            train_k=0,
            method="why_blocked",
            model=f"{knn_k}-nn",
            acc=acc,
            f1=f1,
            fit_s=fit_s,
            build_test_s=build_test_s,
            pred_s=pred_s,
            latency_ms=latency_ms,
        ),
        y_pred,
    )


def eval_why_blocked_streamed(
    train: List[Item],
    test_files: List[str],
    expected_test_count: int,
    knn_k: int,
) -> Tuple[ResultRow, np.ndarray, List[Item]]:
    """Evaluate blocked Why while retaining only one test execution at a time."""
    if knn_k not in KNN_K_VALUES:
        raise ValueError(f"Unsupported k-NN neighbourhood size: {knn_k}")

    label_map = build_label_mapping(train)
    t0 = time.perf_counter()
    train_tids = encode_labels(train, label_map)
    train_ln = [item.lineage for item in train]
    fit_s = time.perf_counter() - t0

    y_true_values: List[int] = []
    y_pred_values: List[int] = []
    test_metadata: List[Item] = []
    train_inv_cache: OrderedDict[int, Dict[int, Sequence[int]]] = OrderedDict()
    load_s = 0.0
    pred_s = 0.0

    iterator = iter_why_test_items(test_files)
    processed = 0
    while True:
        load_started = time.perf_counter()
        try:
            query = next(iterator)
        except StopIteration:
            break
        load_s += time.perf_counter() - load_started

        if query.template_id not in label_map:
            raise ValueError(f"Unknown test template: {query.template_id}")
        build_why_cache([query], build_inverted=False)

        score_started = time.perf_counter()
        prediction = predict_why_blocked_item(
            query,
            train,
            train_tids,
            train_ln,
            knn_k,
            train_inv_cache,
        )
        pred_s += time.perf_counter() - score_started

        y_true_values.append(label_map[query.template_id])
        y_pred_values.append(prediction)

        query.lineage = set()
        query.atomic_why = set()
        query.witnesses = []
        query.why_list = []
        query.why_inv = {}
        query.token_tf = []
        query.witness_tf = []
        test_metadata.append(query)

        processed += 1
        if processed % 25 == 0 or processed == expected_test_count:
            print(
                f"[WHY] processed={processed}/{expected_test_count} "
                f"load={load_s:.1f}s score={pred_s:.1f}s",
                flush=True,
            )
            gc.collect()

    if processed != expected_test_count:
        raise RuntimeError(
            f"Expected {expected_test_count} test rows, streamed {processed}"
        )

    y_true = np.asarray(y_true_values, dtype=np.int32)
    y_pred = np.asarray(y_pred_values, dtype=np.int32)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    latency_ms = (pred_s / max(1, processed)) * 1000.0
    print(
        f"[WHY STREAM] test_load={load_s:.3f}s scoring={pred_s:.3f}s",
        flush=True,
    )

    return (
        ResultRow(
            train_k=0,
            method="why_blocked",
            model=f"{knn_k}-nn",
            acc=acc,
            f1=f1,
            fit_s=fit_s,
            build_test_s=0.0,
            pred_s=pred_s,
            latency_ms=latency_ms,
        ),
        y_pred,
        test_metadata,
    )


"""
    CONFUSION MATRIX PLOTTING
"""


def save_figure(fig: plt.Figure, basename: str) -> None:
    """Save one article figure as both vector PDF and preview PNG."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(
        os.path.join(FIGURES_DIR, f"{basename}.pdf"),
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(FIGURES_DIR, f"{basename}.png"),
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, label_map, basename):
    """
    Plot and save a normalized confusion matrix.

    Rows are normalized by the number of true instances per class.
    """
    inv_label_map = {v: k for k, v in label_map.items()}

    encoded_labels = np.arange(len(inv_label_map), dtype=np.int32)
    cm_counts = confusion_matrix(y_true, y_pred, labels=encoded_labels)
    row_totals = cm_counts.sum(axis=1, keepdims=True)
    cm = np.divide(
        cm_counts.astype(float),
        row_totals,
        out=np.zeros_like(cm_counts, dtype=float),
        where=row_totals != 0,
    )

    labels = [f"Q{inv_label_map[i]}" for i in range(len(inv_label_map))]

    os.makedirs(FIGURES_DIR, exist_ok=True)
    matrix_path = os.path.join(FIGURES_DIR, f"{basename}.csv")
    with open(matrix_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted"] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + [f"{value:.8f}" for value in row])

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(
        cm,
        cmap="Blues",
        annot=False,
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_xlabel("Predicted template")
    ax.set_ylabel("True template")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_figure(fig, basename)


# ============================================================
# ISOLATED EXECUTION AND RESULTS EXPORT
# ============================================================


RESULT_FIELDS = [
    "train_k",
    "method",
    "model",
    "acc",
    "f1",
    "fit_s",
    "build_test_s",
    "pred_s",
    "latency_ms",
]


@dataclass(frozen=True)
class RunConfig:
    """One independently executed experimental configuration."""

    train_k: int
    method: str
    model: str
    knn_k: Optional[int] = None

    @property
    def result_model(self) -> str:
        return f"{self.knn_k}-nn" if self.model == "knn" else self.model

    @property
    def key(self) -> Tuple[int, str, str]:
        return (self.train_k, self.method, self.result_model)


PREDICTION_FIELDS = [
    "train_k",
    "method",
    "model",
    "query_id",
    "query_name",
    "true_template",
    "predicted_template",
    "correct",
]

PER_CLASS_FIELDS = [
    "train_k",
    "method",
    "model",
    "template_id",
    "precision",
    "recall",
    "f1",
    "support",
]

PSD_ABLATION_FIELDS = [
    "train_k",
    "model",
    "removed_feature",
    "acc",
    "f1",
    "delta_acc",
    "delta_f1",
]


def config_stem(config: RunConfig) -> str:
    """Return a stable filename component for one configuration."""
    raw = f"train_{config.train_k}_{config.method}_{config.result_model}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def prediction_path(config: RunConfig) -> str:
    """Return the per-execution prediction path for one configuration."""
    return os.path.join(PREDICTIONS_DIR, f"{config_stem(config)}.csv")


def write_csv_atomic(
    path: str,
    fieldnames: List[str],
    rows: List[Dict[str, Any]],
) -> None:
    """Write a complete CSV through an atomic replacement."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def save_predictions(
    config: RunConfig,
    test: List[Item],
    label_map: Dict[int, int],
    prediction: np.ndarray,
) -> str:
    """Persist every true and predicted template used by an evaluation."""
    if len(test) != len(prediction):
        raise ValueError("Prediction count does not match the test split")

    inv_label_map = {encoded: original for original, encoded in label_map.items()}
    rows: List[Dict[str, Any]] = []
    for item, predicted_label in zip(test, prediction):
        predicted_template = inv_label_map.get(int(predicted_label))
        if predicted_template is None:
            raise ValueError(f"Unknown predicted class label: {predicted_label}")
        rows.append(
            {
                "train_k": config.train_k,
                "method": config.method,
                "model": config.result_model,
                "query_id": item.query_id,
                "query_name": item.query_name,
                "true_template": item.template_id,
                "predicted_template": predicted_template,
                "correct": int(item.template_id == predicted_template),
            }
        )

    path = prediction_path(config)
    write_csv_atomic(path, PREDICTION_FIELDS, rows)
    return path


def save_per_class_metrics(
    config: RunConfig,
    test: List[Item],
    label_map: Dict[int, int],
    prediction: np.ndarray,
) -> None:
    """Upsert precision, recall, and F1 for every template class."""
    y_true = encode_labels(test, label_map)
    labels = np.arange(len(label_map), dtype=np.int32)
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        prediction,
        labels=labels,
        zero_division=0,
    )
    inv_label_map = {encoded: original for original, encoded in label_map.items()}

    existing: List[Dict[str, Any]] = []
    if os.path.exists(PER_CLASS_CSV) and os.path.getsize(PER_CLASS_CSV) > 0:
        with open(PER_CLASS_CSV, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != PER_CLASS_FIELDS:
                raise ValueError(
                    f"Unexpected per-class schema in {PER_CLASS_CSV}: "
                    f"{reader.fieldnames}"
                )
            for row in reader:
                key = (int(row["train_k"]), row["method"], row["model"])
                if key != config.key:
                    existing.append(row)

    for label in labels:
        existing.append(
            {
                "train_k": str(config.train_k),
                "method": config.method,
                "model": config.result_model,
                "template_id": str(inv_label_map[int(label)]),
                "precision": f"{precision[label]:.8f}",
                "recall": f"{recall[label]:.8f}",
                "f1": f"{f1_values[label]:.8f}",
                "support": str(int(support[label])),
            }
        )

    existing.sort(
        key=lambda row: (
            int(row["train_k"]),
            row["method"],
            row["model"],
            int(row["template_id"]),
        )
    )
    write_csv_atomic(PER_CLASS_CSV, PER_CLASS_FIELDS, existing)


def is_psd_xai_config(config: RunConfig) -> bool:
    """Return whether a configuration produces the reported PSD XAI analysis."""
    return (
        config.train_k == max(TRAIN_K_LIST)
        and config.method == "scalars"
        and config.model in XAI_MODELS
    )


def xai_npz_path(config: RunConfig) -> str:
    """Return the compact PSD attribution artifact path."""
    return os.path.join(XAI_DIR, f"psd_{config.model}_train_{config.train_k}.npz")


def _normalise_xgboost_contributions(
    raw: np.ndarray,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert XGBoost contribution output to samples x features x classes."""
    values = np.asarray(raw)
    if values.ndim == 4 and values.shape[-1] == 1:
        values = values[..., 0]

    if values.ndim == 3:
        if values.shape == (n_samples, n_classes, n_features + 1):
            base = values[:, :, -1]
            return np.transpose(values[:, :, :-1], (0, 2, 1)), base
        if values.shape == (n_samples, n_features + 1, n_classes):
            base = values[:, -1, :]
            return values[:, :-1, :], base

    if values.ndim == 2:
        if n_classes == 1 and values.shape == (n_samples, n_features + 1):
            return values[:, :-1, None], values[:, -1, None]
        expected = n_classes * (n_features + 1)
        if values.shape == (n_samples, expected):
            reshaped = values.reshape(n_samples, n_classes, n_features + 1)
            base = reshaped[:, :, -1]
            return np.transpose(reshaped[:, :, :-1], (0, 2, 1)), base

    raise ValueError(
        "Unexpected XGBoost contribution shape: "
        f"{values.shape}; expected samples={n_samples}, features={n_features}, "
        f"classes={n_classes}"
    )


def compute_psd_attributions(
    config: RunConfig,
    artifacts: VectorEvaluationArtifacts,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute class-specific SHAP attributions for a fitted PSD classifier."""
    Xtr = np.asarray(artifacts.Xtr, dtype=np.float64)
    Xte = np.asarray(artifacts.Xte, dtype=np.float64)
    n_samples, n_features = Xte.shape
    n_classes = len(artifacts.label_map)

    if n_features != len(PSD_FEATURE_NAMES):
        raise ValueError(
            f"PSD feature count is {n_features}, expected {len(PSD_FEATURE_NAMES)}"
        )

    if config.model == "xgboost":
        booster = artifacts.classifier.get_booster()
        matrix = DMatrix(Xte)
        try:
            raw = booster.predict(matrix, pred_contribs=True, strict_shape=True)
            target_margins = booster.predict(
                matrix,
                output_margin=True,
                strict_shape=True,
            )
        except TypeError:
            raw = booster.predict(matrix, pred_contribs=True)
            target_margins = booster.predict(matrix, output_margin=True)
        shap_values, base_values = _normalise_xgboost_contributions(
            raw,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
        )
        target_margins = np.asarray(target_margins)
        if target_margins.ndim == 1:
            target_margins = target_margins[:, None]
        model_values = Xte

    elif config.model == "logreg":
        pipeline = artifacts.classifier
        scaler = next(
            step for step in pipeline.named_steps.values() if isinstance(step, StandardScaler)
        )
        model = next(
            step
            for step in pipeline.named_steps.values()
            if isinstance(step, LogisticRegression)
        )
        Xtr_scaled = np.asarray(scaler.transform(Xtr), dtype=np.float64)
        Xte_scaled = np.asarray(scaler.transform(Xte), dtype=np.float64)
        background_mean = np.mean(Xtr_scaled, axis=0)
        centred = Xte_scaled - background_mean

        shap_values = np.zeros(
            (n_samples, n_features, n_classes),
            dtype=np.float64,
        )
        base_values = np.zeros((n_samples, n_classes), dtype=np.float64)
        for class_position, encoded_class in enumerate(model.classes_):
            encoded_class = int(encoded_class)
            coefficients = model.coef_[class_position]
            shap_values[:, :, encoded_class] = centred * coefficients
            baseline = float(model.intercept_[class_position]) + float(
                np.dot(background_mean, coefficients)
            )
            base_values[:, encoded_class] = baseline
        target_margins = np.asarray(model.decision_function(Xte_scaled))
        if target_margins.ndim == 1:
            target_margins = target_margins[:, None]
        model_values = Xte_scaled

    else:
        raise ValueError(f"PSD XAI is not implemented for {config.model}")

    reconstructed = np.sum(shap_values, axis=1) + base_values
    if reconstructed.shape != target_margins.shape:
        raise ValueError(
            f"Invalid {config.model} attribution output shape: "
            f"reconstructed={reconstructed.shape}, "
            f"model={target_margins.shape}"
        )
    if not np.allclose(
        reconstructed,
        target_margins,
        rtol=1e-4,
        atol=1e-4,
    ):
        max_error = float(np.max(np.abs(reconstructed - target_margins)))
        raise ValueError(
            f"Invalid {config.model} attribution reconstruction; "
            f"maximum output-margin error is {max_error:.6g}"
        )

    return (
        np.asarray(shap_values, dtype=np.float32),
        np.asarray(base_values, dtype=np.float32),
        np.asarray(model_values, dtype=np.float32),
    )


def save_psd_xai_artifacts(
    config: RunConfig,
    test: List[Item],
    prediction: np.ndarray,
    artifacts: VectorEvaluationArtifacts,
) -> str:
    """Persist PSD inputs and class-specific attributions without pickling models."""
    os.makedirs(XAI_DIR, exist_ok=True)
    shap_values, base_values, model_values = compute_psd_attributions(
        config,
        artifacts,
    )
    inv_label_map = {
        encoded: original for original, encoded in artifacts.label_map.items()
    }
    template_ids = np.asarray(
        [inv_label_map[i] for i in range(len(inv_label_map))],
        dtype=np.int32,
    )

    path = xai_npz_path(config)
    tmp_path = f"{path}.tmp.{os.getpid()}.npz"
    attribution_method = (
        "XGBoost TreeSHAP contributions"
        if config.model == "xgboost"
        else "linear SHAP decomposition after standardisation"
    )
    np.savez_compressed(
        tmp_path,
        shap_values=shap_values,
        base_values=base_values,
        raw_feature_values=np.asarray(artifacts.Xte, dtype=np.float32),
        model_feature_values=model_values,
        y_true=artifacts.yte.astype(np.int32),
        y_pred=np.asarray(prediction, dtype=np.int32),
        template_ids=template_ids,
        query_ids=np.asarray([item.query_id for item in test], dtype=np.int64),
        query_names=np.asarray([item.query_name for item in test], dtype=str),
        feature_names=np.asarray(PSD_FEATURE_NAMES, dtype=str),
        attribution_method=np.asarray(attribution_method),
        explained_output=np.asarray("class-specific raw decision margin"),
    )
    os.replace(tmp_path, path)

    summary_rows: List[Dict[str, Any]] = []
    global_values = np.mean(np.abs(shap_values), axis=(0, 2))
    global_total = float(np.sum(global_values))
    for feature, importance in zip(PSD_FEATURE_NAMES, global_values):
        summary_rows.append(
            {
                "model": config.model,
                "scope": "global",
                "template_id": "",
                "feature": feature,
                "mean_abs_shap": f"{float(importance):.10g}",
                "relative_importance": f"{ratio(float(importance), global_total):.10g}",
            }
        )

    class_values = np.mean(np.abs(shap_values), axis=0).T
    for encoded_class, template_id in enumerate(template_ids):
        class_total = float(np.sum(class_values[encoded_class]))
        for feature, importance in zip(
            PSD_FEATURE_NAMES,
            class_values[encoded_class],
        ):
            summary_rows.append(
                {
                    "model": config.model,
                    "scope": "class",
                    "template_id": str(int(template_id)),
                    "feature": feature,
                    "mean_abs_shap": f"{float(importance):.10g}",
                    "relative_importance": f"{ratio(float(importance), class_total):.10g}",
                }
            )

    summary_path = os.path.join(
        XAI_DIR,
        f"psd_{config.model}_importance_train_{config.train_k}.csv",
    )
    write_csv_atomic(
        summary_path,
        [
            "model",
            "scope",
            "template_id",
            "feature",
            "mean_abs_shap",
            "relative_importance",
        ],
        summary_rows,
    )
    return path


def upsert_psd_ablation(
    config: RunConfig,
    artifacts: VectorEvaluationArtifacts,
    baseline_prediction: np.ndarray,
) -> None:
    """Measure the effect of removing each PSD feature from one classifier."""
    Xtr = np.asarray(artifacts.Xtr)
    Xte = np.asarray(artifacts.Xte)
    baseline_acc = float(accuracy_score(artifacts.yte, baseline_prediction))
    baseline_f1 = float(
        f1_score(artifacts.yte, baseline_prediction, average="macro", zero_division=0)
    )

    new_rows: List[Dict[str, Any]] = [
        {
            "train_k": str(config.train_k),
            "model": config.model,
            "removed_feature": "none",
            "acc": f"{baseline_acc:.8f}",
            "f1": f"{baseline_f1:.8f}",
            "delta_acc": "0.00000000",
            "delta_f1": "0.00000000",
        }
    ]

    for feature_index, feature_name in enumerate(PSD_FEATURE_NAMES):
        Xtr_reduced = np.delete(Xtr, feature_index, axis=1)
        Xte_reduced = np.delete(Xte, feature_index, axis=1)
        classifier = fit_vector_model(config.model, Xtr_reduced, artifacts.ytr)
        prediction = predict_vector_model(
            config.model,
            classifier,
            Xte_reduced,
            ytr=artifacts.ytr,
        )
        acc = float(accuracy_score(artifacts.yte, prediction))
        f1_value = float(
            f1_score(artifacts.yte, prediction, average="macro", zero_division=0)
        )
        new_rows.append(
            {
                "train_k": str(config.train_k),
                "model": config.model,
                "removed_feature": feature_name,
                "acc": f"{acc:.8f}",
                "f1": f"{f1_value:.8f}",
                "delta_acc": f"{acc - baseline_acc:.8f}",
                "delta_f1": f"{f1_value - baseline_f1:.8f}",
            }
        )

    existing: List[Dict[str, Any]] = []
    if os.path.exists(PSD_ABLATION_CSV) and os.path.getsize(PSD_ABLATION_CSV) > 0:
        with open(PSD_ABLATION_CSV, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != PSD_ABLATION_FIELDS:
                raise ValueError(
                    f"Unexpected ablation schema in {PSD_ABLATION_CSV}: "
                    f"{reader.fieldnames}"
                )
            for row in reader:
                if not (
                    int(row["train_k"]) == config.train_k
                    and row["model"] == config.model
                ):
                    existing.append(row)

    existing.extend(new_rows)
    existing.sort(
        key=lambda row: (
            int(row["train_k"]),
            row["model"],
            row["removed_feature"],
        )
    )
    write_csv_atomic(PSD_ABLATION_CSV, PSD_ABLATION_FIELDS, existing)


def result_as_dict(row: ResultRow) -> Dict[str, str]:
    """Convert a result to the stable CSV representation."""
    return {
        "train_k": str(row.train_k),
        "method": row.method,
        "model": row.model,
        "acc": f"{row.acc:.4f}",
        "f1": f"{row.f1:.4f}",
        "fit_s": f"{row.fit_s:.6f}",
        "build_test_s": f"{row.build_test_s:.6f}",
        "pred_s": f"{row.pred_s:.6f}",
        "latency_ms": f"{row.latency_ms:.6f}",
    }


def csv_row_key(row: Dict[str, str]) -> Tuple[int, str, str]:
    """Return the unique key of a stored result row."""
    return (int(row["train_k"]), row["method"], row["model"])


def read_result_rows(path: str) -> List[Dict[str, str]]:
    """Read existing results, returning an empty list when absent."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_FIELDS:
            raise ValueError(
                f"Unexpected result schema in {path}: {reader.fieldnames}"
            )
        return list(reader)


def completed_result_keys(path: str) -> Set[Tuple[int, str, str]]:
    """Return the configurations already present in the result file."""
    return {csv_row_key(row) for row in read_result_rows(path)}


def has_psd_ablation(config: RunConfig) -> bool:
    """Return whether the selected PSD model has a complete ablation table."""
    if not os.path.exists(PSD_ABLATION_CSV):
        return False
    with open(PSD_ABLATION_CSV, newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["train_k"]) == config.train_k
            and row["model"] == config.model
        ]
    expected = {"none", *PSD_FEATURE_NAMES}
    return {row["removed_feature"] for row in rows} == expected


def configuration_artifacts_complete(
    config: RunConfig,
    completed_results: Set[Tuple[int, str, str]],
) -> bool:
    """Require both aggregate results and the artifacts used by later figures."""
    if config.key not in completed_results or not os.path.exists(prediction_path(config)):
        return False
    if is_psd_xai_config(config):
        return os.path.exists(xai_npz_path(config)) and has_psd_ablation(config)
    return True


def upsert_result(path: str, result: ResultRow) -> None:
    """Checkpoint one result atomically, replacing the same configuration."""
    rows = [row for row in read_result_rows(path) if csv_row_key(row) != (
        result.train_k,
        result.method,
        result.model,
    )]
    rows.append(result_as_dict(result))
    rows.sort(key=lambda row: csv_row_key(row))

    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def append_controller_error(config: RunConfig, returncode: int) -> None:
    """Upsert a worker failure without duplicating earlier attempts."""
    fields = ["train_k", "method", "model", "returncode"]
    rows: List[Dict[str, Any]] = []
    if os.path.exists(ERRORS_CSV) and os.path.getsize(ERRORS_CSV) > 0:
        with open(ERRORS_CSV, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fields:
                raise ValueError(f"Unexpected error schema in {ERRORS_CSV}")
            rows = [
                row
                for row in reader
                if (
                    int(row["train_k"]),
                    row["method"],
                    row["model"],
                )
                != config.key
            ]
    rows.append(
        {
            "train_k": str(config.train_k),
            "method": config.method,
            "model": config.result_model,
            "returncode": str(returncode),
        }
    )
    rows.sort(
        key=lambda row: (
            int(row["train_k"]),
            row["method"],
            row["model"],
        )
    )
    write_csv_atomic(ERRORS_CSV, fields, rows)


def clear_controller_error(config: RunConfig) -> None:
    """Remove a previous failure after the same configuration succeeds."""
    fields = ["train_k", "method", "model", "returncode"]
    if not os.path.exists(ERRORS_CSV) or os.path.getsize(ERRORS_CSV) == 0:
        return
    with open(ERRORS_CSV, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected error schema in {ERRORS_CSV}")
        rows = [
            row
            for row in reader
            if (
                int(row["train_k"]),
                row["method"],
                row["model"],
            )
            != config.key
        ]
    write_csv_atomic(ERRORS_CSV, fields, rows)


def all_run_configs() -> List[RunConfig]:
    """Build the complete set of article configurations."""
    configs: List[RunConfig] = []
    for train_k in TRAIN_K_LIST:
        for method in VECTOR_METHODS:
            if method in KNN_VECTOR_METHODS:
                for knn_k in KNN_K_VALUES:
                    configs.append(RunConfig(train_k, method, "knn", knn_k))
            for model in SUPERVISED_MODELS:
                configs.append(RunConfig(train_k, method, model))

        for method in SET_METHODS:
            for knn_k in KNN_K_VALUES:
                configs.append(RunConfig(train_k, method, "knn", knn_k))
    return configs


def validate_run_config(config: RunConfig) -> None:
    """Reject method/model combinations not used in the article."""
    if config.train_k not in TRAIN_K_LIST:
        raise ValueError(f"Unsupported training size: {config.train_k}")

    if config.method in VECTOR_METHODS:
        if config.model == "knn":
            if config.method not in KNN_VECTOR_METHODS:
                raise ValueError(f"Vector k-NN is not reported for {config.method}")
            if config.knn_k not in KNN_K_VALUES:
                raise ValueError(f"Unsupported k-NN value: {config.knn_k}")
        elif config.model not in SUPERVISED_MODELS:
            raise ValueError(f"Unsupported vector model: {config.model}")
        elif config.knn_k is not None:
            raise ValueError("--knn-k applies only to the knn model")
        return

    if config.method in SET_METHODS:
        if config.model != "knn" or config.knn_k not in KNN_K_VALUES:
            raise ValueError("Set-based methods require knn with k equal to 1 or 5")
        return

    raise ValueError(f"Unknown method: {config.method}")


def maybe_plot_selected_confusion(
    config: RunConfig,
    train: List[Item],
    test: List[Item],
    prediction: np.ndarray,
) -> None:
    """Generate the four confusion matrices reported for the largest history."""
    if config.train_k != max(TRAIN_K_LIST):
        return

    basename: Optional[str] = None
    if config.method == "scalars" and config.model == "xgboost":
        basename = "confusion_psd_xgboost"
    elif config.method == "token_tfidf" and config.model == "random_forest":
        basename = "confusion_token_tf_rf"
    elif (
        config.method == "lineage_jaccard"
        and config.model == "knn"
        and config.knn_k == 5
    ):
        basename = "confusion_lineage_knn"
    elif (
        config.method == "why_blocked"
        and config.model == "knn"
        and config.knn_k == 5
    ):
        basename = "confusion_blocked_knn"

    if basename is None:
        return

    label_map = build_label_mapping(train)
    y_true = encode_labels(test, label_map)
    plot_confusion_matrix(
        y_true,
        prediction,
        label_map,
        basename,
    )


def selected_confusion_configs() -> List[Tuple[RunConfig, str]]:
    """Return the confusion matrices selected for the article."""
    train_k = max(TRAIN_K_LIST)
    return [
        (RunConfig(train_k, "scalars", "xgboost"), "confusion_psd_xgboost"),
        (
            RunConfig(train_k, "token_tfidf", "random_forest"),
            "confusion_token_tf_rf",
        ),
        (
            RunConfig(train_k, "lineage_jaccard", "knn", 5),
            "confusion_lineage_knn",
        ),
        (
            RunConfig(train_k, "why_blocked", "knn", 5),
            "confusion_blocked_knn",
        ),
    ]


def plot_saved_confusion(config: RunConfig, basename: Optional[str] = None) -> bool:
    """Recreate a confusion matrix using only a saved prediction CSV."""
    path = prediction_path(config)
    if not os.path.exists(path):
        print(f"[PLOT SKIP] missing predictions: {path}")
        return False

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PREDICTION_FIELDS:
            raise ValueError(f"Unexpected prediction schema in {path}")
        rows = list(reader)

    true_templates = np.asarray(
        [int(row["true_template"]) for row in rows],
        dtype=np.int32,
    )
    predicted_templates = np.asarray(
        [int(row["predicted_template"]) for row in rows],
        dtype=np.int32,
    )
    templates = sorted(set(true_templates) | set(predicted_templates))
    label_map = {template: position for position, template in enumerate(templates)}
    y_true = np.asarray([label_map[value] for value in true_templates], dtype=np.int32)
    y_pred = np.asarray(
        [label_map[value] for value in predicted_templates],
        dtype=np.int32,
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        label_map,
        basename or f"confusion_{config_stem(config)}",
    )
    return True


def plot_accuracy_by_training_size() -> bool:
    """Plot the article's selected methods across all training-history sizes."""
    rows = read_result_rows(RESULTS_CSV)
    if not rows:
        print(f"[PLOT SKIP] no aggregate results: {RESULTS_CSV}")
        return False

    indexed = {csv_row_key(row): row for row in rows}
    series = [
        ("Blocked Why-provenance", "why_blocked", "5-nn", "#7A5195", "o"),
        ("Lineage", "lineage_jaccard", "5-nn", "#2F6B9A", "s"),
        ("PSD + XGBoost", "scalars", "xgboost", "#D9782D", "D"),
        ("Token TF + Random Forest", "token_tfidf", "random_forest", "#2E8B57", "^")
    ]

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    plotted = 0
    for label, method, model, colour, marker in series:
        points = [
            (train_k, indexed[(train_k, method, model)])
            for train_k in TRAIN_K_LIST
            if (train_k, method, model) in indexed
        ]
        if not points:
            print(f"[PLOT SKIP SERIES] {method}/{model}")
            continue
        plotted += 1
        ax.plot(
            [train_k for train_k, _ in points],
            [float(row["acc"]) for _, row in points],
            label=label,
            color=colour,
            marker=marker,
            linewidth=1.8,
            markersize=4.5,
        )

    if plotted == 0:
        plt.close(fig)
        return False
    ax.set_xlabel("Training executions per template")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(TRAIN_K_LIST)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "accuracy_vs_training_size")
    return True


def load_xai_artifact(model: str) -> Optional[Dict[str, np.ndarray]]:
    """Load one selected PSD attribution artifact into independent arrays."""
    config = RunConfig(max(TRAIN_K_LIST), "scalars", model)
    path = xai_npz_path(config)
    if not os.path.exists(path):
        print(f"[PLOT SKIP] missing XAI artifact: {path}")
        return None
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def plot_psd_global_importance() -> bool:
    """Compare normalised global PSD importance across the two classifiers."""
    loaded = {
        model: load_xai_artifact(model)
        for model in ("xgboost", "logreg")
    }
    available = {model: data for model, data in loaded.items() if data is not None}
    if not available:
        return False

    importances: Dict[str, np.ndarray] = {}
    for model, data in available.items():
        values = np.mean(np.abs(data["shap_values"]), axis=(0, 2))
        importances[model] = values / max(float(np.sum(values)), np.finfo(float).eps)

    combined = np.mean(np.vstack(list(importances.values())), axis=0)
    order = np.argsort(combined)
    y = np.arange(len(PSD_FEATURE_NAMES), dtype=float)
    width = 0.34 if len(importances) > 1 else 0.55
    colours = {"xgboost": "#D9782D", "logreg": "#2F6B9A"}
    labels = {"xgboost": "XGBoost", "logreg": "Logistic Regression"}

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    models = list(importances)
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * width
    for offset, model in zip(offsets, models):
        ax.barh(
            y + offset,
            importances[model][order],
            height=width,
            color=colours[model],
            label=labels[model],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([PSD_FEATURE_LABELS[index] for index in order])
    ax.set_xlabel("Share of mean absolute SHAP value")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "xai_psd_global_importance")
    return True


def plot_psd_class_importance() -> bool:
    """Plot class-specific relative PSD importance for XGBoost."""
    data = load_xai_artifact("xgboost")
    if data is None:
        return False
    values = np.mean(np.abs(data["shap_values"]), axis=0).T
    row_totals = values.sum(axis=1, keepdims=True)
    relative = np.divide(
        values,
        row_totals,
        out=np.zeros_like(values),
        where=row_totals != 0,
    )
    labels = [f"Q{int(value)}" for value in data["template_ids"]]

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    sns.heatmap(
        relative,
        cmap="YlGnBu",
        xticklabels=PSD_FEATURE_LABELS,
        yticklabels=labels,
        vmin=0.0,
        cbar_kws={"label": "Relative mean absolute SHAP value"},
        ax=ax,
    )
    ax.set_xlabel("PSD feature")
    ax.set_ylabel("Explained template")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save_figure(fig, "xai_psd_xgboost_by_template")
    return True


def plot_psd_true_class_beeswarm(model: str) -> bool:
    """Plot PSD contributions towards each execution's true template."""
    data = load_xai_artifact(model)
    if data is None:
        return False
    shap_values = data["shap_values"]
    raw_values = data["raw_feature_values"]
    y_true = data["y_true"].astype(np.int32)
    n_samples, n_features, _ = shap_values.shape
    rows = np.arange(n_samples)[:, None]
    features = np.arange(n_features)[None, :]
    true_values = shap_values[rows, features, y_true[:, None]]
    order = np.argsort(np.mean(np.abs(true_values), axis=0))

    rng = np.random.default_rng(RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    scatter = None
    for display_position, feature_index in enumerate(order):
        feature_values = raw_values[:, feature_index]
        ranks = np.argsort(np.argsort(feature_values, kind="mergesort"), kind="mergesort")
        colour = ranks / max(1, n_samples - 1)
        jitter = rng.uniform(-0.24, 0.24, size=n_samples)
        scatter = ax.scatter(
            true_values[:, feature_index],
            display_position + jitter,
            c=colour,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=6,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
        )
    ax.axvline(0.0, color="0.35", linewidth=0.8)
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels([PSD_FEATURE_LABELS[index] for index in order])
    ax.set_xlabel("SHAP contribution to the true template")
    ax.grid(axis="x", alpha=0.2)
    if scatter is not None:
        colour_bar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colour_bar.set_ticks([0.0, 1.0])
        colour_bar.set_ticklabels(["Low", "High"])
        colour_bar.set_label("Feature value")
    fig.tight_layout()
    save_figure(fig, f"xai_psd_{model}_beeswarm")
    return True


def plot_psd_ablation() -> bool:
    """Plot macro-F1 loss after removing each PSD descriptor."""
    if not os.path.exists(PSD_ABLATION_CSV):
        print(f"[PLOT SKIP] missing ablation results: {PSD_ABLATION_CSV}")
        return False
    with open(PSD_ABLATION_CSV, newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["removed_feature"] != "none"
            and int(row["train_k"]) == max(TRAIN_K_LIST)
        ]
    if not rows:
        return False

    indexed = {
        (row["model"], row["removed_feature"]): -float(row["delta_f1"])
        for row in rows
    }
    x = np.arange(len(PSD_FEATURE_NAMES), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for offset, model, label, colour in [
        (-width / 2, "xgboost", "XGBoost", "#D9782D"),
        (width / 2, "logreg", "Logistic Regression", "#2F6B9A"),
    ]:
        if not any(key[0] == model for key in indexed):
            continue
        values = [indexed.get((model, feature), np.nan) for feature in PSD_FEATURE_NAMES]
        ax.bar(x + offset, values, width=width, label=label, color=colour)
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(PSD_FEATURE_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Macro-F1 loss after feature removal")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "xai_psd_ablation")
    return True


def generate_article_figures() -> None:
    """Regenerate every article figure supported by persisted artifacts."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for config, basename in selected_confusion_configs():
        plot_saved_confusion(config, basename)
    plot_accuracy_by_training_size()
    plot_psd_global_importance()
    plot_psd_class_importance()
    plot_psd_true_class_beeswarm("xgboost")
    plot_psd_true_class_beeswarm("logreg")
    plot_psd_ablation()
    print(f"[PLOTS] available figures written to {FIGURES_DIR}")


def run_one_config(config: RunConfig, create_plots: bool = True) -> ResultRow:
    """Load and evaluate exactly one configuration in the worker process."""
    validate_run_config(config)
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    why_test_files: List[str] = []
    why_test_count = 0
    if config.method == "why_blocked":
        train, why_test_files, why_test_count = prepare_streamed_why_splits(
            config.train_k
        )
        test: List[Item] = []
    elif config.method in VECTOR_METHODS:
        pool, test = load_all_splits([config.method], [])
    else:
        pool, test = load_all_splits([], [config.method])

    if config.method != "why_blocked":
        validate_splits(pool, test, required_train_k=config.train_k)
        train = select_k_per_template(pool, k=config.train_k, seed=RANDOM_SEED)
        del pool
        gc.collect()
    print(
        f"[WORKER] train_k={config.train_k} method={config.method} "
        f"model={config.result_model} train={len(train)} "
        f"test={why_test_count if config.method == 'why_blocked' else len(test)}"
    )

    vector_artifacts: Optional[VectorEvaluationArtifacts] = None
    if config.method in VECTOR_METHODS:
        result, prediction, vector_artifacts = eval_vector_method(
            train,
            test,
            config.method,
            config.model,
            knn_k=config.knn_k,
            collect_artifacts=is_psd_xai_config(config),
        )
    elif config.method == "lineage_jaccard":
        result, prediction = eval_jaccard_custom(
            train,
            test,
            "lineage",
            knn_k=int(config.knn_k),
        )
    elif config.method == "atomic_why_jaccard":
        if not any(item.atomic_why for item in train + test):
            raise ValueError("Atomic Why-provenance is empty in every execution")
        result, prediction = eval_jaccard_custom(
            train,
            test,
            "atomic_why",
            knn_k=int(config.knn_k),
        )
    elif config.method == "why_blocked":
        # Only selected training provenance remains resident. Test executions
        # are loaded, classified, and released one at a time.
        build_why_cache(train, build_inverted=False)
        gc.collect()
        if not any(item.why_list for item in train):
            raise ValueError("Why-provenance is empty in every training execution")
        result, prediction, test = eval_why_blocked_streamed(
            train,
            why_test_files,
            why_test_count,
            knn_k=int(config.knn_k),
        )
    else:
        raise ValueError(config.method)

    result.train_k = config.train_k
    label_map = build_label_mapping(train)
    upsert_result(RESULTS_CSV, result)
    saved_predictions = save_predictions(
        config,
        test,
        label_map,
        prediction,
    )
    save_per_class_metrics(config, test, label_map, prediction)
    if create_plots:
        maybe_plot_selected_confusion(config, train, test, prediction)

    if is_psd_xai_config(config):
        if vector_artifacts is None:
            raise RuntimeError("PSD XAI configuration did not retain fitted artifacts")
        xai_path = save_psd_xai_artifacts(
            config,
            test,
            prediction,
            vector_artifacts,
        )
        upsert_psd_ablation(config, vector_artifacts, prediction)
        print(f"[XAI] attributions={xai_path} ablation={PSD_ABLATION_CSV}")

    print(
        f"[OK] acc={result.acc:.4f} f1={result.f1:.4f} "
        f"latency={result.latency_ms:.3f} ms result={RESULTS_CSV} "
        f"predictions={saved_predictions}"
    )
    return result


def parse_args() -> argparse.Namespace:
    """Parse controller filters and the internal worker arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate provenance representations in isolated processes. "
            "Existing CSV results are skipped automatically."
        )
    )
    parser.add_argument("--method", choices=VECTOR_METHODS + SET_METHODS)
    parser.add_argument("--model", choices=["knn"] + SUPERVISED_MODELS)
    parser.add_argument("--train-k", type=int, choices=TRAIN_K_LIST)
    parser.add_argument("--knn-k", type=int, choices=KNN_K_VALUES)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run matching configurations even when their results exist",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="regenerate figures from saved CSV and NPZ artifacts without loading data",
    )
    parser.add_argument(
        "--all-confusions",
        action="store_true",
        help="with --plots-only, generate a matrix for every saved configuration",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="store experiment artifacts without creating or replacing figures",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def filtered_configs(args: argparse.Namespace) -> List[RunConfig]:
    """Apply optional command-line filters to the full experiment matrix."""
    configs = all_run_configs()
    if args.method is not None:
        configs = [config for config in configs if config.method == args.method]
    if args.model is not None:
        configs = [config for config in configs if config.model == args.model]
    if args.train_k is not None:
        configs = [config for config in configs if config.train_k == args.train_k]
    if args.knn_k is not None:
        configs = [config for config in configs if config.knn_k == args.knn_k]
    return configs


def run_plots_only(args: argparse.Namespace) -> int:
    """Recreate figures without loading provenance or fitting any model."""
    generate_article_figures()
    filters_used = any(
        value is not None
        for value in (args.method, args.model, args.train_k, args.knn_k)
    )
    if args.all_confusions or filters_used:
        configs = filtered_configs(args)
        created = 0
        for config in configs:
            if plot_saved_confusion(config):
                created += 1
        print(f"[PLOTS] additional confusion matrices created={created}")
    return 0


def run_controller(args: argparse.Namespace) -> int:
    """Run matching configurations sequentially in fresh child processes."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configs = filtered_configs(args)
    if not configs:
        raise ValueError("No valid experimental configurations match the filters")

    completed = completed_result_keys(RESULTS_CSV)
    failures = 0
    script = os.path.abspath(__file__)

    print(
        f"[CONTROLLER] selected={len(configs)} complete_with_artifacts="
        f"{sum(configuration_artifacts_complete(config, completed) for config in configs)}"
    )

    for position, config in enumerate(configs, start=1):
        if not args.force and configuration_artifacts_complete(config, completed):
            print(f"[SKIP {position}/{len(configs)}] {config.key}")
            continue

        command = [
            sys.executable,
            script,
            "--worker",
            "--train-k",
            str(config.train_k),
            "--method",
            config.method,
            "--model",
            config.model,
        ]
        if config.knn_k is not None:
            command.extend(["--knn-k", str(config.knn_k)])
        if args.skip_plots:
            command.append("--skip-plots")

        print(f"\n[START {position}/{len(configs)}] {config.key}", flush=True)
        worker = subprocess.run(command, check=False)
        if worker.returncode == 0:
            completed.add(config.key)
            clear_controller_error(config)
        else:
            failures += 1
            append_controller_error(config, worker.returncode)
            reason = "killed (probable OOM)" if worker.returncode == -9 else "failed"
            print(
                f"[WORKER {reason}] key={config.key} "
                f"returncode={worker.returncode}",
                flush=True,
            )

    if not args.skip_plots:
        generate_article_figures()
    print(
        f"\n[DONE] completed={len(configs) - failures} "
        f"failures={failures} results={RESULTS_CSV}"
    )
    return 1 if failures else 0


def main() -> int:
    """Run one worker or coordinate the requested isolated experiments."""
    args = parse_args()
    if args.all_confusions and not args.plots_only:
        raise ValueError("--all-confusions requires --plots-only")
    if args.plots_only and args.skip_plots:
        raise ValueError("--plots-only cannot be combined with --skip-plots")
    if args.plots_only:
        if args.worker:
            raise ValueError("--plots-only cannot be combined with --worker")
        return run_plots_only(args)
    if args.worker:
        if args.method is None or args.model is None or args.train_k is None:
            raise ValueError("A worker requires --method, --model, and --train-k")
        config = RunConfig(
            train_k=args.train_k,
            method=args.method,
            model=args.model,
            knn_k=args.knn_k,
        )
        run_one_config(config, create_plots=not args.skip_plots)
        return 0
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
