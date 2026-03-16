import csv
import glob
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Optional, Set

import numpy as np
import pyarrow.parquet as pq

from scipy import sparse
from scipy.sparse import csr_matrix

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

"""
    Global configuration parameters for training and evaluation.
    Modify these values to adapt the experiments to your environment.
"""

# ------------------------------------------------------------------
# Data paths
# ------------------------------------------------------------------
DATA_ROOT = "."  # Root directory containing dataset folders

# Glob patterns to locate training and testing parquet files
TRAIN_GLOB = os.path.join(DATA_ROOT, "**", "q*_train", "*.parquet")
TEST_GLOB = os.path.join(DATA_ROOT, "**", "q*_test", "*.parquet")


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
# Machine learning models evaluated
# ------------------------------------------------------------------
ML_MODELS = [
    "knn",  # k-nearest neighbors
    "logreg",  # logistic regression
    "naive_bayes",  # multinomial Naive Bayes
    "random_forest",  # random forest classifier
    "xgboost",  # gradient boosted trees
]


# ------------------------------------------------------------------
# KNN parameters
# ------------------------------------------------------------------
SKLEARN_KNN_K = 5  # Number of neighbors for sklearn KNN
CUSTOM_KNN_K = 5  # Number of neighbors for custom KNN implementation


# ------------------------------------------------------------------
# Hash dimensions for feature hashing
# ------------------------------------------------------------------
HASH_D_LINEAGE = 1 << 15  # Hash dimension for lineage vectors
HASH_D_ATOMIC_WHY = 1 << 14  # Hash dimension for atomic why vectors
HASH_D_TOKEN_TF = 1 << 15  # Hash dimension for token TF-IDF features
HASH_D_WITNESS_TF = 1 << 14  # Hash dimension for witness TF-IDF features


# ------------------------------------------------------------------
# Blocked why-provenance matching parameters
# ------------------------------------------------------------------
WHY_LN_TOPR = 50  # Number of top lineage candidates
WHY_MAX_CANDIDATES = 25  # Maximum candidate matches to evaluate
WHY_AVG_W_TOL = 0.75  # Average weight tolerance threshold
WHY_HIST_COS_MIN = 0.80  # Minimum cosine similarity for histogram filtering

"""
    DATA MODEL
"""


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
    witnesses: List[frozenset[int]]

    token_tf: List[Tuple[int, int]]
    witness_tf: List[Tuple[int, int]]

    nwitness_unique: int
    nwitness_total: int
    ntokens_unique: int
    ntokens_total: int

    # Auxiliary structures used by blocked why-provenance matching
    why_list: List[List[int]] = field(default_factory=list)
    why_inv: Dict[int, List[int]] = field(default_factory=dict)
    avg_w: float = 0.0
    hist_w: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.float32))


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


def _as_why_list(x: Any) -> List[frozenset[int]]:
    """
    Convert a witness collection into a list of frozensets of integers.

    Each witness is normalized as an immutable set for stable downstream use.
    """
    if x is None:
        return []
    out: List[frozenset[int]] = []
    for w in x:
        if w is None:
            continue
        out.append(frozenset(int(v) for v in w))
    return out


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
        or "lineage_jaccard_full" in set_methods
        or "lineage_jaccard_hashed" in set_methods
        or "why_blocked" in set_methods
    ):
        cols.add("lineage_hash")

    if (
        "atomic_why_vec" in vector_methods
        or "atomic_why_jaccard_full" in set_methods
        or "atomic_why_jaccard_hashed" in set_methods
    ):
        cols.add("monoids_hash")

    if "why_blocked" in set_methods:
        cols.add("witnesses")

    return sorted(cols)


def load_items_one_file(path: str, columns: List[str]) -> List[Item]:
    """
    Load one parquet file and convert its rows into Item objects.

    Only the requested columns are read when they exist in the schema.
    """
    schema_names = pq.read_schema(path).names
    use_cols = [c for c in columns if c in schema_names]
    table = pq.read_table(path, columns=use_cols)
    rows = table.to_pylist()

    items: List[Item] = []
    for r in rows:
        qid = _safe_int(r.get("query_id"), -1)
        tid = _safe_int(r.get("template_id"), -1)
        qname = str(r.get("query_name", qid))

        items.append(
            Item(
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
        )
    return items


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


def wsize_hist(why_list: List[List[int]], max_bin: int = 8) -> np.ndarray:
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


def build_why_cache(items: List[Item]) -> None:
    """
    Precompute auxiliary structures for blocked why-provenance similarity.

    For each item, this function builds:
    - why_list: sorted witnesses as lists of integers
    - why_inv: inverted index from token to witness indices
    - avg_w: average witness size
    - hist_w: normalized witness-size histogram
    """
    for it in items:
        wl: List[List[int]] = []
        inv: Dict[int, List[int]] = defaultdict(list)

        for j, w in enumerate(it.witnesses):
            lst = sorted(int(x) for x in w)
            wl.append(lst)
            for t in lst:
                inv[t].append(j)

        it.why_list = wl
        it.why_inv = dict(inv)
        it.avg_w = float(np.mean([len(x) for x in wl])) if wl else 0.0
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


def jaccard_witness_list(a: List[int], b: List[int]) -> float:
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


# acho que pode desaparecer
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity assuming the input vectors are already normalized.
    """
    return float(np.dot(a, b))


def predict_knn_vote(scores: np.ndarray, train_tids: np.ndarray, k: int) -> int:
    """
    Predict a class label using weighted k-nearest-neighbor voting.

    The top-k highest-scoring neighbors are selected, and their scores
    are accumulated per training template label. Ties are broken by
    preferring the smaller template identifier.
    """
    if scores.size == 0:
        return -1

    k = min(k, len(scores))
    idx = np.argpartition(scores, len(scores) - k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]

    vote = defaultdict(float)
    for i in idx:
        vote[train_tids[i]] += scores[i]

    best_tid = max(vote.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return int(best_tid)


def why_similarity_soft_blocked_cached(
    q_wl: List[List[int]],
    q_inv: Dict[int, List[int]],
    cand_wl: List[List[int]],
    cand_inv: Dict[int, List[int]],
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
        X_wl: List[List[int]], Y_wl: List[List[int]], Y_inv: Dict[int, List[int]]
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
                v = jaccard_witness_list(x, Y_wl[j])
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


def hashed_set(s: Set[int], dim: int) -> Set[int]:
    """
    Hash all elements of a set into a fixed-dimensional index space.
    """
    return {_hash_to_index(v, dim) for v in s}


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


def fit_vector_model(model_name: str, Xtr, ytr):
    """
    Fit a vector-based classifier on the training data.

    Depending on the selected model, this may include feature scaling
    and the use of sparse or dense representations.
    """
    if model_name == "knn":
        scaler = (
            StandardScaler(with_mean=False)
            if sparse.issparse(Xtr)
            else StandardScaler()
        )
        Xtr_s = scaler.fit_transform(Xtr)
        nn = NearestNeighbors(n_neighbors=SKLEARN_KNN_K, metric="cosine")
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
        # Random forest expects dense input in this implementation.
        Xtr_d = Xtr.toarray() if sparse.issparse(Xtr) else Xtr
        return RandomForestClassifier(random_state=RANDOM_SEED).fit(Xtr_d, ytr)

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
            c = Counter(neigh_labels[i].tolist())
            out[i] = max(c.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        return out

    if model_name == "random_forest":
        Xte = Xte.toarray() if sparse.issparse(Xte) else Xte
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


"""
    EVALUATION: VECTOR METHODS
"""


def eval_vector_method(
    train: List[Item],
    test: List[Item],
    method: str,
    model_name: str,
) -> Tuple[ResultRow, np.ndarray]:
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
    clf = fit_vector_model(model_name, Xtr, ytr)
    fit_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    pred = predict_vector_model(model_name, clf, Xte, ytr=ytr)
    pred_s = time.perf_counter() - t2

    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
    latency_ms = ((build_test_s + pred_s) / max(1, len(test))) * 1000.0

    return (
        ResultRow(
            train_k=0,
            method=method,
            model=model_name,
            acc=acc,
            f1=f1,
            fit_s=fit_s,
            build_test_s=build_test_s,
            pred_s=pred_s,
            latency_ms=latency_ms,
        ),
        pred,
    )


"""    
    EVALUATION: CUSTOM PROVENANCE NEAREST NEIGHBOR
"""


def eval_jaccard_custom(
    train: List[Item],
    test: List[Item],
    method: str,
    hashed: bool,
    dim: Optional[int] = None,
) -> Tuple[ResultRow, np.ndarray]:
    """
    Evaluate a custom k-NN classifier based on Jaccard similarity.

    Depending on the selected method, this function compares either
    lineage sets or atomic why-provenance sets. Optionally, the sets
    can be hashed into a fixed-dimensional space before comparison.
    """
    if method == "lineage":
        get_rep = lambda it: it.lineage
    elif method == "atomic_why":
        get_rep = lambda it: it.atomic_why
    else:
        raise ValueError(method)

    label_map = build_label_mapping(train)

    # Build the training representations used during similarity search.
    t0 = time.perf_counter()
    if hashed:
        assert dim is not None
        train_reps = [hashed_set(get_rep(it), dim) for it in train]
    else:
        train_reps = [get_rep(it) for it in train]

    train_tids = encode_labels(train, label_map)
    fit_s = time.perf_counter() - t0

    y_true = encode_labels(test, label_map)
    y_pred = np.full_like(y_true, -1)

    # Compare each test instance against all training representations.
    t1 = time.perf_counter()
    for i, q in enumerate(test):
        qrep = hashed_set(get_rep(q), dim) if hashed else get_rep(q)
        scores = np.empty(len(train_reps), dtype=np.float32)
        for j, tr in enumerate(train_reps):
            scores[j] = jaccard_set(qrep, tr)
        y_pred[i] = predict_knn_vote(scores, train_tids, CUSTOM_KNN_K)
    pred_s = time.perf_counter() - t1

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    build_test_s = 0.0
    latency_ms = ((build_test_s + pred_s) / max(1, len(test))) * 1000.0

    name = f"{method}_jaccard_hashed" if hashed else f"{method}_jaccard_full"

    return (
        ResultRow(
            train_k=0,
            method=name,
            model=f"{CUSTOM_KNN_K}-nn",
            acc=acc,
            f1=f1,
            fit_s=fit_s,
            build_test_s=build_test_s,
            pred_s=pred_s,
            latency_ms=latency_ms,
        ),
        y_pred,
    )


def eval_why_blocked(
    train: List[Item],
    test: List[Item],
) -> Tuple[ResultRow, np.ndarray]:
    """
    Evaluate the blocked why-provenance nearest-neighbor method.

    Candidate training instances are first filtered using lineage
    similarity and lightweight witness statistics. A more expensive
    blocked why-provenance similarity is then computed only for the
    gated candidates.
    """
    label_map = build_label_mapping(train)

    # Store the encoded labels and lineage representations used for gating.
    t0 = time.perf_counter()
    train_tids = encode_labels(train, label_map)
    train_ln = [it.lineage for it in train]
    fit_s = time.perf_counter() - t0

    y_true = encode_labels(test, label_map)
    y_pred = np.full_like(y_true, -1)

    t1 = time.perf_counter()
    for qi, q in enumerate(test):
        # First-stage ranking using lineage Jaccard similarity.
        ln_scores = np.array(
            [jaccard_set(q.lineage, tr) for tr in train_ln], dtype=np.float32
        )
        R = min(WHY_LN_TOPR, len(train))
        top_idx = np.argpartition(ln_scores, -R)[-R:]
        top_idx = top_idx[np.argsort(ln_scores[top_idx])[::-1]]

        # Gate the top lineage candidates using witness statistics.
        gated: List[int] = []
        for i in top_idx.tolist():
            a = train[i].avg_w
            if max(q.avg_w, a) > 0.0:
                rel = abs(q.avg_w - a) / max(q.avg_w, a)
                if rel > WHY_AVG_W_TOL:
                    continue
            if cosine(q.hist_w, train[i].hist_w) < WHY_HIST_COS_MIN:
                continue
            gated.append(i)
            if len(gated) >= WHY_MAX_CANDIDATES:
                break

        # Fallback: keep the best lineage-ranked candidates if gating is empty.
        if not gated:
            gated = top_idx[: min(len(top_idx), WHY_MAX_CANDIDATES)].tolist()

        # Second-stage similarity using blocked why-provenance matching.
        scores = np.empty(len(train), dtype=np.float32)
        scores.fill(0.0)
        for i in gated:
            scores[i] = float(
                why_similarity_soft_blocked_cached(
                    q.why_list,
                    q.why_inv,
                    train[i].why_list,
                    train[i].why_inv,
                    WHY_MAX_CANDIDATES,
                )
            )

        y_pred[qi] = predict_knn_vote(scores, train_tids, CUSTOM_KNN_K)

    pred_s = time.perf_counter() - t1

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    build_test_s = 0.0
    latency_ms = ((build_test_s + pred_s) / max(1, len(test))) * 1000.0

    # Estimate the memory footprint of the cached lineage and why structures.
    why_tokens = sum(len(w) for it in train for w in it.why_list)
    inv_postings = sum(len(v) for it in train for v in it.why_inv.values())

    return (
        ResultRow(
            train_k=0,
            method="why_blocked",
            model=f"{CUSTOM_KNN_K}-nn",
            acc=acc,
            f1=f1,
            fit_s=fit_s,
            build_test_s=build_test_s,
            pred_s=pred_s,
            latency_ms=latency_ms,
        ),
        y_pred,
    )


"""
    CONFUSION MATRIX PLOTTING
"""


def plot_confusion_matrix(y_true, y_pred, label_map, title, out_file):
    """
    Plot and save a normalized confusion matrix.

    Rows are normalized by the number of true instances per class.
    """
    inv_label_map = {v: k for k, v in label_map.items()}

    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = [f"Q{inv_label_map[i]}" for i in range(len(inv_label_map))]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        cmap="Blues",
        annot=False,
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
    )
    plt.xlabel("Predicted template")
    plt.ylabel("True template")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


# ============================================================
# RESULTS EXPORT
# ============================================================


def save_results_csv(path: str, rows: List[ResultRow]) -> None:
    """
    Save evaluation results to a CSV file.
    """
    header = [
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
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(
                [
                    r.train_k,
                    r.method,
                    r.model,
                    f"{r.acc:.4f}",
                    f"{r.f1:.4f}",
                    f"{r.fit_s:.6f}",
                    f"{r.build_test_s:.6f}",
                    f"{r.pred_s:.6f}",
                    f"{r.latency_ms:.6f}",
                ]
            )


"""
    MAIN
"""


def main() -> None:
    """
    Run the full evaluation pipeline.

    This includes:
    - loading train/test data
    - building optional why-provenance caches
    - evaluating all configured methods and models
    - generating selected confusion matrices
    - exporting the final CSV summary
    """
    np.random.seed(RANDOM_SEED)

    pool, test = load_all_splits(DATA_ROOT, VECTOR_METHODS, SET_METHODS)

    if "why_blocked" in SET_METHODS:
        build_why_cache(pool)
        build_why_cache(test)

    c_pool = Counter(it.template_id for it in pool)
    c_test = Counter(it.template_id for it in test)

    print(f"[POOL] templates={len(c_pool)} rows={len(pool)}")
    print(f"[TEST] templates={len(c_test)} rows={len(test)}")
    print(
        f"[POOL] min/max per template = {min(c_pool.values())} / {max(c_pool.values())}"
    )
    print(
        f"[TEST] min/max per template = {min(c_test.values())} / {max(c_test.values())}"
    )

    all_rows: List[ResultRow] = []

    # Store selected predictions for later confusion-matrix plotting.
    psd_xgb_pred = None
    lineage_knn_pred = None
    blocked_knn_pred = None
    token_tf_rf_pred = None
    y_true_global = None
    label_map_global = None

    for k in TRAIN_K_LIST:
        print("\n" + "=" * 100)
        print(f"[RUN] train_k_per_template={k} seed={RANDOM_SEED}")

        train = select_k_per_template(pool, k=k, seed=RANDOM_SEED)
        print(f"[TRAIN] size={len(train)}")

        if y_true_global is None:
            label_map_global = build_label_mapping(train)
            y_true_global = encode_labels(test, label_map_global)

        for method in VECTOR_METHODS:
            print(f"\n[VECTOR] {method}")
            for model_name in ML_MODELS:
                try:
                    row, pred = eval_vector_method(train, test, method, model_name)
                    row.train_k = k
                    all_rows.append(row)

                    if method == "scalars" and model_name == "xgboost":
                        psd_xgb_pred = pred

                    if method == "token_tfidf" and model_name == "random_forest":
                        token_tf_rf_pred = pred

                    print(
                        f"  {model_name:12s} "
                        f"acc={row.acc:.4f} f1={row.f1:.4f} "
                        f"fit={row.fit_s:.3f}s pred={row.pred_s:.3f}s "
                        f"latency={row.latency_ms:.3f} ms "
                    )
                except Exception as e:
                    print(f"  {model_name:12s} SKIP ({e})")

        if "lineage_jaccard" in SET_METHODS:
            print("\n[CUSTOM] lineage_jaccard")
            row, pred = eval_jaccard_custom(train, test, "lineage", hashed=False)
            row.train_k = k
            all_rows.append(row)
            lineage_knn_pred = pred
            print(
                f"  {row.model:12s} acc={row.acc:.4f} f1={row.f1:.4f} "
                f"fit={row.fit_s:.3f}s pred={row.pred_s:.3f}s "
            )

        if "atomic_why_jaccard" in SET_METHODS:
            print("\n[CUSTOM] atomic_why_jaccard")
            if any(it.atomic_why for it in train) and any(it.atomic_why for it in test):
                row, pred = eval_jaccard_custom(train, test, "atomic_why", hashed=False)
                row.train_k = k
                all_rows.append(row)
                print(
                    f"  {row.model:12s} acc={row.acc:.4f} f1={row.f1:.4f} "
                    f"fit={row.fit_s:.3f}s pred={row.pred_s:.3f}s "
                )
            else:
                print("  SKIP (missing atomic why)")

        if "why_blocked" in SET_METHODS:
            print("\n[CUSTOM] why_blocked")
            if any(it.why_list for it in train) and any(it.why_list for it in test):
                row, pred = eval_why_blocked(train, test)
                row.train_k = k
                all_rows.append(row)
                blocked_knn_pred = pred
                print(
                    f"  {row.model:12s} acc={row.acc:.4f} f1={row.f1:.4f} "
                    f"fit={row.fit_s:.3f}s pred={row.pred_s:.3f}s "
                )
            else:
                print("  SKIP (missing witnesses)")

    if (
        psd_xgb_pred is not None
        and y_true_global is not None
        and label_map_global is not None
    ):
        plot_confusion_matrix(
            y_true_global,
            psd_xgb_pred,
            label_map_global,
            "PSD + XGBoost",
            "confusion_psd_xgboost.pdf",
        )

    if (
        lineage_knn_pred is not None
        and y_true_global is not None
        and label_map_global is not None
    ):
        plot_confusion_matrix(
            y_true_global,
            lineage_knn_pred,
            label_map_global,
            "Lineage + 5NN",
            "confusion_lineage_knn.pdf",
        )

    if (
        blocked_knn_pred is not None
        and y_true_global is not None
        and label_map_global is not None
    ):
        plot_confusion_matrix(
            y_true_global,
            blocked_knn_pred,
            label_map_global,
            "Why Blocked + 5NN",
            "confusion_blocked_knn.pdf",
        )

    if (
        token_tf_rf_pred is not None
        and y_true_global is not None
        and label_map_global is not None
    ):
        plot_confusion_matrix(
            y_true_global,
            token_tf_rf_pred,
            label_map_global,
            "Token TF + Random Forest",
            "confusion_token_tf_rf.pdf",
        )


if __name__ == "__main__":
    main()
