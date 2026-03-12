"""
Similarity metrics for provenance-based query comparison.

All pairwise metrics return a value in [0, 1] where 1 means identical and
0 means completely dissimilar.  Distance metrics are the complement (1 − sim).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..provenance.lineage import Lineage
from ..provenance.why_provenance import AtomicWhyProvenance, BlockedWhyProvenance


# ---------------------------------------------------------------------------
# Lineage similarities
# ---------------------------------------------------------------------------


def lineage_jaccard(a: Lineage, b: Lineage) -> float:
    """Jaccard similarity between two query-level lineage sets."""
    return a.jaccard(b)


# ---------------------------------------------------------------------------
# Why-provenance similarities
# ---------------------------------------------------------------------------


def atomic_jaccard(a: AtomicWhyProvenance, b: AtomicWhyProvenance) -> float:
    """Jaccard similarity over the set of atomic witnesses."""
    return a.jaccard(b)


def average_witness_jaccard(
    a: AtomicWhyProvenance, b: AtomicWhyProvenance
) -> float:
    """Symmetric average best-match Jaccard between witness sets."""
    sim_ab = a.average_witness_jaccard(b)
    sim_ba = b.average_witness_jaccard(a)
    return (sim_ab + sim_ba) / 2.0


def blocked_jaccard(a: BlockedWhyProvenance, b: BlockedWhyProvenance) -> float:
    """Jaccard similarity over the set of blocked witnesses."""
    return a.jaccard(b)


def blocked_relation_overlap(
    a: BlockedWhyProvenance, b: BlockedWhyProvenance
) -> float:
    """Relation-level overlap (ignores tuple IDs)."""
    return a.relation_overlap(b)


# ---------------------------------------------------------------------------
# Vector-based similarities
# ---------------------------------------------------------------------------


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two feature vectors.

    Parameters
    ----------
    u, v:
        1-D numeric arrays of the same length.

    Returns
    -------
    float
        Value in [-1, 1]; typically [0, 1] for non-negative feature vectors.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


# ---------------------------------------------------------------------------
# Distance matrices (for kNN and visualisation)
# ---------------------------------------------------------------------------


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute an (n × n) cosine distance matrix from feature matrix *X*.

    Parameters
    ----------
    X:
        (n_samples, n_features) array.

    Returns
    -------
    np.ndarray
        Symmetric distance matrix ``D`` where ``D[i, j] = 1 − cosine_sim(X[i], X[j])``.
    """
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    X_norm = X / norms
    sim = X_norm @ X_norm.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def jaccard_distance_matrix(lineages: Sequence[Lineage]) -> np.ndarray:
    """Compute an (n × n) Jaccard distance matrix from a list of lineage sets.

    Parameters
    ----------
    lineages:
        Sequence of :class:`~provenance_qs.provenance.Lineage` objects.

    Returns
    -------
    np.ndarray
        Symmetric distance matrix ``D`` where
        ``D[i, j] = 1 − jaccard(lineages[i], lineages[j])``.
    """
    n = len(lineages)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - lineages[i].jaccard(lineages[j])
            D[i, j] = d
            D[j, i] = d
    return D
