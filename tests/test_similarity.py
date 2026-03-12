"""Tests for similarity metrics."""

import numpy as np
import pytest

from provenance_qs.provenance.lineage import Lineage
from provenance_qs.provenance.why_provenance import (
    AtomicWhyProvenance,
    BlockedWhyProvenance,
)
from provenance_qs.similarity.metrics import (
    lineage_jaccard,
    atomic_jaccard,
    blocked_jaccard,
    blocked_relation_overlap,
    average_witness_jaccard,
    cosine_similarity,
    cosine_distance_matrix,
    jaccard_distance_matrix,
)


class TestLineageJaccard:
    def test_identical(self):
        lin = Lineage([("t", 1), ("t", 2)])
        assert lineage_jaccard(lin, lin) == pytest.approx(1.0)

    def test_disjoint(self):
        a = Lineage([("t", 1)])
        b = Lineage([("t", 2)])
        assert lineage_jaccard(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        a = Lineage([("t", 1), ("t", 2)])
        b = Lineage([("t", 2), ("t", 3)])
        assert lineage_jaccard(a, b) == pytest.approx(1 / 3)


class TestAtomicJaccard:
    def test_identical(self):
        wp = AtomicWhyProvenance([[("t", 1)]])
        assert atomic_jaccard(wp, wp) == pytest.approx(1.0)

    def test_disjoint(self):
        a = AtomicWhyProvenance([[("t", 1)]])
        b = AtomicWhyProvenance([[("t", 2)]])
        assert atomic_jaccard(a, b) == pytest.approx(0.0)


class TestAverageWitnessJaccard:
    def test_identical(self):
        wp = AtomicWhyProvenance([[("t", 1), ("t", 2)]])
        assert average_witness_jaccard(wp, wp) == pytest.approx(1.0)

    def test_symmetric(self):
        a = AtomicWhyProvenance([[("t", 1)], [("t", 2)]])
        b = AtomicWhyProvenance([[("t", 1)], [("t", 3)]])
        sim_ab = average_witness_jaccard(a, b)
        sim_ba = average_witness_jaccard(b, a)
        assert sim_ab == pytest.approx(sim_ba)


class TestBlockedJaccard:
    def test_identical(self):
        bwp = BlockedWhyProvenance([[("orders", 1), ("customer", 2)]])
        assert blocked_jaccard(bwp, bwp) == pytest.approx(1.0)

    def test_disjoint(self):
        a = BlockedWhyProvenance([[("t", 1)]])
        b = BlockedWhyProvenance([[("t", 2)]])
        assert blocked_jaccard(a, b) == pytest.approx(0.0)


class TestBlockedRelationOverlap:
    def test_identical(self):
        bwp = BlockedWhyProvenance([[("orders", 1), ("customer", 2)]])
        assert blocked_relation_overlap(bwp, bwp) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = BlockedWhyProvenance([[("orders", 1)]])
        b = BlockedWhyProvenance([[("customer", 2)]])
        assert blocked_relation_overlap(a, b) == pytest.approx(0.0)


class TestCosineSimilarity:
    def test_identical(self):
        u = np.array([1.0, 0.0, 1.0])
        assert cosine_similarity(u, u) == pytest.approx(1.0)

    def test_orthogonal(self):
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 1.0])
        assert cosine_similarity(u, v) == pytest.approx(0.0)

    def test_zero_vector(self):
        u = np.zeros(3)
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(u, v) == pytest.approx(0.0)

    def test_known_value(self):
        u = np.array([1.0, 1.0])
        v = np.array([1.0, 0.0])
        # cos = 1/sqrt(2)
        assert cosine_similarity(u, v) == pytest.approx(1 / np.sqrt(2))


class TestCosineDistanceMatrix:
    def test_shape(self):
        X = np.random.rand(5, 10)
        D = cosine_distance_matrix(X)
        assert D.shape == (5, 5)

    def test_diagonal_zero(self):
        X = np.random.rand(4, 6)
        D = cosine_distance_matrix(X)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)

    def test_symmetric(self):
        X = np.random.rand(4, 6)
        D = cosine_distance_matrix(X)
        np.testing.assert_allclose(D, D.T, atol=1e-6)

    def test_range(self):
        X = np.abs(np.random.rand(10, 8))
        D = cosine_distance_matrix(X)
        assert D.min() >= -1e-9
        assert D.max() <= 1.0 + 1e-9


class TestJaccardDistanceMatrix:
    def test_shape(self):
        lins = [
            Lineage([("t", i), ("t", i + 1)]) for i in range(4)
        ]
        D = jaccard_distance_matrix(lins)
        assert D.shape == (4, 4)

    def test_diagonal_zero(self):
        lins = [Lineage([("t", i)]) for i in range(3)]
        D = jaccard_distance_matrix(lins)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-9)

    def test_symmetric(self):
        lins = [Lineage([("t", i), ("u", i + 1)]) for i in range(5)]
        D = jaccard_distance_matrix(lins)
        np.testing.assert_allclose(D, D.T, atol=1e-9)
