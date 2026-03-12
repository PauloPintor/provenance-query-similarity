"""Tests for Lineage."""

import pytest
from provenance_qs.provenance.lineage import Lineage


def make_lineage(*pairs):
    return Lineage(pairs)


class TestLineageBasics:
    def test_empty(self):
        lin = Lineage([])
        assert len(lin) == 0
        assert list(lin) == []

    def test_contains(self):
        lin = make_lineage(("orders", 1), ("customer", 2))
        assert ("orders", 1) in lin
        assert ("customer", 2) in lin
        assert ("orders", 99) not in lin

    def test_len(self):
        lin = make_lineage(("a", 1), ("a", 2), ("b", 3))
        assert len(lin) == 3

    def test_deduplication(self):
        lin = Lineage([("a", 1), ("a", 1), ("b", 2)])
        assert len(lin) == 2

    def test_relations(self):
        lin = make_lineage(("orders", 1), ("orders", 2), ("customer", 3))
        assert lin.relations == frozenset({"orders", "customer"})

    def test_tuples_for_relation(self):
        lin = make_lineage(("orders", 10), ("orders", 20), ("customer", 5))
        assert lin.tuples_for_relation("orders") == frozenset({10, 20})
        assert lin.tuples_for_relation("customer") == frozenset({5})
        assert lin.tuples_for_relation("nonexistent") == frozenset()


class TestLineageSetOps:
    def test_union(self):
        a = make_lineage(("t", 1), ("t", 2))
        b = make_lineage(("t", 2), ("t", 3))
        u = a.union(b)
        assert len(u) == 3
        assert ("t", 1) in u and ("t", 3) in u

    def test_intersection(self):
        a = make_lineage(("t", 1), ("t", 2))
        b = make_lineage(("t", 2), ("t", 3))
        inter = a.intersection(b)
        assert len(inter) == 1
        assert ("t", 2) in inter

    def test_jaccard_identical(self):
        lin = make_lineage(("t", 1), ("t", 2))
        assert lin.jaccard(lin) == pytest.approx(1.0)

    def test_jaccard_disjoint(self):
        a = make_lineage(("t", 1))
        b = make_lineage(("t", 2))
        assert a.jaccard(b) == pytest.approx(0.0)

    def test_jaccard_partial(self):
        a = make_lineage(("t", 1), ("t", 2))
        b = make_lineage(("t", 2), ("t", 3))
        # intersection = {2}, union = {1,2,3} → 1/3
        assert a.jaccard(b) == pytest.approx(1 / 3)

    def test_jaccard_both_empty(self):
        a = Lineage([])
        b = Lineage([])
        assert a.jaccard(b) == pytest.approx(1.0)


class TestLineageSerialization:
    def test_round_trip(self):
        original = make_lineage(("orders", 1), ("customer", 2), ("orders", 5))
        d = original.to_dict()
        restored = Lineage.from_dict(d)
        assert set(original) == set(restored)

    def test_to_dict_groups_by_relation(self):
        lin = make_lineage(("orders", 1), ("orders", 3), ("customer", 2))
        d = lin.to_dict()
        assert set(d.keys()) == {"orders", "customer"}
        assert sorted(d["orders"]) == [1, 3]
        assert d["customer"] == [2]
