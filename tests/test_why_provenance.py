"""Tests for AtomicWhyProvenance and BlockedWhyProvenance."""

import pytest
from provenance_qs.provenance.why_provenance import (
    AtomicWhyProvenance,
    BlockedWhyProvenance,
)
from provenance_qs.provenance.lineage import Lineage


# ---------------------------------------------------------------------------
# AtomicWhyProvenance
# ---------------------------------------------------------------------------

class TestAtomicWhyProvenance:
    def _make(self, *witnesses):
        return AtomicWhyProvenance(witnesses)

    def test_empty(self):
        wp = AtomicWhyProvenance([])
        assert len(wp) == 0

    def test_len(self):
        wp = self._make(
            [("orders", 1), ("customer", 5)],
            [("orders", 2), ("customer", 5)],
        )
        assert len(wp) == 2

    def test_duplicate_witnesses_collapsed(self):
        wp = self._make(
            [("t", 1), ("t", 2)],
            [("t", 2), ("t", 1)],  # same witness, different order
        )
        assert len(wp) == 1

    def test_relations(self):
        wp = self._make(
            [("orders", 1), ("customer", 5)],
            [("inventory", 3)],
        )
        assert wp.relations == frozenset({"orders", "customer", "inventory"})

    def test_lineage_property(self):
        wp = self._make(
            [("orders", 1), ("customer", 5)],
            [("orders", 2), ("customer", 5)],
        )
        lin = wp.lineage
        assert isinstance(lin, Lineage)
        assert ("orders", 1) in lin
        assert ("orders", 2) in lin
        assert ("customer", 5) in lin

    def test_jaccard_identical(self):
        wp = self._make([("t", 1), ("t", 2)])
        assert wp.jaccard(wp) == pytest.approx(1.0)

    def test_jaccard_disjoint(self):
        a = self._make([("t", 1)])
        b = self._make([("t", 2)])
        assert a.jaccard(b) == pytest.approx(0.0)

    def test_jaccard_partial(self):
        shared = [("t", 1)]
        a = AtomicWhyProvenance([[("t", 1)], [("t", 2)]])
        b = AtomicWhyProvenance([[("t", 1)], [("t", 3)]])
        # witnesses: a={frozenset{(t,1)}, frozenset{(t,2)}}, b={frozenset{(t,1)}, frozenset{(t,3)}}
        # intersection size=1, union size=3 → 1/3
        assert a.jaccard(b) == pytest.approx(1 / 3)

    def test_average_witness_jaccard_identical(self):
        wp = self._make([("t", 1), ("t", 2)])
        assert wp.average_witness_jaccard(wp) == pytest.approx(1.0)

    def test_average_witness_jaccard_empty(self):
        a = AtomicWhyProvenance([])
        b = AtomicWhyProvenance([])
        assert a.average_witness_jaccard(b) == pytest.approx(1.0)

    def test_average_witness_jaccard_one_empty(self):
        a = AtomicWhyProvenance([[("t", 1)]])
        b = AtomicWhyProvenance([])
        assert a.average_witness_jaccard(b) == pytest.approx(0.0)

    def test_serialisation_round_trip(self):
        wp = self._make(
            [("orders", 1), ("customer", 5)],
            [("orders", 2)],
        )
        data = wp.to_list()
        restored = AtomicWhyProvenance.from_list(data)
        assert wp.jaccard(restored) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BlockedWhyProvenance
# ---------------------------------------------------------------------------

class TestBlockedWhyProvenance:
    def _make(self, *witnesses):
        return BlockedWhyProvenance(witnesses)

    def test_empty(self):
        bwp = BlockedWhyProvenance([])
        assert len(bwp) == 0

    def test_len(self):
        bwp = self._make(
            [("orders", 1), ("orders", 2), ("customer", 5)],
            [("orders", 3), ("customer", 5)],
        )
        assert len(bwp) == 2

    def test_relations(self):
        bwp = self._make(
            [("orders", 1), ("customer", 5)],
            [("inventory", 3)],
        )
        assert bwp.relations == frozenset({"orders", "customer", "inventory"})

    def test_lineage_property(self):
        bwp = self._make(
            [("orders", 1), ("customer", 5)],
        )
        lin = bwp.lineage
        assert isinstance(lin, Lineage)
        assert ("orders", 1) in lin
        assert ("customer", 5) in lin

    def test_blocking_groups_by_relation(self):
        bwp = self._make([("orders", 1), ("orders", 2), ("customer", 5)])
        w = list(bwp)[0]
        assert w["orders"] == frozenset({1, 2})
        assert w["customer"] == frozenset({5})

    def test_jaccard_identical(self):
        bwp = self._make([("t", 1), ("t", 2)])
        assert bwp.jaccard(bwp) == pytest.approx(1.0)

    def test_jaccard_disjoint(self):
        a = self._make([("t", 1)])
        b = self._make([("t", 2)])
        assert a.jaccard(b) == pytest.approx(0.0)

    def test_relation_overlap_identical(self):
        bwp = self._make([("orders", 1), ("customer", 2)])
        assert bwp.relation_overlap(bwp) == pytest.approx(1.0)

    def test_relation_overlap_disjoint(self):
        a = self._make([("orders", 1)])
        b = self._make([("customer", 2)])
        assert a.relation_overlap(b) == pytest.approx(0.0)

    def test_relation_overlap_partial(self):
        a = self._make([("orders", 1), ("customer", 2)])
        b = self._make([("orders", 1), ("inventory", 3)])
        # relations: a={orders,customer}, b={orders,inventory}
        # shared={orders}, union={orders,customer,inventory} → 1/3
        assert a.relation_overlap(b) == pytest.approx(1 / 3)

    def test_serialisation_round_trip(self):
        bwp = self._make(
            [("orders", 1), ("customer", 5)],
            [("orders", 2)],
        )
        data = bwp.to_list()
        restored = BlockedWhyProvenance.from_list(data)
        assert bwp.jaccard(restored) == pytest.approx(1.0)
