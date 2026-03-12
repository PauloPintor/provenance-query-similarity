"""Tests for provenance feature extractors."""

import numpy as np
import pytest

from provenance_qs.provenance.lineage import Lineage
from provenance_qs.provenance.why_provenance import (
    AtomicWhyProvenance,
    BlockedWhyProvenance,
)
from provenance_qs.provenance.features import (
    LineageFrequencyExtractor,
    WitnessFrequencyExtractor,
    TokenFrequencyExtractor,
    CompactStructuralDescriptor,
    BlockedWitnessFrequencyExtractor,
    ProvenanceFeatureExtractor,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_lineages():
    return [
        Lineage([("orders", 1), ("orders", 2), ("customer", 5)]),
        Lineage([("orders", 1), ("customer", 6)]),
        Lineage([("inventory", 10), ("customer", 5)]),
    ]


def make_atomic_provs():
    return [
        AtomicWhyProvenance([[("orders", 1), ("customer", 5)], [("orders", 2)]]),
        AtomicWhyProvenance([[("orders", 1), ("customer", 6)]]),
        AtomicWhyProvenance([[("inventory", 10)]]),
    ]


def make_blocked_provs():
    return [
        BlockedWhyProvenance([[("orders", 1), ("orders", 2), ("customer", 5)]]),
        BlockedWhyProvenance([[("orders", 1), ("customer", 6)]]),
        BlockedWhyProvenance([[("inventory", 10)]]),
    ]


# ---------------------------------------------------------------------------
# LineageFrequencyExtractor
# ---------------------------------------------------------------------------

class TestLineageFrequencyExtractor:
    def test_fit_transform_shape(self):
        lins = make_lineages()
        ext = LineageFrequencyExtractor()
        X = ext.fit_transform(lins)
        assert X.shape == (3, len(ext.vocabulary_))
        assert X.shape[1] > 0

    def test_vocabulary_contains_all_tokens(self):
        lins = make_lineages()
        ext = LineageFrequencyExtractor()
        ext.fit(lins)
        # all (relation, tuple_id) pairs should be in vocabulary
        all_tokens = {ref for lin in lins for ref in lin}
        assert set(ext.vocabulary_.keys()) == all_tokens

    def test_min_df_filters(self):
        lins = make_lineages()
        ext = LineageFrequencyExtractor(min_df=2)
        X = ext.fit_transform(lins)
        # Only tokens in ≥2 lineages survive; ("orders",1) and ("customer",5) qualify
        assert ("orders", 1) in ext.vocabulary_
        # ("inventory", 10) only appears once
        assert ("inventory", 10) not in ext.vocabulary_

    def test_transform_counts(self):
        lin = Lineage([("t", 1), ("t", 1), ("t", 2)])
        # frozenset dedup: only 2 unique refs
        ext = LineageFrequencyExtractor()
        ext.fit([lin])
        X = ext.transform([lin])
        assert X.shape == (1, 2)
        assert X.sum() == pytest.approx(2.0)  # 1 count each

    def test_empty_input(self):
        ext = LineageFrequencyExtractor()
        ext.fit(make_lineages())
        X = ext.transform([])
        assert X.shape == (0, len(ext.vocabulary_))


# ---------------------------------------------------------------------------
# TokenFrequencyExtractor
# ---------------------------------------------------------------------------

class TestTokenFrequencyExtractor:
    def test_fit_transform_shape(self):
        lins = make_lineages()
        ext = TokenFrequencyExtractor()
        X = ext.fit_transform(lins)
        # 3 distinct relations: orders, customer, inventory
        assert X.shape == (3, 3)

    def test_counts_reflect_tuple_frequency(self):
        # orders appears twice in first lineage
        lin = Lineage([("orders", 1), ("orders", 2), ("customer", 5)])
        ext = TokenFrequencyExtractor()
        ext.fit([lin])
        X = ext.transform([lin])
        orders_idx = ext.vocabulary_["orders"]
        assert X[0, orders_idx] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# WitnessFrequencyExtractor
# ---------------------------------------------------------------------------

class TestWitnessFrequencyExtractor:
    def test_fit_transform_shape(self):
        provs = make_atomic_provs()
        ext = WitnessFrequencyExtractor()
        X = ext.fit_transform(provs)
        assert X.shape[0] == 3
        assert X.shape[1] > 0

    def test_shared_witness_counts(self):
        shared_w = [("t", 1)]
        a = AtomicWhyProvenance([shared_w, [("t", 2)]])
        b = AtomicWhyProvenance([shared_w])
        ext = WitnessFrequencyExtractor()
        X = ext.fit_transform([a, b])
        shared_key = frozenset([("t", 1)])
        idx = ext.vocabulary_[shared_key]
        assert X[0, idx] == pytest.approx(1.0)
        assert X[1, idx] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CompactStructuralDescriptor
# ---------------------------------------------------------------------------

class TestCompactStructuralDescriptor:
    def test_lineage_descriptor_shape(self):
        lins = make_lineages()
        desc = CompactStructuralDescriptor()
        X = desc.transform_lineage(lins)
        assert X.shape == (3, 4)

    def test_lineage_empty(self):
        desc = CompactStructuralDescriptor()
        X = desc.transform_lineage([Lineage([])])
        assert X.shape == (1, 4)
        assert X[0, 0] == pytest.approx(0.0)  # n_tuples

    def test_atomic_descriptor_shape(self):
        provs = make_atomic_provs()
        desc = CompactStructuralDescriptor()
        X = desc.transform_atomic(provs)
        assert X.shape == (3, 5)

    def test_blocked_descriptor_shape(self):
        provs = make_blocked_provs()
        desc = CompactStructuralDescriptor()
        X = desc.transform_blocked(provs)
        assert X.shape == (3, 6)

    def test_no_nan_values(self):
        desc = CompactStructuralDescriptor()
        for X in [
            desc.transform_lineage(make_lineages()),
            desc.transform_atomic(make_atomic_provs()),
            desc.transform_blocked(make_blocked_provs()),
        ]:
            assert not np.any(np.isnan(X))


# ---------------------------------------------------------------------------
# BlockedWitnessFrequencyExtractor
# ---------------------------------------------------------------------------

class TestBlockedWitnessFrequencyExtractor:
    def test_fit_transform_shape(self):
        provs = make_blocked_provs()
        ext = BlockedWitnessFrequencyExtractor()
        X = ext.fit_transform(provs)
        assert X.shape[0] == 3
        assert X.shape[1] > 0


# ---------------------------------------------------------------------------
# ProvenanceFeatureExtractor (combined)
# ---------------------------------------------------------------------------

class TestProvenanceFeatureExtractor:
    def test_lineage_only(self):
        lins = make_lineages()
        ext = ProvenanceFeatureExtractor(
            use_lineage=True, use_token=False,
            use_witness=False, use_blocked=False, use_structural=False,
        )
        X = ext.fit_transform(lineages=lins)
        assert X.shape[0] == 3
        assert X.shape[1] > 0

    def test_token_only(self):
        lins = make_lineages()
        ext = ProvenanceFeatureExtractor(
            use_lineage=False, use_token=True,
            use_witness=False, use_blocked=False, use_structural=False,
        )
        X = ext.fit_transform(lineages=lins)
        assert X.shape == (3, 3)  # 3 relations

    def test_structural_with_lineage(self):
        lins = make_lineages()
        ext = ProvenanceFeatureExtractor(
            use_lineage=False, use_token=False,
            use_witness=False, use_blocked=False, use_structural=True,
        )
        X = ext.fit_transform(lineages=lins)
        assert X.shape == (3, 4)

    def test_combined_shapes_sum(self):
        lins = make_lineages()
        ext_lin = ProvenanceFeatureExtractor(
            use_lineage=True, use_token=False, use_witness=False,
            use_blocked=False, use_structural=False,
        )
        ext_tok = ProvenanceFeatureExtractor(
            use_lineage=False, use_token=True, use_witness=False,
            use_blocked=False, use_structural=False,
        )
        ext_combined = ProvenanceFeatureExtractor(
            use_lineage=True, use_token=True, use_witness=False,
            use_blocked=False, use_structural=False,
        )
        X_lin = ext_lin.fit_transform(lineages=lins)
        X_tok = ext_tok.fit_transform(lineages=lins)
        X_combined = ext_combined.fit_transform(lineages=lins)
        assert X_combined.shape[1] == X_lin.shape[1] + X_tok.shape[1]

    def test_requires_lineage_when_use_lineage_true(self):
        ext = ProvenanceFeatureExtractor(use_lineage=True)
        with pytest.raises(ValueError, match="lineages required"):
            ext.fit(lineages=None)

    def test_witness_features(self):
        provs = make_atomic_provs()
        ext = ProvenanceFeatureExtractor(
            use_lineage=False, use_token=False,
            use_witness=True, use_blocked=False, use_structural=False,
        )
        X = ext.fit_transform(atomic_provs=provs)
        assert X.shape[0] == 3

    def test_blocked_features(self):
        provs = make_blocked_provs()
        ext = ProvenanceFeatureExtractor(
            use_lineage=False, use_token=False,
            use_witness=False, use_blocked=True, use_structural=False,
        )
        X = ext.fit_transform(blocked_provs=provs)
        assert X.shape[0] == 3

    def test_no_features(self):
        lins = make_lineages()
        ext = ProvenanceFeatureExtractor(
            use_lineage=False, use_token=False,
            use_witness=False, use_blocked=False, use_structural=False,
        )
        X = ext.fit_transform(lineages=lins)
        assert X.shape == (3, 0)
