"""
Feature extraction from provenance representations.

This module bridges the provenance layer and the machine-learning layer by
converting provenance objects into numeric feature vectors.

Five feature families are implemented:

1. **Lineage frequency features** – token-frequency vector over
   (relation, tuple_id) pairs in the lineage.
2. **Witness frequency features** – frequency vector over complete witnesses in
   Why-provenance.
3. **Token frequency features** – relation-level bag-of-words ignoring tuple IDs.
4. **Compact structural descriptors** – a small fixed-length vector capturing
   aggregate structural statistics of the provenance (relation counts,
   witness counts, average witness sizes, etc.).
5. **Blocked witness frequency features** – frequency vector built from
   relation-level witness fingerprints.

All extractors implement a scikit-learn-compatible ``fit`` / ``transform``
interface so they can be used inside ``Pipeline`` objects.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional

import numpy as np

from .lineage import Lineage, TupleRef
from .why_provenance import AtomicWhyProvenance, BlockedWhyProvenance


class LineageFrequencyExtractor:
    """Transform :class:`Lineage` objects into token-frequency vectors.

    Each dimension of the output vector corresponds to a unique
    ``(relation, tuple_id)`` pair seen during :meth:`fit`.

    Parameters
    ----------
    min_df:
        Minimum number of documents (query provenances) a token must appear
        in to be included in the vocabulary.
    """

    def __init__(self, min_df: int = 1) -> None:
        self.min_df = min_df
        self.vocabulary_: dict[TupleRef, int] = {}

    def fit(self, lineages: Iterable[Lineage]) -> "LineageFrequencyExtractor":
        """Build vocabulary from *lineages*."""
        doc_freq: Counter[TupleRef] = Counter()
        for lin in lineages:
            doc_freq.update(set(lin))
        qualified = sorted(
            (token for token, df in doc_freq.items() if df >= self.min_df)
        )
        self.vocabulary_ = {token: idx for idx, token in enumerate(qualified)}
        return self

    def transform(self, lineages: Iterable[Lineage]) -> np.ndarray:
        """Encode each lineage as a binary membership vector."""
        vocab = self.vocabulary_
        n_features = len(vocab)
        rows = []
        for lin in lineages:
            vec = np.zeros(n_features, dtype=np.float32)
            for token in lin:
                if token in vocab:
                    vec[vocab[token]] += 1.0
            rows.append(vec)
        return np.vstack(rows) if rows else np.empty((0, n_features), dtype=np.float32)

    def fit_transform(self, lineages: List[Lineage]) -> np.ndarray:
        """Convenience: :meth:`fit` followed by :meth:`transform`."""
        return self.fit(lineages).transform(lineages)


class WitnessFrequencyExtractor:
    """Transform :class:`AtomicWhyProvenance` objects into witness-count vectors.

    Each dimension corresponds to a unique (frozen) witness seen during
    :meth:`fit`.

    Parameters
    ----------
    min_df:
        Minimum number of query provenances that must contain a witness for it
        to be retained in the vocabulary.
    """

    def __init__(self, min_df: int = 1) -> None:
        self.min_df = min_df
        self.vocabulary_: dict[frozenset, int] = {}

    def fit(
        self, provenances: Iterable[AtomicWhyProvenance]
    ) -> "WitnessFrequencyExtractor":
        doc_freq: Counter = Counter()
        for prov in provenances:
            doc_freq.update(prov)  # each witness counted once per provenance
        qualified = sorted(
            (w for w, df in doc_freq.items() if df >= self.min_df), key=str
        )
        self.vocabulary_ = {witness: idx for idx, witness in enumerate(qualified)}
        return self

    def transform(self, provenances: Iterable[AtomicWhyProvenance]) -> np.ndarray:
        vocab = self.vocabulary_
        n_features = len(vocab)
        rows = []
        for prov in provenances:
            vec = np.zeros(n_features, dtype=np.float32)
            cnt: Counter = Counter(prov)
            for witness, count in cnt.items():
                if witness in vocab:
                    vec[vocab[witness]] += count
            rows.append(vec)
        return np.vstack(rows) if rows else np.empty((0, n_features), dtype=np.float32)

    def fit_transform(self, provenances: List[AtomicWhyProvenance]) -> np.ndarray:
        return self.fit(provenances).transform(provenances)


class TokenFrequencyExtractor:
    """Relation-level bag-of-words, ignoring tuple IDs.

    Each dimension corresponds to a unique *relation name* seen during
    :meth:`fit`.  The value is the number of times that relation appears in
    the lineage (i.e. number of tuples from that relation).

    Parameters
    ----------
    min_df:
        Minimum document frequency for a relation to be retained.
    """

    def __init__(self, min_df: int = 1) -> None:
        self.min_df = min_df
        self.vocabulary_: dict[str, int] = {}

    def fit(self, lineages: Iterable[Lineage]) -> "TokenFrequencyExtractor":
        doc_freq: Counter[str] = Counter()
        for lin in lineages:
            doc_freq.update({rel for rel, _ in lin})
        qualified = sorted(rel for rel, df in doc_freq.items() if df >= self.min_df)
        self.vocabulary_ = {rel: idx for idx, rel in enumerate(qualified)}
        return self

    def transform(self, lineages: Iterable[Lineage]) -> np.ndarray:
        vocab = self.vocabulary_
        n_features = len(vocab)
        rows = []
        for lin in lineages:
            vec = np.zeros(n_features, dtype=np.float32)
            for rel, _ in lin:
                if rel in vocab:
                    vec[vocab[rel]] += 1.0
            rows.append(vec)
        return np.vstack(rows) if rows else np.empty((0, n_features), dtype=np.float32)

    def fit_transform(self, lineages: List[Lineage]) -> np.ndarray:
        return self.fit(lineages).transform(lineages)


class CompactStructuralDescriptor:
    """Extract a fixed-length structural descriptor from provenance objects.

    The descriptor encodes aggregate statistics:

    For :class:`Lineage`:
        ``[n_tuples, n_relations, max_rel_freq, mean_rel_freq]``

    For :class:`AtomicWhyProvenance`:
        ``[n_witnesses, n_tuples_total, mean_witness_size, max_witness_size,
           n_distinct_relations]``

    For :class:`BlockedWhyProvenance`:
        ``[n_witnesses, n_distinct_relations, mean_witness_relations,
           max_witness_relations, mean_witness_tuples, max_witness_tuples]``

    The extractor is *stateless* (no :meth:`fit` step required).
    """

    def transform_lineage(self, lineages: Iterable[Lineage]) -> np.ndarray:
        rows = []
        for lin in lineages:
            n_tuples = len(lin)
            rel_freq: Counter[str] = Counter(rel for rel, _ in lin)
            n_relations = len(rel_freq)
            freqs = list(rel_freq.values()) if rel_freq else [0]
            rows.append([
                float(n_tuples),
                float(n_relations),
                float(max(freqs)),
                float(sum(freqs) / len(freqs)),
            ])
        return np.array(rows, dtype=np.float32) if rows else np.empty((0, 4), dtype=np.float32)

    def transform_atomic(
        self, provenances: Iterable[AtomicWhyProvenance]
    ) -> np.ndarray:
        rows = []
        for prov in provenances:
            witnesses = list(prov)
            n_witnesses = len(witnesses)
            sizes = [len(w) for w in witnesses]
            all_refs = [ref for w in witnesses for ref in w]
            n_tuples_total = len(all_refs)
            n_distinct_relations = len({rel for rel, _ in all_refs}) if all_refs else 0
            mean_size = float(sum(sizes) / len(sizes)) if sizes else 0.0
            max_size = float(max(sizes)) if sizes else 0.0
            rows.append([
                float(n_witnesses),
                float(n_tuples_total),
                mean_size,
                max_size,
                float(n_distinct_relations),
            ])
        return np.array(rows, dtype=np.float32) if rows else np.empty((0, 5), dtype=np.float32)

    def transform_blocked(
        self, provenances: Iterable[BlockedWhyProvenance]
    ) -> np.ndarray:
        rows = []
        for prov in provenances:
            witnesses = list(prov)
            n_witnesses = len(witnesses)
            all_rels = {rel for w in witnesses for rel in w}
            n_distinct_relations = len(all_rels)
            per_w_rels = [len(w) for w in witnesses]
            per_w_tuples = [sum(len(tids) for tids in w.values()) for w in witnesses]
            mean_wr = float(sum(per_w_rels) / len(per_w_rels)) if per_w_rels else 0.0
            max_wr = float(max(per_w_rels)) if per_w_rels else 0.0
            mean_wt = float(sum(per_w_tuples) / len(per_w_tuples)) if per_w_tuples else 0.0
            max_wt = float(max(per_w_tuples)) if per_w_tuples else 0.0
            rows.append([
                float(n_witnesses),
                float(n_distinct_relations),
                mean_wr,
                max_wr,
                mean_wt,
                max_wt,
            ])
        return np.array(rows, dtype=np.float32) if rows else np.empty((0, 6), dtype=np.float32)


class BlockedWitnessFrequencyExtractor:
    """Frequency vector built from relation-set fingerprints of blocked witnesses.

    Each witness is characterised by its *set of relation names* (ignoring tuple
    IDs).  The extractor builds a vocabulary of such relation-sets and counts how
    often each appears in a query's blocked provenance.

    Parameters
    ----------
    min_df:
        Minimum document frequency for a fingerprint to be retained.
    """

    def __init__(self, min_df: int = 1) -> None:
        self.min_df = min_df
        self.vocabulary_: dict[frozenset, int] = {}

    def _fingerprint(self, blocked_witness: dict) -> frozenset:
        return frozenset(blocked_witness.keys())

    def fit(
        self, provenances: Iterable[BlockedWhyProvenance]
    ) -> "BlockedWitnessFrequencyExtractor":
        doc_freq: Counter = Counter()
        for prov in provenances:
            seen_fps = {self._fingerprint(w) for w in prov}
            doc_freq.update(seen_fps)
        qualified = sorted(
            (fp for fp, df in doc_freq.items() if df >= self.min_df), key=str
        )
        self.vocabulary_ = {fp: idx for idx, fp in enumerate(qualified)}
        return self

    def transform(self, provenances: Iterable[BlockedWhyProvenance]) -> np.ndarray:
        vocab = self.vocabulary_
        n_features = len(vocab)
        rows = []
        for prov in provenances:
            vec = np.zeros(n_features, dtype=np.float32)
            for w in prov:
                fp = self._fingerprint(w)
                if fp in vocab:
                    vec[vocab[fp]] += 1.0
            rows.append(vec)
        return np.vstack(rows) if rows else np.empty((0, n_features), dtype=np.float32)

    def fit_transform(self, provenances: List[BlockedWhyProvenance]) -> np.ndarray:
        return self.fit(provenances).transform(provenances)


class ProvenanceFeatureExtractor:
    """High-level extractor that combines multiple feature families.

    This is the primary entry point for converting provenance objects into
    feature matrices ready for ML models.

    Parameters
    ----------
    use_lineage:
        Include lineage token-frequency features.
    use_token:
        Include relation-level token-frequency features.
    use_witness:
        Include atomic witness-frequency features.
    use_blocked:
        Include blocked witness-frequency features.
    use_structural:
        Include compact structural descriptor features.
    min_df:
        Minimum document frequency passed to all sub-extractors.
    """

    def __init__(
        self,
        *,
        use_lineage: bool = True,
        use_token: bool = True,
        use_witness: bool = False,
        use_blocked: bool = False,
        use_structural: bool = True,
        min_df: int = 1,
    ) -> None:
        self.use_lineage = use_lineage
        self.use_token = use_token
        self.use_witness = use_witness
        self.use_blocked = use_blocked
        self.use_structural = use_structural
        self.min_df = min_df

        self._lineage_ext: Optional[LineageFrequencyExtractor] = (
            LineageFrequencyExtractor(min_df=min_df) if use_lineage else None
        )
        self._token_ext: Optional[TokenFrequencyExtractor] = (
            TokenFrequencyExtractor(min_df=min_df) if use_token else None
        )
        self._witness_ext: Optional[WitnessFrequencyExtractor] = (
            WitnessFrequencyExtractor(min_df=min_df) if use_witness else None
        )
        self._blocked_ext: Optional[BlockedWitnessFrequencyExtractor] = (
            BlockedWitnessFrequencyExtractor(min_df=min_df) if use_blocked else None
        )
        self._struct_desc = CompactStructuralDescriptor() if use_structural else None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        lineages: Optional[List[Lineage]] = None,
        atomic_provs: Optional[List[AtomicWhyProvenance]] = None,
        blocked_provs: Optional[List[BlockedWhyProvenance]] = None,
    ) -> "ProvenanceFeatureExtractor":
        if self._lineage_ext is not None:
            if lineages is None:
                raise ValueError("lineages required when use_lineage=True")
            self._lineage_ext.fit(lineages)
        if self._token_ext is not None:
            if lineages is None:
                raise ValueError("lineages required when use_token=True")
            self._token_ext.fit(lineages)
        if self._witness_ext is not None:
            if atomic_provs is None:
                raise ValueError("atomic_provs required when use_witness=True")
            self._witness_ext.fit(atomic_provs)
        if self._blocked_ext is not None:
            if blocked_provs is None:
                raise ValueError("blocked_provs required when use_blocked=True")
            self._blocked_ext.fit(blocked_provs)
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        lineages: Optional[List[Lineage]] = None,
        atomic_provs: Optional[List[AtomicWhyProvenance]] = None,
        blocked_provs: Optional[List[BlockedWhyProvenance]] = None,
    ) -> np.ndarray:
        n = self._infer_n(lineages, atomic_provs, blocked_provs)
        parts: list[np.ndarray] = []

        if self._lineage_ext is not None:
            parts.append(self._lineage_ext.transform(lineages))
        if self._token_ext is not None:
            parts.append(self._token_ext.transform(lineages))
        if self._witness_ext is not None:
            parts.append(self._witness_ext.transform(atomic_provs))
        if self._blocked_ext is not None:
            parts.append(self._blocked_ext.transform(blocked_provs))
        if self._struct_desc is not None:
            # Use whichever provenance is available for structural features
            if lineages is not None:
                parts.append(self._struct_desc.transform_lineage(lineages))
            elif atomic_provs is not None:
                parts.append(self._struct_desc.transform_atomic(atomic_provs))
            elif blocked_provs is not None:
                parts.append(self._struct_desc.transform_blocked(blocked_provs))

        if not parts:
            return np.empty((n, 0), dtype=np.float32)
        return np.hstack(parts)

    def fit_transform(
        self,
        lineages: Optional[List[Lineage]] = None,
        atomic_provs: Optional[List[AtomicWhyProvenance]] = None,
        blocked_provs: Optional[List[BlockedWhyProvenance]] = None,
    ) -> np.ndarray:
        return self.fit(
            lineages=lineages,
            atomic_provs=atomic_provs,
            blocked_provs=blocked_provs,
        ).transform(
            lineages=lineages,
            atomic_provs=atomic_provs,
            blocked_provs=blocked_provs,
        )

    @staticmethod
    def _infer_n(lineages, atomic_provs, blocked_provs) -> int:
        for lst in (lineages, atomic_provs, blocked_provs):
            if lst is not None:
                return len(lst)
        return 0
