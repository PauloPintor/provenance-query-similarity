"""
Why-provenance representations for database queries.

Why-provenance captures *how* query results are derived by recording the
alternative "witnesses" – minimal sets of base tuples that together prove the
existence of a result tuple.

This module provides two flavours:

AtomicWhyProvenance
    Each witness is a plain frozenset of (relation, tuple_id) pairs.
    All witnesses across every result tuple of a query are collected into a
    single set-of-sets structure, which forms the query-level representation.

BlockedWhyProvenance
    Witnesses are grouped ("blocked") by source relation, turning each witness
    into a mapping  relation_name → frozenset[tuple_id].  This exposes which
    relations contribute to each derivation independently of individual IDs.

References
----------
Buneman, P., Khanna, S., & Tan, W.-C. (2001).
Why and Where: A Characterization of Data Provenance.
ICDT 2001, LNCS 1973, 316–330.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Dict, FrozenSet, Set, Tuple


TupleRef = Tuple[str, int]  # (relation_name, tuple_id)
Witness = FrozenSet[TupleRef]  # one minimal proof
BlockedWitness = Dict[str, FrozenSet[int]]  # relation → tuple IDs


class AtomicWhyProvenance:
    """Query-level atomic Why-provenance.

    Parameters
    ----------
    witnesses:
        Iterable of *witnesses*, each being an iterable of
        ``(relation_name, tuple_id)`` pairs.

    Examples
    --------
    >>> wp = AtomicWhyProvenance([
    ...     [("orders", 1), ("customer", 5)],
    ...     [("orders", 2), ("customer", 5)],
    ... ])
    >>> len(wp)
    2
    >>> wp.relations
    frozenset({'orders', 'customer'})
    """

    def __init__(self, witnesses: Iterable[Iterable[TupleRef]]) -> None:
        self._witnesses: FrozenSet[Witness] = frozenset(
            frozenset(w) for w in witnesses
        )

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._witnesses)

    def __iter__(self):
        return iter(self._witnesses)

    def __repr__(self) -> str:  # pragma: no cover
        return f"AtomicWhyProvenance(n_witnesses={len(self._witnesses)})"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def relations(self) -> FrozenSet[str]:
        """Distinct relation names appearing in *any* witness."""
        return frozenset(rel for w in self._witnesses for rel, _ in w)

    @property
    def lineage(self):
        """Flatten all witnesses into a :class:`~provenance_qs.provenance.Lineage`."""
        from .lineage import Lineage

        return Lineage(ref for w in self._witnesses for ref in w)

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def jaccard(self, other: "AtomicWhyProvenance") -> float:
        """Jaccard similarity over the *set of witnesses*.

        Two queries are more similar when they share more witnesses.
        Returns 1.0 when both have no witnesses.
        """
        a, b = self._witnesses, other._witnesses
        union_size = len(a | b)
        if union_size == 0:
            return 1.0
        return len(a & b) / union_size

    def average_witness_jaccard(self, other: "AtomicWhyProvenance") -> float:
        """Average best-match Jaccard similarity between witness sets.

        For every witness in *self* we find its most-similar counterpart in
        *other* (via element-level Jaccard), then average those scores.
        Asymmetric; call both directions and average for a symmetric measure.
        """
        if not self._witnesses:
            return 1.0 if not other._witnesses else 0.0
        if not other._witnesses:
            return 0.0

        total = 0.0
        for w_self in self._witnesses:
            best = max(
                _witness_jaccard(w_self, w_other) for w_other in other._witnesses
            )
            total += best
        return total / len(self._witnesses)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_list(self) -> list[list[list]]:
        """Serialise to a nested list (JSON-compatible)."""
        return [sorted(list(ref) for ref in w) for w in self._witnesses]

    @classmethod
    def from_list(cls, data: list[list[list]]) -> "AtomicWhyProvenance":
        """Reconstruct from :meth:`to_list` output."""
        return cls([[(r, t) for r, t in w] for w in data])


# ---------------------------------------------------------------------------


class BlockedWhyProvenance:
    """Query-level blocked Why-provenance.

    Each witness is represented as a mapping
    ``{relation_name: frozenset(tuple_ids)}`` so that per-relation
    contributions can be analysed independently.

    Parameters
    ----------
    witnesses:
        Iterable of witnesses.  Each witness should be an iterable of
        ``(relation_name, tuple_id)`` pairs (same format as
        :class:`AtomicWhyProvenance`); the constructor groups them by
        relation internally.

    Examples
    --------
    >>> bwp = BlockedWhyProvenance([
    ...     [("orders", 1), ("orders", 2), ("customer", 5)],
    ...     [("orders", 3), ("customer", 5)],
    ... ])
    >>> len(bwp)
    2
    """

    def __init__(self, witnesses: Iterable[Iterable[TupleRef]]) -> None:
        blocked: list[BlockedWitness] = []
        for raw_witness in witnesses:
            grouped: dict[str, set[int]] = defaultdict(set)
            for rel, tid in raw_witness:
                grouped[rel].add(tid)
            blocked.append({rel: frozenset(ids) for rel, ids in grouped.items()})
        self._witnesses: tuple[BlockedWitness, ...] = tuple(blocked)

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._witnesses)

    def __iter__(self):
        return iter(self._witnesses)

    def __repr__(self) -> str:  # pragma: no cover
        return f"BlockedWhyProvenance(n_witnesses={len(self._witnesses)})"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def relations(self) -> FrozenSet[str]:
        """Distinct relation names across all blocked witnesses."""
        return frozenset(rel for w in self._witnesses for rel in w)

    @property
    def lineage(self):
        """Flatten into a :class:`~provenance_qs.provenance.Lineage`."""
        from .lineage import Lineage

        refs: list[TupleRef] = [
            (rel, tid)
            for w in self._witnesses
            for rel, tids in w.items()
            for tid in tids
        ]
        return Lineage(refs)

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def jaccard(self, other: "BlockedWhyProvenance") -> float:
        """Jaccard similarity using *blocked* witness overlap.

        Witnesses are compared by converting each blocked witness back to its
        canonical frozenset of (relation, tuple_id) pairs.
        """
        def flatten_blocked_witness(bw: BlockedWitness) -> Witness:
            return frozenset(
                (rel, tid) for rel, tids in bw.items() for tid in tids
            )

        a_set: Set[Witness] = {flatten_blocked_witness(w) for w in self._witnesses}
        b_set: Set[Witness] = {flatten_blocked_witness(w) for w in other._witnesses}
        union_size = len(a_set | b_set)
        if union_size == 0:
            return 1.0
        return len(a_set & b_set) / union_size

    def relation_overlap(self, other: "BlockedWhyProvenance") -> float:
        """Fraction of relations shared between the two provenances.

        A coarser structural similarity that ignores exact tuple IDs.
        """
        a_rels, b_rels = self.relations, other.relations
        union_size = len(a_rels | b_rels)
        if union_size == 0:
            return 1.0
        return len(a_rels & b_rels) / union_size

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_list(self) -> list[dict]:
        """Serialise to a list of dicts (JSON-compatible)."""
        return [
            {rel: sorted(tids) for rel, tids in w.items()}
            for w in self._witnesses
        ]

    @classmethod
    def from_list(cls, data: list[dict]) -> "BlockedWhyProvenance":
        """Reconstruct from :meth:`to_list` output."""
        witnesses = [
            [(rel, tid) for rel, tids in w.items() for tid in tids]
            for w in data
        ]
        return cls(witnesses)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _witness_jaccard(a: Witness, b: Witness) -> float:
    union_size = len(a | b)
    if union_size == 0:
        return 1.0
    return len(a & b) / union_size
