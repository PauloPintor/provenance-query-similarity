"""
Lineage-based query-level provenance.

A *lineage set* for a query Q evaluated against a database D is the union of
the tuple-level lineage of every result tuple in Q(D).  Each element of the
lineage set is a (relation_name, tuple_id) pair.

References
----------
Cheney, J., Chiticariu, L., & Tan, W.-C. (2009).
Provenance in Databases: Why, How, and Where.
Foundations and Trends in Databases, 1(4), 379–474.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import FrozenSet, Tuple


TupleRef = Tuple[str, int]  # (relation_name, tuple_id)


class Lineage:
    """Query-level lineage set.

    Parameters
    ----------
    tuples:
        Iterable of ``(relation_name, tuple_id)`` pairs that constitute the
        lineage of *all* result tuples produced by a single query execution.

    Examples
    --------
    >>> lin = Lineage([("orders", 1), ("orders", 3), ("customer", 2)])
    >>> ("orders", 1) in lin
    True
    >>> len(lin)
    3
    """

    def __init__(self, tuples: Iterable[TupleRef]) -> None:
        self._tuples: FrozenSet[TupleRef] = frozenset(tuples)

    # ------------------------------------------------------------------
    # Containers protocol
    # ------------------------------------------------------------------

    def __contains__(self, item: object) -> bool:
        return item in self._tuples

    def __len__(self) -> int:
        return len(self._tuples)

    def __iter__(self):
        return iter(self._tuples)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Lineage(n_tuples={len(self._tuples)})"

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def union(self, other: "Lineage") -> "Lineage":
        """Return the union of two lineage sets."""
        return Lineage(self._tuples | other._tuples)

    def intersection(self, other: "Lineage") -> "Lineage":
        """Return the intersection of two lineage sets."""
        return Lineage(self._tuples & other._tuples)

    def jaccard(self, other: "Lineage") -> float:
        """Jaccard similarity between two lineage sets.

        Returns 1.0 when both sets are empty (trivially identical).
        """
        union_size = len(self._tuples | other._tuples)
        if union_size == 0:
            return 1.0
        inter_size = len(self._tuples & other._tuples)
        return inter_size / union_size

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def relations(self) -> FrozenSet[str]:
        """Set of distinct relation names referenced by the lineage."""
        return frozenset(rel for rel, _ in self._tuples)

    def tuples_for_relation(self, relation: str) -> FrozenSet[int]:
        """Return the tuple IDs belonging to *relation*."""
        return frozenset(tid for rel, tid in self._tuples if rel == relation)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dictionary."""
        grouped: dict[str, list[int]] = {}
        for rel, tid in sorted(self._tuples):
            grouped.setdefault(rel, []).append(tid)
        return grouped

    @classmethod
    def from_dict(cls, data: dict) -> "Lineage":
        """Reconstruct a :class:`Lineage` from :meth:`to_dict` output."""
        tuples: list[TupleRef] = [
            (rel, tid) for rel, tids in data.items() for tid in tids
        ]
        return cls(tuples)
