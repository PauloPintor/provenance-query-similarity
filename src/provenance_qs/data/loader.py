"""
Data loading and persistence utilities.

Provenance records are stored in **JSON Lines** (``.jsonl``) format – one
JSON object per line.  Each record has the following schema::

    {
        "query_id":    <str>,
        "template_id": <int>,
        "sql":         <str>,
        "lineage":     { <relation>: [<tuple_id>, ...], ... },
        "atomic_why":  [ [ [<relation>, <tuple_id>], ... ], ... ],
        "blocked_why": [ { <relation>: [<tuple_id>, ...], ... }, ... ]
    }

:class:`ProvenanceDataset` wraps a list of such records and provides
convenience accessors that reconstruct :class:`~provenance_qs.provenance.Lineage`,
:class:`~provenance_qs.provenance.AtomicWhyProvenance`, and
:class:`~provenance_qs.provenance.BlockedWhyProvenance` objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..provenance.lineage import Lineage
from ..provenance.why_provenance import AtomicWhyProvenance, BlockedWhyProvenance


@dataclass
class ProvenanceRecord:
    """Single record associating a query execution with its provenance."""

    query_id: str
    template_id: int
    sql: str
    lineage_data: Dict[str, List[int]] = field(default_factory=dict)
    atomic_why_data: List[List[List]] = field(default_factory=list)
    blocked_why_data: List[Dict[str, List[int]]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Lazy provenance accessors
    # ------------------------------------------------------------------

    @property
    def lineage(self) -> Lineage:
        """Reconstruct a :class:`Lineage` from the stored data."""
        return Lineage.from_dict(self.lineage_data)

    @property
    def atomic_why_provenance(self) -> AtomicWhyProvenance:
        """Reconstruct an :class:`AtomicWhyProvenance` from stored data."""
        return AtomicWhyProvenance.from_list(self.atomic_why_data)

    @property
    def blocked_why_provenance(self) -> BlockedWhyProvenance:
        """Reconstruct a :class:`BlockedWhyProvenance` from stored data."""
        return BlockedWhyProvenance.from_list(self.blocked_why_data)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "template_id": self.template_id,
            "sql": self.sql,
            "lineage": self.lineage_data,
            "atomic_why": self.atomic_why_data,
            "blocked_why": self.blocked_why_data,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProvenanceRecord":
        return cls(
            query_id=d["query_id"],
            template_id=d["template_id"],
            sql=d.get("sql", ""),
            lineage_data=d.get("lineage", {}),
            atomic_why_data=d.get("atomic_why", []),
            blocked_why_data=d.get("blocked_why", []),
        )


class ProvenanceDataset:
    """In-memory dataset of :class:`ProvenanceRecord` objects.

    Parameters
    ----------
    records:
        Sequence of :class:`ProvenanceRecord` instances.
    """

    def __init__(self, records: Sequence[ProvenanceRecord]) -> None:
        self._records: List[ProvenanceRecord] = list(records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ProvenanceRecord]:
        return iter(self._records)

    def __getitem__(self, idx: int) -> ProvenanceRecord:
        return self._records[idx]

    # ------------------------------------------------------------------
    # Batch accessors
    # ------------------------------------------------------------------

    @property
    def template_ids(self) -> List[int]:
        """Ordered list of template IDs (one per record)."""
        return [r.template_id for r in self._records]

    @property
    def lineages(self) -> List[Lineage]:
        """Reconstruct all :class:`Lineage` objects."""
        return [r.lineage for r in self._records]

    @property
    def atomic_why_provenances(self) -> List[AtomicWhyProvenance]:
        """Reconstruct all :class:`AtomicWhyProvenance` objects."""
        return [r.atomic_why_provenance for r in self._records]

    @property
    def blocked_why_provenances(self) -> List[BlockedWhyProvenance]:
        """Reconstruct all :class:`BlockedWhyProvenance` objects."""
        return [r.blocked_why_provenance for r in self._records]

    def filter_by_template(self, template_id: int) -> "ProvenanceDataset":
        """Return a new dataset containing only records for *template_id*."""
        return ProvenanceDataset(
            [r for r in self._records if r.template_id == template_id]
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]]) -> "ProvenanceDataset":
        return cls([ProvenanceRecord.from_dict(d) for d in data])


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def save_provenance_jsonl(
    dataset: ProvenanceDataset,
    path: str | Path,
) -> None:
    """Persist *dataset* to a JSON Lines file at *path*.

    Parameters
    ----------
    dataset:
        Dataset to serialise.
    path:
        Destination file path (will be created or overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in dataset:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_provenance_jsonl(path: str | Path) -> ProvenanceDataset:
    """Load a :class:`ProvenanceDataset` from a JSON Lines file.

    Parameters
    ----------
    path:
        Source file path.

    Returns
    -------
    ProvenanceDataset
    """
    path = Path(path)
    records: List[ProvenanceRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(ProvenanceRecord.from_dict(json.loads(line)))
    return ProvenanceDataset(records)
