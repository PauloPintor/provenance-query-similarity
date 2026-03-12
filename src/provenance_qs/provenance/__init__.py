"""
provenance_qs.provenance
========================

Provenance representations for database queries.

Classes
-------
Lineage
    Tuple-level lineage lifted to the query level.
AtomicWhyProvenance
    Atomic Why-provenance: a collection of minimal witnesses per result tuple.
BlockedWhyProvenance
    Blocked Why-provenance: witnesses grouped by source relation.
ProvenanceFeatureExtractor
    Converts provenance objects into numeric feature vectors suitable for ML.
"""

from .lineage import Lineage
from .why_provenance import AtomicWhyProvenance, BlockedWhyProvenance
from .features import ProvenanceFeatureExtractor

__all__ = [
    "Lineage",
    "AtomicWhyProvenance",
    "BlockedWhyProvenance",
    "ProvenanceFeatureExtractor",
]
