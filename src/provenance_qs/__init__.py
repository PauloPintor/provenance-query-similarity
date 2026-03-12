"""
provenance_qs – Data Provenance-based Query Similarity.

This package implements provenance-based representations for database queries
and provides utilities for evaluating their effectiveness as a signal for
measuring semantic similarity between queries.

Modules
-------
provenance
    Lineage, Why-provenance (atomic and blocked), and feature extractors.
similarity
    Similarity metrics defined over provenance representations.
models
    ML classifiers and evaluation utilities.
data
    TPC-DS query generation and data-loading helpers.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("provenance-query-similarity")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
