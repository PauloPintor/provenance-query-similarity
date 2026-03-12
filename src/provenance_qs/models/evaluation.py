"""
Evaluation utilities for query-template classifiers.

Provides:
- ``evaluate_classifier``      – compute standard metrics on a held-out set.
- ``cross_validate_classifier`` – stratified k-fold cross-validation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold

from .classifiers import QueryTemplateClassifier


def evaluate_classifier(
    clf: QueryTemplateClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    average: str = "macro",
) -> Dict[str, float]:
    """Evaluate *clf* on a held-out test set.

    Parameters
    ----------
    clf:
        A fitted :class:`QueryTemplateClassifier`.
    X_test:
        Feature matrix for the test set.
    y_test:
        True labels for the test set.
    average:
        Averaging strategy forwarded to scikit-learn metrics
        (``'macro'``, ``'micro'``, ``'weighted'``).

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    y_pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, average=average, zero_division=0)
        ),
        "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
    }


def cross_validate_classifier(
    clf: QueryTemplateClassifier,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    average: str = "macro",
    random_state: int = 42,
) -> Dict[str, List[float]]:
    """Stratified k-fold cross-validation.

    Parameters
    ----------
    clf:
        Unfitted (or newly constructed) :class:`QueryTemplateClassifier`.
        **Note**: the estimator is cloned for each fold so the original object
        is left untouched.
    X:
        Full feature matrix.
    y:
        Labels.
    n_splits:
        Number of folds.
    average:
        Averaging strategy for precision / recall / F1.
    random_state:
        Seed for the :class:`~sklearn.model_selection.StratifiedKFold`.

    Returns
    -------
    dict
        Each key maps to a list of per-fold metric values:
        ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    from sklearn.base import clone

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    results: Dict[str, List[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_clf = QueryTemplateClassifier(clone(clf.estimator))
        fold_clf.fit(X_train, y_train)
        metrics = evaluate_classifier(fold_clf, X_test, y_test, average=average)
        for key, val in metrics.items():
            results[key].append(val)

    return results


def summarise_cv_results(
    cv_results: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """Summarise cross-validation results as mean ± std.

    Parameters
    ----------
    cv_results:
        Output of :func:`cross_validate_classifier`.

    Returns
    -------
    dict
        For each metric: ``{'mean': ..., 'std': ...}``.
    """
    return {
        metric: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
        for metric, vals in cv_results.items()
    }
