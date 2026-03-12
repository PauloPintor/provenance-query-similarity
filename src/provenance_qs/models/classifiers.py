"""
ML classifiers for query-template identification.

Supported models
----------------
- ``knn``   – k-Nearest Neighbours
- ``lr``    – Logistic Regression
- ``rf``    – Random Forest
- ``nb``    – Gaussian Naïve Bayes
- ``xgb``   – XGBoost

All classifiers expose the standard scikit-learn ``fit`` / ``predict`` /
``predict_proba`` interface and can therefore be used inside
``sklearn.pipeline.Pipeline`` objects.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier as _XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGBOOST_AVAILABLE = False

ModelName = Literal["knn", "lr", "rf", "nb", "xgb"]

_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "knn": {"n_neighbors": 5, "metric": "cosine"},
    "lr": {"max_iter": 1000, "C": 1.0, "solver": "lbfgs"},
    "rf": {"n_estimators": 100, "random_state": 42},
    "nb": {},
    "xgb": {
        "n_estimators": 100,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": 42,
    },
}


def build_classifier(
    name: ModelName,
    **kwargs: Any,
) -> BaseEstimator:
    """Instantiate a classifier by name, merging *kwargs* with defaults.

    Parameters
    ----------
    name:
        One of ``'knn'``, ``'lr'``, ``'rf'``, ``'nb'``, ``'xgb'``.
    **kwargs:
        Override any default hyperparameter for the selected model.

    Returns
    -------
    sklearn-compatible estimator
    """
    params = {**_DEFAULTS.get(name, {}), **kwargs}
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "lr":
        return LogisticRegression(**params)
    if name == "rf":
        return RandomForestClassifier(**params)
    if name == "nb":
        return GaussianNB(**params)
    if name == "xgb":
        if not _XGBOOST_AVAILABLE:
            raise ImportError(  # pragma: no cover
                "xgboost is not installed. Run: pip install xgboost"
            )
        return _XGBClassifier(**params)
    raise ValueError(
        f"Unknown classifier '{name}'. Choose from: knn, lr, rf, nb, xgb."
    )


class QueryTemplateClassifier:
    """High-level wrapper that pairs a provenance feature matrix with an ML model.

    Parameters
    ----------
    model:
        Classifier name (``'knn'``, ``'lr'``, ``'rf'``, ``'nb'``, ``'xgb'``)
        or a pre-built sklearn-compatible estimator.
    model_kwargs:
        Keyword arguments forwarded to :func:`build_classifier` when *model*
        is a string.

    Examples
    --------
    >>> clf = QueryTemplateClassifier("rf")
    >>> clf.fit(X_train, y_train)
    >>> preds = clf.predict(X_test)
    """

    def __init__(
        self,
        model: ModelName | BaseEstimator = "rf",
        **model_kwargs: Any,
    ) -> None:
        if isinstance(model, str):
            self._clf = build_classifier(model, **model_kwargs)
        else:
            self._clf = model
        self.classes_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # sklearn-like interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QueryTemplateClassifier":
        """Train on feature matrix *X* with labels *y*."""
        self._clf.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for *X*."""
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for *X* (if the model supports it)."""
        return self._clf.predict_proba(X)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on *(X, y)*."""
        return float(np.mean(self.predict(X) == y))

    @property
    def estimator(self) -> BaseEstimator:
        """Access the underlying sklearn estimator."""
        return self._clf
