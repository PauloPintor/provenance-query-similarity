"""Tests for ML classifiers and evaluation utilities."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from provenance_qs.models.classifiers import (
    QueryTemplateClassifier,
    build_classifier,
)
from provenance_qs.models.evaluation import (
    evaluate_classifier,
    cross_validate_classifier,
    summarise_cv_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset(n_classes=4, n_samples=80, n_features=20, seed=0):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=10,
        random_state=seed,
    )
    return X.astype(np.float32), y


# ---------------------------------------------------------------------------
# build_classifier
# ---------------------------------------------------------------------------

class TestBuildClassifier:
    @pytest.mark.parametrize("name", ["knn", "lr", "rf", "nb"])
    def test_known_classifiers(self, name):
        clf = build_classifier(name)
        assert clf is not None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown classifier"):
            build_classifier("unknown_model")

    def test_kwarg_override(self):
        clf = build_classifier("knn", n_neighbors=3)
        assert clf.n_neighbors == 3

    def test_rf_n_estimators(self):
        clf = build_classifier("rf", n_estimators=10)
        assert clf.n_estimators == 10


# ---------------------------------------------------------------------------
# QueryTemplateClassifier
# ---------------------------------------------------------------------------

class TestQueryTemplateClassifier:
    @pytest.mark.parametrize("model_name", ["knn", "lr", "rf", "nb"])
    def test_fit_predict(self, model_name):
        X, y = make_dataset()
        clf = QueryTemplateClassifier(model_name)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape
        assert set(preds).issubset(set(y))

    def test_score(self):
        X, y = make_dataset(n_classes=2, n_samples=200)
        clf = QueryTemplateClassifier("rf")
        clf.fit(X, y)
        acc = clf.score(X, y)
        # should overfit on training set → high acc
        assert acc > 0.5

    def test_predict_proba(self):
        X, y = make_dataset()
        clf = QueryTemplateClassifier("lr")
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), len(set(y)))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_classes_attribute(self):
        X, y = make_dataset()
        clf = QueryTemplateClassifier("rf")
        clf.fit(X, y)
        assert clf.classes_ is not None
        assert set(clf.classes_) == set(y)

    def test_accepts_prebuilt_estimator(self):
        from sklearn.naive_bayes import GaussianNB
        clf = QueryTemplateClassifier(GaussianNB())
        X, y = make_dataset(n_classes=3)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(y)


# ---------------------------------------------------------------------------
# evaluate_classifier
# ---------------------------------------------------------------------------

class TestEvaluateClassifier:
    def test_returns_expected_keys(self):
        X, y = make_dataset()
        clf = QueryTemplateClassifier("rf")
        clf.fit(X, y)
        metrics = evaluate_classifier(clf, X, y)
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}

    def test_perfect_on_training_set(self):
        X, y = make_dataset(n_classes=2, n_samples=100, n_features=20)
        clf = QueryTemplateClassifier("rf")
        clf.fit(X, y)
        metrics = evaluate_classifier(clf, X, y)
        # Random forest typically memorises training data
        assert metrics["accuracy"] >= 0.9

    def test_values_in_range(self):
        X, y = make_dataset()
        clf = QueryTemplateClassifier("lr")
        clf.fit(X, y)
        metrics = evaluate_classifier(clf, X, y)
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


# ---------------------------------------------------------------------------
# cross_validate_classifier
# ---------------------------------------------------------------------------

class TestCrossValidate:
    def test_returns_expected_keys(self):
        X, y = make_dataset()
        clf = QueryTemplateClassifier("rf", n_estimators=10)
        results = cross_validate_classifier(clf, X, y, n_splits=3)
        assert set(results.keys()) == {"accuracy", "precision", "recall", "f1"}

    def test_correct_fold_count(self):
        X, y = make_dataset()
        clf = QueryTemplateClassifier("nb")
        results = cross_validate_classifier(clf, X, y, n_splits=3)
        for vals in results.values():
            assert len(vals) == 3

    def test_values_in_range(self):
        X, y = make_dataset()
        clf = QueryTemplateClassifier("nb")
        results = cross_validate_classifier(clf, X, y, n_splits=3)
        for metric, vals in results.items():
            for v in vals:
                assert 0.0 <= v <= 1.0, f"{metric}={v} out of [0,1]"

    def test_original_clf_not_modified(self):
        X, y = make_dataset()
        clf = QueryTemplateClassifier("nb")
        cross_validate_classifier(clf, X, y, n_splits=3)
        # The original clf should not be fitted by cross_validate
        assert clf.classes_ is None


# ---------------------------------------------------------------------------
# summarise_cv_results
# ---------------------------------------------------------------------------

class TestSummariseCvResults:
    def test_mean_and_std_present(self):
        cv = {"accuracy": [0.8, 0.85, 0.9], "f1": [0.75, 0.8, 0.85]}
        summary = summarise_cv_results(cv)
        assert "mean" in summary["accuracy"]
        assert "std" in summary["accuracy"]

    def test_correct_mean(self):
        cv = {"accuracy": [0.8, 0.9, 1.0]}
        summary = summarise_cv_results(cv)
        assert summary["accuracy"]["mean"] == pytest.approx(0.9)

    def test_correct_std(self):
        cv = {"f1": [1.0, 1.0, 1.0]}
        summary = summarise_cv_results(cv)
        assert summary["f1"]["std"] == pytest.approx(0.0)
