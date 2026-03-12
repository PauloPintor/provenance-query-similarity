"""
train_models.py – Train and cross-validate query-template classifiers.

Usage
-----
    python experiments/train_models.py \\
        --input   data/provenance.jsonl \\
        --models  knn lr rf nb xgb \\
        --folds   5 \\
        --feature lineage token structural \\
        --output  results/cv_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from provenance_qs.data.loader import load_provenance_jsonl
from provenance_qs.provenance.features import ProvenanceFeatureExtractor
from provenance_qs.models.classifiers import QueryTemplateClassifier
from provenance_qs.models.evaluation import (
    cross_validate_classifier,
    summarise_cv_results,
)


FEATURE_FLAGS = {
    "lineage":    "use_lineage",
    "token":      "use_token",
    "witness":    "use_witness",
    "blocked":    "use_blocked",
    "structural": "use_structural",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train query-template classifiers using provenance features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",   type=Path, default=Path("data/provenance.jsonl"),
                   help="Provenance JSONL dataset.")
    p.add_argument("--models",  nargs="+", default=["knn", "lr", "rf", "nb", "xgb"],
                   help="Classifiers to train.")
    p.add_argument("--folds",   type=int,  default=5,
                   help="Number of cross-validation folds.")
    p.add_argument("--feature", nargs="+",
                   default=["lineage", "token", "structural"],
                   dest="features",
                   choices=list(FEATURE_FLAGS),
                   help="Feature families to include.")
    p.add_argument("--output",  type=Path, default=Path("results/cv_results.json"),
                   help="Output JSON file for results.")
    p.add_argument("--seed",    type=int,  default=42,
                   help="Random seed.")
    return p.parse_args()


def build_feature_matrix(dataset, features: List[str]):
    kwargs = {v: False for v in FEATURE_FLAGS.values()}
    for feat in features:
        kwargs[FEATURE_FLAGS[feat]] = True

    extractor = ProvenanceFeatureExtractor(**kwargs)
    needs_lineage = kwargs["use_lineage"] or kwargs["use_token"] or kwargs["use_structural"]
    needs_atomic  = kwargs["use_witness"]
    needs_blocked = kwargs["use_blocked"]

    lineages     = dataset.lineages     if needs_lineage else None
    atomic_provs = dataset.atomic_why_provenances if needs_atomic  else None
    blocked_provs = dataset.blocked_why_provenances if needs_blocked else None

    X = extractor.fit_transform(
        lineages=lineages,
        atomic_provs=atomic_provs,
        blocked_provs=blocked_provs,
    )
    return X


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from {args.input} …")
    dataset = load_provenance_jsonl(args.input)
    y = dataset.template_ids
    print(f"  {len(dataset)} records, {len(set(y))} templates.")

    print(f"Building feature matrix (features: {args.features}) …")
    X = build_feature_matrix(dataset, args.features)
    print(f"  Feature matrix shape: {X.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for model_name in args.models:
        print(f"\nCross-validating '{model_name}' ({args.folds} folds) …")
        try:
            clf = QueryTemplateClassifier(model_name)
            cv = cross_validate_classifier(
                clf, X, y,
                n_splits=args.folds,
                random_state=args.seed,
            )
            summary = summarise_cv_results(cv)
            all_results[model_name] = {"cv": cv, "summary": summary}
            acc = summary["accuracy"]
            print(f"  Accuracy: {acc['mean']:.4f} ± {acc['std']:.4f}")
            print(f"  F1:       {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}")
        except Exception as exc:
            print(f"  [ERROR] {model_name}: {exc}", file=sys.stderr)
            all_results[model_name] = {"error": str(exc)}

    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
