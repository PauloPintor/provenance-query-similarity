"""
evaluate.py – Produce a detailed evaluation report comparing classifiers and
feature representations.

Usage
-----
    python experiments/evaluate.py \\
        --input   data/provenance.jsonl \\
        --output  results/evaluation_report.json \\
        --seed    42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from provenance_qs.data.loader import load_provenance_jsonl
from provenance_qs.provenance.features import ProvenanceFeatureExtractor
from provenance_qs.models.classifiers import QueryTemplateClassifier
from provenance_qs.models.evaluation import (
    cross_validate_classifier,
    summarise_cv_results,
)
from provenance_qs.similarity import (
    lineage_jaccard,
    cosine_distance_matrix,
    jaccard_distance_matrix,
)

# Experiment configurations: (label, feature kwargs)
CONFIGS = [
    ("Lineage-only",     {"use_lineage": True,  "use_token": False, "use_witness": False,
                          "use_blocked": False,  "use_structural": False}),
    ("Token-freq",       {"use_lineage": False,  "use_token": True,  "use_witness": False,
                          "use_blocked": False,  "use_structural": False}),
    ("Structural",       {"use_lineage": False,  "use_token": False, "use_witness": False,
                          "use_blocked": False,  "use_structural": True}),
    ("Token+Structural", {"use_lineage": False,  "use_token": True,  "use_witness": False,
                          "use_blocked": False,  "use_structural": True}),
    ("Lineage+Token+Structural", {
        "use_lineage": True, "use_token": True, "use_witness": False,
        "use_blocked": False, "use_structural": True,
    }),
]

CLASSIFIERS = ["knn", "lr", "rf", "nb"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full evaluation of provenance-based query classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  type=Path, default=Path("data/provenance.jsonl"))
    p.add_argument("--output", type=Path, default=Path("results/evaluation_report.json"))
    p.add_argument("--folds",  type=int,  default=5)
    p.add_argument("--seed",   type=int,  default=42)
    return p.parse_args()


def build_X(dataset, kwargs):
    extractor = ProvenanceFeatureExtractor(**kwargs)
    lineages = dataset.lineages if (kwargs.get("use_lineage") or kwargs.get("use_token") or kwargs.get("use_structural")) else None
    atomic   = dataset.atomic_why_provenances if kwargs.get("use_witness") else None
    blocked  = dataset.blocked_why_provenances if kwargs.get("use_blocked") else None
    return extractor.fit_transform(lineages=lineages, atomic_provs=atomic, blocked_provs=blocked)


def main() -> None:
    args = parse_args()

    print(f"Loading {args.input} …")
    dataset = load_provenance_jsonl(args.input)
    y = dataset.template_ids
    print(f"  {len(dataset)} records · {len(set(y))} templates")

    report = {}

    # ------------------------------------------------------------------
    # 1. Per-config, per-classifier cross-validation
    # ------------------------------------------------------------------
    print("\n=== Cross-validation results ===")
    for config_name, feat_kwargs in CONFIGS:
        print(f"\n[{config_name}]")
        X = build_X(dataset, feat_kwargs)
        report[config_name] = {"n_features": int(X.shape[1]), "classifiers": {}}
        for clf_name in CLASSIFIERS:
            try:
                clf = QueryTemplateClassifier(clf_name)
                cv = cross_validate_classifier(clf, X, y, n_splits=args.folds, random_state=args.seed)
                summary = summarise_cv_results(cv)
                report[config_name]["classifiers"][clf_name] = summary
                acc = summary["accuracy"]
                print(f"  {clf_name:6s}  acc={acc['mean']:.4f}±{acc['std']:.4f}  "
                      f"f1={summary['f1']['mean']:.4f}±{summary['f1']['std']:.4f}")
            except Exception as exc:
                print(f"  {clf_name}: ERROR – {exc}", file=sys.stderr)
                report[config_name]["classifiers"][clf_name] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # 2. Lineage Jaccard similarity statistics
    # ------------------------------------------------------------------
    print("\n=== Lineage Jaccard similarity (intra- vs inter-template) ===")
    lineages = dataset.lineages
    labels   = dataset.template_ids
    intra, inter = [], []
    n = len(lineages)
    for i in range(min(n, 200)):
        for j in range(i + 1, min(n, 200)):
            sim = lineage_jaccard(lineages[i], lineages[j])
            if labels[i] == labels[j]:
                intra.append(sim)
            else:
                inter.append(sim)
    if intra:
        print(f"  Intra-template Jaccard: mean={np.mean(intra):.4f} std={np.std(intra):.4f}")
    if inter:
        print(f"  Inter-template Jaccard: mean={np.mean(inter):.4f} std={np.std(inter):.4f}")
    report["lineage_jaccard"] = {
        "intra": {"mean": float(np.mean(intra)) if intra else None,
                  "std":  float(np.std(intra))  if intra else None},
        "inter": {"mean": float(np.mean(inter)) if inter else None,
                  "std":  float(np.std(inter))  if inter else None},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport saved → {args.output}")


if __name__ == "__main__":
    main()
