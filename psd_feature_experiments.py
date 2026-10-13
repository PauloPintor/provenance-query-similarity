#!/usr/bin/env python3
"""PSD group and single-descriptor experiments using the article's evaluator.

No SQL is executed, no split is regenerated, and no existing output is changed.
The supplied evaluation script is imported, never invoked through its controller.
Its PSD construction, training selection, fitting, preprocessing and prediction
functions are reused. Only PSD counts and query identifiers are loaded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np


FEATURES = (
    "witness_unique", "witness_total", "token_unique", "token_total",
    "token_unique_ratio", "witness_unique_ratio", "witness_token_ratio",
)
GROUPS = {
    "full_7": FEATURES,
    "counts_4": FEATURES[:4],
    "ratios_3": FEATURES[4:],
    "reduced_5": tuple(f for f in FEATURES if f not in {
        "witness_total", "witness_unique_ratio",
    }),
    "tokens_3": ("token_unique", "token_total", "token_unique_ratio"),
    "witnesses_3": ("witness_unique", "witness_total", "witness_unique_ratio"),
    "no_witness_counts_5": FEATURES[2:],
}
GROUP_LABELS = {
    "full_7": "All seven descriptors", "counts_4": "Four counts",
    "ratios_3": "Three ratios", "reduced_5": "Five descriptors",
    "tokens_3": "Token descriptors", "witnesses_3": "Witness descriptors",
    "no_witness_counts_5": "Without witness counts",
}
MODELS = ("xgboost", "logreg", "random_forest", "5-nn")
MODEL_LABELS = {
    "xgboost": "XGBoost", "logreg": "Logistic\nRegression",
    "random_forest": "Random\nForest", "5-nn": "5-NN",
}
FEATURE_LABELS = (
    r"$|\mathcal{W}|$", r"$|\mathcal{W}_{\mathrm{tot}}|$",
    r"$|\mathcal{T}|$", r"$|\mathcal{T}_{\mathrm{tot}}|$",
    r"$|\mathcal{T}|/|\mathcal{T}_{\mathrm{tot}}|$",
    r"$|\mathcal{W}|/|\mathcal{W}_{\mathrm{tot}}|$",
    r"$|\mathcal{W}|/|\mathcal{T}|$",
)
TEMPLATES = (3, 12, 13, 18, 20, 48, 66, 84, 89, 98)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def load_evaluator(path):
    name = "_article_psd_evaluator"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import evaluation script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for name in (
        "load_all_splits", "select_k_per_template", "build_label_mapping",
        "encode_labels", "build_scalars_matrix", "fit_vector_model",
        "predict_vector_model", "required_columns", "list_files",
    ):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"Evaluation script is missing {name}()")
    if tuple(module.PSD_FEATURE_NAMES) != FEATURES:
        raise ValueError("PSD feature names/order differ from the article protocol")
    allowed = {"query_id", "query_name", "template_id", "nwitness_unique",
               "nwitness_total", "ntokens_unique", "ntokens_total"}
    columns = set(module.required_columns(["scalars"], []))
    if columns != allowed:
        raise ValueError(f"Unexpected PSD columns: {sorted(columns)}")
    return module


def item_key(item):
    return int(item.template_id), str(item.query_name)


def validate_partitions(train, test, templates, train_k, test_k):
    key_sets = []
    for split, items, expected in (("train", train, train_k), ("test", test, test_k)):
        counts = Counter(int(it.template_id) for it in items)
        target = {int(t): expected for t in templates}
        if counts != target:
            raise ValueError(f"Unexpected {split} counts: {dict(sorted(counts.items()))}; "
                             f"expected {target}. Check templates and glob patterns.")
        keys = [item_key(it) for it in items]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate (template_id, query_name) in {split}")
        if any(not name.strip() for _, name in keys):
            raise ValueError(f"Empty query_name in {split}")
        key_sets.append(set(keys))
    overlap = key_sets[0] & key_sets[1]
    if overlap:
        raise ValueError(f"Train/test overlap: {sorted(overlap)[:5]}")


def validate_matrix(X, size):
    if X.shape != (size, len(FEATURES)) or not np.isfinite(X).all():
        raise ValueError("Invalid PSD matrix shape or non-finite feature values")
    if np.any(X < 0) or np.any(X[:, 0] > X[:, 1]) or np.any(X[:, 2] > X[:, 3]):
        raise ValueError("Invalid PSD counts: negative or distinct count exceeds total")
    expected = np.column_stack((
        X[:, 2].astype(float) / np.maximum(1, X[:, 3].astype(float)),
        X[:, 0].astype(float) / np.maximum(1, X[:, 1].astype(float)),
        X[:, 0].astype(float) / np.maximum(1, X[:, 2].astype(float)),
    ))
    if not np.allclose(X[:, 4:], expected, rtol=1e-6, atol=1e-7):
        raise ValueError("Stored PSD ratios do not agree with the four counts")


def check_xai_reference(path, test, Xte, yte, label_map):
    with np.load(path, allow_pickle=False) as data:
        if tuple(data["feature_names"].tolist()) != FEATURES:
            raise ValueError("XAI reference has a different PSD feature order")
        tids = data["template_ids"]
        keys = [(int(tids[int(y)]), str(name))
                for y, name in zip(data["y_true"], data["query_names"])]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate query identities in XAI reference")
        index = {key: i for i, key in enumerate(keys)}
        if set(index) != {item_key(it) for it in test}:
            raise ValueError("Test queries differ from the XAI reference")
        order = [index[item_key(it)] for it in test]
        if not np.array_equal(Xte, data["raw_feature_values"][order]):
            raise ValueError("Test PSD values differ from the XAI reference")
        ref_labels = np.array([label_map[int(tids[int(v)])]
                               for v in data["y_true"][order]])
        if not np.array_equal(yte, ref_labels):
            raise ValueError("Test labels differ from the XAI reference")
    print("[CHECK] Test queries, labels and PSD values match the XAI reference", flush=True)


def dataset_fingerprint(items, X):
    digest = hashlib.sha256()
    for it in items:
        digest.update(json.dumps([int(it.template_id), int(it.query_id), str(it.query_name)],
                                 ensure_ascii=True).encode() + b"\n")
    digest.update(np.asarray(X, dtype="<f4").tobytes())
    return digest.hexdigest()


def describe_features(output, datasets):
    stats, repetitions, correlations = [], [], []
    for split, items, matrix in datasets:
        tids = np.array([it.template_id for it in items])
        for tid in [None, *sorted(set(tids.tolist()))]:
            X = matrix if tid is None else matrix[tids == tid]
            tid_label = "all" if tid is None else int(tid)
            for j, feature in enumerate(FEATURES):
                col = X[:, j].astype(float)
                stats.append(dict(split=split, template_id=tid_label, feature=feature,
                                  n=len(col), n_unique=len(np.unique(col)),
                                  min=float(col.min()), max=float(col.max()),
                                  mean=float(col.mean()), std=float(col.std())))
            equal = int(np.sum(X[:, 0] == X[:, 1]))
            repetitions.append(dict(
                split=split, template_id=tid_label, n=len(X),
                equal_witness_counts=equal, repeated_witness_executions=len(X) - equal,
                equal_witness_counts_fraction=equal / len(X),
                ratio_equal_one=int(np.sum(X[:, 5] == 1)),
                zero_total_witnesses=int(np.sum(X[:, 1] == 0)),
            ))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(matrix.astype(float), rowvar=False)
        for i, first in enumerate(FEATURES):
            for j, second in enumerate(FEATURES):
                value = corr[i, j]
                correlations.append(dict(split=split, feature_a=first, feature_b=second,
                                         pearson_r=float(value) if np.isfinite(value) else ""))
    write_csv(output / "psd_descriptor_statistics.csv", stats)
    write_csv(output / "psd_witness_repetition.csv", repetitions)
    write_csv(output / "psd_correlations.csv", correlations)


def evaluate(evaluator, model, Xtr, ytr, Xte):
    np.random.seed(evaluator.RANDOM_SEED)
    name, k = ("knn", 5) if model == "5-nn" else (model, None)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fitted = evaluator.fit_vector_model(name, Xtr, ytr, knn_k=k)
        prediction = evaluator.predict_vector_model(name, fitted, Xte, ytr=ytr)
    messages = [dict(category=w.category.__name__, message=str(w.message)) for w in recorded]
    return np.asarray(prediction), messages


def evaluate_configuration(evaluator, output, kind, variant, features, model,
                           Xtr, ytr, Xte, yte, test, label_map, train_k, baseline=None):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    indices = [FEATURES.index(feature) for feature in features]
    t0 = time.perf_counter()
    prediction, messages = evaluate(evaluator, model, Xtr[:, indices], ytr, Xte[:, indices])
    elapsed = time.perf_counter() - t0
    labels = np.arange(len(label_map))
    if prediction.shape != yte.shape or not np.isin(prediction, labels).all():
        raise ValueError("Classifier returned invalid predictions")
    acc = float(accuracy_score(yte, prediction))
    f1 = float(f1_score(yte, prediction, labels=labels, average="macro", zero_division=0))
    common = dict(analysis=kind, variant=variant, model=model)
    row = dict(**common, train_k_per_template=train_k, train_total=len(ytr),
               test_total=len(yte), n_features=len(features), features=";".join(features),
               accuracy=acc, macro_f1=f1, errors=int(np.sum(yte != prediction)),
               delta_accuracy=acc - baseline["accuracy"] if baseline else 0.0,
               delta_macro_f1=f1 - baseline["macro_f1"] if baseline else 0.0,
               # Diagnostic wall time, not the article's amortised test-time metric.
               configuration_elapsed_s=elapsed, warning_count=len(messages))
    inverse = {v: k for k, v in label_map.items()}
    precision, recall, class_f1, support = precision_recall_fscore_support(
        yte, prediction, labels=labels, zero_division=0)
    per_class = [dict(**common, template_id=int(inverse[int(label)]),
                      precision=float(precision[label]), recall=float(recall[label]),
                      f1=float(class_f1[label]), support=int(support[label]),
                      false_positives=int(np.sum((prediction == label) & (yte != label))),
                      false_negatives=int(np.sum((prediction != label) & (yte == label))))
                 for label in labels]
    predictions = [dict(query_id=int(it.query_id), query_name=it.query_name,
                        true_template=int(inverse[int(actual)]),
                        predicted_template=int(inverse[int(pred)]), correct=int(actual == pred))
                   for it, actual, pred in zip(test, yte, prediction)]
    stem = f"{kind}__{variant}__{model}"
    write_csv(output / "predictions" / f"{stem}.csv", predictions)
    print(f"[RESULT] {variant:26s} {model:14s} acc={acc:.6f} "
          f"macro-F1={f1:.6f} errors={row['errors']} warnings={len(messages)}", flush=True)
    return row, per_class, [dict(**common, **message) for message in messages]


def check_baselines(path, rows, train_k):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        references = list(csv.DictReader(stream))
    checks = []
    for row in rows:
        matching = [ref for ref in references if ref.get("method") == "scalars"
                    and ref.get("model") == row["model"]
                    and int(ref.get("train_k", -1)) == train_k]
        if len(matching) != 1:
            raise ValueError(f"Expected one reference baseline for {row['model']}, "
                             f"found {len(matching)}")
        ref = matching[0]
        old_acc, old_f1 = float(ref["acc"]), float(ref["f1"])
        # The article CSV rounds scores (often to four decimals), whereas the
        # new runner retains full precision. Compare at the reference precision.
        acc_tolerance = .5 * 10.0 ** Decimal(ref["acc"]).as_tuple().exponent + 1e-12
        f1_tolerance = .5 * 10.0 ** Decimal(ref["f1"]).as_tuple().exponent + 1e-12
        match = (abs(row["accuracy"] - old_acc) <= acc_tolerance
                 and abs(row["macro_f1"] - old_f1) <= f1_tolerance)
        checks.append(dict(model=row["model"], reference_accuracy=old_acc,
                           current_accuracy=row["accuracy"], reference_macro_f1=old_f1,
                           current_macro_f1=row["macro_f1"],
                           accuracy_tolerance=acc_tolerance, f1_tolerance=f1_tolerance,
                           match=match))
    return checks


def make_figures(output, results, per_class):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    folder = output / "figures"
    folder.mkdir(exist_ok=True)

    def save(fig, stem):
        fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(folder / f"{stem}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    models = [model for model in MODELS if any(r["model"] == model for r in results)]
    groups = list(GROUPS)
    lookup = {(r["variant"], r["model"]): r for r in results if r["analysis"] == "group"}
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    positions = np.arange(len(models))
    for offset, variant, color, label in (
        (-.19, "counts_4", "#2f6f9f", "Four counts"),
        (.19, "full_7", "#df7627", "Four counts + three ratios"),
    ):
        scores = [lookup[(variant, model)]["macro_f1"] for model in models]
        bars = ax.bar(positions + offset, scores, width=.36, color=color, label=label)
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in scores], padding=3, fontsize=8)
    ax.set_xticks(positions, [MODEL_LABELS[model] for model in models])
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.16)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=2, frameon=False)
    ax.grid(axis="y", color="#dddddd", linewidth=.6)
    ax.set_axisbelow(True)
    save(fig, "psd_counts_vs_ratios")

    values = np.array([[lookup[(group, model)]["macro_f1"] for model in models]
                       for group in groups])
    fig, ax = plt.subplots(figsize=(7.2, 5.0), layout="constrained")
    img = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models])
    ax.set_yticks(range(len(groups)), [GROUP_LABELS[g] for g in groups])
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f"{values[i, j]:.4f}", ha="center", va="center",
                color="black" if values[i, j] > .6 else "white")
    fig.colorbar(img, ax=ax, label="Macro-F1")
    save(fig, "psd_group_comparison")

    singles = [r for r in results if r["analysis"] == "single"]
    if not singles:
        return
    model = "xgboost"
    lookup = {r["variant"]: r for r in singles}
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    vals = [lookup[f]["macro_f1"] for f in FEATURES]
    bars = ax.barh(np.arange(7), vals, color="#df7627", height=.62)
    ax.set_yticks(range(7), FEATURE_LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Macro-F1 (XGBoost)")
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals], padding=4)
    full = next(r["macro_f1"] for r in results
                if r["variant"] == "full_7" and r["model"] == model)
    ax.axvline(full, color="#333333", linestyle="--", label=f"All seven: {full:.4f}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="x", color="#dddddd", linewidth=.6)
    ax.set_axisbelow(True)
    save(fig, "psd_single_descriptor_f1")

    classes = sorted({r["template_id"] for r in per_class})
    lookup = {(r["variant"], r["template_id"]): r["f1"]
              for r in per_class if r["analysis"] == "single" and r["model"] == model}
    values = np.array([[lookup[(feature, tid)] for tid in classes] for feature in FEATURES])
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    img = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(classes)), [f"Q{tid}" for tid in classes])
    ax.set_yticks(range(7), FEATURE_LABELS)
    ax.set_xlabel("Query template")
    for i, j in np.ndindex(values.shape):
        ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", fontsize=9,
                color="black" if values[i,j] > .6 else "white")
    fig.colorbar(img, ax=ax, label="Per-template F1 (single-descriptor XGBoost)")
    save(fig, "psd_single_descriptor_by_template")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-script", type=Path, default=Path("main_2.py"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("PROV_DATA_ROOT", ".")))
    parser.add_argument("--train-glob", help="Optional explicit glob, e.g. '/data/q*_train_2/*.parquet'")
    parser.add_argument("--test-glob", help="Optional explicit glob, e.g. '/data/q*_test_2/*.parquet'")
    parser.add_argument("--templates", type=int, nargs="+", default=list(TEMPLATES))
    parser.add_argument("--train-k", type=int, default=1000)
    parser.add_argument("--test-k", type=int, default=250)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-leave-one-out", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--reference-results", type=Path,
                        help="Optional final_template_results CSV; stop if full-PSD scores differ")
    parser.add_argument("--reference-xai", type=Path,
                        help="Optional PSD XAI NPZ; verify test identities, labels and feature values")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory only; default: PSD_Experiments/run_<timestamp>")
    args = parser.parse_args(argv)
    if min(args.train_k, args.test_k) < 1:
        parser.error("Training and test counts must be positive")
    if len(set(args.templates)) != len(args.templates) or len(set(args.models)) != len(args.models):
        parser.error("Duplicate templates or models")
    if not args.skip_single and "xgboost" not in args.models:
        parser.error("Single-descriptor experiments require xgboost; add it or use --skip-single")
    return args


def run(args):
    evaluator_path = args.evaluation_script.resolve()
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"Evaluation script not found: {evaluator_path}. "
                                "Use --evaluation-script with the script used for the article.")
    output = (args.output_dir or Path("PSD_Experiments") /
              datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; nothing overwritten: {output}")
    evaluator = load_evaluator(evaluator_path)
    data_root = args.data_root.resolve()
    evaluator.TRAIN_GLOB = args.train_glob or str(data_root / "q*_train" / "*.parquet")
    evaluator.TEST_GLOB = args.test_glob or str(data_root / "q*_test" / "*.parquet")
    train_files = evaluator.list_files(evaluator.TRAIN_GLOB)
    test_files = evaluator.list_files(evaluator.TEST_GLOB)
    overlap = set(map(os.path.realpath, train_files)) & set(map(os.path.realpath, test_files))
    if overlap:
        raise ValueError(f"The same input files appear in train and test: {overlap}")
    pool, test = evaluator.load_all_splits(["scalars"], [])
    validate_partitions(pool, test, args.templates, args.train_k, args.test_k)
    train = evaluator.select_k_per_template(pool, args.train_k, evaluator.RANDOM_SEED)
    label_map = evaluator.build_label_mapping(train)
    ytr, yte = evaluator.encode_labels(train, label_map), evaluator.encode_labels(test, label_map)
    Xtr, Xte = evaluator.build_scalars_matrix(train), evaluator.build_scalars_matrix(test)
    validate_matrix(Xtr, len(train))
    validate_matrix(Xte, len(test))
    if args.reference_xai:
        check_xai_reference(args.reference_xai, test, Xte, yte, label_map)
    if args.reference_results and not args.reference_results.is_file():
        raise FileNotFoundError(args.reference_results)
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    manifest = dict(
        status="running", started_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_script=str(evaluator_path),
        evaluation_script_sha256=hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        random_seed=int(evaluator.RANDOM_SEED), feature_names=FEATURES,
        group_definitions=GROUPS, label_map=label_map,
        train_sha256=dataset_fingerprint(train, Xtr), test_sha256=dataset_fingerprint(test, Xte),
        train_files=[dict(path=str(Path(p).resolve()), size=Path(p).stat().st_size,
                          mtime_ns=Path(p).stat().st_mtime_ns) for p in train_files],
        test_files=[dict(path=str(Path(p).resolve()), size=Path(p).stat().st_size,
                         mtime_ns=Path(p).stat().st_mtime_ns) for p in test_files],
        python=sys.version, platform=platform.platform(),
        versions={name: importlib.metadata.version(name) for name in
                  ("numpy", "scipy", "scikit-learn", "xgboost", "pyarrow")},
        notes=["Fixed experimental comparisons, not feature selection on the test set.",
               "All model fitting and preprocessing reuse the supplied evaluation script.",
               "Configuration elapsed seconds are diagnostic, not article test-processing times.",
               "Ablation deletes coordinates; remaining ratios keep their original definitions."])
    write_json(output / "manifest.json", manifest)
    try:
        describe_features(output, (("train", train, Xtr), ("test", test, Xte)))
        write_csv(output / "split_manifest.csv", [
            dict(split=split, row_order=i, template_id=int(it.template_id),
                 query_id=int(it.query_id), query_name=it.query_name)
            for split, items in (("train", train), ("test", test))
            for i, it in enumerate(items)])
        results, per_class, warning_rows = [], [], []
        baselines = {}
        configs = [("group", "full_7", FEATURES, model) for model in args.models]
        configs += [("group", group, features, model)
                    for group, features in GROUPS.items() if group != "full_7"
                    for model in args.models]
        if not args.skip_leave_one_out:
            configs += [("leave_one_out", f"without_{feature}",
                         tuple(f for f in FEATURES if f != feature), model)
                        for feature in FEATURES for model in args.models]
        if not args.skip_single:
            configs += [("single", feature, (feature,), "xgboost") for feature in FEATURES]
        for index, (kind, variant, features, model) in enumerate(configs, start=1):
            print(f"[START {index}/{len(configs)}] {variant} / {model}", flush=True)
            row, classes, messages = evaluate_configuration(
                evaluator, output, kind, variant, features, model, Xtr, ytr, Xte, yte,
                test, label_map, args.train_k, baseline=baselines.get(model))
            results.append(row)
            per_class.extend(classes)
            warning_rows.extend(messages)
            if variant == "full_7":
                baselines[model] = row
            write_csv(output / "psd_group_results.csv", [r for r in results if r["analysis"] == "group"])
            write_csv(output / "psd_single_results.csv", [r for r in results if r["analysis"] == "single"])
            write_csv(output / "psd_leave_one_out_results.csv",
                      [r for r in results if r["analysis"] == "leave_one_out"])
            write_csv(output / "psd_per_class_metrics.csv", per_class)
            write_csv(output / "warnings.csv", warning_rows)
            if index == len(args.models) and args.reference_results:
                checks = check_baselines(args.reference_results, results, args.train_k)
                write_csv(output / "baseline_comparison.csv", checks)
                if not all(c["match"] for c in checks):
                    raise ValueError("Full-PSD baseline differs from the reference. Reduced-feature "
                                     "experiments were not run. Inspect baseline_comparison.csv, "
                                     "data, script and library versions before continuing.")
                print("[CHECK] All full-PSD baseline scores match the reference", flush=True)
        if not args.skip_plots:
            make_figures(output, results, per_class)
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        configurations=len(results), warning_count=len(warning_rows))
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
    print(f"[DONE] {len(results)} configurations; output={output}", flush=True)
    return output


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, ImportError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
