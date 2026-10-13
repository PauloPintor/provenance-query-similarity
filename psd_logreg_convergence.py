#!/usr/bin/env python3
"""Check PSD Logistic Regression convergence without changing earlier outputs.

Requires the existing psd_feature_experiments.py alongside this file and the
evaluation script used for the article. Only three PSD groups are fitted.
Iteration limits are selected using training convergence, never test scores.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import warnings


VARIANTS = ("full_7", "counts_4", "reduced_5")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-script", type=Path, default=Path("main_2.py"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--reference-run", type=Path, required=True,
                        help="Completed run directory from psd_feature_experiments.py")
    parser.add_argument("--train-glob")
    parser.add_argument("--test-glob")
    parser.add_argument("--iteration-limits", nargs="+", type=int,
                        default=[1000, 5000, 20000])
    parser.add_argument("--output-dir", type=Path,
                        help="New directory only; default: PSD_Convergence/run_<timestamp>")
    args = parser.parse_args(argv)
    if any(n < 1 for n in args.iteration_limits):
        parser.error("Iteration limits must be positive")
    if args.iteration_limits != sorted(set(args.iteration_limits)):
        parser.error("Iteration limits must be strictly increasing")
    return args


def logreg_step(pipeline):
    from sklearn.linear_model import LogisticRegression
    steps = [(name, step) for name, step in pipeline.named_steps.items()
             if isinstance(step, LogisticRegression)]
    if len(steps) != 1:
        raise ValueError("Expected one LogisticRegression step in the original pipeline")
    return steps[0]


def clone_with_limit(pipeline, limit):
    from sklearn.base import clone
    name, _ = logreg_step(pipeline)
    result = clone(pipeline)
    result.set_params(**{f"{name}__max_iter": limit})
    return result


def fit_checked(fit, variant, phase, limit):
    from sklearn.exceptions import ConvergenceWarning
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = fit()
    elapsed = time.perf_counter() - start
    _, estimator = logreg_step(model)
    iterations = int(max(estimator.n_iter_))
    warning_count = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    row = dict(variant=variant, phase=phase, max_iter=int(estimator.max_iter),
               n_iter=iterations, convergence_warnings=warning_count,
               converged=warning_count == 0 and iterations < estimator.max_iter,
               fit_s=elapsed)
    if limit is not None and estimator.max_iter != limit:
        raise ValueError("The fitted iteration limit differs from the requested limit")
    messages = [dict(variant=variant, phase=phase, max_iter=int(estimator.max_iter),
                     category=w.category.__name__, message=str(w.message)) for w in caught]
    print(f"[FIT] {phase} / {variant}: max_iter={row['max_iter']} "
          f"n_iter={iterations} converged={row['converged']}", flush=True)
    return model, row, messages


def convergence_rounds(baselines, matrices, ytrain, limits, record):
    """No test features, labels or scores are available to the stopping rule."""
    original_limit = max(logreg_step(m)[1].max_iter for m in baselines.values())
    if limits[0] <= original_limit:
        raise ValueError(f"Iteration limits must exceed the original limit ({original_limit})")
    for limit in limits:
        fitted, checks = {}, []
        for variant, baseline in baselines.items():
            candidate = clone_with_limit(baseline, limit)
            model, row, messages = fit_checked(
                lambda: candidate.fit(matrices[variant], ytrain), variant, "check", limit)
            fitted[variant] = model
            checks.append(row)
            record(row, messages)
        if all(row["converged"] for row in checks):
            return limit, fitted
    raise ValueError("Not all three groups converged. No final comparison was produced. "
                     "Inspect convergence_attempts.csv and warnings.csv before increasing the cap.")


def score_predictions(evaluator, model, Xtest, ytest, test, label_map,
                      variant, phase, output, helper):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    prediction = np.asarray(evaluator.predict_vector_model("logreg", model, Xtest))
    labels = np.arange(len(label_map))
    if prediction.shape != ytest.shape or not np.isin(prediction, labels).all():
        raise ValueError("Invalid predictions")
    inverse = {value: key for key, value in label_map.items()}
    common = dict(phase=phase, variant=variant, model="logreg")
    row = dict(**common, n_features=Xtest.shape[1], test_total=len(test),
               accuracy=float(accuracy_score(ytest, prediction)),
               macro_f1=float(f1_score(ytest, prediction, labels=labels,
                                       average="macro", zero_division=0)),
               errors=int(np.sum(prediction != ytest)),
               max_iter=int(logreg_step(model)[1].max_iter))
    predictions = [dict(query_id=int(it.query_id), query_name=it.query_name,
                        true_template=int(inverse[int(actual)]),
                        predicted_template=int(inverse[int(pred)]), correct=int(actual == pred))
                   for it, actual, pred in zip(test, ytest, prediction)]
    helper.write_csv(output / "predictions" / f"{phase}__{variant}__logreg.csv", predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ytest, prediction, labels=labels, zero_division=0)
    classes = [dict(**common, template_id=int(inverse[int(label)]),
                    precision=float(precision[label]), recall=float(recall[label]),
                    f1=float(f1[label]), support=int(support[label])) for label in labels]
    print(f"[RESULT] {phase} / {variant}: acc={row['accuracy']:.6f} "
          f"macro-F1={row['macro_f1']:.6f} errors={row['errors']}", flush=True)
    return row, prediction, predictions, classes


def verify_reference(reference_rows, new_row, reference_predictions, new_predictions):
    variant = new_row["variant"]
    matching = [r for r in reference_rows if r["variant"] == variant and r["model"] == "logreg"]
    if len(matching) != 1:
        raise ValueError(f"Expected one reference result for {variant} / logreg")
    old = matching[0]
    for metric in ("accuracy", "macro_f1"):
        if abs(float(old[metric]) - new_row[metric]) > 1e-10:
            raise ValueError(f"Original {variant} {metric} does not reproduce the reference")
    def indexed(rows):
        result = {(int(r["true_template"]), r["query_name"]): int(r["predicted_template"])
                  for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate reference prediction identity")
        return result
    if indexed(reference_predictions) != indexed(new_predictions):
        raise ValueError(f"Original {variant} predictions do not reproduce the reference")


def run(args):
    import numpy as np
    import psd_feature_experiments as helper

    source = args.evaluation_script.resolve()
    reference_dir = args.reference_run.resolve()
    reference = json.loads((reference_dir / "manifest.json").read_text())
    if reference["status"] != "complete":
        raise ValueError("The reference run is not complete")
    if sha256(source) != reference["evaluation_script_sha256"]:
        raise ValueError("Evaluation script differs from the reference run. Use the unchanged main_2.py.")
    if sha256(helper.__file__) != reference["runner_sha256"]:
        raise ValueError("psd_feature_experiments.py differs from the reference run")
    if tuple(reference["feature_names"]) != helper.FEATURES:
        raise ValueError("PSD feature order differs from the reference")
    for variant in VARIANTS:
        if tuple(reference["group_definitions"][variant]) != helper.GROUPS[variant]:
            raise ValueError(f"Reference group definition differs for {variant}")
    versions = {name: importlib.metadata.version(name) for name in reference["versions"]}
    if versions != reference["versions"]:
        raise ValueError("Library versions differ from the reference run; use the original environment")
    output = (args.output_dir or Path("PSD_Convergence") /
              datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        raise FileExistsError(f"Output exists; nothing overwritten: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "predictions").mkdir()
    manifest = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                    evaluation_script=str(source), evaluation_script_sha256=sha256(source),
                    runner_sha256=sha256(__file__), reference_run=str(reference_dir),
                    reference_manifest_sha256=sha256(reference_dir / "manifest.json"),
                    versions=versions, python=sys.version, variants=list(VARIANTS),
                    iteration_limits=args.iteration_limits,
                    stopping_rule="First common cap where every group converges on training data",
                    changed_model_parameter="max_iter only",
                    notes=["No earlier result or input is overwritten.",
                           "No test scores are used to choose an iteration limit.",
                           "Candidate fit_s values are diagnostics, not prediction latency.",
                           "The separate full-PSD timing follows the original evaluator protocol."])
    helper.write_json(output / "manifest.json", manifest)
    print(f"[OUTPUT] {output}", flush=True)
    attempts, messages = [], []
    def record(row, caught):
        attempts.append(row)
        messages.extend(caught)
        helper.write_csv(output / "convergence_attempts.csv", attempts)
        helper.write_csv(output / "warnings.csv", messages)
    try:
        print(f"[LOAD] Importing {source}", flush=True)
        evaluator = helper.load_evaluator(source)
        if evaluator.RANDOM_SEED != reference["random_seed"]:
            raise ValueError("Random seed differs from the reference")
        for name in ("eval_vector_method", "save_psd_xai_artifacts"):
            if not callable(getattr(evaluator, name, None)):
                raise ValueError(f"Evaluation script is missing {name}()")
        root = args.data_root.resolve()
        evaluator.TRAIN_GLOB = args.train_glob or str(root / "q*_train" / "*.parquet")
        evaluator.TEST_GLOB = args.test_glob or str(root / "q*_test" / "*.parquet")
        print("[LOAD] Reading only PSD counts and query identifiers", flush=True)
        pool, test = evaluator.load_all_splits(["scalars"], [])
        settings = reference["arguments"]
        train_k, test_k = settings["train_k"], settings["test_k"]
        helper.validate_partitions(pool, test, settings["templates"], train_k, test_k)
        train = evaluator.select_k_per_template(pool, train_k, evaluator.RANDOM_SEED)
        label_map = evaluator.build_label_mapping(train)
        ytrain, ytest = evaluator.encode_labels(train, label_map), evaluator.encode_labels(test, label_map)
        Xtrain, Xtest = evaluator.build_scalars_matrix(train), evaluator.build_scalars_matrix(test)
        helper.validate_matrix(Xtrain, len(train))
        helper.validate_matrix(Xtest, len(test))
        for split, items, X in (("train", train, Xtrain), ("test", test, Xtest)):
            fingerprint = helper.dataset_fingerprint(items, X)
            if fingerprint != reference[f"{split}_sha256"]:
                raise ValueError(f"{split} queries, order or PSD values differ from the reference")
            manifest[f"{split}_sha256"] = fingerprint
        manifest.update(train_total=len(train), test_total=len(test), random_seed=evaluator.RANDOM_SEED)
        print("[CHECK] Original scripts, versions, train/test identities and PSD values match", flush=True)
        indices = {v: [helper.FEATURES.index(f) for f in helper.GROUPS[v]] for v in VARIANTS}
        matrices = {v: Xtrain[:, indices[v]] for v in VARIANTS}
        baselines, original_rows, original_predictions = {}, {}, {}
        score_rows, class_rows = [], []
        reference_rows = read_csv(reference_dir / "psd_group_results.csv")
        for variant in VARIANTS:
            model, row, caught = fit_checked(
                lambda: evaluator.fit_vector_model("logreg", matrices[variant], ytrain),
                variant, "original", None)
            record(row, caught)
            baselines[variant] = model
            score, pred, predictions, classes = score_predictions(
                evaluator, model, Xtest[:, indices[variant]], ytest, test, label_map,
                variant, "original", output, helper)
            score_rows.append(score)
            class_rows.extend(classes)
            helper.write_csv(output / "scores.csv", score_rows)
            verify_reference(reference_rows, score, read_csv(reference_dir / "predictions" /
                             f"group__{variant}__logreg.csv"), predictions)
            original_rows[variant], original_predictions[variant] = score, pred
        print("[CHECK] All three original scores and prediction files reproduced", flush=True)
        limit, fitted = convergence_rounds(baselines, matrices, ytrain, args.iteration_limits, record)
        manifest["selected_max_iter"] = limit
        manifest["original_parameters"] = logreg_step(baselines["full_7"])[1].get_params()
        comparisons, final_predictions = [], {}
        for variant in VARIANTS:
            score, pred, _, classes = score_predictions(
                evaluator, fitted[variant], Xtest[:, indices[variant]], ytest, test, label_map,
                variant, "converged", output, helper)
            score_rows.append(score)
            class_rows.extend(classes)
            original = original_rows[variant]
            comparisons.append(dict(variant=variant, n_features=len(indices[variant]),
                features=";".join(helper.GROUPS[variant]), original_accuracy=original["accuracy"],
                converged_accuracy=score["accuracy"], original_macro_f1=original["macro_f1"],
                converged_macro_f1=score["macro_f1"], original_errors=original["errors"],
                converged_errors=score["errors"],
                changed_predictions=int(np.sum(original_predictions[variant] != pred)),
                max_iter=limit, n_iter=int(max(logreg_step(fitted[variant])[1].n_iter_))))
            final_predictions[variant] = pred
            helper.write_csv(output / "scores.csv", score_rows)
            helper.write_csv(output / "comparison.csv", comparisons)
            helper.write_csv(output / "per_class_metrics.csv", class_rows)

        # Refit the full model once via the original timed evaluation path.
        # This keeps timing definitions and XAI artifacts aligned with the article.
        original_fit = evaluator.fit_vector_model
        def fit_final(model_name, Xtr, ytr, knn_k=None):
            if model_name != "logreg":
                raise ValueError("Only Logistic Regression is allowed in this verification")
            model = clone_with_limit(baselines["full_7"], limit)
            model, row, caught = fit_checked(lambda: model.fit(Xtr, ytr),
                                             "full_7", "timed_refit", limit)
            record(row, caught)
            if not row["converged"]:
                raise ValueError("The timed full-model refit did not converge")
            return model
        evaluator.fit_vector_model = fit_final
        try:
            result, pred, artifacts = evaluator.eval_vector_method(
                train, test, "scalars", "logreg", collect_artifacts=True)
        finally:
            evaluator.fit_vector_model = original_fit
        if not np.array_equal(pred, final_predictions["full_7"]):
            raise ValueError("Timed refit predictions differ from the converged full model")
        from dataclasses import asdict
        result.train_k = train_k
        helper.write_csv(output / "full_psd_result.csv", [asdict(result)])
        original_xai_dir = evaluator.XAI_DIR
        evaluator.XAI_DIR = str(output / "XAI")
        try:
            config = evaluator.RunConfig(train_k=train_k, method="scalars", model="logreg")
            evaluator.save_psd_xai_artifacts(config, test, pred, artifacts)
        finally:
            evaluator.XAI_DIR = original_xai_dir
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                        fit_count=len(attempts), warning_count=len(messages))
        helper.write_json(output / "manifest.json", manifest)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        helper.write_json(output / "manifest.json", manifest)
        raise
    print(f"[DONE] Three groups converged at a common cap of {limit}. Output: {output}", flush=True)
    return output


def main(argv=None):
    args = parse_args(argv)
    print("[START] PSD Logistic Regression convergence verification", flush=True)
    try:
        run(args)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())