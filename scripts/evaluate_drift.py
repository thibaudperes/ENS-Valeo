"""Measure pseudo-drift against the classifier, then the simple detectors.

Pseudo-drift: each rare defect removed from training, and blurred train images. Blurred images
are always scored by models that did not see their source image. Writes results/drift.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from valeo_qc.data import LABEL_IDS, load_test, load_train
from valeo_qc.drift import (
    METHODS,
    flag_drift,
    out_of_fold_scores,
    scores_by_lib,
    thresholds_by_lib,
)
from valeo_qc.embedding_cache import cache_path, load_embeddings
from valeo_qc.metrics import GOOD, minimum_cost_labels
from valeo_qc.validation import fit_and_predict, make_folds

RARE_DEFECTS = ("Boucle plate", "Lift-off blanc", "Lift-off noir", "Short circuit MOS")
FEATURE_SETS = {"crop": ("crop",), "full": ("full",), "crop+full": ("crop", "full")}
FALSE_ALARM_RATES = (0.005, 0.01, 0.02, 0.05)
OUTPUT = Path("results/drift.json")


def features_for(views: dict[str, np.ndarray], feature_set: str) -> np.ndarray:
    parts = [views[view] for view in FEATURE_SETS[feature_set]]
    parts = [part / np.linalg.norm(part, axis=1, keepdims=True) for part in parts]
    return np.concatenate(parts, axis=1)


def good_rates(probabilities: np.ndarray) -> dict[str, float]:
    return {
        "argmax_good": float((probabilities.argmax(axis=1) == GOOD).mean()),
        "minimum_cost_good": float((minimum_cost_labels(probabilities) == GOOD).mean()),
        "n": len(probabilities),
    }


def classifier_on_pseudo_drift(train, blurred, views, blur_views, folds) -> dict[str, dict]:
    labels = train["label"].to_numpy()
    features = np.concatenate([views["crop"], views["full"]], axis=1)
    blur_features = np.concatenate([blur_views["crop"], blur_views["full"]], axis=1)
    results = {}
    for defect in RARE_DEFECTS:
        held_out = labels == LABEL_IDS[defect]
        probabilities = fit_and_predict(features[~held_out], labels[~held_out], features[held_out])
        results[defect] = good_rates(probabilities)

    blur_folds = pd.Series(folds, index=train["filename"]).loc[blurred["filename"]].to_numpy()
    probabilities = np.zeros((len(blurred), 7))
    for fold in np.unique(blur_folds):
        selected = blur_folds == fold
        probabilities[selected] = fit_and_predict(
            features[folds != fold], labels[folds != fold], blur_features[selected]
        )
    results["blur"] = good_rates(probabilities)
    return results


def detection(known: np.ndarray, known_libs, known_labels, drift: np.ndarray, drift_libs) -> dict:
    report = {
        "auroc": roc_auc_score(
            np.r_[np.zeros(len(known)), np.ones(len(drift))], np.r_[known, drift]
        )
    }
    for rate in FALSE_ALARM_RATES:
        thresholds = thresholds_by_lib(known, known_libs, rate)
        known_flags = flag_drift(known, known_libs, thresholds)
        report[f"at_{rate}"] = {
            "recall": float(flag_drift(drift, drift_libs, thresholds).mean()),
            "false_alarm_good": float(known_flags[known_labels == GOOD].mean()),
            "false_alarm_defects": float(known_flags[known_labels != GOOD].mean()),
        }
    return report


def evaluate(method, features, blur_features, train, blurred, folds) -> dict[str, dict]:
    libs = train["lib"].to_numpy()
    labels = train["label"].to_numpy()
    results = {}
    for defect in RARE_DEFECTS:
        held_out = labels == LABEL_IDS[defect]
        known = out_of_fold_scores(method, features[~held_out], libs[~held_out], folds[~held_out])
        drift = scores_by_lib(
            method, features[~held_out], libs[~held_out], features[held_out], libs[held_out]
        )
        results[defect] = detection(
            known, libs[~held_out], labels[~held_out], drift, libs[held_out]
        )

    known = out_of_fold_scores(method, features, libs, folds)
    blur_libs = blurred["lib"].to_numpy()
    blur_folds = pd.Series(folds, index=train["filename"]).loc[blurred["filename"]].to_numpy()
    drift = np.empty(len(blurred))
    for fold in np.unique(blur_folds):
        selected = blur_folds == fold
        drift[selected] = scores_by_lib(
            method,
            features[folds != fold],
            libs[folds != fold],
            blur_features[selected],
            blur_libs[selected],
        )
    results["blur"] = detection(known, libs, labels, drift, blur_libs)
    return results


def main() -> None:
    train, test = load_train(), load_test()
    folds = make_folds(train)
    with np.load(cache_path("blur_crop")) as cache:
        blurred = train.set_index("filename").loc[cache["filenames"]].reset_index()
    views = {v: load_embeddings(cache_path(v), train["filename"]) for v in ("crop", "full")}
    blur_views = {
        v: load_embeddings(cache_path(f"blur_{v}"), blurred["filename"]) for v in ("crop", "full")
    }
    test_views = {v: load_embeddings(cache_path(v), test["filename"]) for v in ("crop", "full")}

    results = {"classifier": classifier_on_pseudo_drift(train, blurred, views, blur_views, folds)}
    print(json.dumps(results["classifier"], indent=1), flush=True)

    results["detectors"] = {}
    for method in METHODS:
        for feature_set in FEATURE_SETS:
            features = features_for(views, feature_set)
            blur_features = features_for(blur_views, feature_set)
            name = f"{method}_{feature_set}"
            report = evaluate(method, features, blur_features, train, blurred, folds)
            libs = train["lib"].to_numpy()
            known = out_of_fold_scores(method, features, libs, folds)
            test_scores = scores_by_lib(
                method,
                features,
                libs,
                features_for(test_views, feature_set),
                test["lib"].to_numpy(),
            )
            report["test_flagged"] = {
                f"at_{rate}": float(
                    flag_drift(
                        test_scores, test["lib"].to_numpy(), thresholds_by_lib(known, libs, rate)
                    ).mean()
                )
                for rate in FALSE_ALARM_RATES
            }
            results["detectors"][name] = report
            summary = {
                key: round(item["auroc"], 3) for key, item in report.items() if "auroc" in item
            }
            print(name, summary, report["test_flagged"], flush=True)

    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
