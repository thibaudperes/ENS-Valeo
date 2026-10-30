"""Evaluate the full decision rule on known images and pseudo-drift, and write the submission.

The expected penalty combines these rates with the shares read from the leaderboard: they only
serve this analysis and the final parameters below are fixed by hand. Writes results/decision.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from valeo_qc.data import LABEL_IDS, load_test, load_train
from valeo_qc.decision import apply_temperature, decide, fit_temperature
from valeo_qc.drift import flag_drift, scores_by_lib, thresholds_by_lib
from valeo_qc.embedding_cache import cache_path, load_embeddings
from valeo_qc.metrics import DRIFT, GOOD, MISSING, minimum_cost_labels
from valeo_qc.submission import SUBMISSIONS_DIR, write_submission
from valeo_qc.validation import fit_and_predict, make_folds, score_predictions

RARE_DEFECTS = ("Boucle plate", "Lift-off blanc", "Lift-off noir", "Short circuit MOS")
FALSE_ALARM_RATE = 0.005
GOOD_QUANTILES = (1.0, 0.9, 0.75, 0.5, 0.25)
CHOSEN_GOOD_QUANTILE = 0.25
LEADERBOARD = Path("results/leaderboard_constants.json")
OUTPUT = Path("results/decision.json")


def infer(train_part, query):
    """Classifier probabilities and both distances of `query`, from `train_part` only."""
    combined, crop, labels, libs = train_part
    query_combined, query_crop, query_libs = query
    is_good = labels == GOOD
    return {
        "probabilities": fit_and_predict(combined, labels, query_combined),
        "drift": scores_by_lib("knn", crop, libs, query_crop, query_libs),
        "good": scores_by_lib("knn", crop[is_good], libs[is_good], query_crop, query_libs),
    }


def part(data, mask):
    return tuple(values[mask] for values in data)


def main() -> None:
    train, test = load_train(), load_test()
    folds = make_folds(train)
    labels, libs = train["label"].to_numpy(), train["lib"].to_numpy()

    def load(view_names, filenames):
        return [load_embeddings(cache_path(view), filenames) for view in view_names]

    crop, full = load(("crop", "full"), train["filename"])
    data = (np.concatenate([crop, full], axis=1), crop, labels, libs)

    known = {
        "probabilities": np.zeros((len(train), 7)),
        "drift": np.empty(len(train)),
        "good": np.empty(len(train)),
    }
    for fold in np.unique(folds):
        held_out = folds == fold
        query = (data[0][held_out], crop[held_out], libs[held_out])
        for key, values in infer(part(data, ~held_out), query).items():
            known[key][held_out] = values

    temperature = fit_temperature(known["probabilities"], labels)
    drift_thresholds = thresholds_by_lib(known["drift"], libs, FALSE_ALARM_RATE)
    known_flags = flag_drift(known["drift"], libs, drift_thresholds)
    good_scores = known["good"][labels == GOOD]
    calibrated = apply_temperature(known["probabilities"], temperature)

    pseudo = {}
    for defect in RARE_DEFECTS:
        held_out = labels == LABEL_IDS[defect]
        query = (data[0][held_out], crop[held_out], libs[held_out])
        pseudo[defect] = {**infer(part(data, ~held_out), query), "libs": libs[held_out]}

    blur_crop_files = np.load(cache_path("blur_crop"))["filenames"]
    blur_crop, blur_full = load(("blur_crop", "blur_full"), pd.Series(blur_crop_files))
    position = pd.Series(np.arange(len(train)), index=train["filename"])
    blur_index = position.loc[blur_crop_files].to_numpy()
    blur_libs, blur_folds = libs[blur_index], folds[blur_index]
    blur_combined = np.concatenate([blur_crop, blur_full], axis=1)
    blur = {
        key: np.zeros((len(blur_index), 7) if key == "probabilities" else len(blur_index))
        for key in known
    }
    for fold in np.unique(blur_folds):
        held_out = blur_folds == fold
        query = (blur_combined[held_out], blur_crop[held_out], blur_libs[held_out])
        for key, values in infer(part(data, folds != fold), query).items():
            blur[key][held_out] = values
    pseudo["blur"] = {**blur, "libs": blur_libs}

    shares = json.loads(LEADERBOARD.read_text())["shares"]
    results = {
        "temperature": temperature,
        "false_alarm_rate": FALSE_ALARM_RATE,
        "known_minimum_cost_uncalibrated": score_predictions(
            labels, minimum_cost_labels(known["probabilities"])
        ),
        "rules": {},
    }
    for quantile in GOOD_QUANTILES:
        good_thresholds = thresholds_by_lib(good_scores, libs[labels == GOOD], 1 - quantile)
        near = ~flag_drift(known["good"], libs, good_thresholds)
        known_labels = decide(calibrated, known_flags, near)
        passes, detected = {}, {}
        for name, item in pseudo.items():
            item_near = ~flag_drift(item["good"], item["libs"], good_thresholds)
            item_flags = flag_drift(item["drift"], item["libs"], drift_thresholds)
            item_labels = decide(
                apply_temperature(item["probabilities"], temperature), item_flags, item_near
            )
            passes[name] = float((item_labels == GOOD).mean())
            detected[name] = float((item_labels == DRIFT).mean())
        report = score_predictions(labels, known_labels)
        mean_pass = float(np.mean(list(passes.values())))
        recall_good = report["recall_GOOD"]
        gain = 100 * shares["good"] * recall_good
        loss = (
            9000 * shares["drift"] * mean_pass
            + 10000 * shares["defects"] * (report["false_good_rate"])
        )
        results["rules"][f"good_quantile_{quantile}"] = {
            **report,
            "pseudo_drift_declared_good": passes,
            "pseudo_drift_detected": detected,
            "mean_pseudo_drift_declared_good": mean_pass,
            "expected_change_vs_never_good": loss - gain,
        }
        print(
            quantile,
            f"penalty {report['mean_penalty']:.2f} recall_GOOD {recall_good:.3f}",
            f"false_good {report['false_good_rate']:.4f} mean_pass {mean_pass:.4f}",
            f"gain {gain:.1f} loss {loss:.1f}",
            {key: round(value, 3) for key, value in passes.items()},
            flush=True,
        )

    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")

    test_crop, test_full = load(("crop", "full"), test["filename"])
    test_libs = test["lib"].to_numpy()
    final = infer(data, (np.concatenate([test_crop, test_full], axis=1), test_crop, test_libs))
    good_thresholds = thresholds_by_lib(good_scores, libs[labels == GOOD], 1 - CHOSEN_GOOD_QUANTILE)
    predictions = decide(
        apply_temperature(final["probabilities"], temperature),
        flag_drift(final["drift"], test_libs, drift_thresholds),
        ~flag_drift(final["good"], test_libs, good_thresholds),
    )
    path = SUBMISSIONS_DIR / "classifier_and_drift.csv"
    write_submission(predictions, path)
    counts = pd.Series(predictions).value_counts().sort_index().to_dict()
    print(path, f"T={temperature:.3f}", counts, "Missing", int((predictions == MISSING).sum()))


if __name__ == "__main__":
    main()
