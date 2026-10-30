"""Evaluate the decision rule on known images and pseudo-drift, for several GOOD filters.

The expected penalty combines these rates with the shares read from the leaderboard: they only
serve this analysis and the final parameters are fixed by hand in valeo_qc.pipeline.
Writes results/decision.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from valeo_qc.data import LABEL_IDS, load_train
from valeo_qc.decision import apply_temperature, decide, fit_temperature
from valeo_qc.drift import flag_drift, thresholds_by_lib
from valeo_qc.embedding_cache import CACHE_DIR, cache_path
from valeo_qc.metrics import DRIFT, GOOD, minimum_cost_labels
from valeo_qc.pipeline import (
    FALSE_ALARM_RATE,
    infer,
    load_views,
    out_of_fold_inference,
    subset,
)
from valeo_qc.validation import make_folds, score_predictions

RARE_DEFECTS = ("Boucle plate", "Lift-off blanc", "Lift-off noir", "Short circuit MOS")
GOOD_QUANTILES = (1.0, 0.9, 0.75, 0.5, 0.25)
LEADERBOARD = Path("results/leaderboard_constants.json")
OUTPUT = Path("results/decision.json")


def main() -> None:
    train = load_train()
    folds = make_folds(train)
    labels, libs = train["label"].to_numpy(), train["lib"].to_numpy()

    combined, crop = load_views(train["filename"], CACHE_DIR)
    data = (combined, crop, labels, libs)
    known = out_of_fold_inference(data, folds)

    temperature = fit_temperature(known["probabilities"], labels)
    drift_thresholds = thresholds_by_lib(known["drift"], libs, FALSE_ALARM_RATE)
    known_flags = flag_drift(known["drift"], libs, drift_thresholds)
    good_scores = known["good"][labels == GOOD]
    calibrated = apply_temperature(known["probabilities"], temperature)

    pseudo = {}
    for defect in RARE_DEFECTS:
        held_out = labels == LABEL_IDS[defect]
        query = (combined[held_out], crop[held_out], libs[held_out])
        pseudo[defect] = {**infer(subset(data, ~held_out), query), "libs": libs[held_out]}

    blur_crop_files = np.load(cache_path("blur_crop"))["filenames"]
    blur_combined, blur_crop = load_views(pd.Series(blur_crop_files), CACHE_DIR, prefix="blur_")
    position = pd.Series(np.arange(len(train)), index=train["filename"])
    blur_index = position.loc[blur_crop_files].to_numpy()
    blur_libs, blur_folds = libs[blur_index], folds[blur_index]
    blur = {
        key: np.zeros((len(blur_index), 7) if key == "probabilities" else len(blur_index))
        for key in known
    }
    for fold in np.unique(blur_folds):
        held_out = blur_folds == fold
        query = (blur_combined[held_out], blur_crop[held_out], blur_libs[held_out])
        for key, values in infer(subset(data, folds != fold), query).items():
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


if __name__ == "__main__":
    main()
