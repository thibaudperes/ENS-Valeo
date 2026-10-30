"""Compare patch-based drift detection with the global kNN on the same pseudo-drift.

Writes results/patch_drift.json. Thresholds flag 0.5% of known images, per component.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from valeo_qc.data import LABEL_IDS, load_test, load_train
from valeo_qc.drift import flag_drift, thresholds_by_lib
from valeo_qc.metrics import GOOD
from valeo_qc.patch_drift import out_of_fold_patch_scores, patch_scores_by_lib
from valeo_qc.pipeline import FALSE_ALARM_RATE
from valeo_qc.validation import make_folds

PATCH_DIR = Path("data/processed/patches")
RARE_DEFECTS = ("Boucle plate", "Lift-off blanc", "Lift-off noir", "Short circuit MOS")
GLOBAL_RESULTS = Path("results/drift.json")
OUTPUT = Path("results/patch_drift.json")


def report(known, known_libs, known_labels, drift, drift_libs) -> dict[str, float]:
    thresholds = thresholds_by_lib(known, known_libs, FALSE_ALARM_RATE)
    flags = flag_drift(known, known_libs, thresholds)
    return {
        "auroc": roc_auc_score(
            np.r_[np.zeros(len(known)), np.ones(len(drift))], np.r_[known, drift]
        ),
        "recall": float(flag_drift(drift, drift_libs, thresholds).mean()),
        "false_alarm_good": float(flags[known_labels == GOOD].mean()),
        "false_alarm_defects": float(flags[known_labels != GOOD].mean()),
    }


def main() -> None:
    train, test = load_train(), load_test()
    labels, libs = train["label"].to_numpy(), train["lib"].to_numpy()
    folds = make_folds(train)
    names = pd.read_csv(PATCH_DIR / "crop.csv").iloc[:, 0]
    stored = np.load(PATCH_DIR / "crop.npy", mmap_mode="r")
    position = pd.Series(np.arange(len(names)), index=names.to_numpy())
    patches = np.asarray(stored[position.loc[train["filename"]].to_numpy()])
    test_patches = np.asarray(stored[position.loc[test["filename"]].to_numpy()])

    results = {}
    for defect in RARE_DEFECTS:
        start = time.perf_counter()
        held_out = labels == LABEL_IDS[defect]
        known = out_of_fold_patch_scores(patches[~held_out], libs[~held_out], folds[~held_out])
        drift = patch_scores_by_lib(
            patches[~held_out], libs[~held_out], patches[held_out], libs[held_out]
        )
        results[defect] = report(known, libs[~held_out], labels[~held_out], drift, libs[held_out])
        print(defect, results[defect], f"{time.perf_counter() - start:.0f} s", flush=True)

    known = out_of_fold_patch_scores(patches, libs, folds)
    blur_names = pd.read_csv(PATCH_DIR / "blur_crop.csv").iloc[:, 0]
    blurred = train.set_index("filename").loc[blur_names].reset_index()
    blur_patches = np.load(PATCH_DIR / "blur_crop.npy")
    blur_libs = blurred["lib"].to_numpy()
    blur_folds = pd.Series(folds, index=train["filename"]).loc[blur_names].to_numpy()
    drift = np.empty(len(blurred))
    for fold in np.unique(blur_folds):
        selected = blur_folds == fold
        drift[selected] = patch_scores_by_lib(
            patches[folds != fold],
            libs[folds != fold],
            blur_patches[selected],
            blur_libs[selected],
        )
    results["blur"] = report(known, libs, labels, drift, blur_libs)
    print("blur", results["blur"], flush=True)

    test_libs = test["lib"].to_numpy()
    scores = patch_scores_by_lib(patches, libs, test_patches, test_libs)
    flagged = flag_drift(scores, test_libs, thresholds_by_lib(known, libs, FALSE_ALARM_RATE))
    results["test_flagged"] = float(flagged.mean())

    global_results = json.loads(GLOBAL_RESULTS.read_text())["detectors"]["knn_crop"]
    results["global_knn"] = {
        name: {"auroc": global_results[name]["auroc"], **global_results[name]["at_0.005"]}
        for name in (*RARE_DEFECTS, "blur")
    }
    results["global_knn"]["test_flagged"] = global_results["test_flagged"]["at_0.005"]
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print("test flagged", results["test_flagged"], flush=True)


if __name__ == "__main__":
    main()
