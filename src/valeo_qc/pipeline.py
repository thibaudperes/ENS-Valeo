"""From cached embeddings to final labels: classifier, drift detector and decision rule."""

from pathlib import Path

import numpy as np
import pandas as pd

from valeo_qc.decision import apply_temperature, decide, fit_temperature
from valeo_qc.drift import flag_drift, scores_by_lib, thresholds_by_lib
from valeo_qc.embedding_cache import cache_path, load_embeddings
from valeo_qc.metrics import GOOD
from valeo_qc.validation import fit_and_predict, make_folds

FALSE_ALARM_RATE = 0.005  # known images flagged as drift, per component
GOOD_QUANTILE = 0.25  # closest share of known GOOD images among which GOOD can be declared
DRIFT_METHOD = "knn"


def load_views(filenames: pd.Series, cache_dir: Path, prefix: str = "") -> tuple:
    """Classifier features (both views) and detector features (crop view)."""
    crop = load_embeddings(cache_path(f"{prefix}crop", cache_dir), filenames)
    full = load_embeddings(cache_path(f"{prefix}full", cache_dir), filenames)
    return np.concatenate([crop, full], axis=1), crop


def subset(data: tuple, mask: np.ndarray) -> tuple:
    return tuple(values[mask] for values in data)


def infer(train_part: tuple, query: tuple) -> dict[str, np.ndarray]:
    """Classifier probabilities and both distances of `query`, from `train_part` only."""
    combined, crop, labels, libs = train_part
    query_combined, query_crop, query_libs = query
    is_good = labels == GOOD
    return {
        "probabilities": fit_and_predict(combined, labels, query_combined),
        "drift": scores_by_lib(DRIFT_METHOD, crop, libs, query_crop, query_libs),
        "good": scores_by_lib(DRIFT_METHOD, crop[is_good], libs[is_good], query_crop, query_libs),
    }


def out_of_fold_inference(data: tuple, folds: np.ndarray) -> dict[str, np.ndarray]:
    combined, crop, _, libs = data
    known = {
        "probabilities": np.zeros((len(folds), 7)),
        "drift": np.empty(len(folds)),
        "good": np.empty(len(folds)),
    }
    for fold in np.unique(folds):
        held_out = folds == fold
        query = (combined[held_out], crop[held_out], libs[held_out])
        for key, values in infer(subset(data, ~held_out), query).items():
            known[key][held_out] = values
    return known


def predict_test(train: pd.DataFrame, test: pd.DataFrame, cache_dir: Path, use_good: bool):
    """Test labels. Calibration and thresholds come from out-of-fold train predictions only."""
    combined, crop = load_views(train["filename"], cache_dir)
    labels, libs = train["label"].to_numpy(), train["lib"].to_numpy()
    data = (combined, crop, labels, libs)
    known = out_of_fold_inference(data, make_folds(train))

    temperature = fit_temperature(known["probabilities"], labels)
    drift_thresholds = thresholds_by_lib(known["drift"], libs, FALSE_ALARM_RATE)
    is_good = labels == GOOD
    good_thresholds = thresholds_by_lib(known["good"][is_good], libs[is_good], 1 - GOOD_QUANTILE)

    test_libs = test["lib"].to_numpy()
    test_combined, test_crop = load_views(test["filename"], cache_dir)
    final = infer(data, (test_combined, test_crop, test_libs))
    flagged = flag_drift(final["drift"], test_libs, drift_thresholds)
    near_good = ~flag_drift(final["good"], test_libs, good_thresholds)
    if not use_good:
        near_good = np.zeros(len(test), dtype=bool)
    return decide(apply_temperature(final["probabilities"], temperature), flagged, near_good)
