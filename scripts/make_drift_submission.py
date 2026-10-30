"""Predict Drift for images far from the known ones, and Missing for the others.

Never predicts GOOD: this submission measures the detector alone.
"""

import numpy as np

from valeo_qc.data import load_test, load_train
from valeo_qc.drift import flag_drift, out_of_fold_scores, scores_by_lib, thresholds_by_lib
from valeo_qc.embedding_cache import cache_path, load_embeddings
from valeo_qc.metrics import DRIFT, MISSING
from valeo_qc.submission import SUBMISSIONS_DIR, write_submission
from valeo_qc.validation import make_folds

METHOD = "knn"
FALSE_ALARM_RATE = 0.005


def main() -> None:
    train, test = load_train(), load_test()
    libs, test_libs = train["lib"].to_numpy(), test["lib"].to_numpy()
    features = load_embeddings(cache_path("crop"), train["filename"])
    test_features = load_embeddings(cache_path("crop"), test["filename"])

    known = out_of_fold_scores(METHOD, features, libs, make_folds(train))
    thresholds = thresholds_by_lib(known, libs, FALSE_ALARM_RATE)
    scores = scores_by_lib(METHOD, features, libs, test_features, test_libs)
    flagged = flag_drift(scores, test_libs, thresholds)

    path = SUBMISSIONS_DIR / "missing_or_drift.csv"
    write_submission(np.where(flagged, DRIFT, MISSING), path)
    print(f"{path}: {flagged.sum()} of {len(flagged)} images flagged ({flagged.mean():.1%})")


if __name__ == "__main__":
    main()
