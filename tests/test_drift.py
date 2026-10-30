import numpy as np

from valeo_qc.drift import flag_drift, out_of_fold_scores, thresholds_by_lib


def test_out_of_fold_scores_do_not_match_an_image_with_itself():
    # distinct random images: only a leak of the image itself would give a zero distance
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 8))
    libs = np.array(["Die01"] * 60)
    folds = np.tile(np.arange(5), 12)
    assert out_of_fold_scores("knn", features, libs, folds).min() > 0.1


def test_thresholds_are_set_per_component():
    scores = np.concatenate([np.arange(100.0), 1000 + np.arange(100.0)])
    libs = np.array(["Die01"] * 100 + ["Die02"] * 100)
    flagged = flag_drift(scores, libs, thresholds_by_lib(scores, libs, 0.05))
    assert flagged[libs == "Die01"].sum() == 5
    assert flagged[libs == "Die02"].sum() == 5
