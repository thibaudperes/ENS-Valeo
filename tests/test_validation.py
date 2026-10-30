import numpy as np
import pandas as pd
import pytest

from valeo_qc.metrics import DRIFT, GOOD, MISSING
from valeo_qc.validation import make_folds, out_of_fold_probabilities, score_predictions


@pytest.fixture
def frame():
    labels = [GOOD] * 40 + [MISSING] * 40 + [1] * 10 + [2] * 3
    libs = ["Die01", "Die02"] * 20 + ["Die03", "Die04"] * 20 + ["Die01"] * 10 + ["Die02"] * 3
    return pd.DataFrame({"label": labels, "lib": libs})


def test_folds_are_deterministic_and_cover_every_image(frame):
    folds = make_folds(frame)
    assert set(folds) == {0, 1, 2, 3, 4}
    assert folds.tolist() == make_folds(frame).tolist()
    assert make_folds(frame, seed=1).tolist() != folds.tolist()


def test_folds_stratify_on_label_and_lib(frame):
    folds = make_folds(frame)
    counts = pd.crosstab(frame["label"].astype(str) + frame["lib"], folds)
    assert (counts.loc[f"{GOOD}Die01"] == 4).all()
    assert (counts.loc["1Die01"] == 2).all()


def test_rare_cells_do_not_fail_and_stay_spread(frame):
    folds = make_folds(frame)
    assert np.unique(folds[frame["label"] == 2]).size == 3


def test_out_of_fold_probabilities_cover_seven_labels():
    rng = np.random.default_rng(0)
    labels = np.repeat([GOOD, MISSING, 1], 20)
    features = rng.normal(size=(60, 4)) + 5 * labels[:, None]
    folds = np.tile(np.arange(5), 12)

    probabilities = out_of_fold_probabilities(features, labels, folds)

    assert probabilities.shape == (60, 7)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(60), abs=1e-5)
    assert (probabilities[:, DRIFT] == 0).all()
    assert (probabilities.argmax(axis=1) == labels).mean() > 0.9


def test_score_predictions_counts_false_good_and_recall():
    y_true = np.array([GOOD, GOOD, MISSING, MISSING, 1])
    y_pred = np.array([GOOD, MISSING, GOOD, MISSING, DRIFT])
    report = score_predictions(y_true, y_pred)
    assert report["false_good_rate"] == pytest.approx(1 / 3)
    assert report["recall_GOOD"] == 0.5
    assert report["recall_Missing"] == 0.5
    assert report["recall_Boucle plate"] == 0
    assert report["predicted_drift_rate"] == 0.2
    assert report["mean_penalty"] == pytest.approx((100 + 10000 + 1000) / 5)
