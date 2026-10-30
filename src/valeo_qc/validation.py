"""Cross-validation folds, out-of-fold logistic regression and scoring."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from valeo_qc.data import CLASS_NAMES
from valeo_qc.metrics import DRIFT, GOOD, N_CLASSES, mean_penalty, penalty_to_pwa

N_FOLDS = 5
SEED = 42


def make_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Folds stratified on label x lib, with cells too rare to split falling back on the label."""
    cells = df["label"].astype(str) + "_" + df["lib"]
    rare = cells.map(cells.value_counts()) < n_folds
    strata = cells.where(~rare, df["label"].astype(str))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.empty(len(df), dtype=int)
    for fold, (_, validation) in enumerate(splitter.split(df, strata)):
        folds[validation] = fold
    return folds


def out_of_fold_probabilities(
    features: np.ndarray, labels: np.ndarray, folds: np.ndarray, c: float = 0.1
) -> np.ndarray:
    """Class probabilities over all seven labels, each row from a model that did not see it."""
    probabilities = np.zeros((len(labels), N_CLASSES), dtype=np.float32)
    for fold in np.unique(folds):
        validation = folds == fold
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        model.fit(features[~validation], labels[~validation])
        classes = model.classes_
        probabilities[np.ix_(validation, classes)] = model.predict_proba(features[validation])
    return probabilities


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """PWA and the rates it compresses, on labelled images."""
    penalty = mean_penalty(y_true, y_pred)
    not_good = y_true != GOOD
    report = {
        "pwa": penalty_to_pwa(penalty),
        "mean_penalty": penalty,
        "false_good_rate": float((y_pred[not_good] == GOOD).mean()),
    }
    for label in np.unique(y_true):
        report[f"recall_{CLASS_NAMES[label]}"] = float((y_pred[y_true == label] == label).mean())
    report["predicted_drift_rate"] = float((y_pred == DRIFT).mean())
    return report
