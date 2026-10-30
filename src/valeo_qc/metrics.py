"""Penalty-weighted accuracy (PWA) and the cost tables derived from it.

Rows are true classes and columns predicted classes, in label order:
GOOD, the five defects, Drift. The two matrices differ on GOOD -> Drift and defect -> Drift.
"""

import numpy as np

N_CLASSES = 7
GOOD = 0
MISSING = 4
DRIFT = 6


def _penalty_matrix(
    good_to_drift: int, defect_to_drift: int, drift_to_defect: int = 1000
) -> np.ndarray:
    matrix = np.ones((N_CLASSES, N_CLASSES), dtype=np.int64)
    matrix[GOOD, :] = 100
    matrix[:, GOOD] = 10000
    matrix[GOOD, DRIFT] = good_to_drift
    matrix[1:DRIFT, DRIFT] = defect_to_drift
    matrix[DRIFT, 1:DRIFT] = drift_to_defect
    np.fill_diagonal(matrix, 0)
    return matrix


PENALTY_MATRICES = {
    "page": _penalty_matrix(good_to_drift=100, defect_to_drift=1),
    "notebook": _penalty_matrix(good_to_drift=10000, defect_to_drift=1000),
}
DEFAULT_MATRIX = "page"


def get_penalty_matrix(name: str = DEFAULT_MATRIX) -> np.ndarray:
    return PENALTY_MATRICES[name]


def mean_penalty(y_true, y_pred, matrix: str = DEFAULT_MATRIX) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return float(get_penalty_matrix(matrix)[y_true, y_pred].mean())


def penalty_to_pwa(penalty: float, matrix: str = DEFAULT_MATRIX) -> float:
    return 1 - penalty / get_penalty_matrix(matrix).max()


def penalty_weighted_accuracy(y_true, y_pred, matrix: str = DEFAULT_MATRIX) -> float:
    return penalty_to_pwa(mean_penalty(y_true, y_pred, matrix), matrix)


def constant_pwa(priors, drift_share: float, label: int, matrix: str = DEFAULT_MATRIX) -> float:
    """PWA of predicting `label` everywhere, with `priors` over the six non-drift classes."""
    priors = np.asarray(priors, dtype=float)
    shares = np.append(priors / priors.sum() * (1 - drift_share), drift_share)
    return penalty_to_pwa(float(shares @ get_penalty_matrix(matrix)[:, label]), matrix)


def minimum_cost_labels(probabilities, matrix: str = DEFAULT_MATRIX) -> np.ndarray:
    """Label of minimal expected penalty for each row of class probabilities."""
    return np.argmin(np.asarray(probabilities) @ get_penalty_matrix(matrix), axis=1)
