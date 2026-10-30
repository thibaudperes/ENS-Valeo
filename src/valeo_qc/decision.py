"""Calibrate the classifier and turn its probabilities and the drift flags into labels."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from valeo_qc.metrics import DRIFT, GOOD, N_CLASSES, minimum_cost_labels

KNOWN_CLASSES = N_CLASSES - 1  # the classifier never sees drift


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    log_probabilities = np.log(np.clip(probabilities[:, :KNOWN_CLASSES], 1e-12, 1))
    calibrated = np.zeros_like(probabilities)
    calibrated[:, :KNOWN_CLASSES] = np.exp(log_softmax(log_probabilities / temperature, axis=1))
    return calibrated


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Temperature minimising the log loss of out-of-fold probabilities."""

    def log_loss(temperature: float) -> float:
        calibrated = apply_temperature(probabilities, temperature)
        return float(-np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1)).mean())

    return float(minimize_scalar(log_loss, bounds=(0.1, 10), method="bounded").x)


def decide(probabilities: np.ndarray, flagged: np.ndarray, near_good: np.ndarray) -> np.ndarray:
    """Minimum-cost labels, with Drift when flagged and GOOD only near known GOOD images."""
    probabilities = probabilities.copy()
    probabilities[~near_good, GOOD] = 0
    labels = minimum_cost_labels(probabilities)
    labels[flagged] = DRIFT
    return labels
