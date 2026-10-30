import numpy as np
import pytest

from valeo_qc.metrics import (
    DRIFT,
    GOOD,
    MISSING,
    PENALTY_MATRICES,
    constant_pwa,
    get_penalty_matrix,
    mean_penalty,
    minimum_cost_labels,
    penalty_to_pwa,
    penalty_weighted_accuracy,
)

TRAIN_PRIORS = [1235, 71, 270, 104, 6472, 126]


@pytest.mark.parametrize("name", PENALTY_MATRICES)
def test_matrix_shape_diagonal_and_scale(name):
    matrix = get_penalty_matrix(name)
    assert matrix.shape == (7, 7)
    assert np.diag(matrix).sum() == 0
    assert matrix.max() == 10000


def test_page_matrix_cells():
    matrix = get_penalty_matrix("page")
    assert matrix[GOOD].tolist() == [0, 100, 100, 100, 100, 100, 100]
    assert matrix[1].tolist() == [10000, 0, 1, 1, 1, 1, 1]
    assert matrix[DRIFT].tolist() == [10000, 1000, 1000, 1000, 1000, 1000, 0]


def test_notebook_matrix_cells():
    matrix = get_penalty_matrix("notebook")
    assert matrix[GOOD].tolist() == [0, 100, 100, 100, 100, 100, 10000]
    assert matrix[5].tolist() == [10000, 1, 1, 1, 1, 0, 1000]
    assert matrix[DRIFT].tolist() == [10000, 1000, 1000, 1000, 1000, 1000, 0]


def test_matrices_differ_only_on_the_drift_column_above_the_last_row():
    differing = np.argwhere(get_penalty_matrix("page") != get_penalty_matrix("notebook"))
    assert differing.tolist() == [[row, DRIFT] for row in range(DRIFT)]


def test_perfect_prediction_scores_one():
    labels = np.arange(7)
    assert penalty_weighted_accuracy(labels, labels) == 1.0


def test_pwa_by_hand():
    # defect -> GOOD costs 10000, GOOD -> defect 100, drift -> defect 1000, one correct
    y_true, y_pred = [1, 0, 6, 4], [0, 2, 3, 4]
    assert mean_penalty(y_true, y_pred) == pytest.approx(11100 / 4)
    assert penalty_weighted_accuracy(y_true, y_pred) == pytest.approx(1 - 11100 / 40000)


def test_mean_penalty_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="shape mismatch"):
        mean_penalty([0, 1], [0])


@pytest.mark.parametrize("name", PENALTY_MATRICES)
def test_constant_missing_is_the_same_under_both_matrices(name):
    assert constant_pwa(TRAIN_PRIORS, 0.05, MISSING, name) == pytest.approx(0.99358, abs=1e-5)


def test_constant_drift_separates_the_matrices():
    assert constant_pwa(TRAIN_PRIORS, 0.05, DRIFT, "page") > 0.998
    assert constant_pwa(TRAIN_PRIORS, 0.05, DRIFT, "notebook") < 0.8


def test_constant_pwa_matches_labels():
    y_true = np.array([0, 0, 4, 4, 4, 4, 4, 4, 6, 6])
    priors = [2, 0, 0, 0, 6, 0]
    expected = penalty_weighted_accuracy(y_true, np.full(10, MISSING))
    assert constant_pwa(priors, 0.2, MISSING) == pytest.approx(expected)


@pytest.mark.parametrize(
    "p_good, p_drift, expected",
    [(0.995, 1e-6, GOOD), (0.985, 1e-6, MISSING), (0.985, 1e-3, DRIFT)],
)
def test_page_matrix_declares_good_only_above_99_percent(p_good, p_drift, expected):
    # below the threshold, a defect beats Drift only if p_defect > 1000 * p_drift
    probabilities = np.zeros((1, 7))
    probabilities[0, GOOD] = p_good
    probabilities[0, MISSING] = 1 - p_good - p_drift
    probabilities[0, DRIFT] = p_drift
    assert minimum_cost_labels(probabilities, "page")[0] == expected


def test_notebook_matrix_is_the_default():
    assert get_penalty_matrix() is PENALTY_MATRICES["notebook"]


def test_notebook_matrix_prefers_a_defect_to_drift_when_good_is_uncertain():
    probabilities = np.array([[0.9, 0, 0, 0, 0.1, 0, 0]])
    assert minimum_cost_labels(probabilities)[0] == MISSING


@pytest.mark.parametrize(
    "p_good, p_drift, kind",
    [
        (0.992, 0.008, "good"),
        (0.992, 0.0, "good"),
        (0.988, 0.0, "defect"),
        (0.988, 0.012, "defect"),
        (0.0, 0.49, "defect"),
        (0.0, 0.51, "drift"),
        (0.05, 0.73, "drift"),
        (0.05, 0.71, "defect"),
        (0.10, 0.90, "defect"),
    ],
)
def test_notebook_matrix_decision_rule(p_good, p_drift, kind):
    probabilities = np.zeros((1, 7))
    probabilities[0, [GOOD, DRIFT]] = p_good, p_drift
    probabilities[0, MISSING] = 1 - p_good - p_drift
    label = minimum_cost_labels(probabilities)[0]
    assert {"good": label == GOOD, "drift": label == DRIFT, "defect": 1 <= label <= 5}[kind]


def test_penalty_to_pwa_uses_the_matrix_maximum():
    assert penalty_to_pwa(245.0, "page") == pytest.approx(0.9755)
