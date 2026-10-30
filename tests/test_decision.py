import numpy as np

from valeo_qc.decision import apply_temperature, decide, fit_temperature
from valeo_qc.metrics import DRIFT, GOOD, MISSING


def _probabilities(p_good):
    probabilities = np.zeros((1, 7))
    probabilities[0, GOOD], probabilities[0, MISSING] = p_good, 1 - p_good
    return probabilities


def test_good_requires_high_probability_and_proximity_to_known_good():
    yes, no = np.array([True]), np.array([False])
    assert decide(_probabilities(0.995), no, yes)[0] == GOOD
    assert decide(_probabilities(0.98), no, yes)[0] == MISSING
    assert decide(_probabilities(0.995), no, no)[0] == MISSING


def test_flag_overrides_the_classifier():
    assert decide(_probabilities(0.995), np.array([True]), np.array([True]))[0] == DRIFT


def test_temperature_softens_overconfident_probabilities():
    # 70% accuracy at a stated 99% confidence
    labels = np.array([GOOD] * 7 + [MISSING] * 3)
    probabilities = np.vstack([_probabilities(0.99)] * 10)
    temperature = fit_temperature(probabilities, labels)
    assert temperature > 1
    assert apply_temperature(probabilities, temperature)[0, GOOD] < 0.8
