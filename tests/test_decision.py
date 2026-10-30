import numpy as np

from valeo_qc.decision import apply_temperature, decide, fit_temperature
from valeo_qc.metrics import DRIFT, GOOD, MISSING
from valeo_qc.pipeline import out_of_fold_inference


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


def test_out_of_fold_inference_never_sees_the_scored_image():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 8))
    labels = np.tile([GOOD, MISSING], 30)
    libs = np.array(["Die01"] * 60)
    folds = np.tile(np.arange(5), 12)
    known = out_of_fold_inference((features, features, labels, libs), folds)
    assert known["drift"].min() > 0.1
    assert known["good"].min() > 0.1
