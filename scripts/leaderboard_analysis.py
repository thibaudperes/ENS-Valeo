"""Identify the official matrix and the composition of the evaluated images.

Writes results/leaderboard_constants.json. These figures come from the leaderboard: they
describe the metric and the test set, and are not model results.
"""

import json
from pathlib import Path

from valeo_qc.leaderboard import solve_composition
from valeo_qc.metrics import DEFAULT_MATRIX, PENALTY_MATRICES, penalty_to_pwa

SCORE_ALL_DRIFT = 0.808901515151515
SCORE_ALL_MISSING = 0.985961174242424
BENCHMARK_NOTEBOOK = 0.9755256872037915
OUTPUT = Path("results/leaderboard_constants.json")


def main() -> None:
    solutions = {
        name: solve_composition(SCORE_ALL_DRIFT, SCORE_ALL_MISSING, name)
        for name in PENALTY_MATRICES
    }
    composition = solutions[DEFAULT_MATRIX][0]
    n, good, drift = composition.n_images, composition.good, composition.drift
    missing_penalty = {
        "undetected_drift": 1000 * drift / n,
        "good_as_defect": 100 * good / n,
        "other_defects": composition.other_defects / n,
    }
    results = {
        "source": "public leaderboard, constant submissions: not a model result",
        "scores": {"all_drift": SCORE_ALL_DRIFT, "all_missing": SCORE_ALL_MISSING},
        "solutions_by_matrix": {
            name: [found._asdict() for found in items] for name, items in solutions.items()
        },
        "evaluated_images": n,
        "shares": {
            "good": good / n,
            "defects": composition.defects / n,
            "missing": composition.missing / n,
            "other_defects": composition.other_defects / n,
            "drift": drift / n,
        },
        "all_missing_penalty_per_image": missing_penalty,
        "all_missing_pwa_check": penalty_to_pwa(sum(missing_penalty.values())),
        "benchmark_pwa_notebook": BENCHMARK_NOTEBOOK,
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
