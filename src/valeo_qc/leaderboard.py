"""Read the composition of the evaluated images from the scores of constant submissions.

Analysis only: decision code must not import this module, so that model results do not
depend on leaderboard information.
"""

from typing import NamedTuple

from valeo_qc.metrics import DRIFT, GOOD, MISSING, get_penalty_matrix

MAX_TEST_IMAGES = 1055
TOLERANCE = 1e-6


class Composition(NamedTuple):
    n_images: int
    good: int
    missing: int
    other_defects: int
    drift: int

    @property
    def defects(self) -> int:
        return self.missing + self.other_defects


def solve_composition(
    score_all_drift: float, score_all_missing: float, matrix: str
) -> list[Composition]:
    """Every integer composition, up to the test size, that gives exactly both scores."""
    penalties = get_penalty_matrix(matrix)
    max_penalty = int(penalties.max())
    good_to_drift, defect_to_drift = int(penalties[GOOD, DRIFT]), int(penalties[1, DRIFT])
    good_to_defect = int(penalties[GOOD, MISSING])
    other_defect_cost = int(penalties[1, MISSING])
    drift_to_defect = int(penalties[DRIFT, MISSING])

    solutions = []
    for n in range(1, MAX_TEST_IMAGES + 1):
        total_drift = (1 - score_all_drift) * max_penalty * n
        total_missing = (1 - score_all_missing) * max_penalty * n
        if max(abs(x - round(x)) for x in (total_drift, total_missing)) > TOLERANCE:
            continue
        total_drift, total_missing = round(total_drift), round(total_missing)
        for good in range(n + 1):
            defects, remainder = divmod(total_drift - good_to_drift * good, defect_to_drift)
            drift = n - good - defects
            if remainder or defects < 0 or drift < 0:
                continue
            other, remainder = divmod(
                total_missing - good_to_defect * good - drift_to_defect * drift, other_defect_cost
            )
            if not remainder and 0 <= other <= defects:
                solutions.append(Composition(n, good, defects - other, other, drift))
    return solutions
