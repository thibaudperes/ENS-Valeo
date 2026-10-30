import ast
from pathlib import Path

import numpy as np
import pytest

import valeo_qc
from valeo_qc.leaderboard import Composition, solve_composition
from valeo_qc.metrics import DRIFT, MISSING, penalty_weighted_accuracy

SCORE_ALL_DRIFT = 0.808901515151515
SCORE_ALL_MISSING = 0.985961174242424


def test_public_scores_have_a_single_solution_under_the_notebook_matrix():
    solutions = solve_composition(SCORE_ALL_DRIFT, SCORE_ALL_MISSING, "notebook")
    assert solutions == [Composition(528, 61, 374, 25, 68)]
    assert solutions[0].defects == 399


def test_public_scores_have_no_solution_under_the_page_matrix():
    assert solve_composition(SCORE_ALL_DRIFT, SCORE_ALL_MISSING, "page") == []


def test_solution_reproduces_both_scores():
    composition = Composition(528, 61, 374, 25, 68)
    labels = np.repeat([0, 4, 1, 6], [61, 374, 25, 68])
    assert penalty_weighted_accuracy(labels, np.full(528, DRIFT), "notebook") == pytest.approx(
        SCORE_ALL_DRIFT
    )
    assert penalty_weighted_accuracy(labels, np.full(528, MISSING), "notebook") == pytest.approx(
        SCORE_ALL_MISSING
    )
    assert len(labels) == composition.n_images


def test_decision_code_does_not_import_the_leaderboard_module():
    importers = []
    for path in Path(valeo_qc.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            if any("leaderboard" in name for name in names):
                importers.append(path.name)
    assert importers == []
