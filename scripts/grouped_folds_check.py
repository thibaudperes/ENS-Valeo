"""Check that near-duplicates across folds do not inflate the out-of-fold score.

Images with a cosine similarity above a threshold, within a lib, stay in the same fold.
Writes results/grouped_folds_check.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from valeo_qc.data import load_train
from valeo_qc.embedding_cache import cache_path, load_embeddings
from valeo_qc.metrics import minimum_cost_labels
from valeo_qc.validation import (
    N_FOLDS,
    SEED,
    make_folds,
    out_of_fold_probabilities,
    score_predictions,
)

BEST_C = 0.1
THRESHOLDS = (0.97, 0.98, 0.99)
OUTPUT = Path("results/grouped_folds_check.json")


def near_duplicate_groups(embeddings: np.ndarray, libs: np.ndarray, threshold: float) -> np.ndarray:
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    rows, columns = [], []
    for lib in np.unique(libs):
        members = np.flatnonzero(libs == lib)
        first, second = np.nonzero(np.triu(unit[members] @ unit[members].T, k=1) > threshold)
        rows.append(members[first])
        columns.append(members[second])
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    n = len(embeddings)
    graph = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def evaluate(features: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> dict[str, float]:
    probabilities = out_of_fold_probabilities(features, labels, folds, c=BEST_C)
    return score_predictions(labels, minimum_cost_labels(probabilities))


def main() -> None:
    train = load_train()
    labels = train["label"].to_numpy()
    crop = load_embeddings(cache_path("crop"), train["filename"])
    full = load_embeddings(cache_path("full"), train["filename"])
    features = np.concatenate([crop, full], axis=1)

    results = {"random_folds": evaluate(features, labels, make_folds(train))}
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for threshold in THRESHOLDS:
        groups = near_duplicate_groups(crop, train["lib"].to_numpy(), threshold)
        folds = np.empty(len(labels), dtype=int)
        for fold, (_, validation) in enumerate(splitter.split(features, labels, groups)):
            folds[validation] = fold
        sizes = np.bincount(groups)[groups]
        results[f"grouped_{threshold}"] = {
            **evaluate(features, labels, folds),
            "grouped_images": float((sizes > 1).mean()),
            "rare_defects_grouped": float((sizes[np.isin(labels, [1, 2, 3, 5])] > 1).mean()),
        }
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    for name, report in results.items():
        print(name, round(report["mean_penalty"], 2), round(report["false_good_rate"], 4))


if __name__ == "__main__":
    main()
