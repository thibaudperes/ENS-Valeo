"""Out-of-fold logistic regression on frozen embeddings, against the constant baseline.

The train set has no drift image, so these scores measure the six known classes only and
cannot be compared with leaderboard scores. Requires the train dependency group.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import log_loss

from valeo_qc.data import load_train
from valeo_qc.embedding_cache import cache_path, load_embeddings
from valeo_qc.metrics import MISSING, minimum_cost_labels
from valeo_qc.validation import N_FOLDS, SEED, make_folds, out_of_fold_probabilities
from valeo_qc.validation import score_predictions as score

C_VALUES = (0.01, 0.1, 1.0)
FEATURE_SETS = {"crop": ("crop",), "full": ("full",), "crop+full": ("crop", "full")}
OUTPUT = Path("results/baseline_logreg.json")
PROBABILITIES = Path("data/processed/oof_logreg.npz")


def main() -> None:
    train = load_train()
    labels = train["label"].to_numpy()
    folds = make_folds(train)
    views = {
        view: load_embeddings(cache_path(view), train["filename"]) for view in ("crop", "full")
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("baseline-logreg")

    results = {"constant_missing": score(labels, np.full(len(labels), MISSING))}
    best_loss, best_probabilities, best_name = np.inf, None, None
    for name, feature_views in FEATURE_SETS.items():
        features = np.concatenate([views[view] for view in feature_views], axis=1)
        for c in C_VALUES:
            probabilities = out_of_fold_probabilities(features, labels, folds, c=c)
            loss = log_loss(labels, probabilities[:, :6], labels=range(6))
            run = f"{name}_C{c}"
            results[run] = {
                "log_loss": loss,
                "argmax": score(labels, probabilities.argmax(axis=1)),
                "minimum_cost": score(labels, minimum_cost_labels(probabilities)),
            }
            with mlflow.start_run(run_name=run):
                mlflow.log_params({"features": name, "C": c, "n_folds": N_FOLDS, "seed": SEED})
                mlflow.log_metric("log_loss", loss)
                for rule in ("argmax", "minimum_cost"):
                    for key, value in results[run][rule].items():
                        metric = key.replace(" ", "_").replace("-", "_")
                        mlflow.log_metric(f"{rule}_{metric}", value)
            print(run, json.dumps(results[run], indent=None), flush=True)
            if loss < best_loss:
                best_loss, best_probabilities, best_name = loss, probabilities, run

    results["best_by_log_loss"] = best_name
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    PROBABILITIES.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        PROBABILITIES,
        filenames=train["filename"].to_numpy(dtype=str),
        probabilities=best_probabilities,
    )


if __name__ == "__main__":
    main()
