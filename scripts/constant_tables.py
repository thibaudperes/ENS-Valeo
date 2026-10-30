"""Expected PWA of constant predictions, by matrix and drift share, with train priors."""

import numpy as np
import pandas as pd

from valeo_qc.data import load_train
from valeo_qc.metrics import DRIFT, MISSING, N_CLASSES, PENALTY_MATRICES, constant_pwa

DRIFT_SHARES = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25)


def main() -> None:
    priors = np.bincount(load_train()["label"], minlength=N_CLASSES)[:DRIFT]
    table = pd.DataFrame(
        {
            (matrix, name): [constant_pwa(priors, rho, label, matrix) for rho in DRIFT_SHARES]
            for matrix in PENALTY_MATRICES
            for name, label in (("all Missing", MISSING), ("all Drift", DRIFT))
        },
        index=pd.Index(DRIFT_SHARES, name="drift share"),
    )
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
