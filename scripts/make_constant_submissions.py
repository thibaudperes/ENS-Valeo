"""Write the constant submissions used to read the official metric from the leaderboard.

They are a metric analysis, not a model.
"""

from valeo_qc.metrics import DRIFT, MISSING
from valeo_qc.submission import SUBMISSIONS_DIR, write_constant_submission


def main() -> None:
    for name, label in (("all_missing", MISSING), ("all_drift", DRIFT)):
        path = SUBMISSIONS_DIR / f"{name}.csv"
        write_constant_submission(label, path)
        print(f"{path}: label {label}, valid")


if __name__ == "__main__":
    main()
