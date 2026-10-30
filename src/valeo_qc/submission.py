"""Write and validate submission files against the format of the supplied example."""

from pathlib import Path

import pandas as pd

from valeo_qc.data import RAW_DIR, SUBMISSION_EXAMPLE
from valeo_qc.metrics import N_CLASSES

SUBMISSIONS_DIR = Path("submissions")


def validate_submission(submission: pd.DataFrame, example: pd.DataFrame) -> None:
    """Raise ValueError unless the submission only differs from the example by its labels."""
    if list(submission.columns) != list(example.columns):
        raise ValueError(f"columns differ: {list(submission.columns)}")
    if len(submission) != len(example):
        raise ValueError(f"expected {len(example)} rows, got {len(submission)}")
    if not submission.index.equals(example.index):
        raise ValueError("index differs from the example")
    for column in ("filename", "window", "lib"):
        if not submission[column].equals(example[column]):
            raise ValueError(f"{column} differs from the example")
    labels = submission["Label"]
    if not pd.api.types.is_integer_dtype(labels):
        raise ValueError(f"labels must be integers, got {labels.dtype}")
    if not labels.between(0, N_CLASSES - 1).all():
        raise ValueError(f"labels must be in 0..{N_CLASSES - 1}")


def validate_submission_file(path: Path, raw_dir: Path = RAW_DIR) -> None:
    example = pd.read_csv(raw_dir / SUBMISSION_EXAMPLE, index_col=0)
    validate_submission(pd.read_csv(path, index_col=0), example)


def write_constant_submission(label: int, path: Path, raw_dir: Path = RAW_DIR) -> None:
    """Write `label` for every test image, then validate the file as written."""
    example = pd.read_csv(raw_dir / SUBMISSION_EXAMPLE, index_col=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    example.assign(Label=label).to_csv(path)
    validate_submission_file(path, raw_dir)
