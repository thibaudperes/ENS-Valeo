"""Load the challenge labels and metadata from data/raw."""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
TRAIN_LABELS = "Y_train_eVW9jym.csv"
SUBMISSION_EXAMPLE = "Y_random_nKwalR1.csv"
TRAIN_IMAGES = "input_train"
TEST_IMAGES = "input_test"

CLASS_NAMES = [
    "GOOD",
    "Boucle plate",
    "Lift-off blanc",
    "Lift-off noir",
    "Missing",
    "Short circuit MOS",
    "Drift",
]
LABEL_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}
LIBS = ("Die01", "Die02", "Die03", "Die04")
WINDOWS = (2003, 2005)


def _check_metadata(df: pd.DataFrame) -> None:
    if df["filename"].duplicated().any():
        raise ValueError("duplicated filenames")
    if unknown := set(df["lib"]) - set(LIBS):
        raise ValueError(f"unknown lib: {sorted(unknown)}")
    if unknown := set(df["window"]) - set(WINDOWS):
        raise ValueError(f"unknown window: {sorted(unknown)}")


def load_train(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / TRAIN_LABELS, index_col=0)
    _check_metadata(df)
    df["label"] = df["Label"].map(LABEL_IDS)
    if df["label"].isna().any():
        raise ValueError(f"unknown labels: {sorted(df.loc[df['label'].isna(), 'Label'].unique())}")
    df["label"] = df["label"].astype(int)
    df["path"] = [raw_dir / TRAIN_IMAGES / name for name in df["filename"]]
    return df.drop(columns="Label").reset_index(drop=True)


def load_test(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Test metadata, read from the submission example whose labels are random."""
    df = pd.read_csv(raw_dir / SUBMISSION_EXAMPLE, index_col=0)
    _check_metadata(df)
    df["path"] = [raw_dir / TEST_IMAGES / name for name in df["filename"]]
    return df.drop(columns="Label").reset_index(drop=True)
