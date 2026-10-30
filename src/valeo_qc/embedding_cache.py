"""Store image embeddings as float16 on disk, indexed by filename."""

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/processed/embeddings")


def cache_path(view: str, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{view}.npz"


def save_embeddings(path: Path, filenames: list[str], embeddings: np.ndarray) -> None:
    if len(filenames) != len(embeddings):
        raise ValueError(f"{len(filenames)} filenames for {len(embeddings)} embeddings")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, filenames=np.array(filenames), embeddings=embeddings.astype(np.float16))


def load_embeddings(path: Path, filenames: pd.Series) -> np.ndarray:
    """Embeddings as float32, in the order of `filenames`."""
    with np.load(path) as cache:
        positions = pd.Series(np.arange(len(cache["filenames"])), index=cache["filenames"])
        embeddings = cache["embeddings"]
    return embeddings[positions.loc[filenames].to_numpy()].astype(np.float32)
