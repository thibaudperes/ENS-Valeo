import numpy as np
import pandas as pd
import pytest

from valeo_qc.embedding_cache import load_embeddings, save_embeddings


def test_embeddings_are_stored_as_float16_and_loaded_in_the_requested_order(tmp_path):
    path = tmp_path / "cache" / "crop.npz"
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    save_embeddings(path, ["a.png", "b.png", "c.png"], embeddings)

    with np.load(path) as cache:
        assert cache["embeddings"].dtype == np.float16

    loaded = load_embeddings(path, pd.Series(["c.png", "a.png"]))
    assert loaded.dtype == np.float32
    assert loaded.tolist() == [[5.0, 6.0], [1.0, 2.0]]


def test_load_fails_on_an_unknown_filename(tmp_path):
    path = tmp_path / "crop.npz"
    save_embeddings(path, ["a.png"], np.zeros((1, 2)))
    with pytest.raises(KeyError):
        load_embeddings(path, pd.Series(["b.png"]))


def test_save_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError, match="1 filenames for 2 embeddings"):
        save_embeddings(tmp_path / "crop.npz", ["a.png"], np.zeros((2, 2)))
