import pandas as pd
import pytest

from valeo_qc.data import LABEL_IDS, SUBMISSION_EXAMPLE, TRAIN_LABELS, load_test, load_train


def _write_raw(raw_dir, labels=("GOOD", "Missing"), libs=("Die01", "Die04")):
    frame = pd.DataFrame(
        {
            "filename": ["a.png", "b.png"],
            "window": [2003, 2005],
            "lib": list(libs),
            "Label": list(labels),
        }
    )
    frame.to_csv(raw_dir / TRAIN_LABELS)
    frame.assign(Label=[5, 6]).to_csv(raw_dir / SUBMISSION_EXAMPLE)


def test_load_train_maps_labels_and_paths(tmp_path):
    _write_raw(tmp_path)
    train = load_train(tmp_path)
    assert train["label"].tolist() == [LABEL_IDS["GOOD"], LABEL_IDS["Missing"]]
    assert train["path"][0] == tmp_path / "input_train" / "a.png"
    assert "Label" not in train


def test_load_test_drops_the_random_labels(tmp_path):
    _write_raw(tmp_path)
    test = load_test(tmp_path)
    assert list(test.columns) == ["filename", "window", "lib", "path"]
    assert test["path"][1] == tmp_path / "input_test" / "b.png"


def test_load_train_rejects_unknown_label(tmp_path):
    _write_raw(tmp_path, labels=("GOOD", "Unknown"))
    with pytest.raises(ValueError, match="unknown labels"):
        load_train(tmp_path)


def test_load_train_rejects_unknown_lib(tmp_path):
    _write_raw(tmp_path, libs=("Die01", "Die09"))
    with pytest.raises(ValueError, match="unknown lib"):
        load_train(tmp_path)
