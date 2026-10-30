import pandas as pd
import pytest

from valeo_qc.data import SUBMISSION_EXAMPLE
from valeo_qc.metrics import DRIFT, MISSING
from valeo_qc.submission import (
    validate_submission,
    validate_submission_file,
    write_constant_submission,
)


@pytest.fixture
def example():
    return pd.DataFrame(
        {
            "filename": ["a.png", "b.png"],
            "window": [2003, 2005],
            "lib": ["Die01", "Die04"],
            "Label": [5, 1],
        }
    )


@pytest.fixture
def raw_dir(tmp_path, example):
    example.to_csv(tmp_path / SUBMISSION_EXAMPLE)
    return tmp_path


def test_validate_accepts_new_labels(example):
    validate_submission(example.assign(Label=[0, 6]), example)


@pytest.mark.parametrize(
    "alter, message",
    [
        (lambda df: df.drop(columns="lib"), "columns differ"),
        (lambda df: df.iloc[:1], "expected 2 rows"),
        (lambda df: df.set_axis([1, 2]), "index differs"),
        (lambda df: df.assign(filename=["a.png", "c.png"]), "filename differs"),
        (lambda df: df.iloc[::-1].reset_index(drop=True), "filename differs"),
        (lambda df: df.assign(window=[2005, 2005]), "window differs"),
        (lambda df: df.assign(lib=["Die02", "Die04"]), "lib differs"),
        (lambda df: df.assign(Label=[0.5, 1.0]), "must be integers"),
        (lambda df: df.assign(Label=["GOOD", "GOOD"]), "must be integers"),
        (lambda df: df.assign(Label=[0, 7]), "must be in 0..6"),
        (lambda df: df.assign(Label=[0, -1]), "must be in 0..6"),
    ],
)
def test_validate_rejects(example, alter, message):
    with pytest.raises(ValueError, match=message):
        validate_submission(alter(example), example)


@pytest.mark.parametrize("label", [MISSING, DRIFT])
def test_constant_submission_only_changes_labels(raw_dir, example, label):
    path = raw_dir / "out" / "constant.csv"
    write_constant_submission(label, path, raw_dir)
    written = pd.read_csv(path, index_col=0)
    assert written["Label"].tolist() == [label, label]
    pd.testing.assert_frame_equal(written.drop(columns="Label"), example.drop(columns="Label"))


def test_constant_submission_keeps_the_example_layout(raw_dir):
    path = raw_dir / "constant.csv"
    write_constant_submission(MISSING, path, raw_dir)
    example_lines = (raw_dir / SUBMISSION_EXAMPLE).read_text().splitlines()
    written_lines = path.read_text().splitlines()
    assert written_lines[0] == example_lines[0]
    assert [line.rsplit(",", 1)[0] for line in written_lines] == [
        line.rsplit(",", 1)[0] for line in example_lines
    ]


def test_validate_file_rejects_a_truncated_file(raw_dir, example):
    path = raw_dir / "bad.csv"
    example.iloc[:1].to_csv(path)
    with pytest.raises(ValueError, match="expected 2 rows"):
        validate_submission_file(path, raw_dir)
