from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import valeo_qc
from valeo_qc.alignment import ALIGNMENTS, Alignment
from valeo_qc.data import LIBS
from valeo_qc.preprocessing import BACKGROUND, align, normalize_scale


def test_alignments_cover_every_component():
    assert set(ALIGNMENTS) == set(LIBS)


@pytest.mark.parametrize("lib", LIBS)
def test_preprocessing_runs_without_challenge_files(lib, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    alignment = ALIGNMENTS[lib]
    width, height = alignment.base_size
    left, upper, right, lower = alignment.box
    image = Image.fromarray(np.full((height, width), 100, dtype=np.uint8))
    assert align(image, alignment).size == (right - left, lower - upper)


def test_package_does_not_read_supplementary_files():
    sources = Path(valeo_qc.__file__).parent.glob("*.py")
    assert not [path.name for path in sources if "Supp_files" in path.read_text()]


def test_align_returns_the_box_size():
    image = Image.fromarray(np.zeros((630, 540), dtype=np.uint8))
    cropped = align(image, Alignment(55, (340, 120, 500, 680), (540, 630)))
    assert cropped.size == (160, 560)
    assert cropped.mode == "L"


def test_align_without_rotation_keeps_pixel_positions():
    pixels = np.zeros((20, 10), dtype=np.uint8)
    pixels[5, 3] = 200
    cropped = align(Image.fromarray(pixels), Alignment(0, (3, 5, 4, 6), (10, 20)))
    assert np.asarray(cropped).tolist() == [[200]]


def test_normalize_scale_centres_a_smaller_image_on_the_background():
    image = Image.fromarray(np.full((1365, 1170), 7, dtype=np.uint8))
    canvas = np.asarray(normalize_scale(image, (720, 840)))
    assert canvas.shape == (840, 720)
    assert canvas[420, 360] == 7
    assert canvas[0, 0] == BACKGROUND


def test_normalize_scale_crops_an_image_larger_than_the_canvas():
    image = Image.fromarray(np.full((1600, 1400), 7, dtype=np.uint8))
    canvas = np.asarray(normalize_scale(image, (540, 630)))
    assert canvas.shape == (630, 540)
    assert (canvas == 7).all()


def test_align_applies_the_scale_only_to_oversized_images():
    pixels = np.full((840, 720), 7, dtype=np.uint8)
    alignment = Alignment(0, (0, 0, 10, 10), (720, 840))
    assert (np.asarray(align(Image.fromarray(pixels), alignment)) == 7).all()
    oversized = Image.fromarray(np.full((1365, 1170), 7, dtype=np.uint8))
    assert (np.asarray(align(oversized, alignment)) == BACKGROUND).all()
