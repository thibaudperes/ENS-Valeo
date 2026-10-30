"""Rotation and crop of each component, to reach a common orientation."""

from typing import NamedTuple


class Alignment(NamedTuple):
    angle: float  # degrees, counter-clockwise
    box: tuple[int, int, int, int]  # left, upper, right, lower, in the rotated image
    base_size: tuple[int, int]  # width, height of a standard image


# Angles and boxes: challenge notebook Notebook_ENS.ipynb, cell index 3 (`rot_crop_data`).
# Base sizes: measured on the images.
ALIGNMENTS = {
    "Die01": Alignment(55, (340, 120, 500, 680), (540, 630)),
    "Die02": Alignment(-44, (480, 210, 640, 930), (720, 840)),
    "Die03": Alignment(134, (460, 200, 620, 920), (720, 840)),
    "Die04": Alignment(35, (310, 130, 470, 690), (540, 630)),
}
