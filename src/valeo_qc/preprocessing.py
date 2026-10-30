"""Rotate each image to a common orientation and crop the bonding area."""

import numpy as np
from PIL import Image

from valeo_qc.alignment import ALIGNMENTS, Alignment

OVERSIZED_SCALE = 0.54  # measured on the oversized images, see scripts/alignment_report.py
BACKGROUND = 239  # modal bright level of the images
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def normalize_scale(image: Image.Image, base_size: tuple[int, int]) -> Image.Image:
    """Bring an oversized image to the base scale, centred on a canvas of the base size.

    Oversized images are magnified by a constant factor unrelated to their size, so
    resizing them to the base size would leave them about 10% too large.
    """
    width, height = round(image.width * OVERSIZED_SCALE), round(image.height * OVERSIZED_SCALE)
    scaled = image.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new(image.mode, base_size, BACKGROUND)
    canvas.paste(scaled, ((base_size[0] - width) // 2, (base_size[1] - height) // 2))
    return canvas


def align(image: Image.Image, alignment: Alignment) -> Image.Image:
    """Rotate then crop, after bringing oversized images back to the base scale."""
    if image.size != alignment.base_size:
        image = normalize_scale(image, alignment.base_size)
    return image.rotate(alignment.angle, expand=True).crop(alignment.box)


def crop_view(image: Image.Image, lib: str) -> Image.Image:
    """The bonding area, in a common orientation."""
    return align(image, ALIGNMENTS[lib])


def full_view(image: Image.Image, lib: str) -> Image.Image:
    """The whole field of view: anomalies can lie outside the crop."""
    base_size = ALIGNMENTS[lib].base_size
    return image if image.size == base_size else normalize_scale(image, base_size)


VIEWS = {"crop": crop_view, "full": full_view}


def to_model_input(image: Image.Image, size: int) -> np.ndarray:
    """Square, three-channel, ImageNet-normalised array for a pretrained backbone."""
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255
    return (np.stack([pixels] * 3) - IMAGENET_MEAN) / IMAGENET_STD
