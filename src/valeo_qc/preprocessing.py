"""Rotate each image to a common orientation and crop the bonding area."""

from PIL import Image

from valeo_qc.alignment import Alignment

OVERSIZED_SCALE = 0.54  # measured on the oversized images, see scripts/alignment_report.py
BACKGROUND = 239  # modal bright level of the images


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
