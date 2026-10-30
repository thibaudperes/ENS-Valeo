"""Check oversized images against the median crop of the same defects at the base scale.

Oversized train images are nearly all lift-off or flat loop defects, so the reference and the
standard images are restricted to these classes. Prints aggregates only: no image leaves data/.
"""

import numpy as np
import pandas as pd
from PIL import Image

from valeo_qc.alignment import ALIGNMENTS, Alignment
from valeo_qc.data import LABEL_IDS, load_test, load_train
from valeo_qc.preprocessing import align

REFERENCE_LABELS = [LABEL_IDS[name] for name in ("Boucle plate", "Lift-off blanc", "Lift-off noir")]


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def proportional_crop(image: Image.Image, alignment: Alignment) -> Image.Image:
    resized = image.resize(alignment.base_size, Image.Resampling.BILINEAR)
    return align(resized, alignment)


def fixed_crop(image: Image.Image, alignment: Alignment) -> Image.Image:
    return image.rotate(alignment.angle, expand=True).crop(alignment.box)


def main() -> None:
    train, test = load_train(), load_test()
    rows = []
    for lib, alignment in ALIGNMENTS.items():
        sizes = {p: Image.open(p).size for p in pd.concat([train, test]).query("lib == @lib").path}
        is_oversized = {p: size != alignment.base_size for p, size in sizes.items()}
        defects = train[train["lib"].eq(lib) & train["label"].isin(REFERENCE_LABELS)]
        standard = [p for p in defects["path"] if not is_oversized[p]]
        crops = [np.asarray(align(Image.open(p), alignment), float) for p in standard]
        reference = np.median(np.stack(crops), axis=0)
        rows += [
            {
                "lib": lib,
                "images": "standard",
                "method": "scale + centre",
                "corr": correlation(c, reference),
            }
            for c in crops
        ]
        for path in (p for p, oversized in is_oversized.items() if oversized):
            for method, crop in (
                ("scale + centre", align),
                ("proportional", proportional_crop),
                ("fixed box", fixed_crop),
            ):
                pixels = np.asarray(crop(Image.open(path), alignment), float)
                rows.append(
                    {
                        "lib": lib,
                        "images": "oversized",
                        "method": method,
                        "corr": correlation(pixels, reference),
                    }
                )
    report = pd.DataFrame(rows)
    summary = report.groupby(["lib", "images", "method"])["corr"].agg(["count", "min", "median"])
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
