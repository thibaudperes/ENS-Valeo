"""Extract DINOv2 patch tokens of the crop view, reduced by PCA and stored in float16.

Covers train, test and blurred images. Requires the train dependency group.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from extract_embeddings import INPUT_SIZE, MODEL_NAME, batches
from sklearn.decomposition import PCA

from valeo_qc.data import load_test, load_train
from valeo_qc.embedding_cache import cache_path
from valeo_qc.validation import SEED

PATCH_DIR = Path("data/processed/patches")
PCA_DIMENSIONS = 64
PCA_IMAGES = 400
N_PATCHES = 256
BATCH_SIZE = 32


@torch.inference_mode()
def tokens_of(model, images: pd.DataFrame, device: str, blur: bool = False):
    for batch in batches(images, "crop", BATCH_SIZE, blur):
        yield model.forward_features(batch.to(device))[:, 1:].float().cpu().numpy()


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_SIZE)
    model = model.eval().to(device)

    train, test = load_train(), load_test()
    fit_images = train.sample(PCA_IMAGES, random_state=SEED)
    tokens = np.concatenate(list(tokens_of(model, fit_images, device)))
    pca = PCA(PCA_DIMENSIONS, random_state=SEED).fit(tokens.reshape(-1, tokens.shape[-1]))
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA keeps {explained:.1%} of the variance", flush=True)

    with np.load(cache_path("blur_crop")) as cache:
        blurred = train.set_index("filename").loc[cache["filenames"]].reset_index()
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    for name, images, blur in (
        ("crop", pd.concat([train, test], ignore_index=True), False),
        ("blur_crop", blurred, True),
    ):
        start = time.perf_counter()
        path = PATCH_DIR / f"{name}.npy"
        stored = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=(len(images), N_PATCHES, PCA_DIMENSIONS),
        )
        done = 0
        for tokens in tokens_of(model, images, device, blur):
            reduced = pca.transform(tokens.reshape(-1, tokens.shape[-1]))
            stored[done : done + len(tokens)] = reduced.reshape(len(tokens), N_PATCHES, -1)
            done += len(tokens)
        stored.flush()
        pd.Series(images["filename"].to_numpy()).to_csv(PATCH_DIR / f"{name}.csv", index=False)
        size = path.stat().st_size / 1e6
        elapsed = time.perf_counter() - start
        print(f"{path}: {stored.shape}, {size:.0f} MB, {elapsed:.0f} s on {device}", flush=True)


if __name__ == "__main__":
    main()
