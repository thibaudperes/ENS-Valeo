"""Extract frozen DINOv2 embeddings of every train and test image, for each view.

Each embedding joins the class token and the mean of the patch tokens. Requires the
train dependency group.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageFilter

from valeo_qc.data import load_test, load_train
from valeo_qc.embedding_cache import CACHE_DIR, cache_path, save_embeddings
from valeo_qc.metrics import MISSING
from valeo_qc.preprocessing import VIEWS, to_model_input
from valeo_qc.validation import SEED

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
INPUT_SIZE = 224
BLUR_SAMPLE = 500
BLUR_RADIUS = 3


def open_image(path, blur: bool) -> Image.Image:
    image = Image.open(path)
    return image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)) if blur else image


def batches(images: pd.DataFrame, view: str, batch_size: int, blur: bool):
    for start in range(0, len(images), batch_size):
        rows = images.iloc[start : start + batch_size]
        pixels = [
            to_model_input(VIEWS[view](open_image(path, blur), lib), INPUT_SIZE)
            for path, lib in zip(rows["path"], rows["lib"], strict=True)
        ]
        yield torch.from_numpy(np.stack(pixels))


@torch.inference_mode()
def embed(
    model, images: pd.DataFrame, view: str, device: str, batch_size: int, blur: bool = False
) -> np.ndarray:
    embeddings = []
    for done, batch in enumerate(batches(images, view, batch_size, blur)):
        tokens = model.forward_features(batch.to(device))
        joined = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)
        embeddings.append(joined.float().cpu().numpy().astype(np.float16))
        if done % 50 == 0:
            print(f"{view}: {done * batch_size}/{len(images)}", flush=True)
    return np.concatenate(embeddings)


def extract(cache_dir: Path = CACHE_DIR, batch_size: int = 32, blur: bool = False) -> None:
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_SIZE)
    model = model.eval().to(device)
    if blur:
        train = load_train()
        images = train.groupby(train["label"] == MISSING).sample(
            BLUR_SAMPLE // 2, random_state=SEED
        )
        for view in VIEWS:
            embeddings = embed(model, images, view, device, batch_size, blur=True)
            path = cache_path(f"blur_{view}", cache_dir)
            save_embeddings(path, images["filename"].tolist(), embeddings)
        return

    images = pd.concat([load_train(), load_test()], ignore_index=True)
    for view in VIEWS:
        start = time.perf_counter()
        embeddings = embed(model, images, view, device, batch_size)
        path = cache_path(view, cache_dir)
        save_embeddings(path, images["filename"].tolist(), embeddings)
        elapsed = time.perf_counter() - start
        size = path.stat().st_size / 1e6
        print(f"{path}: {embeddings.shape}, {size:.1f} MB, {elapsed:.0f} s on {device}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--blur", action="store_true", help="embed blurred train images only")
    args = parser.parse_args()
    extract(batch_size=args.batch_size, blur=args.blur)


if __name__ == "__main__":
    main()
