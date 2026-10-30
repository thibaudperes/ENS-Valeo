"""Extract frozen DINOv2 embeddings of every train and test image, for each view.

Each embedding joins the class token and the mean of the patch tokens. Requires the
train dependency group.
"""

import argparse
import time

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image

from valeo_qc.data import load_test, load_train
from valeo_qc.embedding_cache import cache_path, save_embeddings
from valeo_qc.preprocessing import VIEWS, to_model_input

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
INPUT_SIZE = 224


def batches(images: pd.DataFrame, view: str, batch_size: int):
    for start in range(0, len(images), batch_size):
        rows = images.iloc[start : start + batch_size]
        pixels = [
            to_model_input(VIEWS[view](Image.open(path), lib), INPUT_SIZE)
            for path, lib in zip(rows["path"], rows["lib"], strict=True)
        ]
        yield torch.from_numpy(np.stack(pixels))


@torch.inference_mode()
def embed(model, images: pd.DataFrame, view: str, device: str, batch_size: int) -> np.ndarray:
    embeddings = []
    for done, batch in enumerate(batches(images, view, batch_size)):
        tokens = model.forward_features(batch.to(device))
        joined = torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1)
        embeddings.append(joined.float().cpu().numpy().astype(np.float16))
        if done % 50 == 0:
            print(f"{view}: {done * batch_size}/{len(images)}", flush=True)
    return np.concatenate(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=INPUT_SIZE)
    model = model.eval().to(device)
    images = pd.concat([load_train(), load_test()], ignore_index=True)
    for view in VIEWS:
        start = time.perf_counter()
        embeddings = embed(model, images, view, device, args.batch_size)
        path = cache_path(view)
        save_embeddings(path, images["filename"].tolist(), embeddings)
        elapsed = time.perf_counter() - start
        size = path.stat().st_size / 1e6
        print(f"{path}: {embeddings.shape}, {size:.1f} MB, {elapsed:.0f} s on {device}", flush=True)


if __name__ == "__main__":
    main()
