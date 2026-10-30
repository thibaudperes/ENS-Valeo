"""Out-of-domain score from local patches: distance to the closest known patches of a component."""

import numpy as np

BANK_SIZE = 10000
TOP_PATCHES = 5
CHUNK_IMAGES = 50


def build_bank(patches: np.ndarray, size: int = BANK_SIZE, seed: int = 42) -> np.ndarray:
    """Random subsample of unit-norm patches, from images of shape (n, patches, dims)."""
    flat = patches.reshape(-1, patches.shape[-1])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(flat), size=min(size, len(flat)), replace=False)
    return _unit(flat[chosen].astype(np.float32))


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-8)


def patch_scores(bank: np.ndarray, patches: np.ndarray, top: int = TOP_PATCHES) -> np.ndarray:
    """Mean cosine distance of the `top` patches of each image farthest from the bank."""
    scores = np.empty(len(patches))
    n_images, n_patches, dims = patches.shape
    for start in range(0, n_images, CHUNK_IMAGES):
        chunk = patches[start : start + CHUNK_IMAGES].astype(np.float32).reshape(-1, dims)
        distances = 1 - (_unit(chunk) @ bank.T).max(axis=1)
        per_image = distances.reshape(-1, n_patches)
        scores[start : start + CHUNK_IMAGES] = np.sort(per_image, axis=1)[:, -top:].mean(axis=1)
    return scores


def patch_scores_by_lib(
    reference: np.ndarray,
    reference_libs: np.ndarray,
    query: np.ndarray,
    query_libs: np.ndarray,
) -> np.ndarray:
    """Score each query against a bank built from the images of its own component only."""
    scores = np.empty(len(query))
    for lib in np.unique(query_libs):
        selected = query_libs == lib
        scores[selected] = patch_scores(
            build_bank(reference[reference_libs == lib]), query[selected]
        )
    return scores


def out_of_fold_patch_scores(
    patches: np.ndarray, libs: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    """Scores of known images, each against a bank built from the other folds."""
    scores = np.empty(len(patches))
    for fold in np.unique(folds):
        held_out = folds == fold
        scores[held_out] = patch_scores_by_lib(
            patches[~held_out], libs[~held_out], patches[held_out], libs[held_out]
        )
    return scores
