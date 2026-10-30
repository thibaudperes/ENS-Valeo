"""Out-of-domain scores: distance of an embedding to the known images of the same component."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def knn_distance(reference: np.ndarray, query: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean cosine distance to the k nearest reference images."""
    neighbours = NearestNeighbors(n_neighbors=k, metric="cosine").fit(reference)
    return neighbours.kneighbors(query)[0].mean(axis=1)


def mahalanobis_distance(
    reference: np.ndarray, query: np.ndarray, n_components: int = 50
) -> np.ndarray:
    """Mahalanobis distance in the main components of the reference images."""
    pca = PCA(n_components=n_components, whiten=True).fit(reference)
    return np.linalg.norm(pca.transform(query), axis=1)


METHODS = {"knn": knn_distance, "mahalanobis": mahalanobis_distance}


def scores_by_lib(
    method: str,
    reference: np.ndarray,
    reference_libs: np.ndarray,
    query: np.ndarray,
    query_libs: np.ndarray,
) -> np.ndarray:
    """Score each query against the reference images of its own component."""
    scores = np.empty(len(query))
    for lib in np.unique(query_libs):
        selected = query_libs == lib
        scores[selected] = METHODS[method](reference[reference_libs == lib], query[selected])
    return scores


def out_of_fold_scores(
    method: str, features: np.ndarray, libs: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    """Scores of known images, each against reference images from the other folds."""
    scores = np.empty(len(features))
    for fold in np.unique(folds):
        held_out = folds == fold
        scores[held_out] = scores_by_lib(
            method, features[~held_out], libs[~held_out], features[held_out], libs[held_out]
        )
    return scores


def thresholds_by_lib(
    scores: np.ndarray, libs: np.ndarray, false_alarm_rate: float
) -> dict[str, float]:
    """Per-component threshold flagging `false_alarm_rate` of the known images."""
    return {
        lib: float(np.quantile(scores[libs == lib], 1 - false_alarm_rate))
        for lib in np.unique(libs)
    }


def flag_drift(scores: np.ndarray, libs: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    return scores > np.array([thresholds[lib] for lib in libs])
