import numpy as np
import cv2
from skimage.feature import hog


def preprocess_mask(mask: np.ndarray, size=(128, 128)) -> np.ndarray:
    """
    mask: 0 background, 255 object (binary silhouette)
    We keep it binary and only resize.
    """
    if mask.ndim != 2:
        raise ValueError("Expected a 2D mask")

    # Resize without smoothing edges
    m = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    # Ensure strict binary 0/255
    m = (m > 0).astype(np.uint8) * 255
    return m


def hog_from_views(views: list[np.ndarray]) -> np.ndarray:
    """
    Compute HOG per view then average-pool across views.
    Returns L2-normalized vector.
    """
    feats = []
    for v in views:
        m = preprocess_mask(v)

        f = hog(
            m,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        feats.append(f)

    feats = np.vstack(feats)
    vec = feats.mean(axis=0)

    # L2 normalize
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # assumes normalized vectors
    return float(1.0 - np.dot(a, b))
