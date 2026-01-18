import numpy as np
import cv2


def hu_from_views(views: list[np.ndarray]) -> np.ndarray:
    """
    View-based descriptor: Hu moments per view (7 values) concatenated over all views.
    Output dimension = 7 * n_views.
    This is MUCH stronger than averaging into 7 values.
    """
    feats = []

    for v in views:
        # ensure strict binary 0/255
        m = ((v > 0).astype(np.uint8)) * 255

        mu = cv2.moments(m)
        hu = cv2.HuMoments(mu).flatten().astype(np.float32)

        # log transform for numerical stability
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

        feats.append(hu)

    vec = np.concatenate(feats, axis=0).astype(np.float32)

    # L2 normalize
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # assumes normalized vectors
    return float(1.0 - np.dot(a, b))
