"""Optional FAISS backend."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fast_ballmapper._validation import validate_cosine_points

try:
    import faiss
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    faiss = None


def _require_faiss() -> None:
    if faiss is None:
        raise ImportError(
            "FAISS is required for method='faiss'. Install fast-ballmapper[faiss]."
        )


def create_euclidean_index(x: np.ndarray):
    """Create a float32 FAISS squared-L2 index."""
    _require_faiss()
    points = x.astype(np.float32, copy=True)
    index = faiss.IndexFlatL2(points.shape[1])
    index.add(points)
    return points, index


def create_cosine_index(x: np.ndarray):
    """Create a normalized float32 FAISS inner-product index."""
    _require_faiss()
    points = x.astype(np.float32, copy=True)
    validate_cosine_points(points)
    faiss.normalize_L2(points)
    index = faiss.IndexFlatIP(points.shape[1])
    index.add(points)
    return points, index


def compute_landmarks_euclidean(
    x: np.ndarray,
    eps: float,
) -> tuple[list[int], list[np.ndarray]]:
    """Compute greedy Euclidean landmarks and covers using FAISS."""
    points, index = create_euclidean_index(x)
    radius = np.nextafter(np.float32(eps**2), np.float32(np.inf))
    covered = np.zeros(len(points), dtype=bool)
    landmarks: list[int] = []
    cover: list[np.ndarray] = []

    for point_index in range(len(points)):
        if covered[point_index]:
            continue
        landmarks.append(point_index)
        limits, _, indices = index.range_search(
            points[point_index : point_index + 1], radius
        )
        point_indices = indices[limits[0] : limits[1]]
        cover.append(point_indices)
        covered[point_indices] = True

    return landmarks, cover


def compute_landmarks_cosine(
    x: np.ndarray,
    eps: float,
) -> tuple[list[int], list[np.ndarray]]:
    """Compute greedy cosine landmarks and covers using FAISS."""
    points, index = create_cosine_index(x)
    radius = np.nextafter(np.float32(1.0 - eps), np.float32(-np.inf))
    covered = np.zeros(len(points), dtype=bool)
    landmarks: list[int] = []
    cover: list[np.ndarray] = []

    for point_index in range(len(points)):
        if covered[point_index]:
            continue
        landmarks.append(point_index)
        limits, _, indices = index.range_search(
            points[point_index : point_index + 1], radius
        )
        point_indices = indices[limits[0] : limits[1]]
        cover.append(point_indices)
        covered[point_indices] = True

    return landmarks, cover


def euclidean_distances_to_all(
    points: np.ndarray,
    index,
    point_index: int,
) -> np.ndarray:
    """Return Euclidean distances from one point to all indexed points."""
    squared_distances, indices = index.search(
        points[point_index : point_index + 1], len(points)
    )
    result = np.empty(len(points), dtype=float)
    result[indices[0]] = np.sqrt(np.maximum(squared_distances[0], 0.0))
    return result


def cosine_distances_to_all(
    points: np.ndarray,
    index,
    point_index: int,
) -> np.ndarray:
    """Return cosine distances from one point to all indexed points."""
    similarities, indices = index.search(
        points[point_index : point_index + 1], len(points)
    )
    result = np.empty(len(points), dtype=float)
    result[indices[0]] = np.maximum(1.0 - similarities[0], 0.0)
    return result


def build_cover_euclidean(
    x: np.ndarray,
    landmarks: Sequence[int],
    eps: float,
    *,
    points: np.ndarray | None = None,
    index=None,
) -> list[np.ndarray]:
    """Build Euclidean epsilon-ball covers for fixed landmarks using FAISS."""
    if points is None or index is None:
        points, index = create_euclidean_index(x)

    radius = np.nextafter(np.float32(eps**2), np.float32(np.inf))
    cover: list[np.ndarray] = []
    for landmark_index in landmarks:
        limits, _, indices = index.range_search(
            points[landmark_index : landmark_index + 1], radius
        )
        cover.append(indices[limits[0] : limits[1]])
    return cover


def build_cover_cosine(
    x: np.ndarray,
    landmarks: Sequence[int],
    eps: float,
    *,
    points: np.ndarray | None = None,
    index=None,
) -> list[np.ndarray]:
    """Build cosine-distance epsilon-ball covers using FAISS."""
    if points is None or index is None:
        points, index = create_cosine_index(x)

    radius = np.nextafter(np.float32(1.0 - eps), np.float32(-np.inf))
    cover: list[np.ndarray] = []
    for landmark_index in landmarks:
        limits, _, indices = index.range_search(
            points[landmark_index : landmark_index + 1], radius
        )
        cover.append(indices[limits[0] : limits[1]])
    return cover
