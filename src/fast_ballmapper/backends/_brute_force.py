"""Float64 brute-force reference backend.

This backend is intentionally simple and exhaustive. It provides the numerical
reference oracle used by validation tests and manuscript experiments.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from fast_ballmapper._radius import closed_ball_radius
from fast_ballmapper._validation import validate_cosine_points
from fast_ballmapper.backends._base import BackendMetadata
from fast_ballmapper.backends._utils import normalize_point_indices

BruteForceMetric = Literal["euclidean", "cosine"]


def prepare_reference_points(x: np.ndarray, metric: BruteForceMetric) -> np.ndarray:
    """Return the float64 representation used by the reference oracle."""
    points = np.asarray(x, dtype=np.float64)
    if metric == "euclidean":
        return points
    validate_cosine_points(points)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def distances_from_reference_point(
    points: np.ndarray,
    point_index: int,
    metric: BruteForceMetric,
) -> np.ndarray:
    """Return float64 distances from one indexed point to every row."""
    query = points[point_index]
    if metric == "euclidean":
        distances = np.linalg.norm(points - query, axis=1)
    else:
        similarities = points @ query
        distances = np.maximum(1.0 - similarities, 0.0)
    return np.asarray(distances, dtype=np.float64)


class BruteForceBackend:
    """Exact float64 reference range-query backend."""

    def __init__(self, x: np.ndarray, metric: BruteForceMetric = "euclidean") -> None:
        self.metric = metric
        self.points = prepare_reference_points(x, metric)
        self.metadata = BackendMetadata(
            name="brute_force",
            is_exact=True,
            metric=metric,
            dtype="float64",
            device="cpu",
            supports_batch_queries=True,
            supports_distances_to_all=True,
            notes="Exhaustive float64 reference oracle.",
        )

    @property
    def n_samples(self) -> int:
        return int(self.points.shape[0])

    def distances_to_all(self, point_index: int) -> np.ndarray:
        return distances_from_reference_point(
            self.points, int(point_index), self.metric
        )

    def query_radius(
        self,
        point_indices: Sequence[int],
        eps: float,
    ) -> list[np.ndarray]:
        indices = normalize_point_indices(point_indices, self.n_samples)
        radius = closed_ball_radius(eps)
        return [
            np.flatnonzero(self.distances_to_all(int(i)) < radius).astype(
                np.intp, copy=False
            )
            for i in indices
        ]


def query_brute_force_range(
    points: np.ndarray,
    point_index: int,
    eps: float,
    metric: BruteForceMetric,
) -> np.ndarray:
    """Backward-compatible one-point reference query helper."""
    distances = distances_from_reference_point(points, point_index, metric)
    return np.flatnonzero(distances < closed_ball_radius(eps)).astype(
        np.intp, copy=False
    )


def build_cover_brute_force(
    x: np.ndarray,
    landmarks: Sequence[int],
    eps: float,
    metric: BruteForceMetric,
) -> list[np.ndarray]:
    """Backward-compatible exact cover constructor."""
    return BruteForceBackend(x, metric).query_radius(landmarks, eps)


def compute_landmarks_brute_force(
    x: np.ndarray,
    eps: float,
    metric: BruteForceMetric,
) -> tuple[list[int], list[np.ndarray]]:
    """Backward-compatible greedy landmark constructor."""
    backend = BruteForceBackend(x, metric)
    uncovered = np.ones(backend.n_samples, dtype=bool)
    landmarks: list[int] = []
    cover: list[np.ndarray] = []
    while np.any(uncovered):
        landmark_index = int(np.argmax(uncovered))
        landmarks.append(landmark_index)
        members = backend.query_radius([landmark_index], eps)[0]
        cover.append(members)
        uncovered[members] = False
    return landmarks, cover
