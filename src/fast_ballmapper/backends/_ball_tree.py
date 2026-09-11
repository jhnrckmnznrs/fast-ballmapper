"""scikit-learn BallTree backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fast_ballmapper._radius import closed_ball_radius
from fast_ballmapper.backends._base import BackendMetadata
from fast_ballmapper.backends._utils import normalize_point_indices

try:
    from sklearn.neighbors import BallTree
except ImportError:  # pragma: no cover - dependency is required by project metadata
    BallTree = None  # type: ignore[assignment,misc]


def create_ball_tree(
    x: np.ndarray,
    metric: str,
    leaf_size: int,
    metric_kwargs: Mapping[str, Any],
):
    """Create a BallTree with one shared metric configuration."""
    if BallTree is None:
        raise ImportError("scikit-learn is required for method='ball_tree'.")
    return BallTree(x, metric=metric, leaf_size=leaf_size, **metric_kwargs)


def _query_radius_closed(tree, queries: np.ndarray, eps: float) -> list[np.ndarray]:
    """Return exact closed-ball memberships for one or more queries."""
    radius = closed_ball_radius(eps)
    indices, distances = tree.query_radius(
        queries,
        r=radius,
        return_distance=True,
        sort_results=False,
    )
    result: list[np.ndarray] = []
    for query_indices, query_distances in zip(indices, distances, strict=True):
        members = np.asarray(query_indices)[np.asarray(query_distances) < radius]
        result.append(np.sort(members.astype(np.intp, copy=False)))
    return result


class BallTreeBackend:
    """Exact CPU range-query backend backed by scikit-learn BallTree."""

    def __init__(
        self,
        x: np.ndarray,
        metric: str = "euclidean",
        leaf_size: int = 40,
        metric_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(leaf_size, (int, np.integer)) or leaf_size <= 0:
            raise ValueError("leaf_size must be a positive integer.")
        self.x = np.asarray(x)
        self.metric = metric
        self.leaf_size = int(leaf_size)
        self.metric_kwargs = dict(metric_kwargs or {})
        self.tree = create_ball_tree(
            self.x,
            self.metric,
            self.leaf_size,
            self.metric_kwargs,
        )
        self.metadata = BackendMetadata(
            name="ball_tree",
            is_exact=True,
            metric=metric,
            dtype=str(self.x.dtype),
            device="cpu",
            supports_batch_queries=True,
            supports_distances_to_all=True,
            notes="Exact scikit-learn BallTree closed-ball queries.",
        )

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])

    def query_radius(
        self,
        point_indices: Sequence[int],
        eps: float,
    ) -> list[np.ndarray]:
        indices = normalize_point_indices(point_indices, self.n_samples)
        return _query_radius_closed(self.tree, self.x[indices], eps)

    def distances_to_all(self, point_index: int) -> np.ndarray:
        return distances_to_all(self.x, self.tree, int(point_index))


def compute_landmarks_ball_tree(
    x: np.ndarray,
    eps: float,
    metric: str,
    leaf_size: int,
    metric_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Backward-compatible greedy landmark constructor."""
    backend = BallTreeBackend(x, metric, leaf_size, metric_kwargs)
    uncovered = np.ones(backend.n_samples, dtype=bool)
    landmarks: list[int] = []
    cover: list[np.ndarray] = []
    while np.any(uncovered):
        landmark_index = int(np.argmax(uncovered))
        landmarks.append(landmark_index)
        point_indices = backend.query_radius([landmark_index], eps)[0]
        cover.append(point_indices)
        uncovered[point_indices] = False
    return landmarks, cover


def distances_to_all(x: np.ndarray, tree, point_index: int) -> np.ndarray:
    """Return distances from one point to every row of ``x`` in row order."""
    distances, indices = tree.query(
        x[point_index : point_index + 1],
        k=x.shape[0],
        return_distance=True,
    )
    result = np.empty(x.shape[0], dtype=float)
    result[indices[0]] = distances[0]
    return result


def build_cover_ball_tree(
    x: np.ndarray,
    landmarks: Sequence[int],
    eps: float,
    metric: str = "euclidean",
    leaf_size: int = 40,
    metric_kwargs: Mapping[str, Any] | None = None,
    *,
    tree=None,
) -> list[np.ndarray]:
    """Backward-compatible fixed-landmark cover constructor."""
    if tree is None:
        return BallTreeBackend(
            x,
            metric,
            leaf_size,
            metric_kwargs,
        ).query_radius(landmarks, eps)
    landmark_array = np.asarray(landmarks, dtype=np.intp)
    return _query_radius_closed(tree, np.asarray(x)[landmark_array], eps)
