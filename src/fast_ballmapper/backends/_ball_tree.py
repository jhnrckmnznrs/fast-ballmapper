"""scikit-learn BallTree backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

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
    return BallTree(
        x,
        metric=metric,
        leaf_size=leaf_size,
        **metric_kwargs,
    )


def compute_landmarks_ball_tree(
    x: np.ndarray,
    eps: float,
    metric: str,
    leaf_size: int,
    metric_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Compute greedy landmarks and covers with BallTree radius queries."""
    tree = create_ball_tree(x, metric, leaf_size, dict(metric_kwargs or {}))
    uncovered = np.ones(x.shape[0], dtype=bool)
    landmarks: list[int] = []
    cover: list[np.ndarray] = []

    while np.any(uncovered):
        landmark_index = int(np.argmax(uncovered))
        landmarks.append(landmark_index)
        point_indices = tree.query_radius(x[landmark_index : landmark_index + 1], eps)[
            0
        ]
        cover.append(point_indices)
        uncovered[point_indices] = False

    return landmarks, cover


def distances_to_all(
    x: np.ndarray,
    tree,
    point_index: int,
) -> np.ndarray:
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
    """Build epsilon-ball covers for fixed landmarks using BallTree."""
    if tree is None:
        tree = create_ball_tree(x, metric, leaf_size, dict(metric_kwargs or {}))

    return [tree.query_radius(x[index : index + 1], eps)[0] for index in landmarks]
