"""Validation helpers shared by Ball Mapper algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def validate_point_cloud(x: np.ndarray, eps: float) -> np.ndarray:
    """Validate and normalize point-cloud inputs."""
    x = np.asarray(x)

    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (n_samples, n_features).")
    if x.shape[0] > 0 and x.shape[1] == 0:
        raise ValueError("x must contain at least one feature.")
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("x must contain numeric values.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values.")
    if not np.isscalar(eps) or not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite, non-negative scalar.")

    return x


def validate_leaf_size(leaf_size: int) -> None:
    """Validate a BallTree leaf size."""
    if not isinstance(leaf_size, (int, np.integer)) or leaf_size <= 0:
        raise ValueError("leaf_size must be a positive integer.")


def normalize_backend_options(
    method: str,
    metric: str,
    leaf_size: int,
    metric_kwargs: Mapping[str, Any] | None,
) -> tuple[str, str, dict[str, Any]]:
    """Normalize backend options and reject unsupported combinations."""
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")

    method_key = method.lower().replace("-", "_")
    metric_key = metric.lower()
    normalized_metric_kwargs = dict(metric_kwargs or {})

    # ``balltree`` is accepted as a forgiving alias, while ``ball_tree`` is the
    # documented snake_case spelling.
    if method_key == "balltree":
        method_key = "ball_tree"

    if method_key not in {"ball_tree", "faiss"}:
        raise ValueError("method must be 'ball_tree' or 'faiss'.")

    if method_key == "ball_tree":
        validate_leaf_size(leaf_size)
    else:
        if metric_key not in {"euclidean", "cosine"}:
            raise ValueError(
                "For method='faiss', metric must be 'euclidean' or 'cosine'."
            )
        if normalized_metric_kwargs:
            raise ValueError("metric_kwargs is supported only with method='ball_tree'.")

    return method_key, metric_key, normalized_metric_kwargs


def validate_start_index(start_index: int | None, n_samples: int) -> None:
    """Validate an optional farthest-point-sampling start index."""
    if start_index is None:
        return
    if not isinstance(start_index, (int, np.integer)):
        raise TypeError("start_index must be an integer or None.")
    if not 0 <= int(start_index) < n_samples:
        raise IndexError("start_index is out of bounds for x.")


def lexicographically_smallest_index(x: np.ndarray) -> int:
    """Return the row index ordered by feature 0, then feature 1, and so on."""
    keys = tuple(x[:, column] for column in range(x.shape[1] - 1, -1, -1))
    return int(np.lexsort(keys)[0])


def validate_cosine_points(points: np.ndarray) -> None:
    """Reject zero vectors, for which cosine distance is undefined."""
    if np.any(np.linalg.norm(points, axis=1) == 0):
        raise ValueError("Cosine distance is undefined for zero vectors.")
