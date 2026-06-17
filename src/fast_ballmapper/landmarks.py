"""Landmark selection and cover construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np

from fast_ballmapper._validation import (
    lexicographically_smallest_index,
    normalize_backend_options,
    validate_point_cloud,
    validate_start_index,
)
from fast_ballmapper.backends._ball_tree import (
    build_cover_ball_tree,
    compute_landmarks_ball_tree,
    create_ball_tree,
    distances_to_all,
)
from fast_ballmapper.backends._faiss import (
    build_cover_cosine,
    build_cover_euclidean,
    compute_landmarks_cosine,
    compute_landmarks_euclidean,
    cosine_distances_to_all,
    create_cosine_index,
    create_euclidean_index,
    euclidean_distances_to_all,
)

Backend = Literal["ball_tree", "faiss"]
Cover = list[np.ndarray]


def compute_landmarks(
    x: np.ndarray,
    eps: float,
    method: Backend = "ball_tree",
    metric: str = "euclidean",
    leaf_size: int = 40,
    metric_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], Cover]:
    """Compute greedy landmarks and their epsilon-ball cover.

    Parameters
    ----------
    x:
        Numeric array with shape ``(n_samples, n_features)``.
    eps:
        Finite, non-negative neighborhood radius.
    method:
        ``"ball_tree"`` or ``"faiss"``.
    metric:
        A BallTree metric, or ``"euclidean"``/``"cosine"`` for FAISS.
    leaf_size:
        Positive BallTree leaf size.
    metric_kwargs:
        Additional keyword arguments for the BallTree metric.

    Returns
    -------
    landmarks, cover:
        Landmark row indices and one array of covered point indices per landmark.
    """
    x = validate_point_cloud(x, eps)
    method_key, metric_key, normalized_metric_kwargs = normalize_backend_options(
        method, metric, leaf_size, metric_kwargs
    )

    if x.shape[0] == 0:
        return [], []

    if method_key == "ball_tree":
        return compute_landmarks_ball_tree(
            x,
            eps,
            metric_key,
            leaf_size,
            normalized_metric_kwargs,
        )
    if metric_key == "euclidean":
        return compute_landmarks_euclidean(x, eps)
    return compute_landmarks_cosine(x, eps)


def compute_landmarks_fps(
    x: np.ndarray,
    eps: float,
    start_index: int | None = None,
    method: Backend = "ball_tree",
    metric: str = "euclidean",
    leaf_size: int = 40,
    metric_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], Cover]:
    """Compute deterministic farthest-point landmarks and an epsilon cover.

    Landmark selection and cover construction use the same backend and metric.
    The result is therefore an epsilon-net under that distance: every point is
    within ``eps`` of a landmark, and distinct selected landmarks are farther
    than ``eps`` apart.
    """
    x = validate_point_cloud(x, eps)
    method_key, metric_key, normalized_metric_kwargs = normalize_backend_options(
        method, metric, leaf_size, metric_kwargs
    )

    if x.shape[0] == 0:
        return [], []

    validate_start_index(start_index, x.shape[0])
    selected_start_index = (
        lexicographically_smallest_index(x) if start_index is None else int(start_index)
    )

    if method_key == "ball_tree":
        tree = create_ball_tree(
            x,
            metric_key,
            leaf_size,
            normalized_metric_kwargs,
        )

        def distance_function(point_index: int) -> np.ndarray:
            return distances_to_all(x, tree, point_index)

        landmarks = _select_landmarks_fps(
            x.shape[0], eps, selected_start_index, distance_function
        )
        cover = build_cover_ball_tree(
            x,
            landmarks,
            eps,
            metric_key,
            leaf_size,
            normalized_metric_kwargs,
            tree=tree,
        )
        return landmarks, cover

    if metric_key == "euclidean":
        points, index = create_euclidean_index(x)

        def distance_function(point_index: int) -> np.ndarray:
            return euclidean_distances_to_all(points, index, point_index)

        landmarks = _select_landmarks_fps(
            x.shape[0], eps, selected_start_index, distance_function
        )
        cover = build_cover_euclidean(x, landmarks, eps, points=points, index=index)
        return landmarks, cover

    points, index = create_cosine_index(x)

    def distance_function(point_index: int) -> np.ndarray:
        return cosine_distances_to_all(points, index, point_index)

    landmarks = _select_landmarks_fps(
        x.shape[0], eps, selected_start_index, distance_function
    )
    cover = build_cover_cosine(x, landmarks, eps, points=points, index=index)
    return landmarks, cover


def _select_landmarks_fps(
    n_samples: int,
    eps: float,
    start_index: int,
    distance_to_all: Callable[[int], np.ndarray],
) -> list[int]:
    """Run metric-agnostic farthest-point sampling."""
    distances = _validated_distances(distance_to_all(start_index), n_samples)
    landmarks = [int(start_index)]

    while True:
        next_index = int(np.argmax(distances))
        max_distance = float(distances[next_index])
        if max_distance <= eps:
            break

        landmarks.append(next_index)
        new_distances = _validated_distances(distance_to_all(next_index), n_samples)
        distances = np.minimum(distances, new_distances)

    return landmarks


def _validated_distances(distances: np.ndarray, n_samples: int) -> np.ndarray:
    result = np.asarray(distances, dtype=float)
    if result.shape != (n_samples,):
        raise RuntimeError("Distance backend returned an invalid distance array.")
    if not np.all(np.isfinite(result)):
        raise ValueError("The selected metric produced non-finite distances.")
    return np.maximum(result, 0.0)
