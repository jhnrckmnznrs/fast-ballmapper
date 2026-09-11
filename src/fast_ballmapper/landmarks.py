"""Landmark selection and cover construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from fast_ballmapper._radius import closed_ball_radius
from fast_ballmapper._validation import (
    lexicographically_smallest_index,
    validate_point_cloud,
    validate_start_index,
)
from fast_ballmapper.backends import (
    CuVSConfig,
    HnswlibConfig,
    RangeQueryBackend,
    make_backend,
)
from fast_ballmapper.faiss import FaissConfig

Backend = Literal[
    "ball_tree",
    "brute_force",
    "ckdtree",
    "faiss",
    "hnswlib",
    "cuvs",
    "cuvs_brute_force",
    "cuvs_cagra",
]
Cover = list[np.ndarray]


def _resolve_backend(
    x: np.ndarray,
    *,
    method: str,
    metric: str,
    leaf_size: int,
    metric_kwargs: Mapping[str, Any] | None,
    faiss_config: FaissConfig | None,
    hnswlib_config: HnswlibConfig | None,
    cuvs_config: CuVSConfig | None,
    backend: RangeQueryBackend | None,
) -> RangeQueryBackend:
    if backend is not None:
        if faiss_config is not None or hnswlib_config is not None or cuvs_config is not None:
            raise ValueError(
                "Backend-specific configuration must be supplied when constructing "
                "the backend object, not together with backend=."
            )
        if metric_kwargs:
            raise ValueError("metric_kwargs cannot be combined with backend=.")
        if backend.n_samples != x.shape[0]:
            raise ValueError(
                "The supplied backend indexes a different number of observations "
                "than x."
            )
        return backend

    # Preserve the historical error wording used by downstream tests/users.
    if method.lower().replace("-", "_") not in {"faiss"} and faiss_config is not None:
        raise ValueError("faiss_config may only be used with method='faiss'.")

    return make_backend(
        x,
        method,
        metric=metric,
        leaf_size=leaf_size,
        metric_kwargs=metric_kwargs,
        faiss_config=faiss_config,
        hnswlib_config=hnswlib_config,
        cuvs_config=cuvs_config,
    )


def compute_landmarks(
    x: np.ndarray,
    eps: float,
    method: Backend | str = "ball_tree",
    metric: str = "euclidean",
    leaf_size: int = 40,
    metric_kwargs: Mapping[str, Any] | None = None,
    faiss_config: FaissConfig | None = None,
    hnswlib_config: HnswlibConfig | None = None,
    cuvs_config: CuVSConfig | None = None,
    *,
    backend: RangeQueryBackend | None = None,
) -> tuple[list[int], Cover]:
    """Compute greedy landmarks and their closed epsilon-ball cover.

    Existing ``method=`` calls remain supported.  New code may instead pass a
    fitted :class:`~fast_ballmapper.backends.RangeQueryBackend` through
    ``backend=``; this is the preferred extension point for custom engines.
    """
    x = validate_point_cloud(x, eps)
    if x.shape[0] == 0:
        return [], []

    selected_backend = _resolve_backend(
        x,
        method=str(method),
        metric=metric,
        leaf_size=leaf_size,
        metric_kwargs=metric_kwargs,
        faiss_config=faiss_config,
        hnswlib_config=hnswlib_config,
        cuvs_config=cuvs_config,
        backend=backend,
    )

    uncovered = np.ones(x.shape[0], dtype=bool)
    landmarks: list[int] = []
    cover: list[np.ndarray] = []
    while np.any(uncovered):
        landmark_index = int(np.argmax(uncovered))
        landmarks.append(landmark_index)
        members = selected_backend.query_radius([landmark_index], eps)[0]
        cover.append(np.asarray(members, dtype=np.intp))
        uncovered[members] = False
    return landmarks, cover


def build_cover(
    x: np.ndarray,
    landmarks: Sequence[int],
    eps: float,
    method: Backend | str = "ball_tree",
    metric: str = "euclidean",
    leaf_size: int = 40,
    metric_kwargs: Mapping[str, Any] | None = None,
    faiss_config: FaissConfig | None = None,
    hnswlib_config: HnswlibConfig | None = None,
    cuvs_config: CuVSConfig | None = None,
    *,
    backend: RangeQueryBackend | None = None,
) -> Cover:
    """Construct closed epsilon-balls around fixed landmark observations."""
    x = validate_point_cloud(x, eps)
    landmark_indices = _validate_landmark_indices(landmarks, n_samples=x.shape[0])
    if not landmark_indices:
        return []

    selected_backend = _resolve_backend(
        x,
        method=str(method),
        metric=metric,
        leaf_size=leaf_size,
        metric_kwargs=metric_kwargs,
        faiss_config=faiss_config,
        hnswlib_config=hnswlib_config,
        cuvs_config=cuvs_config,
        backend=backend,
    )
    return selected_backend.query_radius(landmark_indices, eps)


def _validate_landmark_indices(
    landmarks: Sequence[int],
    n_samples: int,
) -> list[int]:
    """Validate and normalize fixed landmark indices."""
    landmark_indices: list[int] = []
    seen: set[int] = set()
    for landmark in landmarks:
        if not isinstance(landmark, (int, np.integer)):
            raise TypeError("Each landmark must be an integer row index.")
        landmark_index = int(landmark)
        if not 0 <= landmark_index < n_samples:
            raise IndexError(
                f"Landmark index {landmark_index} is out of bounds "
                f"for a dataset containing {n_samples} points."
            )
        if landmark_index in seen:
            raise ValueError(f"Landmark index {landmark_index} appears more than once.")
        seen.add(landmark_index)
        landmark_indices.append(landmark_index)
    return landmark_indices


def compute_landmarks_fps(
    x: np.ndarray,
    eps: float,
    start_index: int | None = None,
    method: Backend | str = "ball_tree",
    metric: str = "euclidean",
    leaf_size: int = 40,
    metric_kwargs: Mapping[str, Any] | None = None,
    *,
    backend: RangeQueryBackend | None = None,
) -> tuple[list[int], Cover]:
    """Compute deterministic farthest-point landmarks and an epsilon cover.

    Farthest-point sampling requires an exhaustive distance vector.  Backends
    advertise this capability through ``metadata.supports_distances_to_all``.
    Approximate candidate-only engines such as HNSW/CAGRA are therefore rejected
    unless the caller supplies a backend that can provide exhaustive distances.

    For backward compatibility, ``method='faiss'`` uses an exact FAISS Flat
    backend, matching the historical implementation.
    """
    x = validate_point_cloud(x, eps)
    if x.shape[0] == 0:
        return [], []
    validate_start_index(start_index, x.shape[0])
    selected_start_index = (
        lexicographically_smallest_index(x) if start_index is None else int(start_index)
    )

    if backend is None and str(method).lower().replace("-", "_") == "faiss":
        selected_backend = make_backend(
            x,
            "faiss",
            metric=metric,
            faiss_config=FaissConfig(factory="Flat"),
        )
    else:
        selected_backend = _resolve_backend(
            x,
            method=str(method),
            metric=metric,
            leaf_size=leaf_size,
            metric_kwargs=metric_kwargs,
            faiss_config=None,
            hnswlib_config=None,
            cuvs_config=None,
            backend=backend,
        )

    if not selected_backend.metadata.supports_distances_to_all:
        raise ValueError(
            f"Backend {selected_backend.metadata.name!r} does not provide the "
            "exhaustive distance vector required by farthest-point sampling."
        )

    landmarks = _select_landmarks_fps(
        x.shape[0],
        eps,
        selected_start_index,
        selected_backend.distances_to_all,
    )
    cover = selected_backend.query_radius(landmarks, eps)
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
        if max_distance < closed_ball_radius(eps):
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
