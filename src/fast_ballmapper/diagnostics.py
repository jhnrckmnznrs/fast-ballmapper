"""Diagnostics for range-query sensitivity near the Ball Mapper boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from fast_ballmapper._radius import closed_ball_radius
from fast_ballmapper._validation import validate_point_cloud
from fast_ballmapper.backends._brute_force import (
    BruteForceMetric,
    distances_from_reference_point,
    prepare_reference_points,
)

DiagnosticMetric = Literal["euclidean", "cosine"]


@dataclass(frozen=True)
class BoundaryDiagnostics:
    """Per-landmark distance margins around the radius ``eps``.

    Attributes
    ----------
    landmarks:
        Landmark row indices, in the supplied order.
    eps:
        Radius around which the margins were computed.
    metric:
        Reference distance used for the diagnostics.
    min_abs_margin:
        ``min_x |d(x, l_i) - eps|`` for every landmark.
    inside_margin:
        Distance from ``eps`` to the farthest point still in the closed ball.
        This is zero when a point lies exactly on the boundary.
    outside_margin:
        Distance from ``eps`` to the nearest point outside the closed ball.
        ``np.inf`` means that no data point lies outside that landmark's ball.
    band_counts:
        Mapping from a non-negative half-width ``delta`` to the number of data
        points satisfying ``|d(x, l_i) - eps| <= delta`` for every landmark.
    """

    landmarks: np.ndarray
    eps: float
    metric: str
    min_abs_margin: np.ndarray
    inside_margin: np.ndarray
    outside_margin: np.ndarray
    band_counts: dict[float, np.ndarray]


def compute_boundary_diagnostics(
    x: np.ndarray,
    landmarks: Sequence[int] | np.ndarray,
    eps: float,
    *,
    metric: DiagnosticMetric = "euclidean",
    deltas: Sequence[float] | np.ndarray | None = None,
) -> BoundaryDiagnostics:
    """Measure how close fixed-landmark queries are to the radius boundary.

    The computation uses the package's float64 brute-force reference distance,
    making it suitable for auditing exact or approximate backends.  ``deltas``
    specifies optional symmetric bands around ``eps`` whose point counts should
    be reported.
    """
    x = validate_point_cloud(x, eps)
    metric_key = str(metric).lower()
    if metric_key not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'.")

    landmark_array = _validate_landmarks(landmarks, len(x))
    band_widths = _validate_deltas(deltas)

    if landmark_array.size == 0:
        empty = np.empty(0, dtype=float)
        return BoundaryDiagnostics(
            landmarks=landmark_array,
            eps=float(eps),
            metric=metric_key,
            min_abs_margin=empty.copy(),
            inside_margin=empty.copy(),
            outside_margin=empty.copy(),
            band_counts={delta: np.empty(0, dtype=np.intp) for delta in band_widths},
        )

    reference_metric = cast(BruteForceMetric, metric_key)
    points = prepare_reference_points(x, reference_metric)
    radius = closed_ball_radius(eps)
    min_abs_margin = np.empty(len(landmark_array), dtype=float)
    inside_margin = np.empty(len(landmark_array), dtype=float)
    outside_margin = np.empty(len(landmark_array), dtype=float)
    band_counts = {
        delta: np.empty(len(landmark_array), dtype=np.intp) for delta in band_widths
    }

    for output_index, landmark_index in enumerate(landmark_array):
        distances = distances_from_reference_point(
            points,
            int(landmark_index),
            reference_metric,
        )
        absolute_margin = np.abs(distances - float(eps))
        min_abs_margin[output_index] = float(np.min(absolute_margin))

        inside = distances < radius
        outside = ~inside
        inside_margin[output_index] = float(
            float(eps) - np.max(distances[inside])
        )
        outside_margin[output_index] = (
            float(np.min(distances[outside]) - float(eps))
            if np.any(outside)
            else np.inf
        )

        for delta, counts in band_counts.items():
            counts[output_index] = int(np.count_nonzero(absolute_margin <= delta))

    return BoundaryDiagnostics(
        landmarks=landmark_array,
        eps=float(eps),
        metric=metric_key,
        min_abs_margin=min_abs_margin,
        inside_margin=inside_margin,
        outside_margin=outside_margin,
        band_counts=band_counts,
    )


def _validate_landmarks(
    landmarks: Sequence[int] | np.ndarray,
    n_samples: int,
) -> np.ndarray:
    result: list[int] = []
    seen: set[int] = set()
    for landmark in landmarks:
        if not isinstance(landmark, (int, np.integer)):
            raise TypeError("Each landmark must be an integer row index.")
        index = int(landmark)
        if not 0 <= index < n_samples:
            raise IndexError(
                f"Landmark index {index} is out of bounds for "
                f"a dataset containing {n_samples} points."
            )
        if index in seen:
            raise ValueError(f"Landmark index {index} appears more than once.")
        seen.add(index)
        result.append(index)
    return np.asarray(result, dtype=np.intp)


def _validate_deltas(
    deltas: Sequence[float] | np.ndarray | None,
) -> tuple[float, ...]:
    if deltas is None:
        return ()

    result: list[float] = []
    seen: set[float] = set()
    for delta in deltas:
        value = float(delta)
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                "Every boundary-band delta must be finite and non-negative."
            )
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
