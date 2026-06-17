"""Helpers for assigning scalar or categorical values to Ball Mapper nodes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np


def color_by_function(
    x: np.ndarray,
    cover: Sequence[np.ndarray],
    func: Callable = np.mean,
) -> np.ndarray:
    """Apply ``func`` to the values covered by each Ball Mapper node."""
    values = np.asarray(x)
    colors = np.zeros(len(cover))

    for cover_index, point_indices in enumerate(cover):
        colors[cover_index] = (
            func(values[point_indices]) if len(point_indices) > 0 else np.nan
        )

    return colors


def color_by_mode(y: np.ndarray, cover: Sequence[np.ndarray]) -> np.ndarray:
    """Assign each ball the most frequent label among its covered points."""
    labels = np.asarray(y)
    colors = np.empty(len(cover), dtype=object)

    for cover_index, point_indices in enumerate(cover):
        colors[cover_index] = (
            Counter(labels[point_indices]).most_common(1)[0][0]
            if len(point_indices) > 0
            else -1
        )

    return colors


def color_by_entropy(y: np.ndarray, cover: Sequence[np.ndarray]) -> np.ndarray:
    """Color balls by base-2 label entropy."""
    labels = np.asarray(y)
    colors = np.zeros(len(cover))

    for cover_index, point_indices in enumerate(cover):
        if len(point_indices) == 0:
            continue
        _, counts = np.unique(labels[point_indices], return_counts=True)
        probabilities = counts / counts.sum()
        colors[cover_index] = -np.sum(probabilities * np.log2(probabilities + 1e-12))

    return colors


def color_by_size(cover: Sequence[np.ndarray]) -> np.ndarray:
    """Return the number of covered points for each ball."""
    return np.array([len(point_indices) for point_indices in cover])


def color_by_density(cover: Sequence[np.ndarray]) -> np.ndarray:
    """Normalize cover sizes to the interval ``[0, 1]``."""
    sizes = np.array([len(point_indices) for point_indices in cover], dtype=float)
    if sizes.size == 0:
        return sizes
    max_size = sizes.max()
    if max_size == 0:
        return np.zeros_like(sizes)
    return sizes / max_size
