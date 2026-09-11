"""Small helpers shared by backend implementations."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np


def normalize_point_indices(point_indices: Sequence[int], n_samples: int) -> np.ndarray:
    arr = np.asarray(list(point_indices), dtype=np.intp)
    if arr.ndim != 1:
        raise ValueError("point_indices must be a one-dimensional sequence.")
    if arr.size and (np.any(arr < 0) or np.any(arr >= n_samples)):
        raise IndexError("A range-query point index is out of bounds.")
    return arr
