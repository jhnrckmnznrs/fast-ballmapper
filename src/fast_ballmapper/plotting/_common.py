"""Shared plotting validation."""

from __future__ import annotations

import numpy as np


def scale_node_sizes(
    sizes,
    node_count: int,
    node_scale: float,
) -> np.ndarray:
    """Validate and scale node sizes without dividing by zero."""
    if not np.isscalar(node_scale) or not np.isfinite(node_scale) or node_scale < 0:
        raise ValueError("node_scale must be a finite, non-negative scalar.")

    values = np.asarray(sizes, dtype=float)
    if values.ndim != 1 or len(values) != node_count:
        raise ValueError("sizes must contain exactly one value per node.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("sizes must contain finite, non-negative values.")
    if values.size == 0:
        return values

    max_size = values.max()
    if max_size == 0:
        return np.zeros_like(values)
    return node_scale * values / max_size
