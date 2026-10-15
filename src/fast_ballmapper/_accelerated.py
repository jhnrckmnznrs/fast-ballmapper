"""Optional batched and compiled candidate checks for Euclidean balls.

The Numba kernel is a *candidate generator*, not the final membership oracle.
All retained candidates are checked with NumPy's direct float64 norm, and
unsafe dynamic-range inputs fall back to the NumPy implementation.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def _validate_engine(engine: str) -> str:
    if engine not in {"scalar", "batch", "numba", "auto"}:
        raise ValueError("engine must be 'scalar', 'batch', 'numba', or 'auto'")
    return "batch" if engine == "auto" else engine


def _numba_candidates(
    points: np.ndarray, center: np.ndarray, eps: float, ids: np.ndarray
) -> np.ndarray | None:
    """Return conservative candidate IDs, or None when fast math is unsafe.

    Screening uses squared distances. The ulp guard is heuristic, not a formal
    proof of agreement with NumPy norm for all floating-point inputs.
    """
    if not (1e-120 < eps < 1e120):
        return None
    if not len(ids) or not center.size:
        return ids if center.size else None
    max_magnitude = max(
        float(np.max(np.abs(center))),
        float(np.max(np.abs(points[ids]))),
    )
    if not np.isfinite(max_magnitude) or max_magnitude > 1e120:
        return None
    d = points.shape[1]
    width = 128.0 * np.finfo(np.float64).eps * (d + 2) * (
        2 * max_magnitude + eps
    )
    bound = eps + width
    if not np.isfinite(bound) or bound > 1e150:
        return None

    return ids[_compiled_screen()(points, center, ids, bound * bound)]


@lru_cache(maxsize=1)
def _compiled_screen():
    """Lazily compile once, so importing the package never requires Numba."""
    from numba import njit  # optional dependency

    @njit
    def screen(data, ref, candidates, cutoff_squared):
        keep = np.empty(len(candidates), dtype=np.bool_)
        for k in range(len(candidates)):
            u = candidates[k]
            squared = 0.0
            for j in range(ref.shape[0]):
                delta = data[u, j] - ref[j]
                squared += delta * delta
            keep[k] = squared <= cutoff_squared
        return keep

    return screen


def filter_candidates(
    points: np.ndarray,
    center: np.ndarray,
    ids: np.ndarray,
    eps: float,
    *,
    engine: str = "batch",
    block_size: int = 16384,
) -> np.ndarray:
    """Filter candidate IDs using the direct float64 norm <= eps predicate.

    'batch' amortizes NumPy overhead by merging owner groups into one block.
    'numba' optionally rejects definite outsiders before final NumPy checks.
    """
    engine = _validate_engine(engine)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if engine == "numba":
        try:
            import numba  # noqa: F401
        except ImportError as exc:
            raise ImportError("engine='numba' requires numba") from exc
    out: list[np.ndarray] = []
    for start in range(0, len(ids), block_size):
        current = ids[start : start + block_size]
        if engine == "numba":
            reduced = _numba_candidates(points, center, eps, current)
            if reduced is not None:
                current = reduced
        if not len(current):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            distances = np.linalg.norm(points[current] - center, axis=1)
        if not np.all(np.isfinite(distances)):
            raise ValueError("distance overflow; rescale coordinates")
        out.append(current[distances <= eps])
    return np.concatenate(out) if out else np.empty(0, dtype=np.intp)
