"""SciPy cKDTree backend."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fast_ballmapper._radius import closed_ball_radius
from fast_ballmapper.backends._base import BackendMetadata
from fast_ballmapper.backends._utils import normalize_point_indices

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - optional at import time
    cKDTree = None  # type: ignore[assignment,misc]


class CKDTreeBackend:
    """Exact Euclidean CPU range-query backend using SciPy ``cKDTree``."""

    def __init__(
        self,
        x: np.ndarray,
        *,
        leaf_size: int = 16,
    ) -> None:
        if cKDTree is None:
            raise ImportError("SciPy is required for method='ckdtree'.")
        if leaf_size <= 0:
            raise ValueError("leaf_size must be positive.")
        self.x = np.asarray(x, dtype=np.float64)
        self.tree = cKDTree(self.x, leafsize=int(leaf_size))
        self.metadata = BackendMetadata(
            name="ckdtree",
            is_exact=True,
            metric="euclidean",
            dtype="float64",
            device="cpu",
            supports_batch_queries=True,
            supports_distances_to_all=True,
            notes="Independent exact SciPy cKDTree implementation.",
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
        radius = closed_ball_radius(eps)
        raw = self.tree.query_ball_point(self.x[indices], r=radius)
        result: list[np.ndarray] = []
        for query_index, candidates in zip(indices, raw, strict=True):
            candidate_array = np.asarray(candidates, dtype=np.intp)
            if candidate_array.size == 0:
                result.append(candidate_array)
                continue
            distances = np.linalg.norm(
                self.x[candidate_array] - self.x[int(query_index)],
                axis=1,
            )
            members = candidate_array[distances < radius]
            result.append(np.sort(np.unique(members)).astype(np.intp, copy=False))
        return result

    def distances_to_all(self, point_index: int) -> np.ndarray:
        point_index = int(point_index)
        distances, indices = self.tree.query(self.x[point_index], k=self.n_samples)
        result = np.empty(self.n_samples, dtype=np.float64)
        result[np.asarray(indices, dtype=np.intp)] = np.asarray(distances, dtype=float)
        return result
