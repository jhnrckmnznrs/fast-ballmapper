"""Optional NVIDIA cuVS GPU backends.

The Python cuVS brute-force and CAGRA APIs expose k-nearest-neighbour queries.
This adapter converts those candidate lists to Ball Mapper closed-radius sets.
For the brute-force backend, ``candidate_k=None`` searches all indexed points
and is exact relative to cuVS's float32 representation. CAGRA is approximate.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from fast_ballmapper._radius import closed_ball_radius, faiss_l2_radius
from fast_ballmapper._validation import validate_cosine_points
from fast_ballmapper.backends._base import BackendMetadata
from fast_ballmapper.backends._configs import CuVSConfig
from fast_ballmapper.backends._utils import normalize_point_indices

try:  # pragma: no cover - optional GPU stack
    import cupy as cp
    from cuvs.neighbors import brute_force as cuvs_brute_force
    from cuvs.neighbors import cagra as cuvs_cagra
except ImportError:  # pragma: no cover
    cp = None
    cuvs_brute_force = None
    cuvs_cagra = None


class CuVSBackend:
    """GPU range-query adapter for cuVS brute-force or CAGRA search."""

    def __init__(
        self,
        x: np.ndarray,
        metric: str = "euclidean",
        config: CuVSConfig | None = None,
    ) -> None:
        if cp is None or cuvs_brute_force is None or cuvs_cagra is None:
            raise ImportError(
                "NVIDIA cuVS and CuPy are required for method='cuvs'. Install "
                "the matching CUDA wheel (for example cuvs-cu12 or cuvs-cu13) "
                "and a compatible CuPy build."
            )
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("cuVS supports only 'euclidean' and 'cosine' here.")
        self.config = config or CuVSConfig()
        algorithm = self.config.algorithm.lower().replace("-", "_")
        if algorithm not in {"brute_force", "cagra"}:
            raise ValueError("CuVSConfig.algorithm must be 'brute_force' or 'cagra'.")
        if algorithm == "cagra" and (
            self.config.candidate_k is None or self.config.candidate_k <= 0
        ):
            raise ValueError("CAGRA requires a positive candidate_k.")
        if self.config.candidate_k is not None and self.config.candidate_k <= 0:
            raise ValueError("candidate_k must be positive or None.")

        self.algorithm = algorithm
        self.metric = metric
        self.original_points = np.asarray(x, dtype=np.float64)
        if metric == "cosine":
            validate_cosine_points(self.original_points)
        self.indexed_points = cp.asarray(np.asarray(x, dtype=np.float32))
        cuvs_metric = "sqeuclidean" if metric == "euclidean" else "cosine"

        if algorithm == "brute_force":
            self.index = cuvs_brute_force.build(
                self.indexed_points,
                metric=cuvs_metric,
            )
            exact = self.config.candidate_k is None or (
                int(self.config.candidate_k) >= len(self.original_points)
            )
        else:
            index_kwargs = dict(self.config.index_params)
            index_kwargs.setdefault("metric", cuvs_metric)
            index_params = cuvs_cagra.IndexParams(**index_kwargs)
            self.index = cuvs_cagra.build(index_params, self.indexed_points)
            exact = False

        self.metadata = BackendMetadata(
            name=f"cuvs:{algorithm}",
            is_exact=exact,
            metric=metric,
            dtype="float32",
            device="gpu",
            supports_batch_queries=True,
            supports_distances_to_all=algorithm == "brute_force" and exact,
            notes=(
                "cuVS kNN candidate search converted to a closed-radius query; "
                "brute_force with candidate_k=None searches all points."
            ),
        )

    @property
    def n_samples(self) -> int:
        return int(self.original_points.shape[0])

    def _search(self, query_indices: np.ndarray, k: int):
        queries = self.indexed_points[query_indices]
        if self.algorithm == "brute_force":
            distances, neighbors = cuvs_brute_force.search(self.index, queries, k)
        else:
            search_params = cuvs_cagra.SearchParams(**dict(self.config.search_params))
            distances, neighbors = cuvs_cagra.search(
                search_params,
                self.index,
                queries,
                k,
            )
        return cp.asnumpy(distances), cp.asnumpy(neighbors)

    def query_radius(
        self,
        point_indices: Sequence[int],
        eps: float,
    ) -> list[np.ndarray]:
        query_indices = normalize_point_indices(point_indices, self.n_samples)
        if query_indices.size == 0:
            return []
        k = (
            self.n_samples
            if self.config.candidate_k is None
            else min(int(self.config.candidate_k), self.n_samples)
        )
        raw_distances, neighbors = self._search(query_indices, k)
        result: list[np.ndarray] = []
        for query_index, distances, candidate_ids in zip(
            query_indices,
            raw_distances,
            neighbors,
            strict=True,
        ):
            candidate_ids = np.asarray(candidate_ids, dtype=np.intp)
            distances = np.asarray(distances, dtype=float)
            valid = candidate_ids >= 0
            candidate_ids = candidate_ids[valid]
            distances = distances[valid]
            if self.metric == "euclidean":
                inside = distances < faiss_l2_radius(eps)
            else:
                inside = distances < closed_ball_radius(eps)
            members = candidate_ids[inside]
            if self.config.exact_verify and members.size:
                members = self._verify(int(query_index), members, eps)
            members = np.append(members, int(query_index))
            result.append(np.sort(np.unique(members)).astype(np.intp, copy=False))
        return result

    def _verify(
        self,
        point_index: int,
        candidate_indices: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        query = self.original_points[point_index]
        candidates = self.original_points[candidate_indices]
        if self.metric == "euclidean":
            distances = np.linalg.norm(candidates - query, axis=1)
        else:
            qnorm = np.linalg.norm(query)
            cnorms = np.linalg.norm(candidates, axis=1)
            similarities = (candidates @ query) / (cnorms * qnorm)
            distances = np.maximum(1.0 - similarities, 0.0)
        return candidate_indices[distances < closed_ball_radius(eps)]

    def distances_to_all(self, point_index: int) -> np.ndarray:
        if not self.metadata.supports_distances_to_all:
            raise NotImplementedError(
                "distances_to_all requires cuVS brute_force with candidate_k=None."
            )
        raw_distances, neighbors = self._search(
            np.asarray([int(point_index)], dtype=np.intp),
            self.n_samples,
        )
        result = np.empty(self.n_samples, dtype=float)
        distances = raw_distances[0]
        ids = neighbors[0].astype(np.intp, copy=False)
        if self.metric == "euclidean":
            distances = np.sqrt(np.maximum(distances, 0.0))
        else:
            distances = np.maximum(distances, 0.0)
        result[ids] = distances
        return result
