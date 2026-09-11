"""Optional hnswlib backend.

hnswlib exposes k-nearest-neighbour search rather than a native radius query.
The adapter therefore requests ``candidate_k`` neighbours and filters those
candidates by the Ball Mapper radius.  It is intentionally marked approximate.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from fast_ballmapper._radius import closed_ball_radius, faiss_l2_radius
from fast_ballmapper._validation import validate_cosine_points
from fast_ballmapper.backends._base import BackendMetadata
from fast_ballmapper.backends._configs import HnswlibConfig
from fast_ballmapper.backends._utils import normalize_point_indices

try:
    import hnswlib
except ImportError:  # pragma: no cover - optional dependency
    hnswlib = None


class HnswlibBackend:
    """Approximate CPU HNSW backend using hnswlib candidate queries."""

    def __init__(
        self,
        x: np.ndarray,
        metric: str = "euclidean",
        config: HnswlibConfig | None = None,
    ) -> None:
        if hnswlib is None:
            raise ImportError(
                "hnswlib is required for method='hnswlib'. Install "
                "fast-ballmapper[hnswlib]."
            )
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("hnswlib supports only 'euclidean' and 'cosine'.")
        self.config = config or HnswlibConfig()
        if self.config.candidate_k <= 0:
            raise ValueError("candidate_k must be positive.")
        if self.config.m <= 0 or self.config.ef_construction <= 0:
            raise ValueError("m and ef_construction must be positive.")

        self.metric = metric
        self.original_points = np.asarray(x, dtype=np.float64)
        if metric == "cosine":
            validate_cosine_points(self.original_points)
        self.indexed_points = np.ascontiguousarray(x, dtype=np.float32)
        space = "l2" if metric == "euclidean" else "cosine"
        self.index = hnswlib.Index(space=space, dim=self.indexed_points.shape[1])
        self.index.init_index(
            max_elements=len(self.indexed_points),
            M=int(self.config.m),
            ef_construction=int(self.config.ef_construction),
            random_seed=int(self.config.random_seed),
        )
        ids = np.arange(len(self.indexed_points), dtype=np.int64)
        self.index.add_items(
            self.indexed_points,
            ids,
            num_threads=int(self.config.num_threads),
        )
        effective_k = min(int(self.config.candidate_k), len(self.indexed_points))
        self.index.set_ef(max(int(self.config.ef_search), effective_k))
        if int(self.config.num_threads) > 0:
            self.index.set_num_threads(int(self.config.num_threads))
        self.metadata = BackendMetadata(
            name="hnswlib",
            is_exact=False,
            metric=metric,
            dtype="float32",
            device="cpu",
            supports_batch_queries=True,
            supports_distances_to_all=False,
            notes=(
                "Approximate kNN candidate search followed by an exact radius "
                "filter over returned candidates."
            ),
        )

    @property
    def n_samples(self) -> int:
        return int(self.indexed_points.shape[0])

    def query_radius(
        self,
        point_indices: Sequence[int],
        eps: float,
    ) -> list[np.ndarray]:
        query_indices = normalize_point_indices(point_indices, self.n_samples)
        if query_indices.size == 0:
            return []
        k = min(int(self.config.candidate_k), self.n_samples)
        labels, raw_distances = self.index.knn_query(
            self.indexed_points[query_indices],
            k=k,
            num_threads=int(self.config.num_threads),
        )
        result: list[np.ndarray] = []
        for query_index, candidate_ids, distances in zip(
            query_indices,
            labels,
            raw_distances,
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
        raise NotImplementedError(
            "hnswlib does not provide the exhaustive distance vector required "
            "for deterministic farthest-point sampling."
        )
