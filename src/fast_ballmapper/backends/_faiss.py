"""Optional FAISS backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from fast_ballmapper._radius import (
    closed_ball_radius,
    faiss_cosine_similarity_threshold,
    faiss_l2_radius,
)
from fast_ballmapper._validation import validate_cosine_points
from fast_ballmapper.faiss import FaissConfig
from fast_ballmapper.backends._base import BackendMetadata
from fast_ballmapper.backends._utils import normalize_point_indices

try:
    import faiss
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    faiss = None

FaissMetric = Literal["euclidean", "cosine"]


@dataclass
class FaissBackend:
    """Constructed FAISS index and the arrays needed for querying."""

    original_points: np.ndarray
    indexed_points: np.ndarray
    verification_points: np.ndarray
    index: Any
    metric: FaissMetric
    config: FaissConfig
    device: Literal["cpu", "gpu"] = "cpu"
    gpu_resources: Any | None = None
    gpu_error: str | None = None


def _faiss_has_gpu_support() -> bool:
    """Return whether the imported FAISS module exposes GPU helpers."""
    return (
        faiss is not None
        and hasattr(faiss, "StandardGpuResources")
        and hasattr(faiss, "index_cpu_to_gpu")
        and hasattr(faiss, "get_num_gpus")
    )


def _available_gpu_count() -> int:
    """Return the number of FAISS-visible GPUs."""
    if not _faiss_has_gpu_support():
        return 0

    try:
        return int(faiss.get_num_gpus())
    except Exception:  # noqa: BLE001
        return 0


def _should_try_gpu(config: FaissConfig) -> bool:
    """Return whether the configuration requests GPU usage."""
    if config.device == "cpu":
        return False

    if config.device == "gpu":
        return True

    return _available_gpu_count() > 0


def _gpu_cloner_options(config: FaissConfig):
    """Construct FAISS GPU cloner options when available."""
    if not hasattr(faiss, "GpuClonerOptions"):
        return None

    options = faiss.GpuClonerOptions()
    options.useFloat16 = bool(config.gpu_use_float16)
    return options


def _try_move_index_to_gpu(index, config: FaissConfig):
    """Move a CPU FAISS index to GPU if possible."""
    if not _should_try_gpu(config):
        return index, "cpu", None, None

    if not _faiss_has_gpu_support():
        message = (
            "The installed FAISS module does not expose GPU support. "
            "Install a GPU-enabled FAISS build to use device='gpu'."
        )
        if config.gpu_fallback_to_cpu:
            return index, "cpu", None, message
        raise RuntimeError(message)

    gpu_count = _available_gpu_count()

    if gpu_count <= 0:
        message = "No FAISS-visible GPU is available."
        if config.gpu_fallback_to_cpu:
            return index, "cpu", None, message
        raise RuntimeError(message)

    if not 0 <= config.gpu_device < gpu_count:
        message = (
            f"Requested gpu_device={config.gpu_device}, but FAISS sees "
            f"{gpu_count} GPU(s)."
        )
        if config.gpu_fallback_to_cpu:
            return index, "cpu", None, message
        raise ValueError(message)

    try:
        resources = faiss.StandardGpuResources()

        if config.gpu_temp_memory is not None:
            resources.setTempMemory(int(config.gpu_temp_memory))

        options = _gpu_cloner_options(config)

        if options is None:
            gpu_index = faiss.index_cpu_to_gpu(
                resources,
                config.gpu_device,
                index,
            )
        else:
            gpu_index = faiss.index_cpu_to_gpu(
                resources,
                config.gpu_device,
                index,
                options,
            )

        return gpu_index, "gpu", resources, None

    except Exception as error:  # noqa: BLE001
        message = (
            "Failed to move FAISS index to GPU. Falling back to CPU. "
            f"Original error: {type(error).__name__}: {error}"
        )

        if config.gpu_fallback_to_cpu:
            return index, "cpu", None, message

        raise RuntimeError(message) from error


def create_faiss_backend(
    x: np.ndarray,
    metric: FaissMetric,
    config: FaissConfig | None = None,
) -> FaissBackend:
    """Construct, train, and populate a configured FAISS index."""
    _require_faiss()

    selected_config = config or FaissConfig()
    if selected_config.query_mode not in {"auto", "range", "knn"}:
        raise ValueError("query_mode must be 'auto', 'range', or 'knn'.")
    if (
        isinstance(selected_config.query_batch_size, (bool, np.bool_))
        or not isinstance(selected_config.query_batch_size, (int, np.integer))
        or selected_config.query_batch_size <= 0
    ):
        raise ValueError("query_batch_size must be a positive integer.")
    original_points = np.array(x, copy=True)

    # normalize_L2 is in-place. Keep index and verification snapshots separate
    # from each other and from the caller's possibly contiguous float32 input.
    indexed_points = np.array(x, dtype=np.float32, order="C", copy=True)

    if metric == "cosine":
        validate_cosine_points(indexed_points)
        faiss.normalize_L2(indexed_points)
        verification_points = _normalize_rows_float64(original_points)
        metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        verification_points = np.asarray(original_points, dtype=np.float64)
        metric_type = faiss.METRIC_L2

    index = faiss.index_factory(
        indexed_points.shape[1],
        selected_config.factory,
        metric_type,
    )

    _set_faiss_parameters(index, selected_config.construction_params)

    if not index.is_trained:
        training_points = _select_training_points(
            indexed_points,
            selected_config.train_size,
            selected_config.train_seed,
        )
        index.train(training_points)

    index.add(indexed_points)

    _set_faiss_parameters(index, selected_config.search_params)

    index, actual_device, gpu_resources, gpu_error = _try_move_index_to_gpu(
        index,
        selected_config,
    )

    _set_faiss_parameters(index, selected_config.search_params)

    return FaissBackend(
        original_points=original_points,
        indexed_points=indexed_points,
        verification_points=verification_points,
        index=index,
        metric=metric,
        config=selected_config,
        device=actual_device,
        gpu_resources=gpu_resources,
        gpu_error=gpu_error,
    )


def _set_faiss_parameters(index, parameters) -> None:
    if not parameters:
        return

    parameter_space = faiss.ParameterSpace()
    for name, value in parameters.items():
        try:
            parameter_space.set_index_parameter(index, name, float(value))
        except RuntimeError as error:
            raise ValueError(
                f"FAISS parameter {name!r} is not supported by "
                f"index {type(index).__name__}."
            ) from error


def _select_training_points(
    points: np.ndarray,
    train_size: int | None,
    seed: int,
) -> np.ndarray:
    if train_size is None or train_size >= len(points):
        return points

    if train_size <= 0:
        raise ValueError("train_size must be positive or None.")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=train_size, replace=False)
    return np.ascontiguousarray(points[indices])


def query_faiss_range(
    backend: FaissBackend,
    point_index: int,
    eps: float,
) -> np.ndarray:
    """Compute an exact or approximate range set for one indexed point."""
    config = backend.config

    if config.query_mode in {"auto", "range"}:
        try:
            indices = _native_range_query(backend, point_index, eps)
        except RuntimeError as error:
            if config.query_mode == "range" or config.candidate_k is None:
                raise RuntimeError(
                    f"The FAISS index {config.factory!r} does not support "
                    "native range search in the installed FAISS version. "
                    "Set query_mode='knn' and provide candidate_k."
                ) from error
            indices = _knn_candidate_query(backend, point_index, eps)
    else:
        indices = _knn_candidate_query(backend, point_index, eps)

    if config.exact_verify:
        indices = _verify_exact_membership(
            backend,
            point_index,
            indices,
            eps,
        )

    indices = np.append(indices, point_index)
    return np.unique(indices.astype(np.intp, copy=False))


def _can_emulate_exact_flat_range_search(backend: FaissBackend) -> bool:
    """Return whether range search can be emulated exactly by full kNN search."""
    return backend.config.factory.strip().lower() == "flat" and backend.metric in {
        "euclidean",
        "cosine",
    }


def _exact_flat_range_query_via_search(
    backend: FaissBackend,
    point_index: int,
    eps: float,
) -> np.ndarray:
    """Emulate exact range search for Flat indexes using all-neighbour search."""
    query = backend.indexed_points[point_index : point_index + 1]
    k = len(backend.indexed_points)

    distances, indices = backend.index.search(query, k)

    distances = distances[0]
    indices = indices[0]

    valid = indices >= 0
    distances = distances[valid]
    indices = indices[valid]

    if backend.config.exact_verify:
        return indices

    if backend.metric == "euclidean":
        inside = distances < faiss_l2_radius(eps)
    else:
        inside = distances > faiss_cosine_similarity_threshold(eps)

    return indices[inside]


def _native_range_query(
    backend: FaissBackend,
    point_index: int,
    eps: float,
) -> np.ndarray:
    query = backend.indexed_points[point_index : point_index + 1]

    if backend.metric == "euclidean":
        radius = faiss_l2_radius(eps)
    else:
        radius = faiss_cosine_similarity_threshold(eps)

    try:
        limits, _, indices = backend.index.range_search(query, radius)
        return indices[limits[0] : limits[1]]
    except RuntimeError as error:
        if backend.config.query_mode == "auto" and _can_emulate_exact_flat_range_search(
            backend
        ):
            return _exact_flat_range_query_via_search(
                backend,
                point_index,
                eps,
            )

        raise error


def _knn_candidate_query(
    backend: FaissBackend,
    point_index: int,
    eps: float,
) -> np.ndarray:
    candidate_k = backend.config.candidate_k

    if candidate_k is None:
        raise ValueError("candidate_k must be provided when query_mode='knn'.")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive.")

    k = min(candidate_k, len(backend.indexed_points))
    query = backend.indexed_points[point_index : point_index + 1]
    distances, indices = backend.index.search(query, k)

    distances = distances[0]
    indices = indices[0]

    valid = indices >= 0
    distances = distances[valid]
    indices = indices[valid]

    if backend.config.exact_verify:
        # Quantized search distances can overestimate the true distance.
        # Verification must see every retrieved ID, not a prefiltered subset.
        return indices

    if backend.metric == "euclidean":
        inside = distances < faiss_l2_radius(eps)
    else:
        inside = distances > faiss_cosine_similarity_threshold(eps)

    return indices[inside]


def _query_faiss_batch(
    backend: FaissBackend, point_indices: np.ndarray, eps: float
) -> list[np.ndarray]:
    """Issue one native range/kNN call for a bounded batch of fixed queries."""
    config = backend.config
    queries = np.ascontiguousarray(backend.indexed_points[point_indices])
    radius = (
        faiss_l2_radius(eps)
        if backend.metric == "euclidean"
        else faiss_cosine_similarity_threshold(eps)
    )
    candidates = None
    k = config.candidate_k
    if config.query_mode in {"auto", "range"}:
        try:
            limits, _, ids = backend.index.range_search(queries, radius)
            candidates = [ids[limits[i] : limits[i + 1]] for i in range(len(queries))]
        except RuntimeError as error:
            if config.query_mode == "range":
                raise RuntimeError(
                    "The FAISS index does not support native range search."
                ) from error
            if _can_emulate_exact_flat_range_search(backend):
                k = len(backend.indexed_points)
            elif k is None:
                raise RuntimeError(
                    "The FAISS index does not support native range search; "
                    "provide candidate_k for automatic kNN fallback."
                ) from error
    if candidates is None:
        if k is None or k <= 0:
            raise ValueError("candidate_k must be positive for kNN queries.")
        distances, ids = backend.index.search(
            queries, min(k, len(backend.indexed_points))
        )
        candidates = []
        for row_ids, row_distances in zip(ids, distances, strict=True):
            keep = row_ids >= 0
            if not config.exact_verify:
                keep &= (
                    row_distances < radius
                    if backend.metric == "euclidean"
                    else row_distances > radius
                )
            candidates.append(row_ids[keep])
    result = []
    for query_index, ids in zip(point_indices, candidates, strict=True):
        if config.exact_verify:
            ids = _verify_exact_membership(backend, int(query_index), ids, eps)
        result.append(np.unique(np.append(ids, query_index)).astype(np.intp))
    return result


def _verify_exact_membership(
    backend: FaissBackend,
    point_index: int,
    candidate_indices: np.ndarray,
    eps: float,
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=np.intp)

    if candidate_indices.size == 0:
        return candidate_indices

    query = backend.verification_points[point_index]
    candidates = backend.verification_points[candidate_indices]

    if backend.metric == "euclidean":
        distances = np.linalg.norm(candidates - query, axis=1)
    else:
        similarities = candidates @ query
        distances = np.maximum(1.0 - similarities, 0.0)

    return candidate_indices[distances < closed_ball_radius(eps)]


def _normalize_rows_float64(x: np.ndarray) -> np.ndarray:
    points = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError("Cosine distance is undefined for zero vectors.")

    return points / norms


def _require_faiss() -> None:
    if faiss is None:
        raise ImportError(
            "FAISS is required for method='faiss'. Install fast-ballmapper[faiss]."
        )


def create_euclidean_index(x: np.ndarray):
    """Create a float32 FAISS squared-L2 index."""
    _require_faiss()
    points = x.astype(np.float32, copy=True)
    index = faiss.IndexFlatL2(points.shape[1])
    index.add(points)
    return points, index


def create_cosine_index(x: np.ndarray):
    """Create a normalized float32 FAISS inner-product index."""
    _require_faiss()
    points = x.astype(np.float32, copy=True)
    validate_cosine_points(points)
    faiss.normalize_L2(points)
    index = faiss.IndexFlatIP(points.shape[1])
    index.add(points)
    return points, index


def compute_landmarks_faiss(
    x: np.ndarray,
    eps: float,
    metric: FaissMetric,
    config: FaissConfig | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Compute greedy landmarks using a configured FAISS index."""
    backend = create_faiss_backend(x, metric, config)

    covered = np.zeros(len(backend.indexed_points), dtype=bool)
    landmarks: list[int] = []
    cover: list[np.ndarray] = []

    for point_index in range(len(backend.indexed_points)):
        if covered[point_index]:
            continue

        landmarks.append(point_index)
        point_indices = query_faiss_range(
            backend,
            point_index,
            eps,
        )
        cover.append(point_indices)
        covered[point_indices] = True

    return landmarks, cover


def euclidean_distances_to_all(
    points: np.ndarray,
    index,
    point_index: int,
) -> np.ndarray:
    """Return Euclidean distances from one point to all indexed points."""
    squared_distances, indices = index.search(
        points[point_index : point_index + 1], len(points)
    )
    result = np.empty(len(points), dtype=float)
    result[indices[0]] = np.sqrt(np.maximum(squared_distances[0], 0.0))
    return result


def cosine_distances_to_all(
    points: np.ndarray,
    index,
    point_index: int,
) -> np.ndarray:
    """Return cosine distances from one point to all indexed points."""
    similarities, indices = index.search(
        points[point_index : point_index + 1], len(points)
    )
    result = np.empty(len(points), dtype=float)
    result[indices[0]] = np.maximum(1.0 - similarities[0], 0.0)
    return result


def build_cover_faiss(
    x: np.ndarray,
    landmarks: Sequence[int],
    eps: float,
    metric: FaissMetric,
    config: FaissConfig | None = None,
) -> list[np.ndarray]:
    """Construct range sets for a fixed collection of landmarks."""
    backend = create_faiss_backend(x, metric, config)

    return [query_faiss_range(backend, int(landmark), eps) for landmark in landmarks]


class FaissRangeBackend:
    """Range-query backend wrapping a configured FAISS index.

    ``metadata.is_exact`` describes the search algorithm, not the numeric
    representation: FAISS indexes float32 vectors, so boundary decisions can
    still differ from the float64 reference oracle for adversarially close
    points.
    """

    def __init__(
        self,
        x: np.ndarray,
        metric: FaissMetric = "euclidean",
        config: FaissConfig | None = None,
    ) -> None:
        self.state = create_faiss_backend(x, metric, config)
        self.metric = metric
        factory = self.state.config.factory
        is_flat = factory.strip().lower() == "flat"
        config = self.state.config
        exhaustive = is_flat and (
            config.query_mode != "knn"
            or (config.candidate_k is not None and config.candidate_k >= len(x))
        )
        self.metadata = BackendMetadata(
            name=f"faiss:{factory}",
            is_exact=exhaustive,
            metric=metric,
            dtype=(
                "float16"
                if self.state.device == "gpu" and config.gpu_use_float16
                else "float32"
            ),
            device=self.state.device,
            supports_batch_queries=True,
            supports_distances_to_all=is_flat,
            notes=(
                "Flat radius/all-neighbor search is exhaustive; capped kNN and "
                "IVF/HNSW are approximate. Precision and batch kernels can change "
                "boundary decisions; float64 verification only filters candidates."
            ),
        )

    @property
    def n_samples(self) -> int:
        return int(self.state.indexed_points.shape[0])

    def query_radius(
        self,
        point_indices: Sequence[int],
        eps: float,
    ) -> list[np.ndarray]:
        indices = normalize_point_indices(point_indices, self.n_samples)
        result = []
        batch_size = self.state.config.query_batch_size
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            if len(batch) == 1:
                result.append(query_faiss_range(self.state, int(batch[0]), eps))
            else:
                result.extend(_query_faiss_batch(self.state, batch, eps))
        return result

    def distances_to_all(self, point_index: int) -> np.ndarray:
        if not self.metadata.supports_distances_to_all:
            raise NotImplementedError(
                "distances_to_all is supported only by an exact FAISS Flat backend."
            )
        if self.metric == "euclidean":
            return euclidean_distances_to_all(
                self.state.indexed_points,
                self.state.index,
                int(point_index),
            )
        return cosine_distances_to_all(
            self.state.indexed_points,
            self.state.index,
            int(point_index),
        )


class FaissFlatBackend(FaissRangeBackend):
    """Convenience exact-search FAISS Flat backend."""

    def __init__(
        self,
        x: np.ndarray,
        metric: FaissMetric = "euclidean",
        *,
        device: str = "cpu",
        exact_verify: bool = False,
    ) -> None:
        super().__init__(
            x,
            metric,
            FaissConfig(
                factory="Flat",
                device=device,  # type: ignore[arg-type]
                exact_verify=exact_verify,
            ),
        )


class FaissIVFBackend(FaissRangeBackend):
    """Convenience FAISS IVF-Flat approximate backend."""

    def __init__(
        self,
        x: np.ndarray,
        metric: FaissMetric = "euclidean",
        *,
        nlist: int = 256,
        nprobe: int = 16,
        candidate_k: int | None = None,
        device: str = "cpu",
        exact_verify: bool = False,
    ) -> None:
        query_mode = "auto" if candidate_k is None else "knn"
        super().__init__(
            x,
            metric,
            FaissConfig(
                factory=f"IVF{int(nlist)},Flat",
                search_params={"nprobe": int(nprobe)},
                candidate_k=candidate_k,
                query_mode=query_mode,
                device=device,  # type: ignore[arg-type]
                exact_verify=exact_verify,
            ),
        )


class FaissHNSWBackend(FaissRangeBackend):
    """Convenience FAISS HNSW approximate backend."""

    def __init__(
        self,
        x: np.ndarray,
        metric: FaissMetric = "euclidean",
        *,
        m: int = 32,
        ef_search: int = 128,
        ef_construction: int = 200,
        candidate_k: int = 1024,
        device: str = "cpu",
        exact_verify: bool = True,
    ) -> None:
        super().__init__(
            x,
            metric,
            FaissConfig(
                factory=f"HNSW{int(m)}",
                construction_params={"efConstruction": int(ef_construction)},
                search_params={"efSearch": int(ef_search)},
                query_mode="knn",
                candidate_k=int(candidate_k),
                device=device,  # type: ignore[arg-type]
                exact_verify=exact_verify,
            ),
        )
