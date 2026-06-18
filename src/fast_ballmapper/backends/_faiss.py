"""Optional FAISS backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from fast_ballmapper._validation import validate_cosine_points
from fast_ballmapper.faiss import FaissConfig

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
    """Move a CPU FAISS index to GPU if possible.

    Returns
    -------
    index:
        The GPU index if conversion succeeds; otherwise the original CPU
        index when fallback is enabled.
    device:
        ``"gpu"`` or ``"cpu"``.
    resources:
        The FAISS GPU resources object, or None.
    error:
        A string describing why CPU fallback was used, or None.
    """
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
    original_points = np.asarray(x)

    indexed_points = np.ascontiguousarray(x, dtype=np.float32)

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

    # Some parameters, such as nprobe, may need to be set again after CPU-to-GPU
    # conversion because the converted index is a new FAISS object.
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

    # A landmark always belongs to its own exact ball.
    indices = np.append(indices, point_index)
    return np.unique(indices.astype(np.intp, copy=False))


def _native_range_query(
    backend: FaissBackend,
    point_index: int,
    eps: float,
) -> np.ndarray:
    query = backend.indexed_points[point_index : point_index + 1]

    if backend.metric == "euclidean":
        radius = np.nextafter(
            np.float32(eps**2),
            np.float32(np.inf),
        )
    else:
        radius = np.nextafter(
            np.float32(1.0 - eps),
            np.float32(-np.inf),
        )

    limits, _, indices = backend.index.range_search(query, radius)
    return indices[limits[0] : limits[1]]


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

    if backend.metric == "euclidean":
        threshold = np.nextafter(
            np.float32(eps**2),
            np.float32(np.inf),
        )
        inside = distances < threshold
    else:
        threshold = np.nextafter(
            np.float32(1.0 - eps),
            np.float32(-np.inf),
        )
        inside = distances > threshold

    return indices[inside]


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

    return candidate_indices[distances < eps]


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
