from types import SimpleNamespace

import numpy as np
import pytest

from fast_ballmapper import compute_landmarks
from fast_ballmapper.backends import _faiss as faiss_backend
from fast_ballmapper.backends._faiss import FaissBackend
from fast_ballmapper.faiss import FaissConfig


def _backend(
    points,
    *,
    metric="euclidean",
    config=None,
    index=None,
):
    original_points = np.asarray(points, dtype=float)
    indexed_points = np.ascontiguousarray(original_points, dtype=np.float32)

    if metric == "cosine":
        verification_points = faiss_backend._normalize_rows_float64(original_points)
    else:
        verification_points = original_points.astype(np.float64, copy=True)

    return FaissBackend(
        original_points=original_points,
        indexed_points=indexed_points,
        verification_points=verification_points,
        index=index,
        metric=metric,
        config=config or FaissConfig(),
    )


def test_faiss_config_defaults_to_exact_flat_search():
    config = FaissConfig()

    assert config.factory == "Flat"
    assert config.construction_params == {}
    assert config.search_params == {}
    assert config.train_size is None
    assert config.query_mode == "auto"
    assert config.candidate_k is None
    assert config.exact_verify is False


def test_training_subset_selection_is_reproducible():
    points = np.arange(80, dtype=np.float32).reshape(20, 4)

    first = faiss_backend._select_training_points(points, train_size=7, seed=42)
    second = faiss_backend._select_training_points(points, train_size=7, seed=42)
    third = faiss_backend._select_training_points(points, train_size=7, seed=7)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)
    assert first.flags.c_contiguous


def test_training_subset_uses_all_points_when_train_size_is_none():
    points = np.arange(20, dtype=np.float32).reshape(5, 4)

    selected = faiss_backend._select_training_points(points, None, seed=42)

    assert selected is points


def test_training_subset_rejects_non_positive_size():
    points = np.arange(20, dtype=np.float32).reshape(5, 4)

    with pytest.raises(ValueError, match="positive"):
        faiss_backend._select_training_points(points, train_size=0, seed=42)


def test_float64_cosine_normalization_produces_unit_rows():
    points = np.array([[3.0, 4.0], [1.0, 1.0]])

    normalized = faiss_backend._normalize_rows_float64(points)

    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)
    np.testing.assert_array_equal(points, np.array([[3.0, 4.0], [1.0, 1.0]]))


def test_float64_cosine_normalization_rejects_zero_vectors():
    points = np.array([[1.0, 0.0], [0.0, 0.0]])

    with pytest.raises(ValueError, match="zero vectors"):
        faiss_backend._normalize_rows_float64(points)


def test_exact_euclidean_verification_removes_out_of_range_candidates():
    backend = _backend([[0.0], [0.1], [1.0]])

    verified = faiss_backend._verify_exact_membership(
        backend,
        point_index=0,
        candidate_indices=np.array([0, 1, 2]),
        eps=0.11,
    )

    np.testing.assert_array_equal(verified, np.array([0, 1]))


def test_exact_cosine_verification_removes_out_of_range_candidates():
    backend = _backend(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        metric="cosine",
    )

    verified = faiss_backend._verify_exact_membership(
        backend,
        point_index=0,
        candidate_indices=np.array([0, 1, 2]),
        eps=0.4,
    )

    np.testing.assert_array_equal(verified, np.array([0, 1]))


def test_query_faiss_range_always_contains_the_landmark(monkeypatch):
    backend = _backend([[0.0], [0.1], [1.0]])

    monkeypatch.setattr(
        faiss_backend,
        "_native_range_query",
        lambda backend, point_index, eps: np.array([1]),
    )

    result = faiss_backend.query_faiss_range(
        backend,
        point_index=2,
        eps=0.2,
    )

    np.testing.assert_array_equal(result, np.array([1, 2]))


def test_query_faiss_range_applies_exact_verification(monkeypatch):
    config = FaissConfig(exact_verify=True)
    backend = _backend([[0.0], [0.1], [1.0]], config=config)

    monkeypatch.setattr(
        faiss_backend,
        "_native_range_query",
        lambda backend, point_index, eps: np.array([0, 1, 2]),
    )

    result = faiss_backend.query_faiss_range(
        backend,
        point_index=0,
        eps=0.11,
    )

    np.testing.assert_array_equal(result, np.array([0, 1]))


class _FallbackIndex:
    def range_search(self, query, radius):
        raise RuntimeError("range search is unsupported")

    def search(self, query, k):
        return (
            np.array([[0.0, 0.01, 4.0]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )


def test_auto_mode_falls_back_to_knn_candidates():
    config = FaissConfig(query_mode="auto", candidate_k=3)
    backend = _backend(
        [[0.0], [0.1], [2.0]],
        config=config,
        index=_FallbackIndex(),
    )

    result = faiss_backend.query_faiss_range(
        backend,
        point_index=0,
        eps=0.5,
    )

    np.testing.assert_array_equal(result, np.array([0, 1]))


def test_range_mode_does_not_silently_fall_back():
    config = FaissConfig(query_mode="range", candidate_k=3)
    backend = _backend(
        [[0.0], [0.1], [2.0]],
        config=config,
        index=_FallbackIndex(),
    )

    with pytest.raises(RuntimeError, match="does not support native range search"):
        faiss_backend.query_faiss_range(
            backend,
            point_index=0,
            eps=0.5,
        )


def test_knn_mode_requires_candidate_k():
    config = FaissConfig(query_mode="knn")
    backend = _backend(
        [[0.0], [1.0]],
        config=config,
        index=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="candidate_k"):
        faiss_backend.query_faiss_range(
            backend,
            point_index=0,
            eps=0.5,
        )


def test_knn_mode_rejects_non_positive_candidate_k():
    config = FaissConfig(query_mode="knn", candidate_k=0)
    backend = _backend(
        [[0.0], [1.0]],
        config=config,
        index=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="positive"):
        faiss_backend.query_faiss_range(
            backend,
            point_index=0,
            eps=0.5,
        )


def test_gpu_support_detection_returns_boolean():
    assert isinstance(faiss_backend._faiss_has_gpu_support(), bool)


def test_available_gpu_count_is_nonnegative_integer():
    assert isinstance(faiss_backend._available_gpu_count(), int)
    assert faiss_backend._available_gpu_count() >= 0


def test_cpu_config_does_not_try_gpu():
    config = FaissConfig(device="cpu")
    assert not faiss_backend._should_try_gpu(config)


def test_gpu_auto_only_tries_when_gpu_available():
    config = FaissConfig(device="auto")
    assert faiss_backend._should_try_gpu(config) == (
        faiss_backend._available_gpu_count() > 0
    )


def test_flat_index_can_use_gpu_or_fallback_to_cpu():
    pytest.importorskip("faiss")

    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )

    config = FaissConfig(
        factory="Flat",
        device="gpu",
        gpu_fallback_to_cpu=True,
    )

    landmarks, cover = compute_landmarks(
        x,
        eps=0.25,
        method="faiss",
        metric="euclidean",
        faiss_config=config,
    )

    assert landmarks == [0, 2]
    assert {int(i) for i in cover[0]} == {0, 1}
    assert {int(i) for i in cover[1]} == {2}


@pytest.mark.skipif(
    not faiss_backend._faiss_has_gpu_support()
    or faiss_backend._available_gpu_count() == 0,
    reason="GPU-enabled FAISS with a visible GPU is required.",
)
def test_flat_index_uses_gpu_when_available():
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )

    backend = faiss_backend.create_faiss_backend(
        x,
        "euclidean",
        FaissConfig(
            factory="Flat",
            device="gpu",
            gpu_fallback_to_cpu=False,
        ),
    )

    assert backend.device == "gpu"
    assert backend.gpu_resources is not None
    assert backend.gpu_error is None
