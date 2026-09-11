import numpy as np
import pytest

from fast_ballmapper import (
    BruteForceBackend,
    CKDTreeBackend,
    RangeQueryBackend,
    build_cover,
    compute_landmarks,
    compute_landmarks_fps,
    make_backend,
)


def _sets(cover):
    return [set(map(int, members)) for members in cover]


def test_factory_returns_runtime_protocol_backend():
    x = np.array([[0.0], [0.5], [1.0]])
    backend = make_backend(x, "brute_force")

    assert isinstance(backend, RangeQueryBackend)
    assert backend.metadata.name == "brute_force"
    assert backend.metadata.is_exact is True
    assert backend.metadata.dtype == "float64"
    assert backend.metadata.device == "cpu"


def test_prebuilt_backend_can_be_reused_for_cover_and_landmarks():
    x = np.array([[0.0], [0.1], [1.0], [1.1]])
    backend = BruteForceBackend(x)

    cover = build_cover(x, [0, 2], 0.11, backend=backend)
    landmarks, greedy_cover = compute_landmarks(x, 0.11, backend=backend)

    assert _sets(cover) == [{0, 1}, {2, 3}]
    assert landmarks == [0, 2]
    assert _sets(greedy_cover) == [{0, 1}, {2, 3}]


def test_prebuilt_backend_rejects_different_dataset_size():
    backend = BruteForceBackend(np.array([[0.0], [1.0]]))
    with pytest.raises(ValueError, match="different number of observations"):
        build_cover(np.array([[0.0], [1.0], [2.0]]), [0], 0.1, backend=backend)


def test_ckdtree_matches_reference_closed_ball_boundary():
    eps = 1.0
    x = np.array([[0.0], [eps], [np.nextafter(eps, np.inf)], [2.0]])
    brute = build_cover(x, [0], eps, method="brute_force")
    kd = build_cover(x, [0], eps, method="ckdtree")

    assert _sets(kd) == _sets(brute) == [{0, 1}]


def test_ckdtree_matches_brute_force_greedy_landmarks():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100, 4))
    eps = 1.2

    brute_landmarks, brute_cover = compute_landmarks(
        x, eps, method="brute_force"
    )
    kd_landmarks, kd_cover = compute_landmarks(x, eps, method="ckdtree")

    assert kd_landmarks == brute_landmarks
    assert _sets(kd_cover) == _sets(brute_cover)


def test_ckdtree_matches_brute_force_fps():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 3))
    eps = 1.0

    brute_landmarks, brute_cover = compute_landmarks_fps(
        x, eps, method="brute_force"
    )
    kd_landmarks, kd_cover = compute_landmarks_fps(x, eps, method="ckdtree")

    assert kd_landmarks == brute_landmarks
    assert _sets(kd_cover) == _sets(brute_cover)


def test_ckdtree_metadata_is_exact_cpu_float64():
    backend = CKDTreeBackend(np.array([[0.0], [1.0]]))
    assert backend.metadata.is_exact is True
    assert backend.metadata.metric == "euclidean"
    assert backend.metadata.dtype == "float64"
    assert backend.metadata.device == "cpu"
    assert backend.metadata.supports_distances_to_all is True


def test_ckdtree_rejects_cosine_factory_request():
    with pytest.raises(ValueError, match="only euclidean"):
        make_backend(np.eye(2), "ckdtree", metric="cosine")
