import numpy as np
import pytest

from fast_ballmapper import FaissConfig, build_cover, compute_landmarks

pytest.importorskip("faiss")


def _as_sets(cover):
    return [set(map(int, point_indices)) for point_indices in cover]


def test_default_faiss_configuration_matches_explicit_flat_index():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 8))

    default_landmarks, default_cover = compute_landmarks(
        x,
        eps=2.0,
        method="faiss",
        metric="euclidean",
    )
    flat_landmarks, flat_cover = compute_landmarks(
        x,
        eps=2.0,
        method="faiss",
        metric="euclidean",
        faiss_config=FaissConfig(factory="Flat"),
    )

    assert default_landmarks == flat_landmarks
    assert _as_sets(default_cover) == _as_sets(flat_cover)


def test_fixed_landmark_flat_cover_matches_ball_tree():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(128, 6))
    landmarks = [0, 11, 37, 90]

    ball_tree_cover = build_cover(
        x,
        landmarks,
        eps=1.75,
        method="ball_tree",
        metric="euclidean",
    )
    faiss_cover = build_cover(
        x,
        landmarks,
        eps=1.75,
        method="faiss",
        metric="euclidean",
        faiss_config=FaissConfig(factory="Flat"),
    )

    assert _as_sets(faiss_cover) == _as_sets(ball_tree_cover)


def test_ivf_with_all_lists_probed_matches_flat_range_sets():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(256, 8)).astype(np.float32)
    landmarks = [0, 17, 91, 203]

    flat_cover = build_cover(
        x,
        landmarks,
        eps=2.25,
        method="faiss",
        metric="euclidean",
        faiss_config=FaissConfig(factory="Flat"),
    )
    ivf_cover = build_cover(
        x,
        landmarks,
        eps=2.25,
        method="faiss",
        metric="euclidean",
        faiss_config=FaissConfig(
            factory="IVF8,Flat",
            search_params={"nprobe": 8},
        ),
    )

    assert _as_sets(ivf_cover) == _as_sets(flat_cover)


def test_exact_verification_makes_every_euclidean_membership_valid():
    rng = np.random.default_rng(314)
    x = rng.normal(size=(256, 8))
    landmarks = [0, 25, 100]
    eps = 2.0

    cover = build_cover(
        x,
        landmarks,
        eps=eps,
        method="faiss",
        metric="euclidean",
        faiss_config=FaissConfig(
            factory="IVF8,Flat",
            search_params={"nprobe": 2},
            exact_verify=True,
        ),
    )

    for landmark, point_indices in zip(landmarks, cover, strict=True):
        distances = np.linalg.norm(x[point_indices] - x[landmark], axis=1)
        assert np.all(distances < eps)
        assert landmark in point_indices


def test_cosine_flat_cover_matches_direct_cosine_distance():
    rng = np.random.default_rng(2718)
    x = rng.normal(size=(80, 5))
    landmarks = [0, 7, 29]
    eps = 0.4

    cover = build_cover(
        x,
        landmarks,
        eps=eps,
        method="faiss",
        metric="cosine",
        faiss_config=FaissConfig(factory="Flat"),
    )

    normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
    expected = []
    for landmark in landmarks:
        distances = 1.0 - normalized @ normalized[landmark]
        expected.append(set(np.flatnonzero(distances < eps).tolist()))

    assert _as_sets(cover) == expected
