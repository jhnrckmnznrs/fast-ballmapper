import numpy as np
import pytest

from fast_ballmapper import build_cover, compute_landmarks, compute_landmarks_fps


def _as_lists(cover):
    return [list(map(int, members)) for members in cover]


def test_brute_force_matches_ball_tree_for_euclidean_fixed_cover():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(50, 4))
    landmarks = [0, 7, 21]
    eps = 1.5

    brute = build_cover(x, landmarks, eps, method="brute_force")
    tree = build_cover(x, landmarks, eps, method="ball_tree")

    assert _as_lists(brute) == _as_lists(tree)


def test_brute_force_cosine_matches_direct_float64_computation():
    rng = np.random.default_rng(321)
    x = rng.normal(size=(40, 5))
    landmarks = [1, 9, 30]
    eps = 0.35

    brute = build_cover(
        x,
        landmarks,
        eps,
        method="brute_force",
        metric="cosine",
    )

    normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
    expected = []
    radius = np.nextafter(np.float64(eps), np.float64(np.inf))
    for landmark in landmarks:
        distances = np.maximum(1.0 - normalized @ normalized[landmark], 0.0)
        expected.append(np.flatnonzero(distances < radius))

    assert _as_lists(brute) == _as_lists(expected)


def test_brute_force_greedy_landmarks_match_ball_tree():
    x = np.array([[0.0], [0.1], [0.2], [1.0], [1.1]])

    brute_landmarks, brute_cover = compute_landmarks(
        x,
        eps=0.15,
        method="brute_force",
    )
    tree_landmarks, tree_cover = compute_landmarks(
        x,
        eps=0.15,
        method="ball_tree",
    )

    assert brute_landmarks == tree_landmarks
    assert _as_lists(brute_cover) == _as_lists(tree_cover)


def test_brute_force_fps_matches_ball_tree():
    x = np.array([[0.0], [0.2], [0.8], [1.0], [1.6]])

    brute_landmarks, brute_cover = compute_landmarks_fps(
        x,
        eps=0.45,
        method="brute_force",
    )
    tree_landmarks, tree_cover = compute_landmarks_fps(
        x,
        eps=0.45,
        method="ball_tree",
    )

    assert brute_landmarks == tree_landmarks
    assert _as_lists(brute_cover) == _as_lists(tree_cover)


def test_brute_force_uses_closed_ball_nextafter_policy():
    eps = 1.0
    x = np.array([[0.0], [eps], [np.nextafter(eps, np.inf)]])

    cover = build_cover(
        x,
        landmarks=[0],
        eps=eps,
        method="brute_force",
    )

    assert _as_lists(cover) == [[0, 1]]


def test_brute_force_rejects_unsupported_metric():
    x = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="euclidean.*cosine"):
        build_cover(
            x,
            landmarks=[0],
            eps=1.0,
            method="brute_force",
            metric="manhattan",
        )
