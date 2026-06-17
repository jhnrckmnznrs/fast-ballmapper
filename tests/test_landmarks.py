import numpy as np
import pytest

from fast_ballmapper import compute_landmarks, compute_landmarks_fps


def _all_points_are_covered(cover, n_samples):
    covered = set()
    for point_indices in cover:
        covered.update(map(int, point_indices))
    return covered == set(range(n_samples))


def test_compute_landmarks_builds_a_cover():
    x = np.array([[0.0], [0.1], [1.0], [1.1]])

    landmarks, cover = compute_landmarks(x, eps=0.11)

    assert landmarks == [0, 2]
    assert _all_points_are_covered(cover, len(x))
    assert [set(map(int, points)) for points in cover] == [{0, 1}, {2, 3}]


def test_compute_landmarks_accepts_snake_case_backend_name():
    x = np.array([[0.0], [1.0]])
    landmarks, _ = compute_landmarks(x, eps=0.1, method="ball_tree")
    assert landmarks == [0, 1]


def test_compute_landmarks_fps_is_deterministic_and_forms_epsilon_net():
    x = np.array([[1.0], [0.0], [0.4], [0.8]])
    eps = 0.41

    landmarks, cover = compute_landmarks_fps(x, eps=eps)

    assert landmarks == [1, 0]
    assert _all_points_are_covered(cover, len(x))
    landmark_points = x[landmarks]
    pairwise_distance = abs(float(landmark_points[0, 0] - landmark_points[1, 0]))
    assert pairwise_distance > eps


def test_metric_kwargs_are_forwarded_to_ball_tree():
    x = np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0]])
    landmarks, cover = compute_landmarks_fps(
        x,
        eps=0.3,
        metric="minkowski",
        metric_kwargs={"p": 1},
    )
    assert landmarks
    assert _all_points_are_covered(cover, len(x))


def test_empty_point_cloud_returns_empty_results():
    x = np.empty((0, 2))
    assert compute_landmarks(x, eps=1.0) == ([], [])
    assert compute_landmarks_fps(x, eps=1.0) == ([], [])


@pytest.mark.parametrize("eps", [-1.0, np.inf, np.nan])
def test_invalid_eps_is_rejected(eps):
    with pytest.raises(ValueError):
        compute_landmarks(np.ones((2, 1)), eps=eps)


def test_invalid_start_index_is_rejected():
    with pytest.raises(IndexError):
        compute_landmarks_fps(np.ones((2, 1)), eps=1.0, start_index=2)


def test_faiss_metric_kwargs_are_rejected_before_import():
    with pytest.raises(ValueError, match="metric_kwargs"):
        compute_landmarks(
            np.ones((2, 1)),
            eps=1.0,
            method="faiss",
            metric_kwargs={"p": 2},
        )
