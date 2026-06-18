import numpy as np
import pytest

from fast_ballmapper import FaissConfig, build_cover


def _as_sets(cover):
    return [set(map(int, point_indices)) for point_indices in cover]


def test_build_cover_ball_tree_uses_fixed_landmarks():
    x = np.array([[0.0], [0.1], [1.0], [1.1]])

    cover = build_cover(
        x,
        landmarks=[0, 2],
        eps=0.11,
        method="ball_tree",
    )

    assert _as_sets(cover) == [{0, 1}, {2, 3}]


def test_build_cover_preserves_landmark_order():
    x = np.array([[0.0], [0.1], [1.0], [1.1]])

    cover = build_cover(
        x,
        landmarks=[2, 0],
        eps=0.11,
        method="ball_tree",
    )

    assert _as_sets(cover) == [{2, 3}, {0, 1}]


def test_build_cover_accepts_numpy_integer_indices():
    x = np.array([[0.0], [1.0]])

    cover = build_cover(
        x,
        landmarks=[np.int64(0), np.int32(1)],
        eps=0.1,
    )

    assert _as_sets(cover) == [{0}, {1}]


def test_build_cover_with_no_landmarks_returns_empty_cover():
    x = np.array([[0.0], [1.0]])

    assert build_cover(x, landmarks=[], eps=0.1) == []


@pytest.mark.parametrize("landmark", [0.0, "0", None])
def test_build_cover_rejects_non_integer_landmark_indices(landmark):
    x = np.array([[0.0], [1.0]])

    with pytest.raises(TypeError, match="integer row index"):
        build_cover(x, landmarks=[landmark], eps=0.1)


@pytest.mark.parametrize("landmark", [-1, 2])
def test_build_cover_rejects_out_of_bounds_landmarks(landmark):
    x = np.array([[0.0], [1.0]])

    with pytest.raises(IndexError, match="out of bounds"):
        build_cover(x, landmarks=[landmark], eps=0.1)


def test_build_cover_rejects_duplicate_landmarks():
    x = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="appears more than once"):
        build_cover(x, landmarks=[0, 0], eps=0.1)


def test_build_cover_rejects_faiss_config_for_ball_tree():
    x = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="method='faiss'"):
        build_cover(
            x,
            landmarks=[0],
            eps=0.1,
            method="ball_tree",
            faiss_config=FaissConfig(),
        )


def test_build_cover_rejects_metric_kwargs_for_faiss_before_import():
    x = np.array([[0.0], [1.0]])

    with pytest.raises(ValueError, match="metric_kwargs"):
        build_cover(
            x,
            landmarks=[0],
            eps=0.1,
            method="faiss",
            metric_kwargs={"p": 2},
        )
