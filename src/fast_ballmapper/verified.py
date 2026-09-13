"""Verified Euclidean selection relative to direct float64 ``norm <= eps``.

Candidate sets accelerate marking only. An exhaustive current-landmark coverage
test makes every landmark decision independent of candidate omissions. Complete
memberships require a separate exhaustive pass; partial covers can lose edges.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from fast_ballmapper._validation import validate_point_cloud
from fast_ballmapper.backends._utils import normalize_point_indices

CandidateProvider = Callable[[int, float], Iterable[int]]


@dataclass
class VerifiedSelection:
    """Landmarks, verified memberships, and explicit distance-work counters."""

    landmarks: list[int]
    cover: list[np.ndarray]
    memberships_complete: bool
    coverage_distance_evaluations: int
    candidate_distance_evaluations: int
    rescued_observations: int
    completion_distance_evaluations: int


def _block_size(value: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError("block_size must be a positive integer.")
    return int(value)


def _points(x: np.ndarray, eps: float) -> np.ndarray:
    x = validate_point_cloud(x, eps)
    if np.iscomplexobj(x):
        raise TypeError("Verified Euclidean selection requires real coordinates.")
    return np.array(x, dtype=np.float64, order="C", copy=True)


def _inside(points: np.ndarray, query: np.ndarray, eps: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(points - query, axis=1)
    if not np.all(np.isfinite(distances)):
        raise ValueError("Nonfinite float64 distance; rescale the input coordinates.")
    return distances <= eps


def build_cover_blocked(
    x: np.ndarray, landmarks: Sequence[int], eps: float, *, block_size: int = 4096
) -> list[np.ndarray]:
    """Complete fixed-landmark balls with O(block_size * dimension) workspace.

    Output storage is additional. No landmark-by-observation distance matrix is
    allocated. Exactness is relative to the stated float64 numerical predicate.
    """
    points = _points(x, eps)
    block_size = _block_size(block_size)
    indices = normalize_point_indices(landmarks, len(points))
    if len(np.unique(indices)) != len(indices):
        raise ValueError("landmarks must be distinct.")
    cover = []
    for landmark in indices:
        chunks = []
        for start in range(0, len(points), block_size):
            block = points[start : start + block_size]
            chunks.append(start + np.flatnonzero(_inside(block, points[landmark], eps)))
        cover.append(np.concatenate(chunks) if chunks else np.empty(0, dtype=np.intp))
    return cover


def compute_landmarks_verified(
    x: np.ndarray,
    eps: float,
    candidates: CandidateProvider | None = None,
    *,
    complete: bool = False,
    block_size: int = 4096,
) -> VerifiedSelection:
    """Preserve the ordered exact greedy sequence with arbitrary candidates.

    ``candidates(i, eps)`` may return empty, incomplete, duplicate, or out-of-ball
    IDs. Invalid IDs are rejected. With no provider, only exact coverage tests
    are used. Each rescued point is assigned to a verified containing ball, so
    the partial memberships still cover the data. Use ``complete=True`` or
    ``build_cover_blocked`` before claiming equality of graphs or node colors.

    Coverage verification can require O(n*m) distances in the worst case; all
    candidate-search work is additional. No unconditional speedup is claimed.
    """
    points = _points(x, eps)
    block_size = _block_size(block_size)
    marked = np.zeros(len(points), dtype=bool)
    landmarks: list[int] = []
    memberships: list[set[int]] = []
    coverage_work = candidate_work = rescued = 0
    for i in range(len(points)):
        if marked[i]:
            continue
        owner = None
        for start in range(0, len(landmarks), block_size):
            ids = landmarks[start : start + block_size]
            coverage_work += len(ids)
            hits = np.flatnonzero(_inside(points[ids], points[i], eps))
            if len(hits):
                owner = start + int(hits[0])
                break
        if owner is not None:
            memberships[owner].add(i)
            marked[i] = True
            rescued += 1
            continue
        landmarks.append(i)
        members = {i}
        if candidates is not None:
            ids = np.unique(
                normalize_point_indices(list(candidates(i, eps)), len(points))
            )
            for start in range(0, len(ids), block_size):
                block = ids[start : start + block_size]
                candidate_work += len(block)
                members.update(map(int, block[_inside(points[block], points[i], eps)]))
        memberships.append(members)
        marked[list(members)] = True
    cover = [np.asarray(sorted(m), dtype=np.intp) for m in memberships]
    if complete:
        cover = build_cover_blocked(points, landmarks, eps, block_size=block_size)
    return VerifiedSelection(
        landmarks,
        cover,
        complete,
        coverage_work,
        candidate_work,
        rescued,
        len(landmarks) * len(points) if complete else 0,
    )
