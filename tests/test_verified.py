from itertools import combinations, product

import networkx as nx
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from fast_ballmapper import (
    BackendMetadata,
    build_cover_blocked,
    build_mapper_sparse,
    certify_component_preservation,
    compute_landmarks,
    compute_landmarks_verified,
    cover_statistics,
)


def independent_reference(x, eps):
    distances = cdist(x, x)
    covered = set()
    landmarks, cover = [], []
    for i in range(len(x)):
        if i not in covered:
            landmarks.append(i)
            row = set(np.flatnonzero(distances[i] <= eps))
            cover.append(row)
            covered.update(row)
    return landmarks, cover


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("provider", ["empty", "full", "adversarial"])
def test_verified_selection_against_independent_distances(seed, provider):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(55, 3))
    x[-2:] = x[:2]
    eps = 0.9
    expected_landmarks, expected_cover = independent_reference(x, eps)
    providers = {
        "empty": lambda i, r: [],
        "full": lambda i, r: range(len(x)),
        # Includes duplicates and false positives while omitting most true IDs.
        "adversarial": lambda i, r: [0, 0, (i + 7) % len(x)],
    }
    partial = compute_landmarks_verified(x, eps, providers[provider], block_size=4)
    assert partial.landmarks == expected_landmarks
    assert not partial.memberships_complete
    assert set().union(*map(set, partial.cover)) == set(range(len(x)))
    assert all(set(a) <= b for a, b in zip(partial.cover, expected_cover, strict=True))
    complete = compute_landmarks_verified(
        x, eps, providers[provider], complete=True, block_size=7
    )
    assert list(map(set, complete.cover)) == expected_cover
    assert complete.memberships_complete
    assert complete.completion_distance_evaluations == len(x) * len(expected_landmarks)


def test_verified_closed_boundary_duplicates_and_zero_radius():
    x = np.array([[0.0], [1.0], [np.nextafter(1.0, np.inf)], [0.0]])
    result = compute_landmarks_verified(x, 1, complete=True, block_size=1)
    assert result.landmarks == [0, 2]
    assert set(result.cover[0]) == {0, 1, 3}
    assert list(map(set, build_cover_blocked(x, [0], 0))) == [{0, 3}]
    empty = compute_landmarks_verified(np.empty((0, 3)), 0, complete=True)
    assert empty.landmarks == empty.cover == []


@pytest.mark.parametrize(
    "ids,exception",
    [([1.2], TypeError), ([True], TypeError), ([-1], IndexError), ([2], IndexError)],
)
def test_verified_rejects_invalid_candidate_ids(ids, exception):
    with pytest.raises(exception):
        compute_landmarks_verified(np.array([[0.0], [1.0]]), 0.5, lambda i, r: ids)


def test_verified_rejects_nonfinite_computed_distances():
    with pytest.raises(ValueError, match="Nonfinite"):
        compute_landmarks_verified(np.array([[-1e308], [1e308]]), 1)


def test_generic_greedy_progress_when_custom_oracle_omits_self():
    class EmptyBackend:
        n_samples = 3
        metadata = BackendMetadata("empty", False, "euclidean", "float64", "cpu")

        def query_radius(self, indices, eps):
            return [np.array([], dtype=int) for _ in indices]

    landmarks, cover = compute_landmarks(
        np.arange(3.0).reshape(-1, 1), 0.1, backend=EmptyBackend()
    )
    assert landmarks == [0, 1, 2]
    assert list(map(set, cover)) == [{0}, {1}, {2}]


@pytest.mark.parametrize("block_size", [1, 2, 16])
def test_sparse_counts_against_set_intersections_and_wide_counts(block_size):
    cover = [np.r_[np.arange(300), 0, 0], np.arange(300), np.arange(299, 310), []]
    graph = build_mapper_sparse(cover, 310, row_block_size=block_size)
    expected = {
        (i, j): len(set(a) & set(b))
        for (i, a), (j, b) in combinations(enumerate(cover), 2)
        if set(a) & set(b)
    }
    assert {
        (i, j): a["witness_count"] for i, j, a in graph.edges(data=True)
    } == expected
    assert graph[0][1]["witness_count"] == 300
    assert set(graph) == {0, 1, 2, 3}
    assert cover_statistics(cover, 310) == {"m": 4, "S": 611, "T": 302}
    assert len(build_mapper_sparse([], 0)) == 0
    with pytest.raises(ValueError):
        build_mapper_sparse(cover, 310, row_block_size=0)


def test_component_certificate_for_all_admissible_small_omissions():
    cover = [{0, 1, 2, 3}, {0, 1, 2, 3}, {2, 3, 4}, {5}]
    budgets = [1, 1, 0, 0]
    certificate = certify_component_preservation(cover, 6, budgets, row_block_size=1)
    assert certificate.certified
    assert certificate.exact_components == 2
    options = [
        [row - set(deleted) for k in range(b + 1) for deleted in combinations(row, k)]
        for row, b in zip(cover, budgets, strict=True)
    ]
    for candidate in product(*options):
        assert nx.number_connected_components(build_mapper_sparse(candidate, 6)) == 2
    inconclusive = certify_component_preservation([{0}, {0}], 1, [1, 1])
    assert not inconclusive.certified
    assert nx.number_connected_components(build_mapper_sparse([{0}, {0}], 1)) == 1


def test_small_observed_nerve_witness_maps_and_contiguity():
    # Check every nonempty simplex at every critical radius in both directions.
    # Observed nerve means a COMMON data witness, not graph clique completion.
    x = np.arange(8.0).reshape(-1, 1)
    distance = cdist(x, x)
    left, right = [0, 3, 7], [1, 5, 6]
    h = max(
        distance[left][:, right].min(axis=1).max(),
        distance[right][:, left].min(axis=1).max(),
    )
    for source, target in [(left, right), (right, left)]:
        forward = {i: target[np.argmin(distance[i, target])] for i in source}
        backward = {j: source[np.argmin(distance[j, source])] for j in target}
        for size in range(1, len(source) + 1):
            for simplex in combinations(source, size):
                for radius in range(8):
                    witnesses = np.flatnonzero(
                        np.all(distance[list(simplex)] <= radius, axis=0)
                    )
                    for witness in witnesses:
                        mapped = [forward[i] for i in simplex]
                        assert np.all(distance[mapped, witness] <= radius + h)
                        union = list(set(simplex) | {backward[j] for j in mapped})
                        assert np.all(distance[union, witness] <= radius + 2 * h)
