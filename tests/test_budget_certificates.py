"""Exact small-instance comparisons, graph edge cases, resource caps."""

from __future__ import annotations
from itertools import combinations

import networkx as nx
import numpy as np
import pytest
from fast_ballmapper.budget_certificates import certify_connectivity
from fast_ballmapper.robustness import check_disconnection_witness


def brute(cover, b):
    from itertools import product

    m = len(cover)
    edge_lists = [[(i, int(x)) for x in set(row)] for i, row in enumerate(cover)]
    options = []
    for i, row in enumerate(edge_lists):
        options.append(
            [s for k in range(min(b[i], len(row)) + 1) for s in combinations(row, k)]
        )
    g = nx.Graph()
    g.add_nodes_from(range(m))
    byx = {}
    for i, row in enumerate(cover):
        for x in set(row):
            byx.setdefault(x, []).append(i)
    for labs in byx.values():
        g.add_edges_from(combinations(labs, 2))
    old = nx.number_connected_components(g)
    for sets in product(*options):
        gone = set().union(*map(set, sets))
        h = nx.Graph()
        h.add_nodes_from(range(m))
        rows = {}
        for i, row in enumerate(cover):
            for x in set(row):
                if (i, x) not in gone:
                    rows.setdefault(x, []).append(i)
        for labs in rows.values():
            h.add_edges_from(combinations(labs, 2))
        if nx.number_connected_components(h) > old:
            return False
    return True


@pytest.mark.parametrize("seed", range(70))
def test_random_against_brute(seed):
    rng = np.random.default_rng(seed + 101)
    n = 5
    m = int(rng.integers(2, 5))
    cover = [
        sorted(set(rng.integers(0, n, size=int(rng.integers(1, 4))).tolist()))
        for _ in range(m)
    ]
    b = [int(rng.integers(0, 2)) for _ in range(m)]
    truth = brute(cover, b)
    for mode in ("auto", "milp"):
        result = certify_connectivity(cover, b, n_samples=n, method=mode)
        assert result.robust == truth, (seed, mode, cover, b, result)
        if not truth:
            assert check_disconnection_witness(cover, n, b, result.deletions)


def test_dp_agreement_connected():
    cover = [[0, 1, 4], [1, 2, 5], [0, 2, 3]]
    for b in ([0, 0, 0], [1, 1, 1], [2, 2, 2]):
        expected = certify_connectivity(cover, b, n_samples=6, method="milp").robust
        assert (
            certify_connectivity(cover, b, n_samples=6, method="dp").robust == expected
        )


def test_resource_limit_inconclusive():
    cover = [[0, 2], [0, 1], [1, 2]]
    a = certify_connectivity(cover, [1, 1, 1], n_samples=3, method="dp", max_states=1)
    assert a.robust is None


def test_invalid_budget_and_empty():
    assert certify_connectivity([], []).robust is True
    assert certify_connectivity([[], []], [0, 0]).robust is True
    with pytest.raises(ValueError):
        certify_connectivity([[0]], [-1])
    with pytest.raises(ValueError):
        certify_connectivity([[0]], [1], method="other")
    with pytest.raises(ValueError):
        certify_connectivity([[0]], [1], n_samples=0)


def test_reject_boolean_indices():
    for method in ("auto", "milp", "dp"):
        with pytest.raises(ValueError):
            certify_connectivity([[True], [0]], [1, 1], n_samples=2, method=method)
