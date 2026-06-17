import networkx as nx
import numpy as np

from fast_ballmapper import build_mapper, compute_edge_overlaps


def test_build_mapper_connects_exactly_overlapping_covers():
    cover = [np.array([0, 1]), np.array([1, 2]), np.array([3])]

    graph = build_mapper(cover)

    assert set(graph.nodes()) == {0, 1, 2}
    assert set(graph.edges()) == {(0, 1)}


def test_compute_edge_overlaps_counts_shared_points():
    cover = [np.array([0, 1, 2]), np.array([1, 2, 3])]
    graph = nx.Graph([(0, 1)])

    assert compute_edge_overlaps(cover, graph) == {(0, 1): 2}


def test_duplicate_point_indices_do_not_create_self_loops():
    cover = [np.array([0, 0]), np.array([0])]
    graph = build_mapper(cover)
    assert set(graph.edges()) == {(0, 1)}
    assert nx.number_of_selfloops(graph) == 0
