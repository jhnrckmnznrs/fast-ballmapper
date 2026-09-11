"""Ball Mapper graph construction and edge statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import networkx as nx
import numpy as np


def build_mapper(cover: Sequence[np.ndarray]) -> nx.Graph:
    """Build the Ball Mapper graph and attach edge witness multiplicities.

    An edge joins two cover elements when at least one data point belongs to
    both.  The edge attribute ``witness_count`` stores the number of distinct
    data points witnessing that overlap.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cover)))
    point_to_covers: dict[int, set[int]] = defaultdict(set)

    for cover_id, point_indices in enumerate(cover):
        for point_index in point_indices:
            point_to_covers[int(point_index)].add(cover_id)

    witness_counts: dict[tuple[int, int], int] = defaultdict(int)
    for cover_id_set in point_to_covers.values():
        cover_ids = sorted(cover_id_set)
        for left_position, left_id in enumerate(cover_ids):
            for right_id in cover_ids[left_position + 1 :]:
                witness_counts[(left_id, right_id)] += 1

    for (left_id, right_id), count in witness_counts.items():
        graph.add_edge(left_id, right_id, witness_count=int(count))

    return graph


def compute_edge_overlaps(
    cover: Sequence[np.ndarray],
    graph: nx.Graph,
) -> dict[tuple[int, int], int]:
    """Return the number of shared distinct data points for every graph edge.

    For graphs returned by :func:`build_mapper`, this is the same quantity
    stored on each edge as ``witness_count``.  The function recomputes counts
    from ``cover`` so it also works for user-supplied graphs.
    """
    cover_sets = [set(map(int, point_indices)) for point_indices in cover]
    return {
        (int(left), int(right)): len(cover_sets[left].intersection(cover_sets[right]))
        for left, right in graph.edges()
    }
