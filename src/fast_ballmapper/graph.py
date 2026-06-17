"""Ball Mapper graph construction and edge statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import networkx as nx
import numpy as np


def build_mapper(cover: Sequence[np.ndarray]) -> nx.Graph:
    """Build a graph whose nodes are balls and whose edges represent overlap."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cover)))
    point_to_covers: dict[int, set[int]] = defaultdict(set)

    for cover_id, point_indices in enumerate(cover):
        for point_index in point_indices:
            point_to_covers[int(point_index)].add(cover_id)

    for cover_id_set in point_to_covers.values():
        cover_ids = sorted(cover_id_set)
        for left_position, left_id in enumerate(cover_ids):
            for right_id in cover_ids[left_position + 1 :]:
                graph.add_edge(left_id, right_id)

    return graph


def compute_edge_overlaps(
    cover: Sequence[np.ndarray],
    graph: nx.Graph,
) -> dict[tuple[int, int], int]:
    """Return the number of shared data points for every graph edge."""
    return {
        (int(left), int(right)): int(len(np.intersect1d(cover[left], cover[right])))
        for left, right in graph.edges()
    }
