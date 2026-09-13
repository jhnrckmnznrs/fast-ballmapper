"""Blocked integer witness products and a sufficient connectivity certificate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from fast_ballmapper.backends._utils import normalize_point_indices
from fast_ballmapper.verified import _block_size


def _incidence(cover: Sequence[np.ndarray], n_samples: int) -> csr_matrix:
    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, (int, np.integer))
        or n_samples < 0
    ):
        raise ValueError("n_samples must be a non-negative integer.")
    rows = [np.unique(normalize_point_indices(row, n_samples)) for row in cover]
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(row) for row in rows], dtype=np.int64)
    ids = np.concatenate(rows) if rows else np.empty(0, dtype=np.intp)
    return csr_matrix(
        (np.ones(len(ids), dtype=np.int64), ids, indptr),
        shape=(len(rows), n_samples),
        dtype=np.int64,
    )


def cover_statistics(cover: Sequence[np.ndarray], n_samples: int) -> dict[str, int]:
    """Return m, S and T using distinct memberships and Python integer sums."""
    incidence = _incidence(cover, n_samples)
    degrees = np.asarray(incidence.sum(axis=0)).ravel()
    return {
        "m": len(cover),
        "S": int(incidence.nnz),
        "T": sum(int(a) * (int(a) - 1) // 2 for a in degrees),
    }


def build_mapper_sparse(
    cover: Sequence[np.ndarray], n_samples: int, *, row_block_size: int = 256
) -> nx.Graph:
    """Compute witness counts in int64 sparse row blocks, retaining all vertices.

    Each intermediate product has at most ``row_block_size * m`` entries.
    Incidence and final graph storage are additional; dense output still costs
    O(m**2). Duplicate observation IDs within a ball count once.
    """
    block_size = _block_size(row_block_size)
    incidence = _incidence(cover, n_samples)
    transpose = incidence.T.tocsr()
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cover)))
    for start in range(0, len(cover), block_size):
        product = (incidence[start : start + block_size] @ transpose).tocoo()
        keep = start + product.row < product.col
        graph.add_edges_from(
            (int(start + row), int(col), {"witness_count": int(count)})
            for row, col, count in zip(
                product.row[keep], product.col[keep], product.data[keep], strict=True
            )
        )
    return graph


@dataclass(frozen=True)
class ComponentCertificate:
    """A failed sufficient certificate is inconclusive, not evidence of damage."""

    certified: bool
    exact_components: int
    protected_components: int
    protected_edges: int


def certify_component_preservation(
    exact_cover: Sequence[np.ndarray],
    n_samples: int,
    budgets: Sequence[int],
    *,
    row_block_size: int = 256,
) -> ComponentCertificate:
    """Certify all conservative omissions within per-ball missing-point budgets.

    Requires fixed landmarks, no added memberships, and retention of all graph
    vertices. A protected edge has exact witness_count > budget[i] + budget[j].
    This checks a sufficient condition using an exact reference cover; it does
    not establish that a candidate oracle satisfies the supplied budgets.
    """
    if len(budgets) != len(exact_cover):
        raise ValueError("Provide one non-negative integer budget per ball.")
    if any(
        isinstance(b, (bool, np.bool_))
        or not isinstance(b, (int, np.integer))
        or b < 0
        or b > n_samples
        for b in budgets
    ):
        raise ValueError("Budgets must be integers in [0, n_samples].")
    graph = build_mapper_sparse(exact_cover, n_samples, row_block_size=row_block_size)
    protected = nx.Graph()
    protected.add_nodes_from(graph)
    protected.add_edges_from(
        (i, j)
        for i, j, attrs in graph.edges(data=True)
        if attrs["witness_count"] > int(budgets[i]) + int(budgets[j])
    )
    exact_count = nx.number_connected_components(graph)
    protected_count = nx.number_connected_components(protected)
    return ComponentCertificate(
        exact_count == protected_count,
        exact_count,
        protected_count,
        protected.number_of_edges(),
    )
