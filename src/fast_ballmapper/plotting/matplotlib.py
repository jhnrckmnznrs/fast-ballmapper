"""Static Ball Mapper plotting with Matplotlib."""

from __future__ import annotations

import networkx as nx
import numpy as np

from fast_ballmapper.plotting._common import scale_node_sizes

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise ImportError("Matplotlib plotting requires fast-ballmapper[plot].") from exc


def draw_ball_mapper(
    graph: nx.Graph,
    colors=None,
    sizes=None,
    layout: str = "spring",
    cmap: str = "viridis",
    with_labels: bool = True,
    node_scale: float = 300,
    ax=None,
):
    """Draw a Ball Mapper graph and return ``(positions, nodes)``."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if layout == "spring":
        positions = nx.spring_layout(graph, seed=42)
    elif layout == "kamada_kawai":
        positions = nx.kamada_kawai_layout(graph)
    elif layout == "spectral":
        positions = nx.spectral_layout(graph)
    else:
        raise ValueError("Unknown layout. Use 'spring', 'kamada_kawai', or 'spectral'.")

    if sizes is None:
        sizes = np.ones(len(graph))
    scaled_sizes = scale_node_sizes(sizes, len(graph), node_scale)

    if colors is None:
        node_kwargs = {"node_color": "lightgray"}
    else:
        color_values = np.asarray(colors)
        if color_values.ndim != 1 or len(color_values) != len(graph):
            raise ValueError("colors must contain exactly one value per node.")
        node_kwargs = {"node_color": color_values, "cmap": cmap}

    nodes = nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=scaled_sizes,
        ax=ax,
        **node_kwargs,
    )
    nx.draw_networkx_edges(graph, positions, alpha=0.5, ax=ax)

    if with_labels:
        nx.draw_networkx_labels(graph, positions, font_size=9, ax=ax)

    ax.set_axis_off()
    return positions, nodes


def add_colorbar(nodes, ax, label: str | None = None):
    """Add a colorbar when the node collection contains scalar colors."""
    if not hasattr(nodes, "cmap") or nodes.get_array() is None:
        return None

    scalar_mappable = plt.cm.ScalarMappable(cmap=nodes.cmap, norm=nodes.norm)
    scalar_mappable.set_array([])
    colorbar = plt.colorbar(scalar_mappable, ax=ax)
    if label:
        colorbar.set_label(label)
    return colorbar
