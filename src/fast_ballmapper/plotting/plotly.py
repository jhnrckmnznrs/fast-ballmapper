"""Interactive Ball Mapper plotting with Plotly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np

from fast_ballmapper.graph import compute_edge_overlaps
from fast_ballmapper.plotting._common import scale_node_sizes

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - optional dependency path
    raise ImportError("Plotly plotting requires fast-ballmapper[plot].") from exc


def draw_ball_mapper_plotly(
    graph: nx.Graph,
    cover: Sequence[np.ndarray],
    colorings: Mapping[str, np.ndarray | None] | None = None,
    sizes=None,
    layout: str = "spring",
    node_scale: float = 20,
    export_html: str | None = None,
    show: bool = True,
):
    """Build an interactive Ball Mapper figure with selectable colorings."""
    node_ids = list(graph.nodes())
    if any(
        not isinstance(node_id, (int, np.integer)) or not 0 <= int(node_id) < len(cover)
        for node_id in node_ids
    ):
        raise ValueError("Graph nodes must be integer indices into cover.")
    node_ids = [int(node_id) for node_id in node_ids]

    if layout == "spring":
        positions = nx.spring_layout(graph, seed=42)
    elif layout == "kamada_kawai":
        positions = nx.kamada_kawai_layout(graph)
    else:
        raise ValueError("Unknown layout. Use 'spring' or 'kamada_kawai'.")

    overlaps = compute_edge_overlaps(cover, graph)
    max_overlap = max(overlaps.values()) if overlaps else 1
    edge_traces = [
        go.Scatter(
            x=[positions[left][0], positions[right][0]],
            y=[positions[left][1], positions[right][1]],
            mode="lines",
            line={"width": 1 + 4 * overlap / max_overlap, "color": "gray"},
            hoverinfo="none",
            showlegend=False,
        )
        for (left, right), overlap in overlaps.items()
    ]

    if sizes is None:
        sizes = np.array([len(point_indices) for point_indices in cover])
    scaled_sizes = scale_node_sizes(sizes, len(cover), node_scale)
    node_sizes = scaled_sizes[node_ids]

    hover_text = [
        f"Ball {node_id}<br>Points: {len(cover[node_id])}" for node_id in node_ids
    ]
    node_x = [positions[node_id][0] for node_id in node_ids]
    node_y = [positions[node_id][1] for node_id in node_ids]

    normalized_colorings = dict(colorings) if colorings else {"None": None}
    node_traces = []
    for coloring_index, (name, values) in enumerate(normalized_colorings.items()):
        marker = {
            "size": node_sizes,
            "line": {"width": 1, "color": "black"},
        }
        if values is None:
            marker.update({"color": "lightgray", "showscale": False})
        else:
            color_values = np.asarray(values)
            if color_values.ndim != 1 or len(color_values) != len(cover):
                raise ValueError(f"Coloring {name!r} must contain one value per cover.")
            marker.update(
                {
                    "color": color_values[node_ids],
                    "colorscale": "Viridis",
                    "showscale": True,
                    "colorbar": {"title": name},
                }
            )

        node_traces.append(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=marker,
                hoverinfo="text",
                text=hover_text,
                visible=coloring_index == 0,
                showlegend=False,
            )
        )

    buttons = [
        {
            "label": name,
            "method": "update",
            "args": [
                {
                    "visible": [True] * len(edge_traces)
                    + [
                        trace_index == coloring_index
                        for trace_index in range(len(node_traces))
                    ]
                }
            ],
        }
        for coloring_index, name in enumerate(normalized_colorings)
    ]

    figure = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            updatemenus=[
                {"buttons": buttons, "direction": "down", "x": 0.02, "y": 0.98}
            ],
            hovermode="closest",
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            showlegend=False,
        ),
    )

    if export_html:
        figure.write_html(export_html)
    if show:
        figure.show()
    return figure
