from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import plotly.graph_objects as go

# Optional imports
try:
    from sklearn.neighbors import BallTree
except ImportError:
    BallTree = None

try:
    import faiss
except ImportError:
    faiss = None

from typing import List, Literal, Mapping, Tuple

import matplotlib.pyplot as plt

# =====================================================
# Landmark Computation
# =====================================================


def computeLandmarks(
    X: np.ndarray,
    eps: float,
    method: Literal["ballTree", "faiss"] = "ballTree",
    metric: str = "euclidean",
    leafSize: int = 40,
    metricKwargs: Mapping[str, object] | None = None,
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Compute landmarks and their coverage sets from data points.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, n_features) representing the data points.
    eps : float
        Radius used to define the neighborhood of each landmark.
    method : str, default="ballTree"
        Backend method for nearest neighbor search. Options are:
        - "ballTree": uses scikit-learn's BallTree.
        - "faiss": uses FAISS library for faster searches on large datasets.
    metric : str, default="euclidean"
        Distance metric. BallTree accepts its supported metrics; FAISS accepts
        only "euclidean" and "cosine".
    leafSize : int, default=40
        Leaf size used by BallTree.
    metricKwargs : mapping or None, default=None
        Extra keyword arguments passed to BallTree's metric, such as
        ``{"p": 3}`` for Minkowski distance or ``{"VI": ...}`` for
        Mahalanobis distance. Not supported by FAISS.

    Returns
    -------
    landmarks : list[int]
        Indices of selected landmark points.
    cover : list[np.ndarray]
        Arrays containing the point indices covered by each landmark.

    Raises
    ------
    ValueError
        If `method` is not one of {"ballTree", "faiss"}.
    ImportError
        If the required library (scikit-learn or faiss) is not installed.

    Example
    -------
    >>> X = np.random.rand(100, 3)
    >>> landmarks, cover = computeLandmarks(X, eps=0.2, method="ballTree")
    """
    X = _validatePointCloud(X, eps)
    methodKey, metricKey, metricKwargs = _normalizeBackendOptions(
        method, metric, leafSize, metricKwargs
    )

    if X.shape[0] == 0:
        return [], []

    if methodKey == "balltree":
        return _computeLandmarksBallTree(X, eps, metricKey, leafSize, metricKwargs)

    if metricKey == "euclidean":
        return _computeLandmarksEuclideanFAISS(X, eps)

    return _computeLandmarksCosineFAISS(X, eps)


def _validatePointCloud(X: np.ndarray, eps: float) -> np.ndarray:
    """Validate and normalize point-cloud inputs used by landmark routines."""
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
    if X.shape[0] > 0 and X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X must contain numeric values.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if not np.isscalar(eps) or not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite, non-negative scalar.")

    return X


def _validateLeafSize(leafSize: int) -> None:
    """Validate BallTree leaf size."""
    if not isinstance(leafSize, (int, np.integer)) or leafSize <= 0:
        raise ValueError("leafSize must be a positive integer.")


def _normalizeBackendOptions(method, metric, leafSize, metricKwargs):
    """Normalize backend options and reject unsupported combinations."""
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if not isinstance(metric, str):
        raise TypeError("metric must be a string.")

    methodKey = method.lower()
    metricKey = metric.lower()
    metricKwargs = dict(metricKwargs or {})

    if methodKey not in {"balltree", "faiss"}:
        raise ValueError("Method must be 'ballTree' or 'faiss'.")

    if methodKey == "balltree":
        _validateLeafSize(leafSize)
    else:
        if metricKey not in {"euclidean", "cosine"}:
            raise ValueError("For 'faiss', metric must be 'euclidean' or 'cosine'.")
        if metricKwargs:
            raise ValueError("metricKwargs is supported only with method='ballTree'.")

    return methodKey, metricKey, metricKwargs


def _validateStartIndex(start_index: int | None, n: int) -> None:
    """Validate an optional FPS start index."""
    if start_index is None:
        return
    if not isinstance(start_index, (int, np.integer)):
        raise TypeError("start_index must be an integer or None.")
    if not 0 <= int(start_index) < n:
        raise IndexError("start_index is out of bounds for X.")


def _lexicographicallySmallestIndex(X: np.ndarray) -> int:
    """Return the row index ordered by feature 0, then feature 1, and so on."""
    keys = tuple(X[:, j] for j in range(X.shape[1] - 1, -1, -1))
    return int(np.lexsort(keys)[0])


def _createBallTree(X, metric, leafSize, metricKwargs):
    """Create a BallTree with one shared metric configuration."""
    if BallTree is None:
        raise ImportError("scikit-learn is required for BallTree method.")
    return BallTree(
        X,
        metric=metric,
        leaf_size=leafSize,
        **metricKwargs,
    )


def _validateCosinePoints(points: np.ndarray) -> None:
    """Reject zero vectors, for which cosine distance is undefined."""
    if np.any(np.linalg.norm(points, axis=1) == 0):
        raise ValueError("Cosine distance is undefined for zero vectors.")


def _computeLandmarksBallTree(X, eps, metric, leafSize, metricKwargs=None):
    """
    Internal helper: compute landmarks using scikit-learn BallTree.
    """
    n = X.shape[0]
    tree = _createBallTree(X, metric, leafSize, dict(metricKwargs or {}))

    uncovered = np.ones(n, dtype=bool)
    landmarks, cover = [], []

    while np.any(uncovered):
        i = int(np.argmax(uncovered))
        landmarks.append(i)
        idx = tree.query_radius(X[i : i + 1], eps)[0]
        cover.append(idx)
        uncovered[idx] = False

    return landmarks, cover


def _computeLandmarksEuclideanFAISS(X, eps):
    """
    Internal helper: compute landmarks using FAISS.
    """
    points, index = _createEuclideanFAISSIndex(X)
    radius = np.nextafter(np.float32(eps**2), np.float32(np.inf))

    covered = np.zeros(len(points), dtype=bool)
    landmarks, cover = [], []

    for i in range(len(points)):
        if not covered[i]:
            landmarks.append(i)
            lims, _, I = index.range_search(points[i : i + 1], radius)
            pts = I[lims[0] : lims[1]]
            cover.append(pts)
            covered[pts] = True

    return landmarks, cover


def _computeLandmarksCosineFAISS(X, eps):
    """
    Internal helper: compute landmarks using FAISS.
    """
    points, index = _createCosineFAISSIndex(X)
    radius = np.nextafter(np.float32(1.0 - eps), np.float32(-np.inf))

    covered = np.zeros(len(points), dtype=bool)
    landmarks, cover = [], []

    for i in range(len(points)):
        if not covered[i]:
            landmarks.append(i)
            lims, _, I = index.range_search(points[i : i + 1], radius)
            pts = I[lims[0] : lims[1]]
            cover.append(pts)
            covered[pts] = True

    return landmarks, cover


# Include farthest point sampling
def computeLandmarksFPS(
    X: np.ndarray,
    eps: float,
    start_index: int | None = None,
    method: Literal["ballTree", "faiss"] = "ballTree",
    metric: str = "euclidean",
    leafSize: int = 40,
    metricKwargs: Mapping[str, object] | None = None,
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Deterministic farthest point sampling and epsilon-ball cover.

    FPS and cover construction use the same distance metric, metric parameters,
    and backend semantics. Consequently, the returned landmarks form an
    epsilon-net of X under that distance: every point is within eps of at least
    one landmark, and distinct landmarks are more than eps apart.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, n_features).
    eps : float
        Radius for Ball Mapper cover.
    start_index : int or None
        Deterministic starting point (default = lexicographically smallest).
    method : str, default="ballTree"
        Distance backend: "ballTree" or "faiss".
    metric : str, default="euclidean"
        Distance metric. BallTree accepts its supported metrics; FAISS accepts
        only "euclidean" and "cosine".
    leafSize : int, default=40
        Leaf size used by BallTree.
    metricKwargs : mapping or None, default=None
        Extra keyword arguments passed to BallTree's metric, such as
        ``{"p": 3}`` for Minkowski distance or ``{"VI": ...}`` for
        Mahalanobis distance. Not supported by FAISS.

    Returns
    -------
    landmarks : list[int]
        Indices of landmark points.
    cover : list[np.ndarray]
        List where cover[i] contains indices of all points within eps of landmarks[i].
    """

    X = _validatePointCloud(X, eps)
    methodKey, metricKey, metricKwargs = _normalizeBackendOptions(
        method, metric, leafSize, metricKwargs
    )

    if X.shape[0] == 0:
        return [], []

    _validateStartIndex(start_index, X.shape[0])

    # --------------------------------------------------
    # 1) FPS landmark selection
    # --------------------------------------------------
    if start_index is None:
        start_index = _lexicographicallySmallestIndex(X)
    else:
        start_index = int(start_index)

    if methodKey == "balltree":
        tree = _createBallTree(X, metricKey, leafSize, metricKwargs)

        def distanceToAll(i):
            distances, indices = tree.query(
                X[i : i + 1],
                k=X.shape[0],
                return_distance=True,
            )
            result = np.empty(X.shape[0], dtype=float)
            result[indices[0]] = distances[0]
            return result

        landmarks = _selectLandmarksFPS(X.shape[0], eps, start_index, distanceToAll)
        cover = _buildCoverBallTree(
            X,
            landmarks,
            eps,
            metricKey,
            leafSize,
            metricKwargs,
            tree=tree,
        )

    elif metricKey == "euclidean":
        points, index = _createEuclideanFAISSIndex(X)

        def distanceToAll(i):
            squaredDistances, indices = index.search(points[i : i + 1], X.shape[0])
            result = np.empty(X.shape[0], dtype=float)
            result[indices[0]] = np.sqrt(np.maximum(squaredDistances[0], 0.0))
            return result

        landmarks = _selectLandmarksFPS(X.shape[0], eps, start_index, distanceToAll)
        cover = _buildCoverEuclideanFAISS(X, landmarks, eps, points=points, index=index)

    else:
        points, index = _createCosineFAISSIndex(X)

        def distanceToAll(i):
            similarities, indices = index.search(points[i : i + 1], X.shape[0])
            result = np.empty(X.shape[0], dtype=float)
            result[indices[0]] = np.maximum(1.0 - similarities[0], 0.0)
            return result

        landmarks = _selectLandmarksFPS(X.shape[0], eps, start_index, distanceToAll)
        cover = _buildCoverCosineFAISS(X, landmarks, eps, points=points, index=index)

    return landmarks, cover


def _selectLandmarksFPS(n, eps, start_index, distanceToAll):
    """Run metric-agnostic FPS using distances supplied by the cover backend."""
    dists = np.asarray(distanceToAll(start_index), dtype=float)
    if dists.shape != (n,):
        raise RuntimeError("Distance backend returned an invalid distance array.")
    if not np.all(np.isfinite(dists)):
        raise ValueError("The selected metric produced non-finite distances.")

    dists = np.maximum(dists, 0.0)
    landmarks = [int(start_index)]

    while True:
        next_index = int(np.argmax(dists))
        maxDist = float(dists[next_index])

        if maxDist <= eps:
            break

        landmarks.append(next_index)
        newDists = np.asarray(distanceToAll(next_index), dtype=float)
        if newDists.shape != (n,):
            raise RuntimeError("Distance backend returned an invalid distance array.")
        if not np.all(np.isfinite(newDists)):
            raise ValueError("The selected metric produced non-finite distances.")
        dists = np.minimum(dists, np.maximum(newDists, 0.0))

    return landmarks


def _createEuclideanFAISSIndex(X):
    """Create a float32 FAISS squared-L2 index."""
    if faiss is None:
        raise ImportError("FAISS is required for method='faiss'.")
    points = X.astype(np.float32, copy=True)
    index = faiss.IndexFlatL2(points.shape[1])
    index.add(points)
    return points, index


def _createCosineFAISSIndex(X):
    """Create a normalized float32 FAISS inner-product index."""
    if faiss is None:
        raise ImportError("FAISS is required for method='faiss'.")
    points = X.astype(np.float32, copy=True)
    _validateCosinePoints(points)
    faiss.normalize_L2(points)
    index = faiss.IndexFlatIP(points.shape[1])
    index.add(points)
    return points, index


# =====================================================
# Reusable cover builders
# =====================================================


def _buildCoverBallTree(
    X: np.ndarray,
    landmarks: List[int],
    eps: float,
    metric: str = "euclidean",
    leafSize: int = 40,
    metricKwargs: Mapping[str, object] | None = None,
    tree=None,
) -> List[np.ndarray]:
    """
    Build epsilon-ball covers for a fixed list of landmarks using BallTree.
    """
    if tree is None:
        tree = _createBallTree(X, metric, leafSize, dict(metricKwargs or {}))

    cover = []
    for i in landmarks:
        idx = tree.query_radius(X[i : i + 1], eps)[0]
        cover.append(idx)

    return cover


def _buildCoverEuclideanFAISS(
    X: np.ndarray,
    landmarks: List[int],
    eps: float,
    points=None,
    index=None,
) -> List[np.ndarray]:
    """
    Build epsilon-ball covers for a fixed list of landmarks using FAISS L2 search.
    """
    if points is None or index is None:
        points, index = _createEuclideanFAISSIndex(X)

    radius = np.nextafter(np.float32(eps**2), np.float32(np.inf))
    cover = []

    for i in landmarks:
        lims, _, I = index.range_search(points[i : i + 1], radius)
        pts = I[lims[0] : lims[1]]
        cover.append(pts)

    return cover


def _buildCoverCosineFAISS(
    X: np.ndarray,
    landmarks: List[int],
    eps: float,
    points=None,
    index=None,
) -> List[np.ndarray]:
    """
    Build epsilon-ball covers for a fixed list of landmarks using FAISS cosine search.

    Assumes eps is a cosine-distance radius, so neighbors satisfy:

        cosine_similarity >= 1 - eps
    """
    if points is None or index is None:
        points, index = _createCosineFAISSIndex(X)

    radius = np.nextafter(np.float32(1.0 - eps), np.float32(-np.inf))
    cover = []

    for i in landmarks:
        lims, _, I = index.range_search(points[i : i + 1], radius)
        pts = I[lims[0] : lims[1]]
        cover.append(pts)

    return cover


# =====================================================
# Ball Mapper Graph Construction
# =====================================================


def buildMapper(cover):
    """
    Build a Ball Mapper graph using an inverted index for faster edge computation.

    Parameters
    ----------
    cover : list[np.ndarray]
        List of arrays containing indices of points covered by each landmark.

    Returns
    -------
    G : networkx.Graph
        Graph object with:
        - Nodes representing landmarks
        - Edges connecting landmarks that share at least one point.

    Notes
    -----
    The inverted index avoids repeated pairwise set intersections.

    Example
    -------
    >>> landmarks, cover = computeLandmarks(X, eps=0.2)
    >>> G = buildMapper(cover)
    """
    G = nx.Graph()
    n = len(cover)
    G.add_nodes_from(range(n))
    pointToCovers = defaultdict(list)

    for coverId, points in enumerate(cover):
        for p in points:
            pointToCovers[p].append(coverId)

    for covers in pointToCovers.values():
        for i in range(len(covers)):
            for j in range(i + 1, len(covers)):
                G.add_edge(covers[i], covers[j])

    return G


# =====================================================
# Ball Mapper Coloring
# =====================================================


def colorByFunction(X, cover, func=np.mean):
    """
    Color Ball Mapper nodes using a function applied to covered points.

    Parameters
    ----------
    X : np.ndarray
        Data array (n_samples, n_features) or scalar values (n_samples,).
    cover : list[np.ndarray]
        Cover sets from landmarks.
    func : callable, default=np.mean
        Function applied to X[cover[i]].

    Returns
    -------
    colors : np.ndarray
        One value per landmark (node).
    """
    X = np.asarray(X)
    colors = np.zeros(len(cover))

    for i, pts in enumerate(cover):
        if len(pts) > 0:
            colors[i] = func(X[pts])
        else:
            colors[i] = np.nan

    return colors


def colorByMode(y, cover):
    """
    Assign each ball the most frequent label among covered points.
    """
    y = np.asarray(y)
    colors = np.empty(len(cover), dtype=object)

    for i, pts in enumerate(cover):
        if len(pts) > 0:
            colors[i] = Counter(y[pts]).most_common(1)[0][0]
        else:
            colors[i] = -1

    return colors


def colorByEntropy(y, cover):
    """
    Color balls by label entropy (heterogeneity).
    """

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-12))

    y = np.asarray(y)
    colors = np.zeros(len(cover))

    for i, pts in enumerate(cover):
        if len(pts) > 0:
            colors[i] = entropy(y[pts])
        else:
            colors[i] = 0.0

    return colors


def colorBySize(cover):
    """
    Color balls by number of covered points.
    """
    return np.array([len(c) for c in cover])


def colorByDensity(cover):
    """Normalize cover sizes to the interval [0, 1]."""
    sizes = np.array([len(c) for c in cover], dtype=float)
    if sizes.size == 0:
        return sizes
    maxSize = sizes.max()
    if maxSize == 0:
        return np.zeros_like(sizes)
    return sizes / maxSize


def _scaleNodeSizes(sizes, count, node_scale):
    """Validate and scale node sizes without dividing by zero."""
    if not np.isscalar(node_scale) or not np.isfinite(node_scale) or node_scale < 0:
        raise ValueError("node_scale must be a finite, non-negative scalar.")

    sizes = np.asarray(sizes, dtype=float)
    if sizes.ndim != 1 or len(sizes) != count:
        raise ValueError("sizes must contain exactly one value per node.")
    if not np.all(np.isfinite(sizes)) or np.any(sizes < 0):
        raise ValueError("sizes must contain finite, non-negative values.")
    if sizes.size == 0:
        return sizes

    maxSize = sizes.max()
    if maxSize == 0:
        return np.zeros_like(sizes)
    return node_scale * sizes / maxSize


# =====================================================
# Ball Mapper with Colors
# =====================================================


def drawBallMapper(
    G,
    colors=None,
    sizes=None,
    layout="spring",
    cmap="viridis",
    with_labels=True,
    node_scale=300,
    ax=None,
):
    """
    Draw a Ball Mapper graph.

    Parameters
    ----------
    G : networkx.Graph
    colors : array-like or None
        Optional node colors.
    sizes : array-like or None
        Optional node sizes.
    layout : str
    cmap : str
    with_labels : bool
    node_scale : float
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    pos : dict
    nodes : matplotlib.collections.PathCollection
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        raise ValueError("Unknown layout. Use 'spring', 'kamada_kawai', or 'spectral'.")

    # Sizes
    if sizes is None:
        sizes = np.ones(len(G))
    sizes = _scaleNodeSizes(sizes, len(G), node_scale)

    # Colors
    if colors is None:
        node_kwargs = dict(node_color="lightgray")
    else:
        colors = np.asarray(colors)
        if colors.ndim != 1 or len(colors) != len(G):
            raise ValueError("colors must contain exactly one value per node.")
        node_kwargs = dict(node_color=colors, cmap=cmap)

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=sizes,
        ax=ax,
        **node_kwargs,
    )

    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

    ax.set_axis_off()
    return pos, nodes


def addColorbar(nodes, ax, label=None):
    """
    Add a colorbar if node colors are present.
    """
    if not hasattr(nodes, "cmap") or nodes.get_array() is None:
        return  # No colors → no colorbar

    sm = plt.cm.ScalarMappable(cmap=nodes.cmap, norm=nodes.norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    if label:
        cbar.set_label(label)


def computeEdgeOverlaps(cover, G):
    """
    Compute overlap size for each edge in the Ball Mapper graph.
    """
    overlaps = {}
    for u, v in G.edges():
        overlaps[(u, v)] = len(np.intersect1d(cover[u], cover[v]))
    return overlaps


def drawBallMapperPlotly(
    G,
    cover,
    colorings=None,
    sizes=None,
    layout="spring",
    node_scale=20,
    export_html=None,
    show=True,
):
    """
    Full-featured interactive Ball Mapper visualization.

    Parameters
    ----------
    G : networkx.Graph
    cover : list[np.ndarray]
        Cover sets for hover info and overlap computation.
    colorings : dict or None
        {name: array-like} for dropdown coloring selection.
    sizes : array-like or None
        Node sizes (e.g., ball sizes).
    layout : str
    node_scale : float
    export_html : str or None
        Path to save interactive HTML.
    show : bool, default=True
        Whether to display the figure immediately.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The constructed interactive figure.
    """

    nodeIds = list(G.nodes())
    if any(
        not isinstance(i, (int, np.integer)) or not 0 <= int(i) < len(cover)
        for i in nodeIds
    ):
        raise ValueError("Graph nodes must be integer indices into cover.")
    nodeIds = [int(i) for i in nodeIds]

    # -----------------------
    # Layout
    # -----------------------
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError("Unknown layout. Use 'spring' or 'kamada_kawai'.")

    # -----------------------
    # Edge overlaps → thickness
    # -----------------------
    overlaps = computeEdgeOverlaps(cover, G)
    max_overlap = max(overlaps.values()) if overlaps else 1

    edge_traces = []
    for (u, v), w in overlaps.items():
        edge_traces.append(
            go.Scatter(
                x=[pos[u][0], pos[v][0]],
                y=[pos[u][1], pos[v][1]],
                mode="lines",
                line=dict(width=1 + 4 * w / max_overlap, color="gray"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # -----------------------
    # Node sizes
    # -----------------------
    if sizes is None:
        sizes = np.array([len(c) for c in cover])
    sizes = _scaleNodeSizes(sizes, len(cover), node_scale)
    nodeSizes = sizes[nodeIds]

    # -----------------------
    # Hover text
    # -----------------------
    hover_text = [f"Ball {i}<br>Points: {len(cover[i])}" for i in nodeIds]

    node_x = [pos[i][0] for i in nodeIds]
    node_y = [pos[i][1] for i in nodeIds]

    # -----------------------
    # Colorings
    # -----------------------
    if not colorings:
        colorings = {"None": None}

    traces = []

    for i, (name, values) in enumerate(colorings.items()):
        marker = dict(
            size=nodeSizes,
            line=dict(width=1, color="black"),
        )

        if values is None:
            marker["color"] = "lightgray"
            marker["showscale"] = False
        else:
            values = np.asarray(values)
            if values.ndim != 1 or len(values) != len(cover):
                raise ValueError(f"Coloring {name!r} must contain one value per cover.")
            marker["color"] = values[nodeIds]
            marker["colorscale"] = "Viridis"
            marker["showscale"] = True
            marker["colorbar"] = dict(title=name)

        trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=marker,
            hoverinfo="text",
            text=hover_text,
            visible=(i == 0),
            showlegend=False,
        )

        traces.append(trace)

    buttons = []
    for i, name in enumerate(colorings):
        buttons.append(
            dict(
                label=name,
                method="update",
                args=[
                    {
                        "visible": [True] * len(edge_traces)
                        + [j == i for j in range(len(traces))]
                    },
                ],
            )
        )

    # -----------------------
    # Figure
    # -----------------------
    fig = go.Figure(
        data=edge_traces + traces,
        layout=go.Layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    x=0.02,
                    y=0.98,
                )
            ],
            hovermode="closest",
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        ),
    )

    if export_html:
        fig.write_html(export_html)

    if show:
        fig.show()

    return fig
