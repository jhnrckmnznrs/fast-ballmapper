# fast-ballmapper

`fast-ballmapper` builds Ball Mapper graph summaries of high-dimensional point
clouds. It supports scikit-learn's BallTree, optional FAISS acceleration,
deterministic farthest-point sampling, NetworkX graph construction, node
coloring, and optional Matplotlib or Plotly visualization.

## Installation

Core package:

```bash
pip install fast-ballmapper
```

With plotting support:

```bash
pip install "fast-ballmapper[plot]"
```

With FAISS support:

```bash
pip install "fast-ballmapper[faiss]"
```

For local development:

```bash
python -m pip install -e ".[dev,plot]"
pytest
```

## Minimal example

```python
import numpy as np

from fast_ballmapper import build_mapper, compute_landmarks

rng = np.random.default_rng(42)
x = rng.random((500, 2))

landmarks, cover = compute_landmarks(
    x,
    eps=0.1,
    method="ball_tree",
    metric="euclidean",
    leaf_size=40,
)
graph = build_mapper(cover)

print(f"Landmarks: {len(landmarks)}")
print(f"Edges: {graph.number_of_edges()}")
```

The PyPI distribution name contains a hyphen, while the Python import package
uses an underscore:

```text
pip install fast-ballmapper
import fast_ballmapper
```

## Farthest-point sampling

```python
from fast_ballmapper import compute_landmarks_fps

landmarks, cover = compute_landmarks_fps(
    x,
    eps=0.1,
    start_index=None,
    method="ball_tree",
    metric="euclidean",
    leaf_size=40,
    metric_kwargs=None,
)
```

When `start_index` is `None`, the lexicographically smallest point is selected
first, making the result deterministic. Landmark selection and cover queries use
the same backend and distance.

## BallTree metrics

Any metric supported by scikit-learn's BallTree can be passed through `metric`.
Parameterized metrics use `metric_kwargs`:

```python
landmarks, cover = compute_landmarks_fps(
    x,
    eps=0.2,
    metric="minkowski",
    metric_kwargs={"p": 3},
)
```

## FAISS

FAISS supports Euclidean and cosine distances in this package:

```python
landmarks, cover = compute_landmarks(
    x,
    eps=0.1,
    method="faiss",
    metric="euclidean",
)
```

For cosine distance, zero vectors are rejected because cosine distance is
undefined for them.

## Graph construction and coloring

```python
from fast_ballmapper import (
    build_mapper,
    color_by_density,
    color_by_entropy,
    color_by_function,
    color_by_mode,
    color_by_size,
)

graph = build_mapper(cover)
sizes = color_by_size(cover)
density = color_by_density(cover)
mean_first_coordinate = color_by_function(x[:, 0], cover)
```

## Matplotlib visualization

```python
import matplotlib.pyplot as plt

from fast_ballmapper.plotting.matplotlib import add_colorbar, draw_ball_mapper

fig, ax = plt.subplots(figsize=(8, 6))
_, nodes = draw_ball_mapper(
    graph,
    colors=sizes,
    sizes=sizes,
    layout="spring",
    node_scale=500,
    ax=ax,
)
add_colorbar(nodes, ax, label="Ball size")
plt.show()
```

## Plotly visualization

```python
from fast_ballmapper.plotting.plotly import draw_ball_mapper_plotly

figure = draw_ball_mapper_plotly(
    graph,
    cover,
    colorings={"Ball size": sizes, "Density": density},
    sizes=sizes,
    show=True,
)
```

Use `export_html="ball_mapper.html"` to create a standalone interactive HTML
file.

## Public API

Core functions are exported directly from `fast_ballmapper`:

- `compute_landmarks`
- `compute_landmarks_fps`
- `build_mapper`
- `compute_edge_overlaps`
- `color_by_function`
- `color_by_mode`
- `color_by_entropy`
- `color_by_size`
- `color_by_density`

Plotting functions live in their optional submodules so importing the core
package does not import Matplotlib or Plotly.

## License

MIT
