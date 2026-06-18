# fast-ballmapper

`fast-ballmapper` builds Ball Mapper graph summaries of high-dimensional point
clouds. It supports scikit-learn's BallTree, optional FAISS acceleration,
deterministic farthest-point sampling, NetworkX graph construction, node
coloring, and optional Matplotlib or Plotly visualization.

The package is designed around two use cases:

1. exact Ball Mapper cover construction using BallTree or exhaustive FAISS
   indexes;
2. experimental approximate range-query construction using configurable FAISS
   indexes.

## Installation

Core package:

```bash
pip install fast-ballmapper
````

With plotting support:

```bash
pip install "fast-ballmapper[plot]"
```

With CPU FAISS support:

```bash
pip install "fast-ballmapper[faiss]"
```

With development tools, plotting, and CPU FAISS support:

```bash
python -m pip install -e ".[dev,plot,faiss]"
pytest
```

The PyPI distribution name contains a hyphen, while the Python import package
uses an underscore:

```text
pip install fast-ballmapper
import fast_ballmapper
```

### GPU FAISS support

GPU support is optional. The `faiss` extra installs the CPU FAISS package. To
use GPU execution, install a GPU-enabled FAISS build separately for your CUDA
and platform.

The package can request GPU execution through `FaissConfig(device="gpu")` or
`FaissConfig(device="auto")`. If GPU support is unavailable and
`gpu_fallback_to_cpu=True`, the backend falls back to CPU execution.

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

## Fixed-landmark cover construction

The function `compute_landmarks` selects landmarks and constructs their cover.
The function `build_cover` only constructs cover sets around an already chosen
landmark collection.

This is useful for comparing different backends or approximate FAISS
configurations on the same landmark set.

```python
import numpy as np

from fast_ballmapper import FaissConfig, build_cover, compute_landmarks

rng = np.random.default_rng(42)
x = rng.random((1000, 8)).astype("float32")

landmarks, exact_cover = compute_landmarks(
    x,
    eps=0.25,
    method="faiss",
    metric="euclidean",
    faiss_config=FaissConfig(factory="Flat"),
)

approximate_cover = build_cover(
    x,
    landmarks,
    eps=0.25,
    method="faiss",
    metric="euclidean",
    faiss_config=FaissConfig(
        factory="IVF64,Flat",
        search_params={"nprobe": 8},
    ),
)
```

Here, the landmark set is fixed. Therefore, differences between `exact_cover`
and `approximate_cover` come from the range-query backend rather than from
different landmark choices.

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

For FAISS, farthest-point sampling uses an exact Flat index. Approximate FAISS
indexes are not used for farthest-point sampling because the algorithm requires
distances from each selected landmark to every data point.

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

BallTree is useful when metric flexibility is important.

## FAISS backend

FAISS supports Euclidean and cosine distances in this package:

```python
from fast_ballmapper import compute_landmarks

landmarks, cover = compute_landmarks(
    x,
    eps=0.1,
    method="faiss",
    metric="euclidean",
)
```

For cosine distance, zero vectors are rejected because cosine distance is
undefined for them.

By default, the FAISS backend uses a Flat index:

```python
from fast_ballmapper import FaissConfig

config = FaissConfig(factory="Flat")
```

A Flat FAISS index is exhaustive. It compares the query with every indexed
vector, so it accelerates distance computation without changing the Ball Mapper
range sets, apart from finite-precision effects near the threshold.

### Configurable FAISS indexes

The FAISS backend can be configured using `FaissConfig`:

```python
from fast_ballmapper import FaissConfig, compute_landmarks

config = FaissConfig(
    factory="IVF256,Flat",
    search_params={"nprobe": 16},
)

landmarks, cover = compute_landmarks(
    x,
    eps=0.1,
    method="faiss",
    metric="euclidean",
    faiss_config=config,
)
```

The `factory` argument is a FAISS index-factory string. Examples include:

```python
FaissConfig(factory="Flat")
FaissConfig(factory="IVF256,Flat", search_params={"nprobe": 16})
FaissConfig(factory="HNSW32", search_params={"efSearch": 64})
FaissConfig(factory="SQ8")
FaissConfig(factory="IVF256,SQ8", search_params={"nprobe": 16})
FaissConfig(factory="IVF256,PQ16x4", query_mode="knn", candidate_k=1024)
```

Approximate indexes may change the computed cover. Non-exhaustive indexes can
omit true ball members, while compressed indexes can also introduce points whose
exact distance lies outside the ball. For controlled comparisons, use
`build_cover` with a fixed landmark set.

### Candidate search and exact verification

Some FAISS configurations use a candidate-limited `k`-nearest-neighbour search
instead of native range search:

```python
config = FaissConfig(
    factory="IVF256,PQ16x4",
    search_params={"nprobe": 16},
    query_mode="knn",
    candidate_k=1024,
)
```

Exact verification can be enabled to recompute exact distances for candidates
and remove points outside the requested radius:

```python
config = FaissConfig(
    factory="IVF256,Flat",
    search_params={"nprobe": 16},
    query_mode="knn",
    candidate_k=1024,
    exact_verify=True,
)
```

Exact verification prevents false-positive ball memberships among the examined
candidates, but it cannot recover true members that were never included in the
candidate set.

### Optional GPU execution

GPU execution can be requested through `FaissConfig`:

```python
config = FaissConfig(
    factory="Flat",
    device="gpu",
    gpu_device=0,
    gpu_fallback_to_cpu=True,
)
```

Use `device="auto"` to use GPU only when a GPU-enabled FAISS build and a
visible GPU are available:

```python
config = FaissConfig(
    factory="Flat",
    device="auto",
)
```

The default is `device="cpu"` for reproducibility.

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

## Experiments

Research scripts can be placed in the top-level `experiments/` directory.
These scripts are not part of the public package API.

A fixed-landmark approximate FAISS experiment can be run with:

```bash
python experiments/run_fixed_landmarks.py \
    --n 5000 \
    --d 32 \
    --seed 0 \
    --target-ball-size 30
```

The fixed-landmark design first selects landmarks exactly and then compares
different range-query backends around the same landmark set. This separates
range-query approximation errors from changes in landmark selection.

## Public API

Core functions and configuration objects are exported directly from
`fast_ballmapper`:

* `FaissConfig`
* `compute_landmarks`
* `compute_landmarks_fps`
* `build_cover`
* `build_mapper`
* `compute_edge_overlaps`
* `color_by_function`
* `color_by_mode`
* `color_by_entropy`
* `color_by_size`
* `color_by_density`

Plotting functions live in their optional submodules so importing the core
package does not import Matplotlib or Plotly.

## License

MIT
