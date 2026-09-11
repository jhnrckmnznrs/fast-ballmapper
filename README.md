# fast-ballmapper

`fast-ballmapper` builds Ball Mapper graph summaries of high-dimensional point
clouds. It is organized around a common range-query backend protocol and ships
with a float64 brute-force reference oracle, scikit-learn BallTree, SciPy
`cKDTree`, optional FAISS/hnswlib engines, and optional NVIDIA cuVS GPU
adapters, together with deterministic farthest-point sampling,
NetworkX graph construction with overlap-witness counts, boundary-sensitivity
diagnostics, node coloring, and optional Matplotlib or Plotly visualization.

**Release status:** version 0.2.0 is the first backend-protocol release. Exact CPU backends are cross-checked against a float64 brute-force oracle; optional GPU backends require platform-specific validation.

The package is designed around three use cases:

1. auditable exact Ball Mapper construction using independent reference,
   BallTree, or `cKDTree` engines;
2. exact or approximate construction using configurable FAISS indexes; and
3. a backend extension interface for hnswlib, NVIDIA cuVS, or user-defined
   range-query engines.

### Closed-ball boundary convention

Ball Mapper uses closed metric balls,
`B(l, eps) = {x : d(x, l) <= eps}`. Floating-point backends do not all expose
the same comparison convention: FAISS range predicates are strict, while
scikit-learn BallTree radius queries are inclusive. `fast-ballmapper`
therefore implements one explicit closed-ball policy. Strict-threshold paths
use the next representable value beyond the requested boundary, via
`np.nextafter`, and exact verification applies the same convention. The
BallTree path queries at that outward radius and then applies a strict
distance filter, so a point exactly at `eps` is retained but a point at the
next representable distance above `eps` is not.

This is a floating-point boundary rule, not a user-tunable tolerance: no
problem-dependent `atol` or `rtol` is introduced.

### Reproducible exactness gate

The post-refactor exactness check can be reproduced without FAISS:

```bash
python experiments/run_exactness_gate.py
```

The script calibrates the mathematical `eps` value from exact k-nearest-
neighbour distances and deliberately does **not** pre-expand it. The backend
then applies the centralized `np.nextafter` policy exactly once when an
underlying query API requires a strict threshold. It checks greedy landmark
identity, fixed-cover membership identity, and graph-edge identity between the
float64 brute-force oracle and exact spatial-tree backends.

## Installation

Core package:

```bash
pip install fast-ballmapper
```

With plotting support:

```bash
pip install "fast-ballmapper[plot]"
```

With CPU FAISS support:

```bash
pip install "fast-ballmapper[faiss]"
```

With GPU FAISS support:

```bash
pip install "fast-ballmapper[faiss-gpu]"
```

With hnswlib support:

```bash
pip install "fast-ballmapper[hnswlib]"
```

NVIDIA cuVS is CUDA-version specific. For example:

```bash
pip install "fast-ballmapper[cuvs-cu12]"  # CUDA 12
# or
pip install "fast-ballmapper[cuvs-cu13]"  # CUDA 13
```

cuVS currently requires Python 3.11+ and a supported NVIDIA GPU/CUDA
environment; see the NVIDIA cuVS installation guide for current platform
requirements.

With the common optional CPU backends and plotting support:

```bash
pip install "fast-ballmapper[all]"
```

CUDA-specific cuVS extras are intentionally not included in `all`; select
either `cuvs-cu12` or `cuvs-cu13` for the target system.

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

## Range-query backend architecture

Ball Mapper itself needs one primitive from a search engine: for an indexed
landmark `i`, return the observations in the closed ball
`B(x[i], eps)`. The public `RangeQueryBackend` protocol makes this operation
explicit. High-level functions still accept the historical `method=` strings,
but new code can construct a backend once and reuse it:

```python
from fast_ballmapper import BruteForceBackend, build_cover, compute_landmarks

backend = BruteForceBackend(x, metric="euclidean")
landmarks, cover = compute_landmarks(x, eps=0.25, backend=backend)
held_fixed = build_cover(x, landmarks, eps=0.25, backend=backend)

print(backend.metadata)
```

Every backend exposes metadata including `name`, `is_exact`, `metric`, `dtype`,
`device`, and whether it can provide exhaustive distance vectors for
farthest-point sampling. External users can implement the same protocol and
pass their own fitted backend through `backend=` without changing the Ball
Mapper algorithms.

The built-in engines are:

| backend | exact? | device | role |
|---|---:|---|---|
| `BruteForceBackend` | yes | CPU | float64 reference oracle |
| `BallTreeBackend` | yes | CPU | metric-flexible exact tree |
| `CKDTreeBackend` | yes | CPU | independent Euclidean exact tree |
| `FaissFlatBackend` | exhaustive search* | CPU/GPU | float32 FAISS baseline |
| `FaissIVFBackend` | no | CPU/GPU | approximate IVF candidate search |
| `FaissHNSWBackend` | no | CPU/GPU | approximate FAISS HNSW |
| `HnswlibBackend` | no | CPU | optional HNSW candidate search |
| `CuVSBackend` | configuration dependent | GPU | optional brute-force/CAGRA search |

`*` FAISS Flat is exhaustive relative to the indexed float32 representation;
adversarial points extremely close to the boundary can still differ from the
float64 reference oracle because of representation rounding.

The factory offers the same functionality without importing concrete classes:

```python
from fast_ballmapper import make_backend

backend = make_backend(x, "ckdtree", metric="euclidean")
```

For hnswlib and cuVS, the Python APIs are k-nearest-neighbour oriented rather
than native arbitrary-radius search APIs. Their adapters therefore request a
candidate set and apply the same closed-ball filter used elsewhere. This makes
`candidate_k` part of the approximation contract, not an invisible
implementation detail.

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

### Brute-force reference oracle

For correctness audits, `method="brute_force"` performs exhaustive float64
Euclidean or cosine queries. It is intentionally simple rather than optimized:
its purpose is to provide a transparent reference against which accelerated
backends can be checked.

```python
reference_cover = build_cover(
    x,
    landmarks,
    eps=0.25,
    method="brute_force",
    metric="euclidean",
)

balltree_cover = build_cover(
    x,
    landmarks,
    eps=0.25,
    method="ball_tree",
    metric="euclidean",
)
```

The same backend can be used with `compute_landmarks` and
`compute_landmarks_fps`. All reference queries use the package's explicit
closed-ball `np.nextafter` boundary convention.

### Boundary-margin diagnostics

The sensitivity of a radius query is concentrated near the boundary
`d(x, l) = eps`. The diagnostic API computes, for every landmark, the smallest
absolute distance to that boundary, the nearest inside and outside margins, and
optional counts in user-specified boundary bands.

```python
from fast_ballmapper import compute_boundary_diagnostics

diagnostics = compute_boundary_diagnostics(
    x,
    landmarks,
    eps=0.25,
    metric="euclidean",
    deltas=[1e-6, 1e-4, 1e-2],
)

print(diagnostics.min_abs_margin)
print(diagnostics.outside_margin)
print(diagnostics.band_counts[1e-4])
```

These quantities are computed with the same float64 reference distance used by
the brute-force backend, making them suitable for explaining when approximate
queries are likely to change memberships.

### Approximation audit reports

`compare_covers` combines membership, graph, witness, boundary, and optional
coloring diagnostics into one fixed-landmark audit report. This is intended for
comparing an accelerated or approximate cover against the brute-force reference
cover.

```python
from fast_ballmapper import compare_covers

report = compare_covers(
    reference_cover,
    approximate_cover,
    x=x,
    landmarks=landmarks,
    eps=0.25,
    metric="euclidean",
    deltas=[1e-6, 1e-4, 1e-2],
    values=response,  # optional scalar values for mean-color auditing
)

print(report.membership_precision, report.membership_recall)
print(report.edge_precision, report.edge_recall)
print(report.missing_edges)

for edge, change in report.witness_changes.items():
    print(edge, change.retained_witnesses, change.lost_witnesses)

if report.colors is not None:
    print(report.colors.mean_absolute_error)
    print(report.colors.theoretical_bound)
```

Per-ball membership precision, recall, and Jaccard scores are available under
`report.balls`. Edge witness changes distinguish retained witnesses from lost
witnesses and newly introduced witnesses, so replacement by a false-positive
point cannot be mistaken for true witness survival. When `values` is supplied,
the report uses mean node colors and evaluates the finite-set bound

```text
abs(color_reference - color_approximate)
    <= local_oscillation * (1 - min(precision, recall)).
```

Boundary diagnostics are attached only when `x`, `landmarks`, and `eps` are
provided together. The two covers must correspond to the same fixed landmarks
in the same order.

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

# Every edge stores the number of distinct data points witnessing the overlap.
for left, right, data in graph.edges(data=True):
    print(left, right, data["witness_count"])

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

Two repository scripts are kept as reproducibility checks rather than public API:

```bash
python experiments/run_exactness_gate.py
python experiments/run_fixed_landmarks.py --n 5000 --d 32 --seed 0
```

`run_exactness_gate.py` compares the float64 brute-force oracle, BallTree, and
SciPy `cKDTree` under the unified closed-ball convention.
`run_fixed_landmarks.py` is a compact FAISS fixed-landmark benchmark. Generated
CSV, audit, and figure outputs are ignored by Git and are not included in source
distributions. The larger manuscript experiment matrix is documented in
`EXPERIMENT_STATUS.md` and can be maintained separately from the core library
release.

## Public API

Core functions, backend abstractions, and configuration objects are exported directly from
`fast_ballmapper`:

* `RangeQueryBackend`, `BackendMetadata`, `make_backend`
* `BruteForceBackend`, `BallTreeBackend`, `CKDTreeBackend`
* `FaissFlatBackend`, `FaissIVFBackend`, `FaissHNSWBackend`
* `HnswlibBackend`, `CuVSBackend`
* `ApproximationAudit`
* `BoundaryDiagnostics`
* `FaissConfig`
* `compare_covers`
* `compute_boundary_diagnostics`
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

## Development and citation

Backend-extension guidance is in [`docs/backend_architecture.md`](docs/backend_architecture.md), and contributor setup is in [`CONTRIBUTING.md`](CONTRIBUTING.md). The repository includes [`CITATION.cff`](CITATION.cff) so GitHub can expose software citation metadata.

For research use, please cite the Ball Mapper methodology relevant to your analysis and cite this software release. The companion range-query manuscript can be added here once it has a persistent publication identifier.
