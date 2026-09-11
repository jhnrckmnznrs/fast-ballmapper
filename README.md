# fast-ballmapper

`fast-ballmapper` constructs Ball Mapper graph summaries of metric data, with a particular focus on auditable range query backends and numerical consistency at the radius boundary.

Ball Mapper repeatedly asks one computational question: for a landmark and a radius, which observations lie in the corresponding closed metric ball? The package separates that mathematical query from the data structure used to answer it. In the mathematical language used by the companion manuscript, an **oracle** is an idealized procedure that returns the answer to a specified query. A backend is a concrete implementation of that procedure. The float64 brute force backend therefore serves as a transparent reference oracle against which accelerated backends can be checked.

Version `0.2.0` introduces a common range query backend interface, independent exact CPU implementations, approximation audits, witness counts, and a unified closed ball convention.

## Main Features

The package provides:

- exact Ball Mapper construction with float64 brute force, scikit-learn BallTree, and SciPy `cKDTree`;
- exhaustive and approximate FAISS backends;
- optional hnswlib and NVIDIA cuVS adapters;
- deterministic greedy landmark construction and farthest point sampling;
- fixed landmark cover construction for backend comparisons;
- graph construction with distinct witness counts on edges;
- boundary margin diagnostics;
- membership, witness, edge, and color audits for approximate covers; and
- optional Matplotlib and Plotly visualization.

## Closed Ball Boundary Convention

Ball Mapper uses closed metric balls:

```text
B(l, eps) = {x : d(x, l) <= eps}.
```

Different libraries expose different numerical comparison rules. Some radius APIs are inclusive, while others use a strict threshold. `fast-ballmapper` implements one common policy so that all supported exact backends represent the same mathematical closed ball.

When a backend uses a strict comparison, the implementation replaces `eps` by the next representable floating point value above it through `np.nextafter`. It then applies the strict comparison at that outward rounded threshold. Therefore, a value represented exactly as `eps` is included, while the next representable value above `eps` is excluded.

This rule is part of the numerical semantics of the package. It is not a user selected tolerance, and no problem dependent `atol` or `rtol` is introduced.

## Installation

Install the core package with:

```bash
pip install fast-ballmapper
```

Optional extras include:

```bash
pip install "fast-ballmapper[plot]"
pip install "fast-ballmapper[faiss]"
pip install "fast-ballmapper[hnswlib]"
```

For NVIDIA cuVS, choose the extra that matches the installed CUDA major version:

```bash
pip install "fast-ballmapper[cuvs-cu12]"
# or
pip install "fast-ballmapper[cuvs-cu13]"
```

The common optional CPU backends and plotting dependencies can be installed with:

```bash
pip install "fast-ballmapper[all]"
```

CUDA specific cuVS extras are intentionally excluded from `all` because the correct package depends on the target system.

For development:

```bash
python -m pip install -e ".[dev,plot,faiss]"
pytest
```

The distribution name contains a hyphen, whereas the Python import package uses an underscore:

```text
pip install fast-ballmapper
import fast_ballmapper
```

## Minimal Example

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

## Range Query Backend Architecture

The public `RangeQueryBackend` protocol makes the search operation explicit. Existing code can continue to use `method=...`, while new code can construct a backend once and reuse it:

```python
from fast_ballmapper import BruteForceBackend, build_cover, compute_landmarks

backend = BruteForceBackend(x, metric="euclidean")
landmarks, cover = compute_landmarks(x, eps=0.25, backend=backend)
fixed_cover = build_cover(x, landmarks, eps=0.25, backend=backend)

print(backend.metadata)
```

Each backend reports metadata such as its name, whether it is intended to be exact, its metric, numerical dtype, execution device, batch query support, and whether it can provide exhaustive distance vectors. Farthest point sampling requires exhaustive distance vectors, so approximate candidate only backends cannot silently change its semantics.

The included backends are:

| Backend | Exactness | Device | Main Role |
|---|---:|---|---|
| `BruteForceBackend` | exact | CPU | transparent float64 reference |
| `BallTreeBackend` | exact | CPU | exact tree with flexible metrics |
| `CKDTreeBackend` | exact | CPU | independent exact Euclidean tree |
| `FaissFlatBackend` | exhaustive over stored vectors | CPU/GPU | optimized float32 baseline |
| `FaissIVFBackend` | approximate | CPU/GPU | inverted file search |
| `FaissHNSWBackend` | approximate | CPU/GPU | FAISS HNSW search |
| `HnswlibBackend` | approximate | CPU | optional HNSW search |
| `CuVSBackend` | configuration dependent | GPU | optional brute force or CAGRA search |

FAISS Flat is exhaustive relative to its stored float32 representation. Consequently, points extremely close to the radius boundary can still differ from a float64 reference because of representation rounding.

A backend can also be created through the factory:

```python
from fast_ballmapper import make_backend

backend = make_backend(x, "ckdtree", metric="euclidean")
```

External users can implement the same protocol and pass a fitted backend through `backend=` without changing landmark selection, graph construction, coloring, or auditing code.

## Fixed Landmark Cover Construction

`compute_landmarks` selects landmarks and constructs their cover. `build_cover` constructs cover sets around a landmark set that has already been chosen.

The second operation is useful when different search backends must be compared on exactly the same landmarks:

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

Because the landmark set is fixed, differences between the two covers arise from range query behavior rather than from different landmark choices.

## Brute Force Reference Oracle

For correctness audits, `method="brute_force"` performs exhaustive float64 Euclidean or cosine queries. It is intentionally simple rather than optimized.

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

The same backend can be used with `compute_landmarks` and `compute_landmarks_fps`.

## Boundary Margin Diagnostics

The sensitivity of a fixed radius query is concentrated near the boundary `d(x, l) = eps`. The diagnostic API computes, for each landmark, the minimum absolute margin to that boundary, the nearest inside and outside margins, and optional counts inside specified boundary bands.

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

These diagnostics use the same float64 reference distance as the brute force backend.

## Approximation Audit Reports

`compare_covers` combines membership, graph, witness, boundary, and optional color diagnostics into one fixed landmark audit report.

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
    values=response,
)

print(report.membership_precision, report.membership_recall)
print(report.edge_precision, report.edge_recall)
print(report.missing_edges)
```

For each ball, precision, recall, and Jaccard scores are available under `report.balls`. Edge witness changes distinguish retained, lost, and newly introduced witnesses. Therefore, replacement by a false positive observation cannot be mistaken for genuine witness survival.

When scalar `values` are supplied, the audit also computes mean node colors and evaluates the finite set bound

```text
abs(color_reference - color_approximate)
    <= local_oscillation * (1 - min(precision, recall)).
```

Boundary diagnostics are attached only when `x`, `landmarks`, and `eps` are provided together. The two covers must use the same fixed landmarks in the same order.

## Farthest Point Sampling

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

When `start_index` is `None`, the lexicographically smallest point is selected first. Therefore, the result is deterministic once the distance and tie rules are fixed.

Approximate candidate only backends are not used for farthest point sampling because the algorithm requires the distance from each selected landmark to every observation.

## FAISS Backend

FAISS supports Euclidean and cosine distances in this package. For cosine distance, zero vectors are rejected because cosine distance is undefined for them.

```python
from fast_ballmapper import FaissConfig, compute_landmarks

config = FaissConfig(factory="Flat")

landmarks, cover = compute_landmarks(
    x,
    eps=0.1,
    method="faiss",
    metric="euclidean",
    faiss_config=config,
)
```

A Flat index examines every stored vector. Therefore, it is exhaustive over its stored numerical representation.

### Configurable FAISS Indexes

Examples include:

```python
FaissConfig(factory="Flat")
FaissConfig(factory="IVF256,Flat", search_params={"nprobe": 16})
FaissConfig(factory="HNSW32", search_params={"efSearch": 64})
FaissConfig(factory="SQ8")
FaissConfig(factory="IVF256,SQ8", search_params={"nprobe": 16})
FaissConfig(factory="IVF256,PQ16x4", query_mode="knn", candidate_k=1024)
```

Indexes that do not search exhaustively can omit true ball members. Compressed indexes can additionally introduce returned candidates whose exact distance lies outside the requested ball.

### Candidate Search and Exact Verification

Some configurations use a `k` nearest neighbor candidate search and then filter candidates by radius:

```python
config = FaissConfig(
    factory="IVF256,Flat",
    search_params={"nprobe": 16},
    query_mode="knn",
    candidate_k=1024,
    exact_verify=True,
)
```

Exact verification removes false positive memberships among the examined candidates. However, it cannot recover true ball members that were never present in the candidate set.

### Optional GPU Execution

GPU execution can be requested through `FaissConfig`:

```python
config = FaissConfig(
    factory="Flat",
    device="gpu",
    gpu_device=0,
    gpu_fallback_to_cpu=True,
)
```

Use `device="auto"` when GPU execution should be used only if the current FAISS installation and hardware support it. The default remains `device="cpu"` for reproducibility.

## Graph Construction and Coloring

```python
from fast_ballmapper import (
    build_mapper,
    color_by_density,
    color_by_function,
    color_by_size,
)

graph = build_mapper(cover)

for left, right, data in graph.edges(data=True):
    print(left, right, data["witness_count"])

sizes = color_by_size(cover)
density = color_by_density(cover)
mean_first_coordinate = color_by_function(x[:, 0], cover)
```

Every edge stores the number of distinct observations witnessing the corresponding overlap.

## Visualization

Matplotlib and Plotly support live in optional submodules so that importing the core package does not import either plotting library.

### Matplotlib

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

### Plotly

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

Use `export_html="ball_mapper.html"` to create a standalone interactive HTML file.

## Reproducibility Checks

Two repository scripts provide compact reproducibility checks:

```bash
python experiments/run_exactness_gate.py
python experiments/run_fixed_landmarks.py --n 5000 --d 32 --seed 0
```

`run_exactness_gate.py` compares the float64 brute force reference, BallTree, and SciPy `cKDTree` under the common closed ball convention. `run_fixed_landmarks.py` provides a compact FAISS benchmark with fixed landmarks.

Generated CSV, audit, and figure outputs are ignored by Git and are not included in source distributions. The larger manuscript experiment matrix is summarized in `EXPERIMENT_STATUS.md`.

## Public API

Core functions, backend abstractions, and configuration objects are exported directly from `fast_ballmapper`:

- `RangeQueryBackend`, `BackendMetadata`, `make_backend`
- `BruteForceBackend`, `BallTreeBackend`, `CKDTreeBackend`
- `FaissFlatBackend`, `FaissIVFBackend`, `FaissHNSWBackend`
- `HnswlibBackend`, `CuVSBackend`
- `ApproximationAudit`
- `BoundaryDiagnostics`
- `FaissConfig`
- `compare_covers`
- `compute_boundary_diagnostics`
- `compute_landmarks`
- `compute_landmarks_fps`
- `build_cover`
- `build_mapper`
- `compute_edge_overlaps`
- `color_by_function`
- `color_by_mode`
- `color_by_entropy`
- `color_by_size`
- `color_by_density`

## Development and Citation

Backend extension guidance is available in [`docs/backend_architecture.md`](docs/backend_architecture.md), and contributor setup is documented in [`CONTRIBUTING.md`](CONTRIBUTING.md). The repository also contains [`CITATION.cff`](CITATION.cff) so GitHub can expose software citation metadata.

For research use, please cite the Ball Mapper methodology relevant to the analysis and cite the software release. The companion range query manuscript can be added after it receives a persistent publication identifier.

## License

MIT
