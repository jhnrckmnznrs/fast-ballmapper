# fast-ballmapper

`fast-ballmapper` is a small Python package for constructing Ball Mapper covers and graphs with interchangeable range-query backends. The package separates the mathematical Ball Mapper construction from the data structure used to answer radius queries, making exact implementations easy to compare and approximate implementations easy to audit.

## Installation

Core package:

```bash
pip install fast-ballmapper
```

Common optional dependencies:

```bash
pip install "fast-ballmapper[faiss]"   # FAISS backends
pip install "fast-ballmapper[plot]"    # Matplotlib and Plotly
pip install "fast-ballmapper[all]"     # common CPU extras
```

For development or manuscript reproduction:

```bash
git clone https://github.com/jhnrckmnznrs/fast-ballmapper.git
cd fast-ballmapper
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,plot,faiss]"
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
)
graph = build_mapper(cover)

print(len(landmarks), graph.number_of_edges())
```

`compute_landmarks` returns the ordered landmark indices and their cover sets. `build_mapper` creates the Ball Mapper graph; every graph edge stores its witness count.

## Core API

| Function or object | Purpose |
|---|---|
| `compute_landmarks` | Greedy Ball Mapper landmark selection and cover construction |
| `build_cover` | Build balls around an existing landmark set |
| `build_mapper` | Construct the overlap graph from a cover |
| `compare_covers` | Audit membership, witness, edge, and optional color differences |
| `compute_boundary_diagnostics` | Measure distances to the radius boundary |
| `make_backend` | Construct a range-query backend explicitly |
| `FaissConfig` | Configure FAISS Flat, IVF, HNSW, CPU, or GPU execution |

The public API is exported directly from `fast_ballmapper`; users normally do not need to import internal modules.

## Range-query backends

| Backend | Role |
|---|---|
| `brute_force` | transparent float64 reference |
| `ball_tree` | exact scikit-learn BallTree queries |
| `ckdtree` | independent exact SciPy cKDTree queries |
| FAISS Flat | exhaustive search over stored float32 vectors |
| FAISS IVF / HNSW | approximate candidate search |
| hnswlib / cuVS | optional external backends |

A fitted backend may also be reused directly:

```python
from fast_ballmapper import BruteForceBackend, build_cover

backend = BruteForceBackend(x)
cover = build_cover(x, landmarks, eps=0.1, backend=backend)
```

### Closed-ball convention

Ball Mapper uses

```text
B(l, eps) = {x : d(x, l) <= eps}.
```

The package applies a common boundary policy across supported backends. The brute-force reference uses float64 distances. FAISS stores vectors in float32, so points extremely close to the boundary can still differ because of numerical representation; experiment outputs are therefore audited against the float64 reference rather than assumed exact.

## Exact computation with reduced memory

The package provides optional algorithms when a complete Ball Mapper cover,
weighted graph, or connected components are needed without always materializing
a NetworkX graph. The original exact range-query methods remain the defaults.

```python
from fast_ballmapper import (
    ball_mapper_components,
    build_mapper_auto,
    compute_landmarks_uncovered,
)

landmarks, cover = compute_landmarks_uncovered(x, eps=0.1)
adjacency = build_mapper_auto(cover, len(x), output="csr")
components, labels = ball_mapper_components(cover, len(x))
```

- `select_landmarks_uncovered` performs exact Euclidean greedy selection by
  comparing a new landmark only with observations not yet covered.
- `build_cover_localized` completes exact closed balls from an independently
  verified covering assignment. Its `auto` strategy estimates geometric
  candidate counts and falls back to blocked exhaustive evaluation when
  localization is unlikely to help.
- `build_mapper_csr` uses blocked sparse witness-count products; the returned
  symmetric CSR matrix has a zero diagonal and integer overlap counts.
- `build_mapper_auto` chooses witness enumeration or sparse multiplication
  from membership statistics.
- `build_mapper_streaming` consumes an iterable of balls once, without
  retaining a separate cover list.
- `ball_mapper_components` returns exact graph component labels in nearly
  linear time in the number of cover memberships, without enumerating edges.
- `compute_mapper_streaming` selects landmarks and computes the weighted graph
  without holding all complete ball membership arrays in memory.

### Fixed-landmark multiscale connectivity

```python
from fast_ballmapper import fixed_landmark_h0, fixed_landmark_h0_from_cover

radii = [0.05, 0.1, 0.2]
profile = fixed_landmark_h0(x, landmarks, radii)
print(profile.component_counts, profile.merges)

# Avoid a second exhaustive distance computation when a trustworthy
# complete cover at max(radii) is already available.
profile = fixed_landmark_h0_from_cover(x, landmarks, radii, max_radius_cover)
```

These functions compute connected components and union events for the graph
filtration **with the same landmarks at every radius**. They are not a replacement
for recomputing greedy landmarks independently at each radius.

The new algorithms target different workloads; they are not uniformly faster
than cKDTree, BallTree, or FAISS Flat. The streaming weighted-graph algorithms
may still require quadratic work in the number of balls containing one point.
All Euclidean comparisons use direct float64 norms, so numerical boundary cases
should be audited when compared with other libraries or lower-precision indexes.

## Approximation audit

Fixed landmarks isolate range-query error from landmark-selection error:

```python
from fast_ballmapper import FaissConfig, build_cover, compare_covers

reference = build_cover(x, landmarks, eps=0.1, method="brute_force")
approximate = build_cover(
    x,
    landmarks,
    eps=0.1,
    method="faiss",
    faiss_config=FaissConfig(
        factory="HNSW32",
        query_mode="knn",
        candidate_k=4096,
        search_params={"efSearch": 64},
        exact_verify=True,
    ),
)

report = compare_covers(reference, approximate)
print(report.membership_recall, report.edge_recall)
```

`exact_verify=True` filters returned candidates by the original-coordinate radius test. It removes false-positive candidates but cannot recover true members omitted by an approximate search.

## Repository layout

```text
fast-ballmapper/
├── src/fast_ballmapper/       package implementation
├── tests/                     unit and regression tests
├── examples/                  two small executable examples
├── experiments/               manuscript validation and benchmark runner
├── README.md                  user, developer, and reproducibility guide
├── pyproject.toml             package and tool configuration
├── CITATION.cff               software citation metadata
└── LICENSE                    MIT license
```

Implementation files are separated by responsibility: landmark/cover construction, graph construction, diagnostics/audits, optional plotting, and backend adapters. This is intentionally more reviewable than a single large implementation file.

## Validation and tests

For ordinary development:

```bash
python -m ruff check .
pytest
```

For the numerical exactness gate used by the manuscript:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python experiments/run_exactness_gate.py
```

The gate compares the float64 brute-force oracle with BallTree and cKDTree on ordered landmarks, memberships, and graph output.

## Reproducing the manuscript experiment matrix

First run the smoke matrix:

```bash
python experiments/run_paper_experiments.py \
  --smoke \
  --output experiments/results/smoke
```

The main `n=20000`, `d=50` experiment is:

```bash
python experiments/run_paper_experiments.py \
  --n 20000 --d 50 \
  --seeds 0 1 2 --repeats 3 \
  --target-ball-size 50 \
  --candidate-k 4096 \
  --threads 1 \
  --output experiments/results/paper-n20000-d50
```

Search-effort sweeps use separate output directories. For IVF:

```bash
for nprobe in 1 2 4 8 16 32 64; do
  python experiments/run_paper_experiments.py \
    --n 20000 --d 50 --seeds 0 1 2 --repeats 3 \
    --target-ball-size 50 --threads 1 \
    --methods ivf --nprobe "$nprobe" \
    --output "experiments/results/ivf-nprobe${nprobe}"
done
```

For HNSW:

```bash
for ef in 32 64 128 256; do
  python experiments/run_paper_experiments.py \
    --n 20000 --d 50 --seeds 0 1 2 --repeats 3 \
    --target-ball-size 50 --candidate-k 4096 --threads 1 \
    --methods hnsw verified_hnsw_partial verified_hnsw_complete \
    --ef-search "$ef" \
    --output "experiments/results/hnsw-ef${ef}"
done
```

Run manuscript timing experiments sequentially rather than concurrently. Each worker already runs in a fresh process with the requested thread limits; concurrent benchmark invocations would introduce CPU, cache, and memory contention.

Each output directory contains three compact review files:

- `manifest.json`: source, environment, arguments, hashes, and dataset metadata;
- `summary.csv`: one row per measured run;
- `validation.json`: overall gate status and failed-run identifiers.

Large per-run worker artifacts and reference arrays are intentionally ignored by Git.

## Reviewer checklist

A compact independent review can be done with:

```bash
python -m pip install -e ".[dev,faiss]"
python -m ruff check .
pytest
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/run_exactness_gate.py
python experiments/run_paper_experiments.py --smoke --output /tmp/fast-ballmapper-smoke
```

The experiment runner records source hashes and backend configuration so reported measurements can be tied to the code that produced them.



## Exact CSR-first numerical workflows

For numerical algorithms or large graphs, keep the weighted adjacency in SciPy
CSR format. The original `build_mapper(cover)` continues to return a NetworkX
graph for compatibility; `build_mapper_auto` returns CSR by default:

```python
from fast_ballmapper import build_mapper_auto, compute_landmarks

landmarks, cover = compute_landmarks(x, eps=0.1, method="ckdtree")
adjacency = build_mapper_auto(cover, len(x))  # scipy.sparse.csr_matrix
# adjacency[i, j] = number of observations shared by balls i and j
```

Automatic graph selection uses a conservative size/witness heuristic:
streaming for tiny sparse covers, sparse matrix multiplication otherwise.
Use `algorithm="csr"` or `algorithm="streaming"` to force a method, and
`output="networkx"` only when NetworkX functionality is required. No edge
weights or isolated vertices are lost in CSR.

When completing a verified ownership-based cover, batched NumPy membership
checks now amortize per-group overhead without changing the float64 radius
predicate:

```python
from fast_ballmapper import select_landmarks_uncovered, build_cover_localized

landmarks, owners = select_landmarks_uncovered(x, eps=0.1)
cover = build_cover_localized(
    x, landmarks, eps=0.1, owners=owners,
    strategy="localized", engine="batch",
)
```

An optional `engine="numba"` uses JIT-compiled conservative candidate
screening followed by direct float64 NumPy verification. Install
`pip install "fast-ballmapper[jit]"` to enable it. JIT warmup has a
first-call cost; for most workloads batching is the recommended default.
The compiled prefilter's floating-point guard is conservative in tested
cases, but is not an interval-arithmetic proof for all possible inputs.
The existing cKDTree remains the recommended general-purpose exact
cover-construction backend.

Run `python experiments/benchmark_dispatch_and_completion.py --smoke` for
a quick correctness-checked comparison, or without `--smoke` for the
full synthetic experiment. Warm-up and steady-state times are separate.


## Exact robustness under per-ball omission budgets

For a **complete fixed-landmark cover**, the new exact-certificate API checks
whether any conservative omission of at most `budgets[i]` observation
memberships from ball `i` can increase the number of connected components.

```python
from fast_ballmapper import certify_connectivity

cover = [[0, 1], [0, 1]]
result = certify_connectivity(cover, [1, 1], method="auto")
assert result.robust is False
assert result.deletions   # independently verified destructive omission witness
```

The result contains `robust: True | False | None`: `True` means the exact
solver completed without finding a permissible disconnection; `False`
means a disconnection and its verified membership-deletion witness;
`None` means *inconclusive* (time or state limit). The method parameter
selects `"auto"` (biconnected blocks + measured-width DP/MILP dispatch),
`"milp"` (pattern-compressed SciPy/HiGHS), or `"dp"` (exact integer-state
min-fill dynamic programming for a connected input). The DP's width cutoff
is measured, not a certified separator width. The SciPy/HiGHS infeasibility
status is numerical solver evidence rather than a formally checked
integer-arithmetic proof.

The fast existing `certify_component_preservation` remains available as a
**sufficient** certificate. These methods do not infer or verify that an
approximate search backend actually respects a claimed omission budget.
They require the complete exact reference cover, fixed landmark indices,
and no added ball memberships. Graph vertices persist even if a ball is
emptied. For reproducible studies, see
`experiments/benchmark_budget_certificates.py`.

## Citation

Software citation metadata are provided in `CITATION.cff`. For research use, cite both the Ball Mapper methodology relevant to the analysis and the software release used for computation.

## License

MIT.
