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

## Citation

Software citation metadata are provided in `CITATION.cff`. For research use, cite both the Ball Mapper methodology relevant to the analysis and the software release used for computation.

## License

MIT.
