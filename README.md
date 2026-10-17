# fast-ballmapper

A Python package for constructing, visualizing, and studying **Ball Mapper** graphs. It supports different radius-query methods without changing the underlying construction.

Given a point cloud `x` and radius `eps`, Ball Mapper selects landmarks, forms balls around them, and joins two landmarks when their balls contain a common observation.

## Install

```bash
pip install fast-ballmapper
```

For plotting, install `fast-ballmapper[plot]`. Optional search backends are available through `[faiss]`, `[hnswlib]`, and `[cuvs-cu12]` or `[cuvs-cu13]`. Use `pip install -e ".[dev,plot]"` when working from a local clone.

## Quick start

```python
import numpy as np
from fast_ballmapper import compute_landmarks, build_mapper

x = np.random.default_rng(42).random((500, 2))
eps = 0.1

landmarks, cover = compute_landmarks(x, eps, method="ckdtree")
graph = build_mapper(cover)

print(f"{len(landmarks)} landmarks, {graph.number_of_edges()} edges")
```

- `landmarks` contains row indices into `x`, in greedy selection order.
- `cover[j]` contains the row indices of observations in the ball around `x[landmarks[j]]`.
- Vertex `j` in `graph` represents that landmark. An edge `(i, j)` means `cover[i]` and `cover[j]` share an observation; its `witness_count` attribute counts the shared observations.

For a plotted example with node sizes, run [`examples/basic_usage.py`](examples/basic_usage.py) after installing the plotting extra.

## Find the function you need

The functions below follow the usual Ball Mapper workflow. Most can be imported directly from `fast_ballmapper`.

| Step | What you want to do | Function(s) | Implementation |
|---|---|---|---|
| 1. Landmarks | Select greedy landmarks and construct their balls | `compute_landmarks` | [`landmarks.py`](src/fast_ballmapper/landmarks.py) |
| 1. Landmarks | Use farthest-point sampling instead | `compute_landmarks_fps` | [`landmarks.py`](src/fast_ballmapper/landmarks.py) |
| 2. Cover | Construct balls around landmarks already chosen | `build_cover` | [`landmarks.py`](src/fast_ballmapper/landmarks.py) |
| 3. Graph | Build the overlap graph (NetworkX) | `build_mapper` | [`graph.py`](src/fast_ballmapper/graph.py) |
| 3. Graph | Recalculate shared-observation counts | `compute_edge_overlaps` | [`graph.py`](src/fast_ballmapper/graph.py) |
| 4. Color | Color by a function, class, entropy, size, or density | `color_by_function`, `color_by_mode`, `color_by_entropy`, `color_by_size`, `color_by_density` | [`coloring.py`](src/fast_ballmapper/coloring.py) |
| 5. Plot | Draw with Matplotlib or Plotly | `draw_ball_mapper`, `draw_ball_mapper_plotly` | [`plotting/`](src/fast_ballmapper/plotting) |
| 6. Compare | Compare exact and approximate covers or inspect radius boundaries | `compare_covers`, `compute_boundary_diagnostics` | [`audit.py`](src/fast_ballmapper/audit.py), [`diagnostics.py`](src/fast_ballmapper/diagnostics.py) |

Plotting functions are imported from `fast_ballmapper.plotting.matplotlib` or `fast_ballmapper.plotting.plotly`, rather than the package root.

## Choose a radius-query method

Pass `method=...` to `compute_landmarks` or `build_cover`. The construction stays the same; the method determines how candidate observations are found.

| Method | Use |
|---|---|
| `"ball_tree"` | Default; exact radius queries using scikit-learn BallTree |
| `"ckdtree"` | Exact radius queries using SciPy cKDTree |
| `"brute_force"` | Simple float64 reference for checking results |
| `"faiss"` | FAISS Flat or approximate IVF/HNSW searches, configured with `FaissConfig` |
| `"hnswlib"`, `"cuvs"` | Additional optional search backends |

The intended balls are **closed**: an observation belongs to a ball when its distance from the landmark is at most `eps`. Numerical boundary behavior can differ across libraries, especially those using float32. Approximate queries may omit members, which can change the selected landmarks or graph.

To compare methods fairly, **hold the landmarks fixed**:

```python
from fast_ballmapper import build_cover, compare_covers

reference = build_cover(x, landmarks, eps, method="brute_force")
other = build_cover(x, landmarks, eps, method="ckdtree")
report = compare_covers(reference, other)

print(report.membership_recall, report.edge_recall)
```

For an approximate FAISS example, see [`examples/approximation_audit.py`](examples/approximation_audit.py). Even with candidate verification, an approximate search cannot automatically recover observations it failed to retrieve.

## Larger computations

The standard `build_mapper(cover)` returns NetworkX, which is convenient for plotting. For larger covers, use sparse adjacency or compute connected components without building the full graph:

```python
from fast_ballmapper import build_mapper_auto, ball_mapper_components

adjacency = build_mapper_auto(cover, len(x))  # SciPy CSR; shared-observation counts
n_components, labels = ball_mapper_components(cover, len(x))
```

Useful alternatives are grouped here instead of replacing the ordinary workflow:

| Task | Function(s) | Implementation |
|---|---|---|
| Return a CSR overlap graph, with explicit algorithm choice | `build_mapper_csr`, `build_mapper_auto`, `build_mapper_streaming` | [`optimized.py`](src/fast_ballmapper/optimized.py) |
| Build a graph while generating balls, without keeping the full cover | `compute_mapper_streaming` | [`optimized.py`](src/fast_ballmapper/optimized.py) |
| Select greedy landmarks using uncovered points, then complete their balls | `compute_landmarks_uncovered`, `select_landmarks_uncovered`, `build_cover_localized` | [`optimized.py`](src/fast_ballmapper/optimized.py) |
| Verify greedy selection or complete a cover with bounded distance workspace | `compute_landmarks_verified`, `build_cover_blocked` | [`verified.py`](src/fast_ballmapper/verified.py) |
| Track connected components across radii with **fixed landmarks** | `fixed_landmark_h0`, `fixed_landmark_h0_from_cover` | [`optimized.py`](src/fast_ballmapper/optimized.py) |
| Check connectivity under limits on missing ball memberships | `certify_component_preservation`, `certify_connectivity` | [`sparse.py`](src/fast_ballmapper/sparse.py), [`budget_certificates.py`](src/fast_ballmapper/budget_certificates.py) |

`build_mapper_auto` returns **CSR by default**; use `output="networkx"` if needed. Its `algorithm` option accepts `"auto"`, `"csr"`, or `"streaming"`. The localized-cover path also supports batched NumPy evaluation and optional Numba acceleration (`pip install "fast-ballmapper[jit]"`). These alternatives are workload-dependent; they are not always faster than cKDTree.

For connectivity certificates, `certify_component_preservation` gives a fast sufficient test. `certify_connectivity` searches more precisely for allowed missing memberships that could disconnect the graph; its `robust` result is `True`, `False`, or `None` (inconclusive). Both require a **complete exact reference cover** and fixed landmarks. The certificate does not establish that a particular approximate backend satisfies the assumed limits.

## Examples, tests, and experiments

| Resource | Where |
|---|---|
| Basic Ball Mapper and plot | [`examples/basic_usage.py`](examples/basic_usage.py) |
| Compare covers from different search methods | [`examples/approximation_audit.py`](examples/approximation_audit.py) |
| Unit and regression tests | [`tests/`](tests) |
| Radius-query exactness checks | [`experiments/run_exactness_gate.py`](experiments/run_exactness_gate.py) |
| Manuscript experiments | [`experiments/run_paper_experiments.py`](experiments/run_paper_experiments.py) |
| Graph and cover benchmarks | [`experiments/benchmark_dispatch_and_completion.py`](experiments/benchmark_dispatch_and_completion.py) |
| Connectivity-certificate benchmarks | [`experiments/benchmark_budget_certificates.py`](experiments/benchmark_budget_certificates.py) |

From a development installation:

```bash
python -m ruff check .
pytest
python experiments/run_paper_experiments.py --smoke --output experiments/results/smoke
```

The experiment runner writes `manifest.json`, `summary.csv`, and `validation.json` so configurations and results can be inspected. More extensive experiment options are available through the scripts' `--help` flags.

## Citation and license

See [`CITATION.cff`](CITATION.cff) for software citation details. Cite the relevant Ball Mapper methodology as well as the software version used in your analysis.

Licensed under [MIT](LICENSE).
