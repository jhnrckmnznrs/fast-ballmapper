# Experiments for the strengthened manuscript

Use a reviewed commit of this source tree and record its SHA. The new runner
implements the proposed comparisons. The older `run_fixed_landmarks.py` is a
legacy Gaussian-only exploratory script; it does not reconstruct the historical
three-geometry matrix described in `EXPERIMENT_STATUS.md`.

## Install and validate

Python 3.12 is recommended for the CPU experiment environment. From the repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,plot,faiss]"
python -m ruff check .
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/run_exactness_gate.py
python experiments/run_paper_experiments.py --smoke --output experiments/results/smoke
```

The exactness gate exits nonzero for any landmark, membership, or edge mismatch.
The smoke matrix covers three geometries, twelve methods, and two graph
implementations (72 small runs). It checks execution and correctness. Its timings
are unsuitable for manuscript performance claims.

An explicit selection of core-only methods works without FAISS:

```bash
python experiments/run_paper_experiments.py --smoke \
  --methods brute_force ball_tree ckdtree ckdtree_inflated \
    verified_none_partial verified_none_complete \
  --output experiments/results/core-smoke
```

## Run the study

```bash
python experiments/run_paper_experiments.py \
  --n 20000 --d 50 --seeds 0 1 2 --repeats 3 \
  --target-ball-size 50 --candidate-k 4096 --threads 1 \
  --output experiments/results/paper-n20000-d50
```

Use a fresh output directory for each invocation; existing results are preserved.
The default study contains 648 isolated runs and includes exhaustive reference
construction and audits. Start with the smoke run, then assess cost on `n=5000`.
Run larger sizes on the experiment machine after assessing memory use. The final
NetworkX graph can require quadratic storage even with blocked construction.

Sweep search effort using separate output directories, for example:

```bash
python experiments/run_paper_experiments.py \
  --n 20000 --d 50 --methods ivf --nprobe 1 \
  --output experiments/results/ivf-nprobe1
python experiments/run_paper_experiments.py \
  --n 20000 --d 50 --methods hnsw verified_hnsw_partial verified_hnsw_complete \
  --candidate-k 4096 --ef-search 32 \
  --output experiments/results/hnsw-ef32
```

Repeat with `--nprobe 2 4 8 16 32 64` **as separate invocations**, and
`--ef-search 64`, `128`, `256`. Each option takes one value. Vary `--candidate-k`
separately: a cap exceeding the largest exact ball removes a cardinality
obstruction but does not guarantee ANN recall. The JSON stores actual FAISS
configuration, including clipped `nlist`, `nprobe`, and candidate cap.

For scaling, vary `--n`, `--d`, and target ball size. Keep data, order, epsilon,
threads, and output requirements identical within a comparison. The runner
randomizes method order using a fixed scheduling seed and runs repetitions in
fresh processes. It performs no hidden warm-up. Report median and dispersion
over repetitions, then variability over data seeds; repetitions of one seed are
not independent datasets.

## What is compared

| Method | Selection | Fixed memberships |
| --- | --- | --- |
| `brute_force` | Float64 exhaustive greedy | Exhaustive float64 |
| `ball_tree`, `ckdtree` | Tree greedy | Batched tree queries |
| `ckdtree_inflated` | Inflated candidates, filtered | Same, `--candidate-eps` controls inflation |
| `flat_scalar`, `flat_batch` | Sequential FAISS Flat greedy with float64 filtering | Batch size 1 or `--batch-size`, with float64 filtering |
| `ivf`, `hnsw` | Approximate greedy, measured separately | Queries at the common reference landmarks |
| `verified_none_partial` | Complete coverage tests, no candidate provider | Verified partial cover |
| `verified_none_complete` | Same | Additional blocked exhaustive reconstruction |
| `verified_hnsw_partial` | HNSW candidates plus complete coverage tests | Verified partial cover |
| `verified_hnsw_complete` | Same | Additional blocked exhaustive reconstruction |

Both inverted membership and blocked sparse graph construction run on the same
method's memberships. Sparse multiplication uses int64 counts after per-ball
deduplication. Its row-block intermediate has at most `row_block_size * m`
entries, in addition to incidence and graph storage.

The verified selector has the reference landmark sequence even with arbitrary
candidate omissions, relative to direct float64 Euclidean comparisons. Complete
current-landmark tests can cost O(n*m); they are included in selection time and
distance counters. A partial cover covers the observations but may lose graph
edges and alter colors. In particular, the no-candidate partial control assigns
each point to one ball and need not have any edges. Use completion when comparing
exact graph output.

`staged_total_seconds` sums index build, greedy selection, a separate fixed-cover
query/completion stage, and graph construction. Selection itself produces
memberships, so ordinary methods repeat membership work in the fixed-cover
stage. This total describes the staged experimental workload; do not present it
as the latency of a single ordinary `compute_landmarks` + `build_mapper` call.
Use stage comparisons to assess batching and graph kernels.

## Reproducibility and numerical interpretation

Each invocation saves:

- `manifest.json`: commit, dirty state, source hash, dependencies, hardware,
  arguments, actual data/landmark hashes, epsilon, and reference output sizes;
- `*_reference.npz`: the generated float32 observations, ordered float64-reference
  landmark IDs, complete memberships, and calibrated epsilon;
- `job_*.json`, `run_*.json`: parameters, actual backend configuration, thread
  pools, stage timings, peak process RSS, audit summaries, witness-survival bins;
- `run_*.npz`: selected landmarks and per-ball membership/color error arrays;
- `summary.csv` and `validation.json`: combined results and validation outcome.

`pipeline_peak_rss_bytes` is the fresh worker's process high-water mark measured
before reference memberships and audits are loaded. It includes imports, input
data, index, transient selection, membership, and graph allocations; it is not
an incremental allocation or a per-stage maximum. It is unavailable on platforms
without `resource`. Reference preparation and auditing are outside reported
pipeline stage times and RSS. These experiments are CPU-only.

The newly specified geometries are standard Gaussian, a four-component Gaussian
mixture, and a noisy trigonometric curve. Float64 direct norms select the sampled
median target-order radius; epsilon is not expanded during calibration. These
generators are new reproducible specifications, not reconstructions of missing
historical generator code.

An outward `nextafter` implements inclusive comparison only for a threshold and
reported distances in the same representation. It does not bound arithmetic,
normalization, squaring, float32 conversion, or quantization error. Batched FAISS
can select a different distance kernel and differ at the numerical boundary.
The runner records the FAISS BLAS threshold and audits each output against the
float64 reference. Both Flat configurations use `exact_verify=True`: returned
candidates are filtered using original-coordinate float64 norms, and that cost
is included in selection/membership timings. Preserve old unverified Flat rows
as a separate configuration; their timings are not measurements of this filtered
configuration. A candidate filter cannot restore points omitted by a native
range search. See the [FAISS implementation notes](https://github.com/facebookresearch/faiss/wiki/Implementation-notes).

For cKDTree approximation parameter u, querying at radius (1+u)*epsilon provides
a candidate superset under the ideal arithmetic pruning contract, then direct
float64 filtering retains the ball. Floating-point boundary equality remains an
empirical gate. See [SciPy's query contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html).

## Gates for the final paper audit

Exact comparisons require identical ordered landmarks, memberships, and witness
counts. Inspect `exact_output_match`; a failed equality cannot support an exact
speedup claim. Approximate/partial runs instead require conservative membership,
valid color bounds, and correct witness computation; differences in graph
topology are measured outcomes. `memberships_complete` records observed equality
to the full reference balls. Requested completion is recorded separately.

On failure, the console and each run's `validation_failures` list identify the
failed checks. `validation.json` includes a `failed_runs` list with filenames,
dataset/method identities and reasons. Worker exceptions are also retained in
the run JSON and summary CSV. Inspect these outputs before rerunning; the
aggregate failure message alone does not identify the cause.

The component certificate uses the *observed* per-ball omission counts as budgets
and the full exact reference. It is an a posteriori sufficient certificate;
failure is inconclusive, and it is not a cheaper substitute for the exact audit.
Witness bins use powers of two as lower bounds, e.g. bin 4 covers counts 4--7.

The persistence claims concern the full observed nerve, fixed landmark sets,
nested filtrations, and uniform radius-error assumptions. The small exhaustive
simplex tests check examples of witness maps and contiguity. These graph/range
experiments do not establish a persistence theorem for single-radius ANN runs
or clique-completed Ball Mapper graphs. Full-scale measurements and a final
claim-to-result review are still required before updating paper tables.
