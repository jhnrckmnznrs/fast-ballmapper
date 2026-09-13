# Repository audit for the manuscript experiments

Reviewed base: `5e8604cf9816947ef24cadcfeef7ab1b8c6a3fca`, including merged
PRs #1 (0.2.0 preparation) and #2 (README polish). Audit date: 2026-09-13.

## Assessment

The release supplied the right backend and audit foundations, but needed updates
before the proposed experiments. This revision fixes confirmed correctness and
measurement issues and implements the missing comparison paths. It establishes
readiness to run the new CPU study. It does not establish a performance advantage
or complete the final scientific audit of the paper.

## Confirmed issues addressed

1. Cosine FAISS construction could normalize a caller-owned contiguous float32
   array in place. Index and verification inputs are now separate snapshots.
2. Flat kNN queries with a candidate cap below n were labeled exact. Metadata now
   accounts for the cap; numerical representation remains a separate caveat.
3. Exact verification previously received only candidates that had already
   passed approximate kNN-distance filtering. It now receives every retrieved ID,
   so quantized distance overestimates cannot prematurely reject those IDs.
4. Fixed-landmark FAISS querying looped over single queries. Bounded native
   batches now expose the intended scalar/batch comparison.
5. Generic greedy selection repeatedly searched the full coverage mask and
   relied on an oracle returning its query point. It now scans forward and
   explicitly enforces self-membership and valid integer IDs.
6. The exactness script exited successfully even when equality fields failed.
   All relevant mismatches, including cKDTree memberships, now fail the gate.
7. Float32 color means were compared with float64 oscillation bounds. The smoke
   matrix exposed spurious tight-bound violations; float64 accumulation fixes
   them, with a regression case using values `[0.1, 0.2]`.
8. The legacy experiment script was insufficient to reconstruct the historical
   three-geometry matrix. A new explicit generator/configuration specification
   now records all data, references, hashes, and settings needed for reruns.

## Paper-to-implementation mapping

| Proposed addition | Implementation and validation |
| --- | --- |
| Exact ordered landmarks despite candidate omissions | `compute_landmarks_verified`; independent SciPy pairwise-distance reference, adversarial/empty candidates, duplicates and boundary cases |
| Exact membership completion | `build_cover_blocked`; agreement with independent reference balls and explicit completion-work count |
| Sparse witness multiplicities | `build_mapper_sparse`; actual row blocking, per-ball deduplication, int64 counts, independent set-intersection checks including 300 shared witnesses |
| Witness-budget connectivity certificate | `certify_component_preservation`; exhaustive small admissible omissions, sufficient-condition and inconclusive cases |
| Fixed-landmark batching | Native FAISS range/kNN batches; call-size/order checks and exact-output gates |
| Inflated cKDTree candidates | `CKDTreeBackend(candidate_eps=...)`; direct filtering and oracle comparisons |
| Changing-landmark observed-nerve bound | Exhaustive small simplex/witness-map/contiguity examples in `tests/test_verified.py`; full persistence computation remains outside this runner |

The earlier paper supplement is a prototype. The repository now supplies actual
sparse row blocking and stronger checks than that prototype. At the final paper
revision, update Appendix C to cite this reviewed commit and these tests, and
replace its earlier statement that the inspected FAISS adapter only loops over
single queries. Preserve red markup for any new LaTeX edits.

## Validation performed

- `python -m ruff check .`: passed.
- `python -m pytest --cov=fast_ballmapper --cov-report=term-missing`:
  **118 passed, 1 skipped**, 81% aggregate coverage. The skipped check requires
  real FAISS GPU hardware. Optional hnswlib/cuVS adapter tests use mocks; the new
  experiment runner uses real CPU FAISS Flat, IVF and HNSW.
- Full `run_exactness_gate.py`: both cases passed across brute force, BallTree,
  and cKDTree. The n=1000, d=16 case produced 297 landmarks / 3,686 edges. The
  n=5000, d=32 case reproduced 1,263 landmarks / 85,588 edges.
- New runner: **72/72 smoke runs passed**, with all three geometries, all twelve
  methods and both graph constructors. The deliberately restrictive settings
  were `--smoke --candidate-k 8 --nprobe 1 --ef-search 16 --row-block-size 7`.
  Approximate queries had genuine omissions; verified completion recovered all
  reference memberships. No smoke timing is presented as a benchmark result.
- Wheel/source build and `twine check`: passed. The source distribution includes
  experiment scripts and excludes generated results.

Validated runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0,
NetworkX 3.6.1, FAISS CPU 1.15.0, Linux x86_64. Source digest over Python files in
`src/` and `experiments/` during the final smoke run:
`0bdd2feb44bef7857b54a787dc9c6acd9cb53d19f4304b059a923cc6aee3b0ac`.
The local validation checkout had pending changes, recorded as such in its
manifest; production runs should use the reviewed commit and a clean checkout.

## Remaining evidence for the manuscript

Run the full matrix and search-effort/scaling sweeps using the
[experiment guide](../experiments/README.md), preserving raw outputs and manifest.
Report stage times, repetitions/seeds, peak memory, m, S, T, E, membership/edge
quality and witness survival. Exact speed comparisons require identical outputs.
Keep the original reported matrix separate until its original data and generator
versions are recovered or its tables are replaced by the new reproducible study.

The persistence assumptions still require fixed landmark sets, full observed
nerves, nested filtrations and uniform radius control. These single-radius CPU
experiments do not verify those assumptions for arbitrary ANN indexes, and no
GPU performance claim has been validated here.
