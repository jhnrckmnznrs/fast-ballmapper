# Changelog

All notable changes to `fast-ballmapper` are documented here.

## Unreleased

### Fixed

- Preserve caller inputs and original-coordinate verification snapshots when
  constructing cosine FAISS indexes.
- Mark capped Flat kNN searches approximate and record actual GPU storage dtype.
- Verify every retrieved kNN candidate before radius rejection when exact
  verification is enabled, avoiding rejection based on quantized overestimates.
- Accumulate audit color means in float64 to avoid spurious tight-bound failures
  from float32 accumulation.
- Make generic greedy selection progress when a custom backend omits the query
  point; reject noninteger observation IDs rather than truncating them.
- Fail the exactness gate on mismatches, including cKDTree membership errors.

### Added

- Bounded native FAISS query batches for fixed landmarks, configured by
  `FaissConfig.query_batch_size`; greedy decisions remain sequential.
- `compute_landmarks_verified`, `VerifiedSelection`, and `build_cover_blocked`.
- `build_mapper_sparse`, `cover_statistics`, and a witness-budget component
  certificate with integer counts and bounded sparse row products.
- Inflated and filtered cKDTree candidate search via `candidate_eps`.
- A CPU experiment runner with three explicit generators, seeds, repetitions,
  isolated processes, stage timings, memory, provenance, and correctness gates.
- Independent-distance, adversarial-candidate, wide-witness, and small observed
  nerve contiguity tests, plus experiment smoke checks in CI.

## 0.2.0 - 2026-09-11

### Added

- Public `RangeQueryBackend` protocol and `BackendMetadata` capability metadata.
- Reusable exact `BruteForceBackend`, `BallTreeBackend`, and SciPy
  `CKDTreeBackend` implementations.
- Reusable `FaissFlatBackend`, `FaissIVFBackend`, and `FaissHNSWBackend`
  classes, plus optional `HnswlibBackend` and NVIDIA `CuVSBackend` adapters.
- `make_backend(...)` factory while preserving the legacy `method=` API.
- Float64 brute-force reference oracle for backend agreement tests.
- Per-landmark boundary-margin diagnostics and boundary-band counts.
- Edge `witness_count` attributes and identity-aware witness auditing.
- `compare_covers` / `ApproximationAudit` for membership, edge, witness,
  boundary, and optional coloring-error diagnostics.
- Reproducible closed-ball exactness gate and fixed-landmark FAISS benchmark.

### Changed

- Standardized Ball Mapper neighborhoods as closed balls and centralized
  floating-point boundary handling with `np.nextafter` where a backend uses a
  strict comparison.
- Kept epsilon calibration unexpanded so outward rounding is applied exactly
  once at backend comparison boundaries.
- Batched fixed-landmark BallTree cover queries.
- Restricted farthest-point sampling to backends that provide exhaustive
  distances, preventing candidate-only approximate engines from silently
  changing FPS semantics.
- Cleaned packaging metadata, optional dependency groups, CI, citation
  metadata, and contributor/backend documentation.

### Validation

- Added exact-boundary, next-float-outside, and zero-radius duplicate tests.
- Added independent SciPy `cKDTree` equivalence tests for greedy landmarks,
  fixed covers, exact-boundary behavior, and farthest-point sampling.
- Added mocked optional-adapter tests for hnswlib and cuVS candidate filtering
  and backend metadata.

## 0.1.0

- Reorganized the project into a PyPI-ready `src` layout.
- Renamed the distribution to `fast-ballmapper` and the import package to
  `fast_ballmapper`.
- Converted the public API and internal Python identifiers to snake_case.
- Split landmark selection, backends, graph construction, coloring, and
  plotting into focused modules.
- Made FAISS, Matplotlib, and Plotly optional extras.
- Added automated tests and package build metadata.
