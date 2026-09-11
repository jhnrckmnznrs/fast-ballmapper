# Changelog

All notable changes to `fast-ballmapper` are documented here.

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
