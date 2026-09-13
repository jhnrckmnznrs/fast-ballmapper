# Range-query backend architecture

`fast-ballmapper` separates Ball Mapper logic from the search engine used to
answer metric-ball queries. A backend implements the public
`RangeQueryBackend` protocol and exposes `BackendMetadata` describing whether
it is exact, its metric, numerical dtype, device, batch-query support, and
whether exhaustive distance vectors are available.

## Core contract

For a fitted data matrix `x`, a backend must return the observation indices in
the closed ball

```text
B(x[i], eps) = {j : d(x[j], x[i]) <= eps}.
```

The package centralizes floating-point boundary handling in `_radius.py` so
that APIs using strict predicates and APIs using inclusive predicates implement
the same mathematical closed ball.

This is a comparison convention, not an arithmetic error bound. Equality across
representations or distance kernels is checked empirically. A supplied backend
must index the same data and row order as `x`; the generic protocol can check
sample count but cannot establish data identity for an arbitrary custom engine.

## Exact backends

- `BruteForceBackend`: transparent float64 reference oracle.
- `BallTreeBackend`: exact scikit-learn BallTree queries with metric
  flexibility.
- `CKDTreeBackend`: independent exact SciPy `cKDTree` implementation for
  Euclidean distance.
- `FaissFlatBackend`: exhaustive relative to its indexed float32
  representation; points adversarially close to the boundary can still differ
  from the float64 reference because of representation rounding.

## Approximate backends

FAISS IVF/HNSW, hnswlib, and cuVS CAGRA may omit true members. Candidate-based
adapters make `candidate_k` explicit and apply the same closed-ball filter to
returned candidates. This separates candidate-generation error from radius
verification.

`FaissRangeBackend` uses true bounded native batches for fixed landmarks. A Flat
index with `query_mode="knn"` and `candidate_k < n` is also approximate. With
`exact_verify=True`, every retrieved kNN ID is verified against original float64
coordinates before radius rejection. A native range query can already have
omitted candidates before verification. Scalar and batched FAISS kernels can
make different numerical boundary decisions.

The generic greedy loop scans the observation order once and explicitly inserts
each landmark into its own neighborhood. For an exact ordered landmark sequence
with arbitrary candidate omissions, use `compute_landmarks_verified`: verified
candidate marking is supplemented by exhaustive current-landmark coverage tests.
Its partial memberships cover the observations but need not reconstruct all
edges; `build_cover_blocked` provides complete reference-predicate memberships.

## Farthest-point sampling

FPS requires the distance from each selected landmark to every observation.
For this reason, `compute_landmarks_fps` rejects backends whose metadata does
not advertise exhaustive distance-vector support. Candidate-only ANN engines
cannot silently substitute an approximate FPS rule.

## Adding a backend

1. Implement `RangeQueryBackend`.
2. Populate truthful `BackendMetadata`.
3. Reuse `_radius.py` for boundary comparisons.
4. Add equivalence tests against `BruteForceBackend` if exact.
5. If approximate, add fixed-landmark audits with `compare_covers` and expose
   all candidate/search-effort parameters.
6. Register the backend in `make_backend(...)` only after the direct class API
   is tested.
