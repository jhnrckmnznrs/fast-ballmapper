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
