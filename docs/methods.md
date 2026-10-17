---
layout: default
title: Search methods
permalink: /methods/
description: Exact and approximate radius-query backends supported by fast-ballmapper.
---

# Search methods

A **range query** asks which observations lie within radius `eps` of a landmark. In `fast-ballmapper`, these queries can use interchangeable backends.

## Common methods

| Method | Description | When to use |
|---|---|---|
| `"ball_tree"` | scikit-learn BallTree | Default exact search |
| `"ckdtree"` | SciPy cKDTree | Exact Euclidean search |
| `"brute_force"` | Direct float64 distance computation | Reference and validation |
| `"faiss"` | Flat, IVF, or HNSW through FAISS | Exhaustive or approximate search, depending on configuration |
| `"hnswlib"` | Approximate graph-based search | Optional candidate search |
| `"cuvs"` | Optional GPU-focused search backends | Supported CUDA environments |

```python
from fast_ballmapper import compute_landmarks

landmarks, cover = compute_landmarks(x, eps=0.1, method="ball_tree")
```

Most users should start with `ball_tree` or `ckdtree`. Benchmark alternative methods on your data before assuming one is faster.

## Exact versus approximate

A complete ball contains every observation satisfying the chosen distance threshold. Approximate candidate searches may omit some members. An approximate `compute_landmarks` call may also change which observations become landmarks.

To compare the range queries themselves, keep the landmarks fixed:

```python
from fast_ballmapper import build_cover, compare_covers

reference = build_cover(x, landmarks, eps=0.1, method="brute_force")
other = build_cover(x, landmarks, eps=0.1, method="ckdtree")

audit = compare_covers(reference, other)
print(audit.membership_recall, audit.edge_recall)
```

A FAISS configuration may request verification of returned candidates against the original coordinates. This removes false-positive candidates but **cannot restore missing true members**.

## Numerical boundary

Balls are closed: the membership condition is `distance <= eps`. FAISS works with float32 vectors in its indexes, whereas the brute-force reference uses float64; these numerical representations may differ at or near a radius boundary.

When exact membership matters, compare the backend against the reference and inspect [boundary diagnostics](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/diagnostics.py).

## Additional installations

```bash
pip install "fast-ballmapper[faiss]"
pip install "fast-ballmapper[hnswlib]"
```

GPU and plotting extras are listed in [`pyproject.toml`](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/pyproject.toml). The [approximation audit example](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/examples/approximation_audit.py) shows a complete fixed-landmark comparison.
