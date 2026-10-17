---
layout: default
title: Advanced computations
permalink: /advanced/
description: Sparse graphs, exact components, optimized cover completion, and connectivity certificates.
---

# Advanced computations

The standard `compute_landmarks` and `build_mapper` functions are a good starting point. The alternatives below are designed for specific computational needs; they are **not universally faster**.

## Sparse graphs

`build_mapper` returns a NetworkX graph, which is convenient for plotting. For numerical work, `build_mapper_auto` instead returns a symmetric SciPy CSR adjacency matrix whose nonzero entries are the integer shared-observation counts.

```python
from fast_ballmapper import build_mapper_auto, ball_mapper_components

adjacency = build_mapper_auto(cover, len(x))
n_components, labels = ball_mapper_components(cover, len(x))
```

Use `build_mapper_auto(cover, len(x), output="networkx")` when a NetworkX object is needed, or `algorithm="csr"` / `"streaming"` to override the automatic selection.

`ball_mapper_components` avoids constructing the entire weighted graph when only the connected-component partition is needed.

## Reduced-memory cover construction

- `compute_landmarks_uncovered` preserves exact greedy landmark selection in Euclidean data by focusing on uncovered observations and then completing the cover.
- `build_cover_localized` uses verified first-cover assignments to restrict candidate distances; it also has a blocked fallback.
- `compute_mapper_streaming` builds the weighted graph without keeping the complete cover as a separate list.
- `build_cover_blocked` constructs complete balls without allocating a full landmark-by-observation distance matrix.

The localized path supports batched NumPy evaluation. An optional JIT prefilter is available with `pip install "fast-ballmapper[jit]"`; it requires numerical care near radius boundaries.

## Connectivity across radii

`fixed_landmark_h0` tracks the connected components of Ball Mapper graphs as the radius varies **while the landmark set stays fixed**.

```python
from fast_ballmapper import fixed_landmark_h0

profile = fixed_landmark_h0(x, landmarks, [0.05, 0.10, 0.15])
print(profile.component_counts)
```

If you already have the **complete** cover at the largest radius, `fixed_landmark_h0_from_cover` can reuse it.

This is not the same as selecting a fresh set of greedy landmarks independently at every radius.

## Connectivity under missing memberships

Two related tools analyze whether losing members from the balls can change graph connectivity:

- `certify_component_preservation` performs a fast **sufficient** test.
- `certify_connectivity` attempts an exact decision using integer optimization and/or tree-decomposition dynamic programming.

```python
from fast_ballmapper import certify_connectivity

result = certify_connectivity(
    cover, [1] * len(cover), n_samples=len(x), method="auto"
)
print(result.robust)  # True, False, or None
```

The integers in the second argument are **upper bounds on how many memberships may be missing from each ball**. A `False` result includes a checked disconnection witness; `None` is inconclusive because a solver limit was reached. Numerical solver infeasibility is not a formally machine-verified integer proof.

These certificates require a complete exact reference cover, fixed landmarks, and omissions only. They do not verify that a search backend actually respects the assumed limits.

For details, inspect [`budget_certificates.py`](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/budget_certificates.py) or run [the certificate benchmark](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/experiments/benchmark_budget_certificates.py).
