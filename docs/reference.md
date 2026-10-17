---
layout: default
title: Function reference
permalink: /reference/
description: Public fast-ballmapper functions and their implementation modules.
---

# Function reference

This is a **map of the public API**, not a duplicate of the source documentation. Most functions below are available from `fast_ballmapper`; plotting functions live in the `plotting` subpackage.

## Landmarks and cover

| Function | Use | Source |
|---|---|---|
| `compute_landmarks` | Greedy landmarks and their radius-ball cover | [landmarks.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/landmarks.py) |
| `compute_landmarks_fps` | Farthest-point landmark selection | [landmarks.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/landmarks.py) |
| `build_cover` | Complete balls around chosen landmarks | [landmarks.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/landmarks.py) |
| `compute_landmarks_uncovered` | Alternative exact Euclidean greedy selection and completion | [optimized.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/optimized.py) |
| `compute_landmarks_verified` | Verify greedy choices when candidate sets may be incomplete | [verified.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/verified.py) |
| `build_cover_blocked` | Complete balls with bounded distance workspace | [verified.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/verified.py) |

## Graph and connectivity

| Function | Use | Source |
|---|---|---|
| `build_mapper` | NetworkX overlap graph with `witness_count` edges | [graph.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/graph.py) |
| `compute_edge_overlaps` | Recompute shared-observation counts | [graph.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/graph.py) |
| `build_mapper_auto` | CSR graph with automatic or explicit construction strategy | [optimized.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/optimized.py) |
| `build_mapper_csr` | Sparse integer witness-count adjacency | [optimized.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/optimized.py) |
| `build_mapper_streaming` | Build the graph from an iterable of balls | [optimized.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/optimized.py) |
| `ball_mapper_components` | Connected components without constructing graph edges | [optimized.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/optimized.py) |
| `fixed_landmark_h0` | Connected-component changes across radii for fixed landmarks | [optimized.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/optimized.py) |

## Color and draw

| Function | Use | Source |
|---|---|---|
| `color_by_function` | Aggregate numerical values within balls | [coloring.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/coloring.py) |
| `color_by_mode`, `color_by_entropy` | Summarize categorical values | [coloring.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/coloring.py) |
| `color_by_size`, `color_by_density` | Describe ball membership | [coloring.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/coloring.py) |
| `draw_ball_mapper` | Draw a graph with Matplotlib | [plotting/matplotlib.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/plotting/matplotlib.py) |
| `draw_ball_mapper_plotly` | Draw an interactive graph with Plotly | [plotting/plotly.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/plotting/plotly.py) |

## Approximation and certificates

| Function | Use | Source |
|---|---|---|
| `compare_covers` | Compare ball memberships and graph edges | [audit.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/audit.py) |
| `compute_boundary_diagnostics` | Check observations near the radius boundary | [diagnostics.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/diagnostics.py) |
| `make_backend` | Create a radius-query backend explicitly | [backends/](https://github.com/jhnrckmnznrs/fast-ballmapper/tree/main/src/fast_ballmapper/backends) |
| `certify_component_preservation` | Fast sufficient component-preservation test | [sparse.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/sparse.py) |
| `certify_connectivity` | Exact robustness decision procedure with possible inconclusive status | [budget_certificates.py](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/src/fast_ballmapper/budget_certificates.py) |

See [Advanced computations]({{ '/advanced/' | relative_url }}) for the remaining streaming, completion, and multiscale functions.
