---
layout: default
title: Ball Mapper workflow
permalink: /workflow/
description: What landmarks, balls, graph edges, and witnesses mean in fast-ballmapper.
---

# Ball Mapper workflow

The package follows the usual construction. The choice of a radius-query implementation is separate from the mathematical steps.

## 1. Choose landmarks

`compute_landmarks(x, eps)` processes observations in their given order. When an observation has not yet been covered, it is chosen as a landmark. Greedy landmark selection and cover construction happen together in this function.

The returned `landmarks` are **observation indices**. To work with a landmark set you already have, use `build_cover` instead.

## 2. Construct the balls

Given a metric `d`, each landmark `l`, and a radius `eps`, the corresponding ball consists of observations `x` for which `d(x, l) <= eps`.

`cover[j]` stores the observation indices in the ball of landmark `landmarks[j]`. The cover is a list of such index arrays.

```python
from fast_ballmapper import build_cover

cover = build_cover(x, landmarks, eps=0.1, method="ckdtree")
```

The package uses **closed balls**. Around floating-point radius boundaries, backends with different numerical precision may disagree; see [Search methods]({{ '/methods/' | relative_url }}).

## 3. Construct the graph

`build_mapper(cover)` creates one graph vertex per ball. It adds an edge `(i, j)` if at least one observation occurs in both `cover[i]` and `cover[j]`.

The stored `witness_count` is the number of distinct shared observations. It is an overlap count, not a geometric distance between landmarks.

```python
from fast_ballmapper import build_mapper

graph = build_mapper(cover)
shared = graph[0][1]["witness_count"] if graph.has_edge(0, 1) else 0
```

## 4. Color and visualize

Node colors can summarize a numerical function, the most common class, class-label entropy, or ball size/density. These operations do not change the graph itself.

Use `draw_ball_mapper` for Matplotlib and `draw_ball_mapper_plotly` for interactive Plotly figures.

## 5. Check approximation effects

If you use an approximate radius-query method, the returned cover can differ from the reference cover. This may also change the chosen landmarks, so to study **range-query error alone**, keep the landmark indices fixed and call `build_cover` with different backends.

Use `compare_covers` to inspect membership and graph differences.

**Next:** [Find these functions in the source →]({{ '/reference/' | relative_url }})
