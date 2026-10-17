---
layout: default
title: Overview
permalink: /
description: Fast, understandable Ball Mapper documentation from data to graph.
---

<div class="eyebrow">Topological data analysis · Python</div>

# Ball Mapper,<br>from data to graph.

<p class="lead">A practical, compact guide to constructing Ball Mapper covers and graphs, choosing search methods, and finding the right functions in <strong>fast-ballmapper</strong>.</p>

<div class="action-row">
  <a class="action primary" href="{{ '/start/' | relative_url }}">Get started →</a>
  <a class="action" href="{{ '/reference/' | relative_url }}">Browse functions</a>
  <a class="action" href="https://github.com/jhnrckmnznrs/fast-ballmapper">View source ↗</a>
</div>

<div class="steps">
  <div class="step"><span class="step-number">01 · SELECT</span><strong>Landmarks</strong><span>Choose representative observations using greedy selection.</span></div>
  <div class="step"><span class="step-number">02 · COVER</span><strong>Balls</strong><span>Find the observations within a chosen radius of each landmark.</span></div>
  <div class="step"><span class="step-number">03 · CONNECT</span><strong>Graph</strong><span>Connect balls that share observations; optionally color and plot.</span></div>
</div>

## The shortest working example

```python
import numpy as np
from fast_ballmapper import compute_landmarks, build_mapper

x = np.random.default_rng(42).random((500, 2))
landmarks, cover = compute_landmarks(x, eps=0.1, method="ckdtree")
graph = build_mapper(cover)

print(len(landmarks), graph.number_of_edges())
```

The graph is a NetworkX graph. Its edge attribute `witness_count` gives the number of observations shared by the corresponding balls.

## Where to go next

- **[Getting started]({{ '/start/' | relative_url }})** — install, construct, color, and draw your first graph.
- **[Ball Mapper workflow]({{ '/workflow/' | relative_url }})** — see exactly what each step represents.
- **[Function reference]({{ '/reference/' | relative_url }})** — find the public function and its Python source file.
- **[Advanced computations]({{ '/advanced/' | relative_url }})** — sparse graphs, streaming, fixed-landmark connectivity, and robustness certificates.

The [README](https://github.com/jhnrckmnznrs/fast-ballmapper#readme) stays short; this site holds the longer explanations.
