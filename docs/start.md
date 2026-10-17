---
layout: default
title: Getting started
permalink: /start/
description: Install fast-ballmapper and build a Ball Mapper graph.
---

# Getting started

If you already know Ball Mapper, you only need three things: a point cloud, a radius, and a graph builder.

## Install

```bash
pip install fast-ballmapper
```

Use `pip install "fast-ballmapper[plot]"` if you want Matplotlib or Plotly visualization.

## Construct a cover and graph

```python
import numpy as np
from fast_ballmapper import compute_landmarks, build_mapper

x = np.random.default_rng(42).random((500, 2))
eps = 0.1

landmarks, cover = compute_landmarks(x, eps, method="ckdtree")
graph = build_mapper(cover)

print(f"{len(landmarks)} landmarks")
print(f"{graph.number_of_edges()} graph edges")
```

`landmarks` contains indices into `x`. For each landmark `landmarks[j]`, `cover[j]` contains indices of the observations within distance `eps`. Graph vertex `j` represents the corresponding ball, not the original observation index.

## Color and draw

```python
from fast_ballmapper import color_by_size
from fast_ballmapper.plotting.matplotlib import draw_ball_mapper
import matplotlib.pyplot as plt

sizes = color_by_size(cover)
draw_ball_mapper(graph, colors=sizes, sizes=sizes)
plt.show()
```

Here the node color and size are based on the number of observations in each ball. The plotting function also accepts other color values computed by [the coloring functions]({{ '/reference/' | relative_url }}#color-and-draw).

## What should I change?

- **Radius:** Change `eps` to adjust the cover and its overlaps.
- **Search method:** Change `method` to select a supported range-query backend.
- **Colors:** Use a measured variable, predicted value, or class label instead of ball size.
- **Graph output:** For numerical work rather than plotting, [use a sparse graph]({{ '/advanced/' | relative_url }}#sparse-graphs).

For a runnable plotting example, see [`examples/basic_usage.py`](https://github.com/jhnrckmnznrs/fast-ballmapper/blob/main/examples/basic_usage.py).

**Next:** [Understand each construction step →]({{ '/workflow/' | relative_url }})
