"""Minimal fast-ballmapper example."""

import numpy as np

from fast_ballmapper import build_mapper, color_by_size, compute_landmarks
from fast_ballmapper.plotting.matplotlib import add_colorbar, draw_ball_mapper


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.random((500, 2))

    landmarks, cover = compute_landmarks(x, eps=0.1)
    graph = build_mapper(cover)
    sizes = color_by_size(cover)

    _, nodes = draw_ball_mapper(graph, colors=sizes, sizes=sizes)
    import matplotlib.pyplot as plt

    add_colorbar(nodes, plt.gca(), label="Ball size")
    plt.show()
    print(f"Selected {len(landmarks)} landmarks")


if __name__ == "__main__":
    main()
