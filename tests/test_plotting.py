import matplotlib
import numpy as np

matplotlib.use("Agg")

from fast_ballmapper import build_mapper
from fast_ballmapper.plotting.matplotlib import draw_ball_mapper
from fast_ballmapper.plotting.plotly import draw_ball_mapper_plotly


def test_matplotlib_plotting_returns_nodes():
    cover = [np.array([0, 1]), np.array([1, 2])]
    graph = build_mapper(cover)
    positions, nodes = draw_ball_mapper(graph, sizes=[2, 2])
    assert set(positions) == {0, 1}
    assert nodes is not None


def test_plotly_plotting_can_skip_display():
    cover = [np.array([0, 1]), np.array([1, 2])]
    graph = build_mapper(cover)
    figure = draw_ball_mapper_plotly(graph, cover, show=False)
    assert len(figure.data) == 2
