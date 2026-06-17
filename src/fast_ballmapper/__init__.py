"""Fast construction and analysis of Ball Mapper graphs."""

from importlib.metadata import PackageNotFoundError, version

from fast_ballmapper.coloring import (
    color_by_density,
    color_by_entropy,
    color_by_function,
    color_by_mode,
    color_by_size,
)
from fast_ballmapper.graph import build_mapper, compute_edge_overlaps
from fast_ballmapper.landmarks import compute_landmarks, compute_landmarks_fps

try:
    __version__ = version("fast-ballmapper")
except PackageNotFoundError:  # pragma: no cover - source checkout fallback
    __version__ = "0.1.0"

__all__ = [
    "build_mapper",
    "color_by_density",
    "color_by_entropy",
    "color_by_function",
    "color_by_mode",
    "color_by_size",
    "compute_edge_overlaps",
    "compute_landmarks",
    "compute_landmarks_fps",
]
