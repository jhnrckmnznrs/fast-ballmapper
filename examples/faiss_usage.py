"""FAISS-backed landmark construction."""

import numpy as np

from fast_ballmapper import build_mapper, compute_landmarks_fps

rng = np.random.default_rng(42)
x = rng.normal(size=(10_000, 8))

landmarks, cover = compute_landmarks_fps(
    x,
    eps=1.5,
    method="faiss",
    metric="euclidean",
)
graph = build_mapper(cover)
print(len(landmarks), graph.number_of_edges())
