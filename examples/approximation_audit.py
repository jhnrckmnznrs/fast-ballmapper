"""Audit an approximate fixed-landmark Ball Mapper cover."""

import numpy as np

from fast_ballmapper import FaissConfig, build_cover, compare_covers, compute_landmarks

rng = np.random.default_rng(42)
x = rng.normal(size=(1000, 8)).astype(np.float32)
response = x[:, 0] + 0.25 * x[:, 1]
eps = 1.5

landmarks, _ = compute_landmarks(
    x, eps=eps, method="brute_force", metric="euclidean"
)
reference_cover = build_cover(
    x, landmarks, eps=eps, method="brute_force", metric="euclidean"
)
approximate_cover = build_cover(
    x,
    landmarks,
    eps=eps,
    method="faiss",
    metric="euclidean",
    faiss_config=FaissConfig(factory="IVF64,Flat", search_params={"nprobe": 8}),
)

report = compare_covers(
    reference_cover,
    approximate_cover,
    x=x,
    landmarks=landmarks,
    eps=eps,
    deltas=[1e-5, 1e-4, 1e-3],
    values=response,
)

print(f"membership recall: {report.membership_recall:.4f}")
print(f"edge recall: {report.edge_recall:.4f}")
print(f"missing edges: {len(report.missing_edges)}")
if report.colors is not None:
    print(f"color MAE: {report.colors.mean_absolute_error:.6g}")
