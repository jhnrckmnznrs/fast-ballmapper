"""Backend protocol examples."""

import numpy as np

from fast_ballmapper import (
    BruteForceBackend,
    CKDTreeBackend,
    HnswlibConfig,
    build_cover,
    compute_landmarks,
    make_backend,
)

rng = np.random.default_rng(42)
x = rng.normal(size=(1000, 8)).astype(np.float32)
eps = 2.0

# Construct once, reuse across Ball Mapper operations.
reference = BruteForceBackend(x)
landmarks, reference_cover = compute_landmarks(x, eps, backend=reference)

# Independent exact CPU implementation.
ckdtree = CKDTreeBackend(x)
ckdtree_cover = build_cover(x, landmarks, eps, backend=ckdtree)
assert all(
    np.array_equal(left, right)
    for left, right in zip(reference_cover, ckdtree_cover, strict=True)
)

# Factory form. Optional dependencies are imported only when requested.
print(make_backend(x, "ball_tree").metadata)

# Example configuration for the optional hnswlib adapter:
# approximate = make_backend(
#     x,
#     "hnswlib",
#     hnswlib_config=HnswlibConfig(ef_search=128, candidate_k=1024),
# )
# approximate_cover = build_cover(x, landmarks, eps, backend=approximate)

_ = HnswlibConfig  # keep the optional example import visible to readers
