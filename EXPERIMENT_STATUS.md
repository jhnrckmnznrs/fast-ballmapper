# Experiment status for the 0.2.0 release

The current implementation uses closed balls `d <= eps`. Backend APIs that
expose strict thresholds implement the same mathematical boundary with a
single outward `np.nextafter` comparison. Epsilon calibration is not
pre-expanded.

## Exactness gate

The post-refactor exactness gate agrees across the float64 brute-force oracle,
scikit-learn BallTree, and SciPy `cKDTree`. In the Gaussian
`n=5000, d=32, target=50` validation case, all three exact CPU implementations
select the same 1263 landmarks and reproduce all 85,588 graph edges.

## Approximate FAISS matrix

The final CPU study used Gaussian, mixture, and noisy-curve data at
`n=20,000, d=50` with seeds 0, 1, and 2. `candidate_k=4096` exceeded the
largest reference-ball size in every run. Fixed-landmark approximate queries
were conservative in this matrix: membership, witness, and edge precision were
all 1.0, and no coloring-bound violations were observed.

Approximation difficulty depended strongly on geometry. Gaussian data required
substantially greater search effort than mixture data, while the noisy curve
was nearly exact at modest search settings. Witness multiplicity strongly
protected edge survival once membership recall entered a useful regime.

At this problem size, exhaustive FAISS Flat remained faster than the tested
approximate configurations. The approximation study should therefore be read
as a controlled fidelity/error-propagation analysis at `n=20,000`, not as a
claim that ANN search is already faster than Flat at that scale.

Generated CSV, audit JSON, and figure payloads are intentionally excluded from
the source distribution. The experiment scripts reproduce them in
`experiments/results/`.
