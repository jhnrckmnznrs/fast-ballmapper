# Gaussian pilot boundary audit

The supplied `paper-pilot-n5000-d32.zip` was produced from clean commit
`4bb9575e35850689ed24053a90786dab46a2527a`. Of 72 runs, 68 passed. The four
failures were Gaussian `flat_scalar` and `flat_batch`, each with inverted and
sparse graph construction. There were no worker exceptions or color-bound
violations, and graph construction agreed with the witness-count reference in
every run.

## Cause

Each failing run added the same single membership: observation 653 in the ball
centered at observation 5 (ball position 4). The saved numerical values were:

| Quantity | Value |
| --- | --- |
| Calibrated radius | 5.995250733644709 |
| Direct float64 distance | 5.99525094106526 |
| Distance minus radius | 2.07420550957238e-7 |
| FAISS reported squared distance | 35.943031311035156 |
| Outward float32 squared threshold | 35.94303512573242 |

The native float32 comparison admits the point, while the declared float64
reference predicate excludes it. Exhaustive search and a shared closed-ball
convention do not imply numerical equivalence between these predicates.

The landmark sequence stayed identical in these four runs, but the added
membership created two graph edges: 87,081 instead of 87,079. Thus the strict
failure was justified. The benchmark configuration was incomplete: the Flat
comparisons disabled original-coordinate verification while being required to
match the float64 reference.

## Correction and validation

The paper runner now enables `exact_verify=True` for both Flat configurations.
This filters retrieved candidates using original-coordinate float64 norms. The
filter runs within the measured selection and membership stages. The public
FAISS defaults and the calibrated epsilon are unchanged. Verification cannot
recover a point omitted by the native range search, so the strict equality gate
remains necessary for every dataset and configuration.

The identical extra membership was reproduced from the uploaded reference data;
enabling verification removed it. All 12 Flat runs across the three geometries
and two graph constructors passed after the change. Their generated data,
epsilon, reference landmarks, and reference memberships were array-identical to
the uploaded files, despite differing NumPy/SciPy versions in the review
environment. Corrected Gaussian runs had zero false-positive or false-negative
memberships and 87,079 edges.

The revised failure-reporting logic was checked against all 72 uploaded rows:
it preserves every original pass/fail decision, including the four failures.
Regression tests exercise the reported boundary pair through both Flat query
paths, rejection of a single extra membership, partial-cover requirements, and
retention of worker exception context.

## Recheck on the experiment machine

After checking out this fix and activating the experiment environment, run:

```bash
python experiments/run_paper_experiments.py \
  --n 5000 --d 32 --seeds 0 --repeats 1 \
  --target-ball-size 50 --candidate-k 4096 --threads 1 \
  --methods flat_scalar flat_batch \
  --output experiments/results/paper-pilot-flat-filtered-01
```

This reruns only the 12 Flat configurations. Keep the original archive. Its
unverified Flat timings describe a different configuration and should not be
relabeled as filtered timings. Once the recheck passes, run the full study from
one clean reviewed commit. The 20,000-point matrix and repeated timing/scaling
evidence remain outstanding.

The original pilot recorded at most 185.99 MiB of pipeline peak RSS. That metric
excludes later audit and parent reference-preparation allocations; it is not an
upper bound on the experiment's total memory requirements.
