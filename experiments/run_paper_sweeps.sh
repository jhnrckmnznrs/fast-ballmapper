#!/usr/bin/env bash
set -euo pipefail

# Run from the fast-ballmapper repository root.
#
# IMPORTANT:
# Run these sequentially. Do not use "&", GNU parallel, xargs -P, etc.
# Concurrent benchmarks would contaminate timing/RSS comparisons.

COMMON=(
    --n 20000
    --d 50
    --seeds 0 1 2
    --repeats 3
    --target-ball-size 50
    --candidate-k 4096
    --threads 1
)

RESULTS="experiments/results"

echo
echo "======================================"
echo "HNSW efSearch sweep"
echo "======================================"

for EF in 32 64 128 256; do
    OUT="${RESULTS}/hnsw-ef${EF}"

    echo
    echo ">>> Running efSearch=${EF}"
    echo ">>> Output: ${OUT}"

    python experiments/run_paper_experiments.py \
        "${COMMON[@]}" \
        --methods \
            hnsw \
            verified_hnsw_partial \
            verified_hnsw_complete \
        --ef-search "${EF}" \
        --output "${OUT}"
done


echo
echo "======================================"
echo "All requested search-effort sweeps finished"
echo "======================================"
