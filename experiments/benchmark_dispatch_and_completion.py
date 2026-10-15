"""Reproducible exact cover and graph dispatch benchmark.

Run with OPENBLAS_NUM_THREADS=1 python experiments/benchmark_dispatch_and_completion.py
The first JIT compilation is reported separately from warm steady-state timing.
All covers and weighted adjacency results are checked for exact equality.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from fast_ballmapper.optimized import (
    build_cover_localized,
    build_mapper_auto,
    build_mapper_csr,
    select_landmarks_uncovered,
    select_mapper_algorithm,
)


def timed(action, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = action()
        times.append(time.perf_counter() - start)
    return result, statistics.median(times)


def generate_data(kind, n, d, seed):
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(size=(n, d))
    if kind == "mixture":
        centers = rng.normal(size=(6, d)) * 2
        groups = rng.integers(0, 6, size=n)
        return centers[groups] + rng.normal(size=(n, d)) * 0.3
    if kind == "gaussian":
        return rng.normal(size=(n, d))
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--with-numba", action="store_true",
        help="Include optional JIT engine (compile time reported separately).",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be >= 1")
    cases = [
        ("uniform", 1500, 2, .065),
        ("uniform", 5000, 2, .06),
        ("uniform", 3000, 3, .25),
        ("mixture", 2000, 10, 1.0),
        ("gaussian", 2000, 50, 7.6),
    ]
    if args.smoke:
        cases = [("uniform", 300, 2, .12)]
    rows = []
    for kind, n, d, eps in cases:
        x = generate_data(kind, n, d, seed=42)
        landmarks, owners = select_landmarks_uncovered(x, eps)
        reference = build_cover_localized(
            x, landmarks, eps, owners, strategy="blocked"
        )
        row = {
            "dataset": kind, "n": n, "d": d, "eps": eps,
            "landmarks": len(landmarks),
        }
        engines = ["scalar", "batch"]
        if args.with_numba:
            engines.append("numba")
        for engine in engines:
            start = time.perf_counter()
            build_cover_localized(
                x, landmarks, eps, owners, strategy="localized", engine=engine
            )
            row[f"{engine}_warmup_seconds"] = time.perf_counter() - start
            got, duration = timed(
                lambda x=x, landmarks=landmarks, eps=eps, owners=owners, engine=engine: build_cover_localized(
                    x, landmarks, eps, owners,
                    strategy="localized", engine=engine
                ), args.repeats
            )
            assert all(
                np.array_equal(a, b)
                for a, b in zip(got, reference, strict=True)
            )
            row[f"{engine}_seconds"] = duration
        adjacency, duration = timed(
            lambda reference=reference, n=n: build_mapper_csr(reference, n), args.repeats
        )
        row["csr_seconds"] = duration
        row["dispatch"] = select_mapper_algorithm(reference, n)
        for algorithm in ("auto", "streaming"):
            result, duration = timed(
                lambda reference=reference, n=n, algorithm=algorithm: build_mapper_auto(
                    reference, n, algorithm=algorithm
                ), args.repeats
            )
            assert (result != adjacency).nnz == 0
            row[f"{algorithm}_graph_seconds"] = duration
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
