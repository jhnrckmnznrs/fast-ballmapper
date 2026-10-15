"""Small, reproducible benchmark for exact Ball Mapper algorithms."""

from __future__ import annotations

import argparse
import time

import numpy as np

from fast_ballmapper import (
    ball_mapper_components,
    build_mapper_auto,
    build_mapper_csr,
    build_mapper_sparse,
    compute_landmarks,
    compute_landmarks_uncovered,
)


def measure(action):
    start = time.perf_counter()
    result = action()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--eps", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n < 1 or args.d < 1 or args.eps < 0:
        parser.error("n and d must be positive, eps must be nonnegative")
    rng = np.random.default_rng(args.seed)
    x = rng.random((args.n, args.d))
    (ref_lm, ref_cov), ref_time = measure(
        lambda: compute_landmarks(x, args.eps, method="ckdtree")
    )
    (new_lm, new_cov), new_time = measure(
        lambda: compute_landmarks_uncovered(x, args.eps)
    )
    assert ref_lm == new_lm
    assert all(
        np.array_equal(a, b)
        for a, b in zip(ref_cov, new_cov, strict=True)
    )
    reference_graph, old_graph_time = measure(
        lambda: build_mapper_sparse(ref_cov, len(x))
    )
    matrix, csr_time = measure(
        lambda: build_mapper_csr(new_cov, len(x))
    )
    auto, auto_time = measure(
        lambda: build_mapper_auto(new_cov, len(x))
    )
    (components, _labels), h0_time = measure(
        lambda: ball_mapper_components(new_cov, len(x))
    )
    assert (matrix != auto).nnz == 0
    assert reference_graph.number_of_edges() == matrix.nnz // 2
    assert reference_graph.number_of_nodes() == len(new_cov)
    assert components == __import__(
        "networkx"
    ).number_connected_components(reference_graph)
    print(f"n={args.n} d={args.d} eps={args.eps} m={len(new_lm)}")
    print(f"reference selection and cover: {ref_time:.5f}s")
    print(f"uncovered selection and cover: {new_time:.5f}s")
    print(f"reference NetworkX graph:      {old_graph_time:.5f}s")
    print(f"exact CSR graph:              {csr_time:.5f}s")
    print(f"automatic CSR graph:          {auto_time:.5f}s")
    print(f"union-find components:        {h0_time:.5f}s")


if __name__ == "__main__":
    main()
