"""Closed-ball exactness gate for independent exact CPU backends.

This experiment is intentionally independent of FAISS. It calibrates epsilon
from exact k-nearest-neighbour distances, then checks that the float64
brute-force reference oracle, scikit-learn BallTree, and SciPy cKDTree produce
the same greedy landmarks, fixed-landmark cover, and Ball Mapper edge set under
the package's unified closed-ball boundary convention.

The default cases reproduce the validation table used in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from fast_ballmapper import build_cover, build_mapper, compare_covers, compute_landmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/raw/exactness_gate_closed_ball.csv"),
    )
    return parser.parse_args()


def gaussian_data(n_samples: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(
        rng.normal(size=(n_samples, dimension)).astype(np.float32)
    )


def calibrate_epsilon(x: np.ndarray, target_ball_size: int) -> float:
    """Return the mathematical target-occupancy radius without pre-expansion."""
    if not 2 <= target_ball_size <= len(x):
        raise ValueError("target_ball_size must lie in [2, n_samples].")
    nn = NearestNeighbors(
        n_neighbors=target_ball_size,
        algorithm="brute",
        metric="euclidean",
        n_jobs=1,
    )
    nn.fit(x)
    distances, _ = nn.kneighbors(x, return_distance=True)
    return float(np.median(distances[:, target_ball_size - 1]))


def normalized_edges(graph) -> set[tuple[int, int]]:
    return {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in graph.edges()
    }


def run_case(
    n_samples: int,
    dimension: int,
    seed: int,
    target_ball_size: int,
) -> dict[str, object]:
    x = gaussian_data(n_samples, dimension, seed)
    epsilon = calibrate_epsilon(x, target_ball_size)

    start = time.perf_counter()
    balltree_landmarks, balltree_selected_cover = compute_landmarks(
        x, epsilon, method="ball_tree", metric="euclidean"
    )
    balltree_landmark_seconds = time.perf_counter() - start

    start = time.perf_counter()
    reference_landmarks, reference_selected_cover = compute_landmarks(
        x, epsilon, method="brute_force", metric="euclidean"
    )
    reference_landmark_seconds = time.perf_counter() - start

    start = time.perf_counter()
    ckdtree_landmarks, ckdtree_selected_cover = compute_landmarks(
        x, epsilon, method="ckdtree", metric="euclidean"
    )
    ckdtree_landmark_seconds = time.perf_counter() - start

    landmark_indices_equal = balltree_landmarks == reference_landmarks
    ckdtree_landmark_indices_equal = ckdtree_landmarks == reference_landmarks
    selected_cover_equal = landmark_indices_equal and all(
        np.array_equal(np.sort(left), np.sort(right))
        for left, right in zip(
            balltree_selected_cover,
            reference_selected_cover,
            strict=True,
        )
    )
    ckdtree_selected_cover_equal = ckdtree_landmark_indices_equal and all(
        np.array_equal(np.sort(left), np.sort(right))
        for left, right in zip(
            ckdtree_selected_cover,
            reference_selected_cover,
            strict=True,
        )
    )

    start = time.perf_counter()
    balltree_cover = build_cover(
        x,
        balltree_landmarks,
        epsilon,
        method="ball_tree",
        metric="euclidean",
    )
    balltree_cover_seconds = time.perf_counter() - start

    start = time.perf_counter()
    reference_cover = build_cover(
        x,
        balltree_landmarks,
        epsilon,
        method="brute_force",
        metric="euclidean",
    )
    reference_cover_seconds = time.perf_counter() - start

    start = time.perf_counter()
    ckdtree_cover = build_cover(
        x,
        balltree_landmarks,
        epsilon,
        method="ckdtree",
        metric="euclidean",
    )
    ckdtree_cover_seconds = time.perf_counter() - start

    audit = compare_covers(reference_cover, balltree_cover)
    ckdtree_audit = compare_covers(reference_cover, ckdtree_cover)
    reference_graph = build_mapper(reference_cover)
    balltree_graph = build_mapper(balltree_cover)
    ckdtree_graph = build_mapper(ckdtree_cover)

    return {
        "dataset": "gaussian",
        "n_samples": n_samples,
        "dimension": dimension,
        "seed": seed,
        "target_ball_size": target_ball_size,
        "epsilon": epsilon,
        "landmark_count_balltree": len(balltree_landmarks),
        "landmark_count_bruteforce": len(reference_landmarks),
        "landmark_count_ckdtree": len(ckdtree_landmarks),
        "landmark_indices_equal": landmark_indices_equal,
        "ckdtree_landmark_indices_equal": ckdtree_landmark_indices_equal,
        "selected_cover_equal": selected_cover_equal,
        "ckdtree_selected_cover_equal": ckdtree_selected_cover_equal,
        "fixed_cover_membership_precision": audit.membership_precision,
        "fixed_cover_membership_recall": audit.membership_recall,
        "fixed_cover_membership_jaccard": audit.membership_jaccard,
        "false_negative_count": audit.membership_false_negatives,
        "false_positive_count": audit.membership_false_positives,
        "ckdtree_membership_precision": ckdtree_audit.membership_precision,
        "ckdtree_membership_recall": ckdtree_audit.membership_recall,
        "ckdtree_membership_jaccard": ckdtree_audit.membership_jaccard,
        "edge_sets_equal": normalized_edges(reference_graph)
        == normalized_edges(balltree_graph),
        "ckdtree_edge_sets_equal": normalized_edges(reference_graph)
        == normalized_edges(ckdtree_graph),
        "edge_count": reference_graph.number_of_edges(),
        "balltree_landmark_seconds": balltree_landmark_seconds,
        "bruteforce_landmark_seconds": reference_landmark_seconds,
        "ckdtree_landmark_seconds": ckdtree_landmark_seconds,
        "balltree_cover_seconds": balltree_cover_seconds,
        "bruteforce_cover_seconds": reference_cover_seconds,
        "ckdtree_cover_seconds": ckdtree_cover_seconds,
    }


def main() -> None:
    args = parse_args()
    cases = [
        (1000, 16, 0, 20),
        (5000, 32, 0, 50),
    ]
    rows = [run_case(*case) for case in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
