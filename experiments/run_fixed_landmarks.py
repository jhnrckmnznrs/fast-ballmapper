"""Fixed-landmark comparison of exact and approximate FAISS indexes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import faiss
import networkx as nx
import numpy as np

from fast_ballmapper import (
    FaissConfig,
    build_cover,
    build_mapper,
    compute_landmarks,
)


def generate_gaussian_data(
    n_samples: int,
    dimension: int,
    seed: int,
) -> np.ndarray:
    """Generate an isotropic Gaussian point cloud."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(
        size=(n_samples, dimension),
        dtype=np.float32,
    )


def choose_epsilon(
    x: np.ndarray,
    target_ball_size: int,
    calibration_queries: int,
    seed: int,
) -> float:
    """Choose epsilon using exact nearest-neighbour distances.

    Epsilon is chosen so that a typical query ball contains approximately
    ``target_ball_size`` points, including the query point itself.
    """
    n_samples = len(x)

    if not 2 <= target_ball_size <= n_samples:
        raise ValueError(
            "target_ball_size must be between 2 and the number of samples."
        )

    rng = np.random.default_rng(seed)
    query_count = min(calibration_queries, n_samples)
    query_indices = rng.choice(
        n_samples,
        size=query_count,
        replace=False,
    )

    points = np.ascontiguousarray(x, dtype=np.float32)
    index = faiss.IndexFlatL2(points.shape[1])
    index.add(points)

    squared_distances, _ = index.search(
        points[query_indices],
        target_ball_size,
    )

    boundary_distances = np.sqrt(
        np.maximum(squared_distances[:, -1], 0.0)
    )

    epsilon = float(np.median(boundary_distances))

    # Keep epsilon as the calibrated mathematical radius.  Closed-ball
    # inclusivity is implemented once inside the exact/query backend via the
    # centralized nextafter threshold policy.
    return epsilon


def range_metrics(
    exact_cover: list[np.ndarray],
    approximate_cover: list[np.ndarray],
) -> dict[str, float | int]:
    """Compare approximate and exact range sets."""
    if len(exact_cover) != len(approximate_cover):
        raise ValueError("The two covers must use the same landmarks.")

    true_positive_total = 0
    exact_total = 0
    approximate_total = 0
    false_negative_total = 0
    false_positive_total = 0

    recalls: list[float] = []
    precisions: list[float] = []
    perfect_queries = 0

    for exact_members, approximate_members in zip(
        exact_cover,
        approximate_cover,
        strict=True,
    ):
        exact_set = set(map(int, exact_members))
        approximate_set = set(map(int, approximate_members))

        true_positives = len(exact_set & approximate_set)
        false_negatives = len(exact_set - approximate_set)
        false_positives = len(approximate_set - exact_set)

        true_positive_total += true_positives
        exact_total += len(exact_set)
        approximate_total += len(approximate_set)
        false_negative_total += false_negatives
        false_positive_total += false_positives

        recall = (
            true_positives / len(exact_set)
            if exact_set
            else 1.0
        )
        precision = (
            true_positives / len(approximate_set)
            if approximate_set
            else float(not exact_set)
        )

        recalls.append(recall)
        precisions.append(precision)

        if exact_set == approximate_set:
            perfect_queries += 1

    micro_recall = (
        true_positive_total / exact_total
        if exact_total
        else 1.0
    )
    micro_precision = (
        true_positive_total / approximate_total
        if approximate_total
        else float(exact_total == 0)
    )

    return {
        "range_micro_recall": micro_recall,
        "range_macro_recall": float(np.mean(recalls)),
        "range_recall_p10": float(np.percentile(recalls, 10)),
        "range_micro_precision": micro_precision,
        "range_macro_precision": float(np.mean(precisions)),
        "perfect_range_fraction": perfect_queries / len(exact_cover),
        "false_negative_count": false_negative_total,
        "false_positive_count": false_positive_total,
    }


def edge_set(graph: nx.Graph) -> set[tuple[int, int]]:
    """Return graph edges as normalized integer pairs."""
    return {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in graph.edges()
    }


def graph_metrics(
    exact_graph: nx.Graph,
    approximate_graph: nx.Graph,
) -> dict[str, float | int]:
    """Compare graphs defined on the same landmark set."""
    exact_edges = edge_set(exact_graph)
    approximate_edges = edge_set(approximate_graph)

    common_edges = exact_edges & approximate_edges
    union_edges = exact_edges | approximate_edges

    edge_recall = (
        len(common_edges) / len(exact_edges)
        if exact_edges
        else float(not approximate_edges)
    )
    edge_precision = (
        len(common_edges) / len(approximate_edges)
        if approximate_edges
        else float(not exact_edges)
    )
    edge_jaccard = (
        len(common_edges) / len(union_edges)
        if union_edges
        else 1.0
    )

    exact_components = nx.number_connected_components(exact_graph)
    approximate_components = nx.number_connected_components(
        approximate_graph
    )

    exact_cycle_rank = (
        exact_graph.number_of_edges()
        - exact_graph.number_of_nodes()
        + exact_components
    )
    approximate_cycle_rank = (
        approximate_graph.number_of_edges()
        - approximate_graph.number_of_nodes()
        + approximate_components
    )

    return {
        "exact_edge_count": len(exact_edges),
        "approximate_edge_count": len(approximate_edges),
        "edge_recall": edge_recall,
        "edge_precision": edge_precision,
        "edge_jaccard": edge_jaccard,
        "exact_components": exact_components,
        "approximate_components": approximate_components,
        "exact_cycle_rank": exact_cycle_rank,
        "approximate_cycle_rank": approximate_cycle_rank,
    }


def largest_power_of_two_at_most(value: float) -> int:
    """Return the largest power of two not exceeding value."""
    if value < 1:
        return 1

    return 2 ** int(math.floor(math.log2(value)))


def product_quantizer_subvectors(dimension: int) -> int:
    """Choose a valid number of PQ subvectors."""
    for candidate in (32, 16, 8, 4, 2, 1):
        if dimension % candidate == 0:
            return candidate

    return 1


def experiment_configurations(
    n_samples: int,
    dimension: int,
    target_ball_size: int,
) -> list[tuple[str, FaissConfig]]:
    """Construct the FAISS configurations used in the experiment."""
    nlist = largest_power_of_two_at_most(math.sqrt(n_samples))
    nlist = min(max(nlist, 16), 256)

    nprobe_values = sorted(
        {
            1,
            2,
            4,
            8,
            16,
            32,
            nlist,
        }
    )
    nprobe_values = [
        value for value in nprobe_values if value <= nlist
    ]

    candidate_k = min(
        n_samples,
        max(512, 10 * target_ball_size),
    )

    pq_subvectors = product_quantizer_subvectors(dimension)

    configurations: list[tuple[str, FaissConfig]] = [
        (
            "flat",
            FaissConfig(factory="Flat"),
        ),
        (
            "sq8",
            FaissConfig(factory="SQ8"),
        ),
        (
            "sq8_verified",
            FaissConfig(
                factory="SQ8",
                exact_verify=True,
            ),
        ),
    ]

    for nprobe in nprobe_values:
        configurations.append(
            (
                f"ivf_flat_nprobe_{nprobe}",
                FaissConfig(
                    factory=f"IVF{nlist},Flat",
                    search_params={"nprobe": nprobe},
                ),
            )
        )

        configurations.append(
            (
                f"ivf_sq8_nprobe_{nprobe}",
                FaissConfig(
                    factory=f"IVF{nlist},SQ8",
                    search_params={"nprobe": nprobe},
                ),
            )
        )

        configurations.append(
            (
                f"ivf_sq8_verified_nprobe_{nprobe}",
                FaissConfig(
                    factory=f"IVF{nlist},SQ8",
                    search_params={"nprobe": nprobe},
                    exact_verify=True,
                ),
            )
        )

        configurations.append(
            (
                f"ivf_pq_nprobe_{nprobe}",
                FaissConfig(
                    factory=f"IVF{nlist},PQ{pq_subvectors}x4",
                    search_params={"nprobe": nprobe},
                    query_mode="knn",
                    candidate_k=candidate_k,
                ),
            )
        )

        configurations.append(
            (
                f"ivf_pq_verified_nprobe_{nprobe}",
                FaissConfig(
                    factory=f"IVF{nlist},PQ{pq_subvectors}x4",
                    search_params={"nprobe": nprobe},
                    query_mode="knn",
                    candidate_k=candidate_k,
                    exact_verify=True,
                ),
            )
        )

    for ef_search in (16, 32, 64, 128, 256):
        configurations.append(
            (
                f"hnsw_efsearch_{ef_search}",
                FaissConfig(
                    factory="HNSW32",
                    construction_params={"efConstruction": 80},
                    search_params={"efSearch": ef_search},
                    query_mode="knn",
                    candidate_k=candidate_k,
                ),
            )
        )

    configurations.extend(
        [
            (
                "pq",
                FaissConfig(
                    factory=f"PQ{pq_subvectors}x4",
                    query_mode="knn",
                    candidate_k=candidate_k,
                ),
            ),
            (
                "pq_verified",
                FaissConfig(
                    factory=f"PQ{pq_subvectors}x4",
                    query_mode="knn",
                    candidate_k=candidate_k,
                    exact_verify=True,
                ),
            ),
        ]
    )

    return configurations


def config_metadata(config: FaissConfig) -> dict[str, Any]:
    """Convert a FAISS configuration to CSV-compatible values."""
    return {
        "factory": config.factory,
        "construction_params": json.dumps(
            dict(config.construction_params),
            sort_keys=True,
        ),
        "search_params": json.dumps(
            dict(config.search_params),
            sort_keys=True,
        ),
        "train_size": config.train_size,
        "train_seed": config.train_seed,
        "query_mode": config.query_mode,
        "candidate_k": config.candidate_k,
        "exact_verify": config.exact_verify,
    }


def covers_are_equal(
    left_cover: list[np.ndarray],
    right_cover: list[np.ndarray],
) -> bool:
    """Check equality of corresponding membership sets."""
    if len(left_cover) != len(right_cover):
        return False

    return all(
        set(map(int, left)) == set(map(int, right))
        for left, right in zip(left_cover, right_cover, strict=True)
    )


def run_experiment(
    n_samples: int,
    dimension: int,
    seed: int,
    target_ball_size: int,
    calibration_queries: int,
) -> list[dict[str, Any]]:
    """Run one fixed-landmark experiment."""
    x = generate_gaussian_data(
        n_samples=n_samples,
        dimension=dimension,
        seed=seed,
    )

    epsilon = choose_epsilon(
        x,
        target_ball_size=target_ball_size,
        calibration_queries=calibration_queries,
        seed=seed + 1,
    )

    exact_config = FaissConfig(factory="Flat")

    start = time.perf_counter()
    landmarks, exact_cover = compute_landmarks(
        x,
        eps=epsilon,
        method="faiss",
        metric="euclidean",
        faiss_config=exact_config,
    )
    exact_landmark_seconds = time.perf_counter() - start

    exact_graph = build_mapper(exact_cover)

    rows: list[dict[str, Any]] = []

    for configuration_name, configuration in experiment_configurations(
        n_samples=n_samples,
        dimension=dimension,
        target_ball_size=target_ball_size,
    ):
        print(
            f"Running {configuration_name}: "
            f"n={n_samples}, d={dimension}, seed={seed}"
        )

        base_row: dict[str, Any] = {
            "dataset": "isotropic_gaussian",
            "n_samples": n_samples,
            "dimension": dimension,
            "seed": seed,
            "target_ball_size": target_ball_size,
            "calibration_queries": calibration_queries,
            "epsilon": epsilon,
            "landmark_count": len(landmarks),
            "exact_landmark_seconds": exact_landmark_seconds,
            "configuration": configuration_name,
            **config_metadata(configuration),
        }

        try:
            start = time.perf_counter()
            approximate_cover = build_cover(
                x,
                landmarks,
                eps=epsilon,
                method="faiss",
                metric="euclidean",
                faiss_config=configuration,
            )
            cover_seconds = time.perf_counter() - start

            approximate_graph = build_mapper(approximate_cover)

            result = {
                **base_row,
                "cover_seconds": cover_seconds,
                "status": "ok",
                "error": "",
                **range_metrics(exact_cover, approximate_cover),
                **graph_metrics(exact_graph, approximate_graph),
            }

            if configuration_name == "flat":
                result["flat_matches_greedy_cover"] = covers_are_equal(
                    exact_cover,
                    approximate_cover,
                )
            else:
                result["flat_matches_greedy_cover"] = ""

        except Exception as error:  # noqa: BLE001
            result = {
                **base_row,
                "cover_seconds": "",
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }

        rows.append(result)

    return rows


def write_rows(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write experiment rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen_fields: set[str] = set()

    for row in rows:
        for field in row:
            if field not in seen_fields:
                seen_fields.add(field)
                fieldnames.append(field)

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact and approximate FAISS range sets using "
            "a fixed exact landmark collection."
        )
    )
    parser.add_argument("--n", type=int, default=5_000)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-ball-size", type=int, default=30)
    parser.add_argument("--calibration-queries", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    output_path = arguments.output
    if output_path is None:
        output_path = Path(
            "experiments/results/raw/"
            f"fixed_landmarks_n{arguments.n}_"
            f"d{arguments.d}_"
            f"seed{arguments.seed}.csv"
        )

    rows = run_experiment(
        n_samples=arguments.n,
        dimension=arguments.d,
        seed=arguments.seed,
        target_ball_size=arguments.target_ball_size,
        calibration_queries=arguments.calibration_queries,
    )

    write_rows(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
