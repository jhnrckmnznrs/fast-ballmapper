"""Reproducible CPU experiment matrix; see experiments/README.md.

Each configuration/repetition runs in a fresh process. Reference construction
and auditing are excluded from stage times and measured pipeline peak RSS.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from fast_ballmapper import (
    CKDTreeBackend,
    FaissConfig,
    FaissRangeBackend,
    build_cover,
    build_cover_blocked,
    build_mapper,
    build_mapper_sparse,
    certify_component_preservation,
    compare_covers,
    compute_landmarks,
    compute_landmarks_verified,
    cover_statistics,
    make_backend,
)

ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "brute_force",
    "ball_tree",
    "ckdtree",
    "ckdtree_inflated",
    "flat_scalar",
    "flat_batch",
    "ivf",
    "hnsw",
    "verified_none_partial",
    "verified_none_complete",
    "verified_hnsw_partial",
    "verified_hnsw_complete",
)


def generate_data(kind, n, d, seed):
    """Explicit new generators; no claim to reconstruct unpublished old runs."""
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        x = rng.normal(size=(n, d))
    elif kind == "mixture":
        centers = rng.normal(0, 3, size=(4, d))
        x = centers[rng.integers(4, size=n)] + rng.normal(0, 0.35, size=(n, d))
    else:
        t = rng.uniform(0, 2 * np.pi, size=n)
        x = np.column_stack(
            [np.sin((j // 2 + 1) * t + (j % 2) * np.pi / 2) for j in range(d)]
        )
        x += rng.normal(0, 0.03, size=x.shape)
    return np.ascontiguousarray(x, dtype=np.float32)


def choose_radius(x, target, queries, seed):
    """Sampled median target-order distance, direct float64, no pre-expansion."""
    points = np.asarray(x, dtype=np.float64)
    ids = np.random.default_rng(seed).choice(
        len(x), min(queries, len(x)), replace=False
    )
    boundaries = [
        np.partition(np.linalg.norm(points - points[i], axis=1), target - 1)[target - 1]
        for i in ids
    ]
    return float(np.median(boundaries))


def _hash_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def provenance():
    dependencies = {}
    for name in ("numpy", "scipy", "scikit-learn", "networkx", "faiss-cpu"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    digest = hashlib.sha256()
    for path in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("experiments/*.py")]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    try:
        commit, dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_sha256": digest.hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "dependencies": dependencies,
        "reference_predicate": "direct float64 Euclidean norm <= epsilon",
        "scope": "CPU Euclidean fixed observation order; newly specified generators",
    }


def prepare_case(output, dataset, args, seed):
    x = generate_data(dataset, args.n, args.d, seed)
    eps = choose_radius(x, args.target_ball_size, args.calibration_queries, seed + 1)
    landmarks, cover = compute_landmarks(x, eps, method="brute_force")
    indptr = np.r_[0, np.cumsum([len(row) for row in cover])]
    path = output / f"{dataset}_seed{seed}_reference.npz"
    np.savez_compressed(
        path,
        x=x,
        epsilon=eps,
        landmarks=landmarks,
        members=np.concatenate(cover),
        indptr=indptr,
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "reference_file": path.name,
        "n": args.n,
        "d": args.d,
        "epsilon": eps,
        "data_sha256": _hash_array(x),
        "landmarks_sha256": _hash_array(np.asarray(landmarks, dtype=np.int64)),
        "max_reference_ball_size": max(map(len, cover)),
        **cover_statistics(cover, len(x)),
    }


def backend_for(x, method, settings):
    if method.startswith("verified_none"):
        return None
    if method == "ckdtree_inflated":
        return CKDTreeBackend(x, candidate_eps=settings["candidate_eps"])
    if method in {"brute_force", "ball_tree", "ckdtree"}:
        return make_backend(x, method)
    config = {
        "device": "cpu",
        "query_batch_size": settings["batch_size"],
        "train_seed": settings["seed"],
    }
    if method.startswith("flat"):
        config.update(
            factory="Flat",
            # The experiment's reference predicate uses original-coordinate
            # float64 norms. Exhaustive float32 search alone can add members
            # at the boundary; include verification in the timed workload.
            exact_verify=True,
            query_batch_size=(1 if method == "flat_scalar" else settings["batch_size"]),
        )
    elif method == "ivf":
        nlist = min(settings["nlist"], max(1, len(x) // 40))
        config.update(
            factory=f"IVF{nlist},Flat",
            exact_verify=True,
            search_params={"nprobe": min(settings["nprobe"], nlist)},
        )
    else:
        config.update(
            factory="HNSW32",
            query_mode="knn",
            exact_verify=True,
            candidate_k=min(settings["candidate_k"], len(x)),
            construction_params={"efConstruction": 80},
            search_params={"efSearch": settings["ef_search"]},
        )
    return FaissRangeBackend(x, config=FaissConfig(**config))


def _timed(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def _peak_rss_bytes():
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        return None


def _equal_graphs(left, right):
    return dict(left.nodes) == dict(right.nodes) and {
        (min(i, j), max(i, j)): a["witness_count"] for i, j, a in left.edges(data=True)
    } == {
        (min(i, j), max(i, j)): a["witness_count"] for i, j, a in right.edges(data=True)
    }


def validation_failures(row):
    """Explain failed checks without relaxing exact-output requirements."""
    failures = []
    method = row["method"]
    if not row["graph_witness_counts_equal"]:
        failures.append("graph_witness_counts_mismatch")
    if row["color_bound_violations"]:
        failures.append(f"color_bound_violations={row['color_bound_violations']}")
    if method not in {"ivf", "hnsw"} and not method.endswith("partial"):
        if not row["landmark_sequence_equal"]:
            failures.append("landmark_sequence_mismatch")
        if not row["fixed_cover_equal"]:
            failures.append(
                f"membership_mismatch: fp={row['membership_false_positives']}, "
                f"fn={row['membership_false_negatives']}"
            )
    elif row["membership_false_positives"]:
        failures.append(
            f"nonconservative_memberships: fp={row['membership_false_positives']}"
        )
    if method.startswith("verified") and not row["covers_all_observations"]:
        failures.append("verified_cover_does_not_cover_all_observations")
    return failures


def run_worker(job):
    from threadpoolctl import threadpool_info, threadpool_limits

    try:
        import faiss

        faiss.omp_set_num_threads(job["threads"])
    except ImportError:
        faiss = None
    threadpool_limits(limits=job["threads"])
    with np.load(job["reference_path"]) as data:
        x, eps = data["x"], float(data["epsilon"])
        fixed_landmarks = list(map(int, data["landmarks"]))
    method = job["method"]
    backend, index_seconds = _timed(lambda: backend_for(x, method, job))
    work = {}
    if method.startswith("verified"):
        provider = (
            None if backend is None else lambda i, r: backend.query_radius([i], r)[0]
        )
        result, selection_seconds = _timed(
            lambda: compute_landmarks_verified(
                x, eps, provider, block_size=job["block_size"]
            )
        )
        selected_landmarks, cover = result.landmarks, result.cover
        work = {
            "coverage_distance_evaluations": result.coverage_distance_evaluations,
            "candidate_distance_evaluations": result.candidate_distance_evaluations,
            "rescued_observations": result.rescued_observations,
        }
        membership_seconds = 0.0
        if method.endswith("complete"):
            cover, membership_seconds = _timed(
                lambda: build_cover_blocked(
                    x, selected_landmarks, eps, block_size=job["block_size"]
                )
            )
        del result
    else:
        selection, selection_seconds = _timed(
            lambda: compute_landmarks(x, eps, backend=backend)
        )
        selected_landmarks = selection[0]
        del selection
        cover, membership_seconds = _timed(
            lambda: build_cover(x, fixed_landmarks, eps, backend=backend)
        )
    build_graph = (
        (lambda: build_mapper(cover))
        if job["graph_method"] == "inverted"
        else lambda: build_mapper_sparse(
            cover, len(x), row_block_size=job["row_block_size"]
        )
    )
    graph, graph_seconds = _timed(build_graph)
    peak = _peak_rss_bytes()  # Capture BEFORE reference loading and audit allocations.
    row = {
        **{k: job[k] for k in ("dataset", "seed", "repeat", "method", "graph_method")},
        "index_seconds": index_seconds,
        "selection_seconds": selection_seconds,
        "membership_seconds": membership_seconds,
        "graph_seconds": graph_seconds,
        "staged_total_seconds": index_seconds
        + selection_seconds
        + membership_seconds
        + graph_seconds,
        "pipeline_peak_rss_bytes": peak,
        "threads": job["threads"],
        "selected_landmark_count": len(selected_landmarks),
        "landmark_sequence_equal": selected_landmarks == fixed_landmarks,
        "membership_completion_requested": method.startswith("verified")
        and method.endswith("complete"),
        "completion_distance_evaluations": len(x) * len(selected_landmarks)
        if method.startswith("verified") and method.endswith("complete")
        else 0,
        "E": graph.number_of_edges(),
        **cover_statistics(cover, len(x)),
        **work,
    }
    with np.load(job["reference_path"]) as data:
        members, indptr = data["members"], data["indptr"]
        reference = [members[a:b] for a, b in zip(indptr[:-1], indptr[1:], strict=True)]
    if method.startswith("verified") and selected_landmarks != fixed_landmarks:
        raise RuntimeError(
            "Verified selection failed the exact landmark-sequence gate."
        )
    audit = compare_covers(reference, cover, values=x[:, 0])
    graph_agrees = _equal_graphs(graph, build_mapper(cover))
    cover_equal = (
        audit.membership_false_positives == audit.membership_false_negatives == 0
    )
    covered = set().union(*(set(map(int, r)) for r in cover))
    row.update(
        fixed_cover_equal=cover_equal,
        memberships_complete=cover_equal,
        exact_output_match=bool(
            row["landmark_sequence_equal"] and cover_equal and graph_agrees
        ),
        graph_witness_counts_equal=graph_agrees,
        covers_all_observations=len(covered) == len(x),
        membership_precision=audit.membership_precision,
        membership_recall=audit.membership_recall,
        membership_false_positives=audit.membership_false_positives,
        membership_false_negatives=audit.membership_false_negatives,
        edge_precision=audit.edge_precision,
        edge_recall=audit.edge_recall,
        edge_jaccard=audit.edge_jaccard,
        color_max_error=audit.colors.max_absolute_error,
        color_bound_violations=int(
            np.count_nonzero(
                audit.colors.absolute_error > audit.colors.theoretical_bound + 1e-12
            )
        ),
    )
    if audit.membership_false_positives == 0:
        cert = certify_component_preservation(
            reference, len(x), audit.balls.false_negative_counts
        )
        row.update({"component_" + k: v for k, v in asdict(cert).items()})
    bins = {}
    for change in audit.witness_changes.values():
        if change.reference_count:
            key = str(2 ** (change.reference_count.bit_length() - 1))
            bucket = bins.setdefault(key, {"reference_edges": 0, "surviving_edges": 0})
            bucket["reference_edges"] += 1
            bucket["surviving_edges"] += int(change.approximate_count > 0)
    row["validation_failures"] = validation_failures(row)
    row["gate_passed"] = not row["validation_failures"]
    details = {
        "row": row,
        "job": job,
        "witness_survival_bins_lower_bound": bins,
        "backend_metadata": asdict(backend.metadata) if backend is not None else None,
        "faiss_config": asdict(backend.state.config)
        if isinstance(backend, FaissRangeBackend)
        else None,
        "threadpools": threadpool_info(),
        "faiss_blas_threshold": int(faiss.cvar.distance_compute_blas_threshold)
        if faiss
        else None,
    }
    np.savez_compressed(
        Path(job["result_path"]).with_suffix(".npz"),
        selected_landmarks=selected_landmarks,
        precision=audit.balls.precision,
        recall=audit.balls.recall,
        false_negatives=audit.balls.false_negative_counts,
        false_positives=audit.balls.false_positive_counts,
        color_error=audit.colors.absolute_error,
        color_bound=audit.colors.theoretical_bound,
    )
    return details


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/results/paper")
    )
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["gaussian", "mixture", "curve"],
        default=["gaussian", "mixture", "curve"],
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--graph-methods",
        nargs="+",
        choices=["inverted", "sparse"],
        default=["inverted", "sparse"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--target-ball-size", type=int, default=50)
    parser.add_argument("--calibration-queries", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--row-block-size", type=int, default=256)
    parser.add_argument("--candidate-k", type=int, default=4096)
    parser.add_argument("--candidate-eps", type=float, default=0.1)
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        job = json.loads(args.worker.read_text())
        try:
            result = run_worker(job)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            result = {
                "row": {
                    **{
                        k: job[k]
                        for k in ("dataset", "seed", "repeat", "method", "graph_method")
                    },
                    "gate_passed": False,
                    "error": message,
                    "validation_failures": [message],
                },
                "error": message,
                "job": job,
            }
        result["row"]["result_file"] = Path(job["result_path"]).name
        Path(job["result_path"]).write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return
    if args.smoke:
        args.n, args.d, args.target_ball_size = 200, 4, 12
        args.seeds, args.repeats = [0], 1
    for name in (
        "n",
        "d",
        "repeats",
        "threads",
        "calibration_queries",
        "batch_size",
        "block_size",
        "row_block_size",
        "candidate_k",
        "nlist",
        "nprobe",
        "ef_search",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"{name} must be positive.")
    if not 2 <= args.target_ball_size <= args.n:
        raise SystemExit("target-ball-size must be between 2 and n.")
    if not np.isfinite(args.candidate_eps) or args.candidate_eps < 0:
        raise SystemExit("candidate-eps must be finite and non-negative.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise SystemExit(
            "Use an empty output directory to preserve previous experiments."
        )
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=args.threads)
    manifest = {
        **provenance(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cases": [],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            case = prepare_case(output, dataset, args, seed)
            manifest["cases"].append(case)
            for repeat in range(args.repeats):
                for method in args.methods:
                    for graph_method in args.graph_methods:
                        job = {
                            **vars(args),
                            **case,
                            "method": method,
                            "repeat": repeat,
                            "graph_method": graph_method,
                            "reference_path": str(output / case["reference_file"]),
                        }
                        job.pop("output")
                        job.pop("worker")
                        jobs.append(job)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.random.default_rng(2026).shuffle(jobs)
    env = dict(os.environ)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = str(args.threads)
    rows = []
    for number, job in enumerate(jobs):
        job["result_path"] = str(output / f"run_{number:05d}.json")
        job_path = output / f"job_{number:05d}.json"
        job_path.write_text(json.dumps(job))
        print(
            f"[{number + 1}/{len(jobs)}] {job['dataset']} "
            f"{job['method']} {job['graph_method']}",
            flush=True,
        )
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path)],
            env=env,
            check=True,
        )
        rows.append(json.loads(Path(job["result_path"]).read_text())["row"])
        if not rows[-1]["gate_passed"]:
            print(
                f"  FAILED {rows[-1]['result_file']}: "
                + "; ".join(rows[-1]["validation_failures"]),
                flush=True,
            )
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=sorted(set().union(*(r.keys() for r in rows)))
            )
            writer.writeheader()
            writer.writerows(rows)
    passed = all(row["gate_passed"] for row in rows)
    failed_runs = [
        {
            key: row[key]
            for key in (
                "result_file",
                "dataset",
                "seed",
                "repeat",
                "method",
                "graph_method",
                "validation_failures",
            )
        }
        for row in rows
        if not row["gate_passed"]
    ]
    (output / "validation.json").write_text(
        json.dumps(
            {"gate_passed": passed, "runs": len(rows), "failed_runs": failed_runs},
            indent=2,
        )
        + "\n"
    )
    if not passed:
        raise SystemExit(
            "Experiment validation FAILED; inspect run JSON and summary.csv."
        )
    print(f"Validated {len(rows)} runs; wrote {output}")


if __name__ == "__main__":
    main()
