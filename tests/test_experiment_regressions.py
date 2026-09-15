import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fast_ballmapper import CKDTreeBackend, FaissConfig, FaissRangeBackend, build_cover
from fast_ballmapper import compare_covers
from fast_ballmapper.backends import _faiss


def _paper_runner():
    path = Path(__file__).resolve().parents[1] / "experiments/run_paper_experiments.py"
    spec = importlib.util.spec_from_file_location("paper_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", ["flat_scalar", "flat_batch"])
def test_paper_flat_removes_reported_gaussian_boundary_false_positive(method):
    pytest.importorskip("faiss")
    # Original observation 653 relative to landmark 5 in the supplied pilot.
    # Its float64 distance exceeds epsilon by about 2.0742e-7, while native
    # float32 FAISS can include it. Generate just the prefix containing the pair.
    x = np.random.default_rng(0).normal(size=(654, 32)).astype(np.float32)[[5, 653]]
    eps = 5.995250733644709
    assert np.linalg.norm(x[0].astype(float) - x[1].astype(float)) > eps
    backend = _paper_runner().backend_for(x, method, {"batch_size": 64, "seed": 0})
    assert list(map(set, backend.query_radius([0, 1], eps))) == [{0}, {1}]


@pytest.mark.parametrize("method", ["flat_scalar", "flat_batch", "ivf", "hnsw"])
def test_paper_gate_still_rejects_one_extra_membership(method):
    row = {
        "method": method,
        "graph_witness_counts_equal": True,
        "color_bound_violations": 0,
        "landmark_sequence_equal": True,
        "fixed_cover_equal": False,
        "membership_false_positives": 1,
        "membership_false_negatives": 0,
        "covers_all_observations": True,
    }
    failures = _paper_runner().validation_failures(row)
    assert len(failures) == 1
    assert "fp=1" in failures[0]


def test_paper_gate_retains_partial_cover_and_landmark_requirements():
    module = _paper_runner()
    row = {
        "method": "verified_none_partial",
        "graph_witness_counts_equal": True,
        "color_bound_violations": 0,
        "landmark_sequence_equal": True,
        "fixed_cover_equal": False,
        "membership_false_positives": 0,
        "membership_false_negatives": 5,
        "covers_all_observations": True,
    }
    assert module.validation_failures(row) == []
    row["covers_all_observations"] = False
    assert module.validation_failures(row) == [
        "verified_cover_does_not_cover_all_observations"
    ]
    row.update(
        method="flat_scalar",
        fixed_cover_equal=True,
        membership_false_negatives=0,
        landmark_sequence_equal=False,
    )
    assert module.validation_failures(row) == ["landmark_sequence_mismatch"]


def test_paper_worker_exception_retains_identity_and_reason(tmp_path, monkeypatch):
    module = _paper_runner()
    job = {
        "dataset": "gaussian",
        "seed": 0,
        "repeat": 0,
        "method": "flat_scalar",
        "graph_method": "sparse",
        "result_path": str(tmp_path / "run_00000.json"),
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(worker=job_path))

    def fail(job):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(module, "run_worker", fail)
    module.main()
    result = json.loads(Path(job["result_path"]).read_text())
    assert result["row"]["dataset"] == "gaussian"
    assert result["row"]["graph_method"] == "sparse"
    assert result["row"]["result_file"] == "run_00000.json"
    assert result["row"]["gate_passed"] is False
    assert result["row"]["validation_failures"] == ["RuntimeError: backend failed"]
    assert result["row"]["error"] == result["error"]


def test_exactness_gate_rejects_membership_mismatch_even_when_graph_agrees():
    path = Path(__file__).resolve().parents[1] / "experiments/run_exactness_gate.py"
    spec = importlib.util.spec_from_file_location("gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {
        key: True
        for key in [
            "landmark_indices_equal",
            "ckdtree_landmark_indices_equal",
            "selected_cover_equal",
            "ckdtree_selected_cover_equal",
            "edge_sets_equal",
            "ckdtree_edge_sets_equal",
        ]
    }
    row.update(
        {
            key: 0
            for key in [
                "false_negative_count",
                "false_positive_count",
                "ckdtree_false_negative_count",
                "ckdtree_false_positive_count",
            ]
        }
    )
    assert module.gate_passed(row)
    row["ckdtree_false_negative_count"] = 1
    assert not module.gate_passed(row)


def test_float32_colors_do_not_create_spurious_tight_bound_violations():
    audit = compare_covers(
        [[0, 1]], [[0]], values=np.array([0.1, 0.2], dtype=np.float32)
    )
    assert audit.colors.absolute_error[0] <= audit.colors.theoretical_bound[0]


@pytest.mark.parametrize("u", [0, 0.1, 0.5, 1])
def test_inflated_ckdtree_candidate_filter_matches_reference(u):
    x = np.random.default_rng(3).normal(size=(120, 3))
    reference = build_cover(x, [0, 4, 30], 1.0, method="brute_force")
    observed = CKDTreeBackend(x, candidate_eps=u).query_radius([0, 4, 30], 1)
    assert list(map(set, observed)) == list(map(set, reference))


def test_cosine_index_does_not_mutate_inputs_and_owns_verification_snapshot():
    pytest.importorskip("faiss")
    x = np.array([[3, 4], [5, 12]], dtype=np.float32)
    before = x.copy()
    backend = FaissRangeBackend(x, metric="cosine")
    np.testing.assert_array_equal(x, before)
    expected = before.astype(float) / np.linalg.norm(
        before.astype(float), axis=1, keepdims=True
    )
    np.testing.assert_array_equal(backend.state.verification_points, expected)
    x[:] = 0
    np.testing.assert_array_equal(backend.state.original_points, before)


@pytest.mark.parametrize("k,expected", [(1, False), (3, True), (10, True)])
def test_flat_capped_knn_metadata(k, expected):
    pytest.importorskip("faiss")
    backend = FaissRangeBackend(
        np.arange(3.0).reshape(-1, 1),
        config=FaissConfig(query_mode="knn", candidate_k=k),
    )
    assert backend.metadata.is_exact is expected


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_true_faiss_batch_calls_and_order_match_scalar(metric):
    pytest.importorskip("faiss")
    x = np.random.default_rng(4).normal(size=(75, 5)).astype(np.float32)
    backend = FaissRangeBackend(x, metric, FaissConfig(query_batch_size=8))
    ids = [3, 1, 4, 3, 2, 40, 74, 6, 8, 11, 12]
    expected = [_faiss.query_faiss_range(backend.state, i, 0.7) for i in ids]
    real_index = backend.state.index
    calls = []

    class Spy:
        def range_search(self, queries, radius):
            calls.append(len(queries))
            return real_index.range_search(queries, radius)

    backend.state.index = Spy()
    actual = backend.query_radius(ids, 0.7)
    assert calls == [8, 3]
    assert list(map(set, actual)) == list(map(set, expected))
    assert backend.query_radius([], 0.7) == []


def test_knn_verification_receives_overestimated_candidates_scalar_and_batch():
    class OverestimatedIndex:
        def search(self, queries, k):
            return np.tile([0, 100, 100], (len(queries), 1)), np.tile(
                [0, 1, 2], (len(queries), 1)
            )

    points = np.array([[0.0], [0.1], [5.0]])
    backend = _faiss.FaissBackend(
        points,
        points.astype(np.float32),
        points,
        OverestimatedIndex(),
        "euclidean",
        FaissConfig(query_mode="knn", candidate_k=3, exact_verify=True),
    )
    assert set(_faiss.query_faiss_range(backend, 0, 0.2)) == {0, 1}
    assert list(
        map(set, _faiss._query_faiss_batch(backend, np.array([0, 1]), 0.2))
    ) == [{0, 1}, {0, 1}]
