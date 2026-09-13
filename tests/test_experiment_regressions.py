import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fast_ballmapper import CKDTreeBackend, FaissConfig, FaissRangeBackend, build_cover
from fast_ballmapper import compare_covers
from fast_ballmapper.backends import _faiss


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
