from types import SimpleNamespace

import numpy as np

from fast_ballmapper import BruteForceBackend, CuVSConfig, HnswlibConfig
from fast_ballmapper.backends._cuvs import CuVSBackend
from fast_ballmapper.backends._hnswlib import HnswlibBackend


def _sets(cover):
    return [set(map(int, members)) for members in cover]


class _FakeHnswIndex:
    def __init__(self, space, dim):
        self.space = space
        self.dim = dim
        self.data = None
        self.ids = None

    def init_index(self, **kwargs):
        return None

    def add_items(self, data, ids, num_threads=-1):
        self.data = np.asarray(data, dtype=np.float32)
        self.ids = np.asarray(ids, dtype=np.int64)

    def set_ef(self, ef):
        self.ef = ef

    def set_num_threads(self, n):
        self.threads = n

    def knn_query(self, queries, k=1, num_threads=-1):
        queries = np.asarray(queries, dtype=np.float32)
        all_labels = []
        all_distances = []
        for query in queries:
            if self.space == "l2":
                distances = np.sum((self.data - query) ** 2, axis=1)
            else:
                q = query / np.linalg.norm(query)
                d = self.data / np.linalg.norm(self.data, axis=1, keepdims=True)
                distances = 1.0 - d @ q
            order = np.argsort(distances)[:k]
            all_labels.append(self.ids[order])
            all_distances.append(distances[order])
        return np.asarray(all_labels), np.asarray(all_distances)


class _FakeCuPy:
    @staticmethod
    def asarray(value):
        return np.asarray(value)

    @staticmethod
    def asnumpy(value):
        return np.asarray(value)


class _FakeCuVSBruteForce:
    @staticmethod
    def build(dataset, metric="sqeuclidean"):
        return (np.asarray(dataset, dtype=np.float32), metric)

    @staticmethod
    def search(index, queries, k):
        data, metric = index
        queries = np.asarray(queries, dtype=np.float32)
        distance_rows = []
        neighbor_rows = []
        for query in queries:
            if metric == "sqeuclidean":
                distances = np.sum((data - query) ** 2, axis=1)
            else:
                q = query / np.linalg.norm(query)
                d = data / np.linalg.norm(data, axis=1, keepdims=True)
                distances = 1.0 - d @ q
            order = np.argsort(distances)[:k]
            distance_rows.append(distances[order])
            neighbor_rows.append(order)
        return np.asarray(distance_rows), np.asarray(neighbor_rows)


class _FakeIndexParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeSearchParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeCuVSCagra:
    IndexParams = _FakeIndexParams
    SearchParams = _FakeSearchParams

    @staticmethod
    def build(index_params, dataset):
        metric = index_params.kwargs.get("metric", "sqeuclidean")
        return (np.asarray(dataset, dtype=np.float32), metric)

    @staticmethod
    def search(search_params, index, queries, k):
        return _FakeCuVSBruteForce.search(index, queries, k)


def test_hnswlib_adapter_filters_candidates_to_closed_ball(monkeypatch):
    import fast_ballmapper.backends._hnswlib as module

    monkeypatch.setattr(module, "hnswlib", SimpleNamespace(Index=_FakeHnswIndex))
    x = np.array([[0.0], [0.5], [1.0], [2.0]], dtype=np.float32)
    backend = HnswlibBackend(
        x,
        config=HnswlibConfig(candidate_k=4, ef_search=4, exact_verify=True),
    )
    reference = BruteForceBackend(x)

    assert _sets(backend.query_radius([0, 2], 1.0)) == _sets(
        reference.query_radius([0, 2], 1.0)
    )
    assert backend.metadata.is_exact is False


def test_cuvs_brute_force_adapter_can_be_exhaustive(monkeypatch):
    import fast_ballmapper.backends._cuvs as module

    monkeypatch.setattr(module, "cp", _FakeCuPy)
    monkeypatch.setattr(module, "cuvs_brute_force", _FakeCuVSBruteForce)
    monkeypatch.setattr(module, "cuvs_cagra", _FakeCuVSCagra)

    x = np.array([[0.0], [0.5], [1.0], [2.0]], dtype=np.float32)
    backend = CuVSBackend(
        x,
        config=CuVSConfig(
            algorithm="brute_force",
            candidate_k=None,
            exact_verify=True,
        ),
    )
    reference = BruteForceBackend(x)

    assert _sets(backend.query_radius([0, 2], 1.0)) == _sets(
        reference.query_radius([0, 2], 1.0)
    )
    assert backend.metadata.is_exact is True
    assert backend.metadata.device == "gpu"


def test_cuvs_cagra_adapter_is_marked_approximate(monkeypatch):
    import fast_ballmapper.backends._cuvs as module

    monkeypatch.setattr(module, "cp", _FakeCuPy)
    monkeypatch.setattr(module, "cuvs_brute_force", _FakeCuVSBruteForce)
    monkeypatch.setattr(module, "cuvs_cagra", _FakeCuVSCagra)

    x = np.array([[0.0], [0.5], [1.0], [2.0]], dtype=np.float32)
    backend = CuVSBackend(
        x,
        config=CuVSConfig(
            algorithm="cagra",
            candidate_k=4,
            exact_verify=True,
        ),
    )

    assert backend.metadata.is_exact is False
    assert _sets(backend.query_radius([0], 1.0)) == [{0, 1, 2}]


def test_hnswlib_missing_dependency_has_clear_error(monkeypatch):
    import fast_ballmapper.backends._hnswlib as module

    monkeypatch.setattr(module, "hnswlib", None)
    x = np.array([[0.0], [1.0]], dtype=np.float32)
    try:
        HnswlibBackend(x)
    except ImportError as error:
        assert "fast-ballmapper[hnswlib]" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected missing hnswlib dependency error.")


def test_cuvs_missing_dependency_has_clear_error(monkeypatch):
    import fast_ballmapper.backends._cuvs as module

    monkeypatch.setattr(module, "cp", None)
    monkeypatch.setattr(module, "cuvs_brute_force", None)
    monkeypatch.setattr(module, "cuvs_cagra", None)
    x = np.array([[0.0], [1.0]], dtype=np.float32)
    try:
        CuVSBackend(x)
    except ImportError as error:
        assert "cuvs-cu12" in str(error) or "cuvs-cu13" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected missing cuVS dependency error.")
