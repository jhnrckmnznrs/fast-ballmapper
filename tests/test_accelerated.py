"""Exact batched/compiled candidate checks and graph dispatch regressions."""

import numpy as np
import pytest

from fast_ballmapper import build_cover_localized, select_landmarks_uncovered
from fast_ballmapper.optimized import (
    build_mapper_auto,
    build_mapper_csr,
    select_mapper_algorithm,
)


@pytest.mark.parametrize(
    "n,d,eps",
    [(100, 1, .1), (300, 2, .12), (500, 3, .23),
     (100, 20, 5.7), (80, 50, 9.), (50, 0, 0.)],
)
def test_completion_engines_match_direct_float64_reference(n, d, eps):
    x = np.random.default_rng(n+d).normal(size=(n, d))
    landmarks, owners = select_landmarks_uncovered(x, eps)
    reference = [
        np.flatnonzero(np.linalg.norm(x - x[index], axis=1) <= eps)
        for index in landmarks
    ]
    for engine in ('scalar', 'batch', 'auto'):
        got = build_cover_localized(
            x, landmarks, eps, owners, strategy='localized',
            engine=engine, block_size=23
        )
        assert all(np.array_equal(a,b) for a,b in zip(reference, got, strict=True))


def test_optional_numba_engine_matches_reference():
    pytest.importorskip('numba')
    rng = np.random.default_rng(40)
    for d in (1, 3, 12, 50):
        x = rng.normal(size=(100, d))
        landmarks, owners = select_landmarks_uncovered(x, 2.5)
        expected = [
            np.flatnonzero(np.linalg.norm(x-x[i], axis=1) <= 2.5)
            for i in landmarks
        ]
        got = build_cover_localized(
            x, landmarks, 2.5, owners, strategy='localized',
            engine='numba', block_size=19
        )
        assert all(np.array_equal(a,b) for a,b in zip(expected, got, strict=True))


def test_adversarial_near_boundary():
    rng = np.random.default_rng(20261015)
    for _ in range(200):
        eps = 10 ** rng.uniform(-7, 7)
        shift = rng.normal() * 10 ** rng.uniform(-7, 7)
        ulp = np.nextafter(eps, np.inf) - eps
        points = np.array([
            [shift], [shift + 2 * eps],
            [shift + eps + rng.integers(-4, 5) * ulp],
            [shift + .5 * eps], [shift + 1.5 * eps],
        ])
        landmarks, owners = select_landmarks_uncovered(points, eps)
        expected = [
            np.flatnonzero(np.linalg.norm(points-points[i], axis=1) <= eps)
            for i in landmarks
        ]
        for engine in ('batch', 'scalar'):
            got = build_cover_localized(
                points, landmarks, eps, owners, strategy='localized',
                engine=engine
            )
            assert all(np.array_equal(a,b)
                       for a,b in zip(expected,got,strict=True))


def test_numba_near_boundary():
    pytest.importorskip('numba')
    rng = np.random.default_rng(120)
    for _ in range(80):
        eps = 10 ** rng.uniform(-7, 7)
        shift = rng.normal() * 10 ** rng.uniform(-7, 7)
        x = np.array([
            [shift], [shift + 2*eps], [shift + np.nextafter(eps, np.inf)],
            [shift + eps], [shift + 3*eps],
        ])
        landmarks, owners = select_landmarks_uncovered(x,eps)
        reference = [
            np.flatnonzero(np.linalg.norm(x - x[i],axis=1) <= eps)
            for i in landmarks
        ]
        got = build_cover_localized(
            x,landmarks,eps,owners,strategy='localized',engine='numba'
        )
        assert all(np.array_equal(a,b)
                   for a,b in zip(reference,got,strict=True))


def test_dispatch_sparse_dense_large_and_override():
    rng = np.random.default_rng(10)
    tiny = [rng.choice(1000, size=3, replace=False) for _ in range(20)]
    dense = [np.arange(90) for _ in range(30)]
    large = [rng.choice(2000, size=6, replace=False) for _ in range(300)]
    assert select_mapper_algorithm(tiny, 1000) == 'streaming'
    assert select_mapper_algorithm(dense, 100) == 'csr'
    assert select_mapper_algorithm(large, 2000) == 'csr'
    for cover, n in ((tiny,1000),(dense,100),(large,2000)):
        expected = build_mapper_csr(cover,n)
        for algorithm in ('csr','streaming','auto'):
            observed = build_mapper_auto(cover,n,algorithm=algorithm)
            assert (expected != observed).nnz == 0
            legacy = build_mapper_auto(
                cover,n,algorithm=algorithm,output='networkx'
            )
            assert legacy.number_of_nodes() == len(cover)
            assert legacy.number_of_edges() == expected.nnz//2


def test_invalid_modes_are_rejected():
    with pytest.raises(ValueError, match='algorithm'):
        build_mapper_auto([],0,algorithm='invalid')
    with pytest.raises(ValueError, match='output'):
        build_mapper_auto([],0,output='invalid')
    with pytest.raises(ValueError, match='engine'):
        build_cover_localized(
            np.array([[0.]]),[0],1.,[0],engine='invalid'
        )
