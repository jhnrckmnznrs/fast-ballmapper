"""Independent exactness and edge-case tests for the optimized algorithms."""
import itertools

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from fast_ballmapper.optimized import (
    ball_mapper_components, build_cover_localized, build_mapper_csr,
    build_mapper_auto, build_mapper_streaming, compute_landmarks_uncovered,
    compute_mapper_streaming, fixed_landmark_h0, iter_cover_blocked,
    select_landmarks_uncovered,
)


def reference_landmarks(x, eps):
    uncovered = np.ones(len(x), dtype=bool)
    lm, cov = [], []
    for i in range(len(x)):
        if not uncovered[i]:
            continue
        lm.append(i)
        members = np.flatnonzero(np.linalg.norm(x-x[i],axis=1)<=eps)
        cov.append(members)
        uncovered[members]=False
    return lm,cov


def reference_adj(cover, n):
    out = np.zeros((len(cover),len(cover)),dtype=np.int64)
    for a,b in itertools.combinations(range(len(cover)),2):
        k = len(set(cover[a]) & set(cover[b]))
        out[a,b] = k
        out[b,a] = k
    return out


@pytest.mark.parametrize('n,d,eps',[(0,2,.1),(1,1,0.),(100,2,.13),
                                   (120,4,.65),(150,15,3.5),(100,30,7.),
                                   (200,3,0.),(140,2,1.1)])
def test_greedy_and_complete_cover(n,d,eps):
    x=np.random.default_rng(n+d).normal(size=(n,d))
    ref_lm,ref_cov = reference_landmarks(x,eps)
    lm,owners=select_landmarks_uncovered(x,eps,block_size=17)
    assert lm==ref_lm
    for strategy in ('localized','blocked','auto'):
        got,stats=build_cover_localized(x,lm,eps,owners,strategy=strategy,
                                         block_size=19,return_stats=True)
        assert len(got)==len(ref_cov)
        assert all(np.array_equal(u,v) for u,v in zip(got,ref_cov,strict=True))
        assert stats.total_possible_point_evaluations == n*len(lm)
    got_lm,got_cov=compute_landmarks_uncovered(x,eps)
    assert got_lm==lm
    assert all(np.array_equal(u,v) for u,v in zip(got_cov,ref_cov,strict=True))


@pytest.mark.parametrize('n,m',[(0,0),(100,15),(200,90),(50,1),(150,40)])
def test_graph_algorithms(n,m):
    rng=np.random.default_rng(n+m)
    cover = [rng.choice(n,size=rng.integers(0,min(n,20)+1),replace=True)
             if n else np.empty(0,dtype=np.intp) for _ in range(m)]
    expected=reference_adj(cover,n)
    for name,adj in [
        ('csr',build_mapper_csr(cover,n,row_block_size=7)),
        ('stream',build_mapper_streaming(iter(cover),n,n_balls=m)),
        ('auto',build_mapper_auto(cover,n)),
    ]:
        assert np.array_equal(adj.toarray(),expected),name
    cnt,lab=ball_mapper_components(iter(cover),n,n_balls=m)
    ref_cnt,ref_lab=connected_components((expected != 0),directed=False)
    assert cnt==ref_cnt
    for i,j in itertools.combinations(range(m),2):
        assert (lab[i]==lab[j])==(ref_lab[i]==ref_lab[j])
    graph=build_mapper_auto(cover,n,output='networkx')
    assert graph.number_of_nodes()==m
    assert all(graph[a][b]['witness_count']==expected[a,b] for a,b in graph.edges())


def test_multiscale():
    rng=np.random.default_rng(17)
    x=rng.random((120,3))
    lm=np.array([0,3,4,11,20,23,38,55,68,99])
    radii=np.array([.6,.0,.18,.3,.18,.45])
    h=fixed_landmark_h0(x,lm,radii,block_size=13)
    assert np.array_equal(h.radii,radii)
    for i,r in enumerate(radii):
        cover=[np.flatnonzero(np.linalg.norm(x-x[j],axis=1)<=r) for j in lm]
        expected=connected_components(reference_adj(cover,len(x)),directed=False)[0]
        assert h.component_counts[i]==expected
    assert len(h.merges)<=len(lm)-1
    assert all(h.merges[i][0]<=h.merges[i+1][0] for i in range(len(h.merges)-1))


def test_streaming_pipeline():
    x=np.random.default_rng(3).uniform(size=(80,2))
    lm,cover=reference_landmarks(x,.18)
    got_lm,got_graph=compute_mapper_streaming(x,.18,block_size=17)
    assert got_lm==lm
    assert np.array_equal(got_graph.toarray(),reference_adj(cover,len(x)))
    assert all(
        np.array_equal(a,b) for a,b in zip(
            iter_cover_blocked(x,lm,.18,block_size=9),cover,strict=True
        )
    )


def test_bad_owner_rejected():
    x=np.array([[0.],[2.],[4.]])
    with pytest.raises(ValueError,match='verified'):
        build_cover_localized(x,[0,1,2],.1,[0,0,2])


def test_uncovered_boundary():
    x=np.array([[0.],[1.],[2.],[3.]])
    lm,cover=compute_landmarks_uncovered(x,1.,strategy='localized')
    assert lm==[0,2]
    assert [list(a) for a in cover]==[[0,1],[1,2,3]]


def test_multiscale_from_complete_cover():
    rng=np.random.default_rng(12)
    x=rng.random((150,2))
    lm=[1,4,20,36,72,85,99]
    radii=[.23,.39,.3,.1,.0,.4]
    full=[np.flatnonzero(np.linalg.norm(x-x[j],axis=1)<=max(radii)) for j in lm]
    from fast_ballmapper.optimized import fixed_landmark_h0_from_cover
    a=fixed_landmark_h0_from_cover(x,lm,radii,full)
    b=fixed_landmark_h0(x,lm,radii)
    np.testing.assert_array_equal(a.component_counts,b.component_counts)
    with pytest.raises(ValueError):
        fixed_landmark_h0_from_cover(x,lm,radii,[np.array([0])]*len(lm))