"""Exact, memory-conscious Ball Mapper constructions.

The optional algorithms here do not modify the existing range-query backends.
All Euclidean membership decisions use direct float64 ``norm <= eps``.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix

from fast_ballmapper._accelerated import filter_candidates, _validate_engine


def _points(x: np.ndarray, eps: float) -> np.ndarray:
    points = np.asarray(x)
    if (
        points.ndim != 2
        or not np.issubdtype(points.dtype, np.number)
        or np.iscomplexobj(points)
    ):
        raise ValueError("x must be a two-dimensional real numeric array")
    if not np.all(np.isfinite(points)):
        raise ValueError("x must contain only finite coordinates")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and non-negative")
    return np.ascontiguousarray(points, dtype=np.float64)


def _indices(
    ids: Sequence[int] | np.ndarray, n: int, *, unique: bool = False
) -> np.ndarray:
    arr = np.asarray(ids)
    if arr.ndim != 1 or (arr.dtype.kind not in 'iu' and arr.size):
        raise TypeError("indices must be a one-dimensional array of integers")
    if np.any(arr < 0) or np.any(arr >= n):
        raise IndexError("index outside the point cloud")
    arr = arr.astype(np.intp, copy=False)
    if unique and len(np.unique(arr)) != len(arr):
        raise ValueError("indices must be distinct")
    return arr


def _within(points: np.ndarray, center: np.ndarray, eps: float) -> np.ndarray:
    with np.errstate(over='ignore', invalid='ignore'):
        ds = np.linalg.norm(points - center, axis=1)
    if not np.all(np.isfinite(ds)):
        raise ValueError("distance overflow; rescale coordinates")
    return ds <= eps


def select_landmarks_uncovered(
    x: np.ndarray, eps: float, *, block_size: int = 16384,
) -> tuple[list[int], np.ndarray]:
    """Exact greedy landmarks and first-cover owners, testing only uncovered rows.

    The landmark sequence is the smallest-index-uncovered sequence of standard
    greedy Ball Mapper. ``owners[x]`` is the ordinal of the first ball covering x.
    Work is proportional to the sum of the number of uncovered points inspected.
    An exact full cover can subsequently be constructed with ``build_cover_localized``.
    """
    points = _points(x, eps)
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, (int, np.integer))
        or block_size <= 0
    ):
        raise ValueError('block_size must be a positive integer')
    n = len(points)
    owners = np.full(n, -1, dtype=np.intp)
    landmarks: list[int] = []
    # Use a monotonically increasing scan for the next uncovered point.
    for candidate in range(n):
        if owners[candidate] >= 0:
            continue
        ordinal = len(landmarks)
        landmarks.append(candidate)
        for start in range(candidate, n, int(block_size)):
            ids = np.arange(start, min(n, start + int(block_size)), dtype=np.intp)
            ids = ids[owners[ids] < 0]
            if not len(ids):
                continue
            members = ids[_within(points[ids], points[candidate], eps)]
            owners[members] = ordinal
    assert n == 0 or np.all(owners >= 0)
    return landmarks, owners


@dataclass(frozen=True)
class CompletionStats:
    strategy: str
    center_distance_evaluations: int
    point_distance_evaluations: int
    total_possible_point_evaluations: int


def build_cover_localized(
    x: np.ndarray,
    landmarks: Sequence[int] | np.ndarray,
    eps: float,
    owners: Sequence[int] | np.ndarray,
    *,
    strategy: Literal['auto', 'localized', 'blocked'] = 'auto',
    block_size: int = 16384,
    max_localized_fraction: float = 0.45,
    return_stats: bool = False,
    engine: Literal['scalar', 'batch', 'numba', 'auto'] = 'auto',
) -> list[np.ndarray] | tuple[list[np.ndarray], CompletionStats]:
    """Build exact complete Euclidean balls from *verified* first-cover owners.

    Every point must have a valid assigned landmark at distance <= eps. Center
    pruning uses direct float64 distances and the metric triangle inequality;
    a candidate is never discarded by an approximate spatial index. 'auto'
    samples geometric admissibility to avoid slow localization in high dimensions.
    'blocked' uses exhaustive bounded-workspace norm computations.
    """
    points = _points(x, eps)
    n = len(points)
    lm = _indices(landmarks, n, unique=True)
    m = len(lm)
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, (int, np.integer))
        or block_size <= 0
    ):
        raise ValueError('block_size must be a positive integer')
    engine = _validate_engine(engine)
    if strategy not in ('auto', 'localized', 'blocked'):
        raise ValueError("strategy must be 'auto', 'localized' or 'blocked'")
    if not np.isfinite(max_localized_fraction) or not 0 < max_localized_fraction <= 1:
        raise ValueError('max_localized_fraction must be in (0, 1]')
    if m == 0:
        if n:
            raise ValueError('nonempty point cloud cannot have an empty covering')
        empty: list[np.ndarray] = []
        stats = CompletionStats('blocked', 0, 0, 0)
        return (empty, stats) if return_stats else empty
    owner = _indices(owners, m)
    if len(owner) != n:
        raise ValueError('owners must have one entry for every observation')
    centers = points[lm]
    # Compute verified owner radii; invalid assignments are not a certificate.
    radii = np.empty(m, dtype=np.float64)
    counts = np.bincount(owner, minlength=m)
    order = np.argsort(owner, kind='stable')
    offsets = np.empty(m+1, dtype=np.intp)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)
    owner_evaluations = 0
    for i in range(m):
        ids = order[offsets[i]:offsets[i+1]]
        if not len(ids):
            radii[i] = 0.0
            continue
        distances = np.linalg.norm(points[ids] - centers[i], axis=1)
        if not np.all(np.isfinite(distances)) or np.any(distances > eps):
            raise ValueError('owners are not a verified radius-epsilon covering')
        radii[i] = float(np.max(distances))
        owner_evaluations += len(ids)
    center_work = 0
    selected = strategy
    # Independently rounded Euclidean norms may violate the triangle-inequality
    # threshold by a few ulps. Include uncertain groups rather than excluding
    # them. The coordinate scale also covers large translated point clouds.
    # This is deliberately conservative but is not a formal interval proof
    # for every possible hardware/dtype combination.
    unit = np.finfo(np.float64).eps
    dimension = points.shape[1]
    max_coord = (
        np.max(np.abs(centers), axis=1)
        if dimension else np.zeros(m, dtype=np.float64)
    )
    subnormal_guard = 16.0 * np.nextafter(0.0, 1.0)

    def may_intersect(j: int, distances: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            extent = (
                np.sqrt(dimension) * (max_coord + max_coord[j])
                + eps + radii
            )
            guard = (64.0 * unit * (dimension + 2)) * extent + subnormal_guard
            threshold = np.nextafter(eps + radii + guard, np.inf)
        return distances <= threshold
    if strategy == 'auto':
        # Estimate the admissible point fraction without computing any full ball.
        # High-dimensional geometry rarely profits from Euclidean group exclusion.
        if points.shape[1] > 16 or m > 4096:
            selected = 'blocked'
        else:
            sample = np.unique(np.linspace(0, m - 1, min(m, 24), dtype=np.intp))
            expected = 0
            for j in sample:
                d = np.linalg.norm(centers - centers[j], axis=1)
                center_work += m
                expected += int(np.sum(counts[may_intersect(int(j), d)]))
            selected = (
                'localized'
                if expected < max_localized_fraction * n * len(sample)
                else 'blocked'
            )
    cover: list[np.ndarray] = []
    distance_work = 0
    if selected == 'blocked':
        for j in range(m):
            chunks = []
            for start in range(0, n, int(block_size)):
                ids = np.arange(start, min(n, start+int(block_size)), dtype=np.intp)
                chunks.append(ids[_within(points[ids], centers[j], eps)])
                distance_work += len(ids)
            cover.append(
                np.concatenate(chunks) if chunks else np.empty(0, dtype=np.intp)
            )
    else:
        for j in range(m):
            d = np.linalg.norm(centers - centers[j], axis=1)
            if not np.all(np.isfinite(d)):
                raise ValueError('center distance overflow; rescale coordinates')
            center_work += m
            admissible = np.flatnonzero(may_intersect(j, d))
            # Batch across groups to amortize NumPy overhead. Retained
            # members still use the direct float64 Euclidean predicate.
            chunks = []
            candidate_groups = []
            for i in admissible:
                ids = order[offsets[i]:offsets[i+1]]
                if not len(ids):
                    continue
                # A point always belongs to its verified owner's ball.
                if i == j:
                    chunks.append(ids)
                elif engine == 'scalar':
                    distance_work += len(ids)
                    chunks.append(ids[_within(points[ids], centers[j], eps)])
                else:
                    candidate_groups.append(ids)
                    distance_work += len(ids)
            if candidate_groups:
                ids = np.concatenate(candidate_groups)
                chunks.append(filter_candidates(
                    points, centers[j], ids, eps, engine=engine,
                    block_size=int(block_size),
                ))
            cover.append(
                np.sort(np.concatenate(chunks))
                if chunks else np.empty(0, dtype=np.intp)
            )
    stats = CompletionStats(
        str(selected), int(center_work), int(distance_work + owner_evaluations),
        int(n*m),
    )
    return (cover, stats) if return_stats else cover


def compute_landmarks_uncovered(
    x: np.ndarray, eps: float, *,
    strategy: Literal['auto', 'localized', 'blocked'] = 'auto',
    block_size: int = 16384,
    engine: Literal['scalar', 'batch', 'numba', 'auto'] = 'auto',
) -> tuple[list[int], list[np.ndarray]]:
    """Drop-in Euclidean alternative returning full greedy landmarks and cover."""
    landmarks, owner = select_landmarks_uncovered(x, eps, block_size=block_size)
    cover = build_cover_localized(
        x, landmarks, eps, owner, strategy=strategy, block_size=block_size,
        engine=engine,
    )
    return landmarks, cover


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.intp)
        self.rank = np.zeros(n, dtype=np.uint8)
        self.n_components = n

    def find(self, u: int) -> int:
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]
            u = int(self.parent[u])
        return u

    def union(self, u: int, v: int) -> bool:
        a, b = self.find(u), self.find(v)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        self.n_components -= 1
        return True


def ball_mapper_components(
    cover: Iterable[Sequence[int] | np.ndarray], n_samples: int, *,
    n_balls: int | None = None,
) -> tuple[int, np.ndarray]:
    """Find exact overlap-graph components without generating edges.

    Input may be a streaming iterable. ``n_balls`` is required for iterators
    without ``len``. Duplicate IDs within a ball are harmless.
    Returns (number of components, component label per ball).
    """
    if (
        isinstance(n_samples, bool)
        or not isinstance(n_samples, (int, np.integer))
        or n_samples < 0
    ):
        raise ValueError('n_samples must be a nonnegative integer')
    if n_balls is None:
        if not hasattr(cover, '__len__'):
            raise ValueError('n_balls is required for streaming covers')
        n_balls = len(cover)  # type: ignore[arg-type]
    if (
        isinstance(n_balls, bool)
        or not isinstance(n_balls, (int, np.integer))
        or n_balls < 0
    ):
        raise ValueError('n_balls must be a nonnegative integer')
    first = np.full(n_samples, -1, dtype=np.intp)
    uf = _UnionFind(n_balls)
    total = 0
    for j, ids in enumerate(cover):
        if j >= n_balls:
            raise ValueError('n_balls smaller than the cover')
        for p in np.unique(_indices(ids, n_samples)):
            if first[p] == -1:
                first[p] = j
            else:
                uf.union(int(first[p]), j)
        total += 1
    if total != n_balls:
        raise ValueError('n_balls larger than the cover')
    roots = np.asarray([uf.find(j) for j in range(n_balls)], dtype=np.intp)
    return uf.n_components, np.unique(roots, return_inverse=True)[1]


def build_mapper_streaming(
    cover: Iterable[Sequence[int] | np.ndarray], n_samples: int, *, n_balls: int,
    output: Literal['csr','networkx'] = 'csr',
):
    """Aggregate exact witness counts while consuming each ball only once.

    Avoids keeping the cover in memory. Work is proportional to the number of
    overlapping ball pairs per observation; use sparse matrix multiplication
    instead when multiplicities are high. CSR weights are witness counts.
    """
    if (isinstance(n_samples, bool) or isinstance(n_balls, bool)
            or not isinstance(n_samples, (int, np.integer))
            or not isinstance(n_balls, (int, np.integer))
            or n_samples < 0 or n_balls < 0):
        raise ValueError('n_samples and n_balls must be nonnegative')
    inverted: dict[int, list[int]] = defaultdict(list)
    weights: dict[tuple[int,int], int] = defaultdict(int)
    n_seen = 0
    for j, ids in enumerate(cover):
        if j >= n_balls:
            raise ValueError('n_balls smaller than the cover')
        for p in np.unique(_indices(ids, n_samples)):
            previous = inverted[int(p)]
            for i in previous:
                weights[(i,j)] += 1
            previous.append(j)
        n_seen += 1
    if n_seen != n_balls:
        raise ValueError('n_balls larger than the cover')
    if output == 'networkx':
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(range(n_balls))
        graph.add_edges_from(
            (i,j,{'witness_count': int(w)}) for (i,j),w in weights.items()
        )
        return graph
    if output != 'csr':
        raise ValueError("output must be 'csr' or 'networkx'")
    rows, cols, vals = [], [], []
    for (i,j),w in weights.items():
        rows.extend((i,j))
        cols.extend((j,i))
        vals.extend((w,w))
    return csr_matrix(
        (np.asarray(vals, dtype=np.int64), (rows,cols)),
        shape=(n_balls,n_balls), dtype=np.int64,
    )


@dataclass(frozen=True)
class ComponentFiltration:
    """Exact fixed-landmark H0 merge events and component counts at given radii."""
    radii: np.ndarray
    component_counts: np.ndarray
    # Each row: [merge radius, landmark i, landmark j], sorted by event radius.
    merges: tuple[tuple[float,int,int], ...]


def fixed_landmark_h0(
    x: np.ndarray, landmarks: Sequence[int], radii: Sequence[float],
    *, max_radius: float | None = None, block_size: int = 4096,
) -> ComponentFiltration:
    """Exact H0 of fixed-landmark Ball Mapper graphs at multiple radii.

    Uses nearest-cover-center stars, not the complete overlap graph. Enumerates
    only membership pairs up to max_radius; does not build intermediate graphs.
    Landmark indices must stay the same at every radius.
    """
    rr = np.asarray(radii, dtype=np.float64)
    if rr.ndim != 1 or not np.all(np.isfinite(rr)) or np.any(rr < 0):
        raise ValueError('radii must be a one-dimensional finite nonnegative array')
    limit = float(np.max(rr)) if len(rr) else 0.0
    if max_radius is not None:
        if not np.isfinite(max_radius) or max_radius < limit:
            raise ValueError('max_radius must be finite and at least max(radii)')
        limit = float(max_radius)
    points = _points(x, limit)
    lm = _indices(landmarks, len(points), unique=True)
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, (int, np.integer))
        or block_size < 1
    ):
        raise ValueError('block_size must be a positive integer')
    centers = points[lm]
    # Fixed-radius memberships may be obtained from an exact range index in a
    # subsequent backend specialization. The blocked oracle is memory-bounded.
    # Accumulate nearest owner and candidate distances one observation block at a time.
    edges: dict[tuple[int,int], float] = {}
    m = len(lm)
    for start in range(0, len(points), block_size):
        data = points[start:start+block_size]
        # Bound the distance workspace by block_size*m. For a huge landmark
        # set callers may select a smaller block_size.
        for row in data:
            d = np.linalg.norm(centers - row, axis=1)
            eligible = np.flatnonzero(d <= limit)
            if len(eligible) < 2:
                continue
            # argmin on all centers is exact relative to our float64 predicate.
            owner = int(np.argmin(d))
            for j in eligible:
                if j == owner:
                    continue
                u,v = (owner,int(j)) if owner < j else (int(j),owner)
                key = (u,v)
                val = float(d[j])
                if key not in edges or val < edges[key]:
                    edges[key] = val
    events = sorted(
        ((weight,u,v) for (u,v), weight in edges.items()),
        key=lambda a:(a[0],a[1],a[2]),
    )
    uf = _UnionFind(m)
    merges: list[tuple[float,int,int]] = []
    counts = np.empty(len(rr), dtype=np.intp)
    order = np.argsort(rr, kind='stable')
    pos = 0
    for k in order:
        while pos < len(events) and events[pos][0] <= rr[k]:
            r,i,j = events[pos]
            if uf.union(i,j):
                merges.append((r,i,j))
            pos += 1
        counts[k] = uf.n_components
    return ComponentFiltration(rr, counts, tuple(merges))


def build_mapper_csr(
    cover: Sequence[Sequence[int] | np.ndarray], n_samples: int, *,
    row_block_size: int = 256,
) -> csr_matrix:
    """Exact symmetric witness-count adjacency in CSR, without NetworkX.

    Sparse blocked A @ A.T; bounded temporary output to row_block_size*m.
    Diagonal is zero, every off-diagonal value is an integer witness count.
    """
    from scipy.sparse import csr_matrix as csr
    if (isinstance(n_samples, bool) or not isinstance(n_samples, (int,np.integer))
            or n_samples < 0):
        raise ValueError('n_samples must be a nonnegative integer')
    if (isinstance(row_block_size, bool)
            or not isinstance(row_block_size, (int,np.integer))
            or row_block_size < 1):
        raise ValueError('row_block_size must be positive')
    m = len(cover)
    rows = [_indices(ids, n_samples) for ids in cover]
    rows = [np.unique(row) for row in rows]
    lengths = np.fromiter((len(a) for a in rows), dtype=np.intp, count=m)
    indptr = np.empty(m+1, dtype=np.intp)
    indptr[0] = 0
    indptr[1:] = np.cumsum(lengths)
    indexes = np.concatenate(rows) if rows else np.empty(0,dtype=np.intp)
    A = csr((np.ones(len(indexes),dtype=np.int64),indexes,indptr),shape=(m,n_samples))
    AT = A.T.tocsr()
    result_rows = []
    result_cols = []
    result_vals = []
    for start in range(0, m, row_block_size):
        product = (A[start:start+row_block_size] @ AT).tocoo()
        r = product.row + start
        keep = r < product.col
        if np.any(keep):
            result_rows.append(r[keep])
            result_cols.append(product.col[keep])
            result_vals.append(product.data[keep])
    if not result_rows:
        return csr((m,m), dtype=np.int64)
    rr = np.concatenate(result_rows)
    cc = np.concatenate(result_cols)
    vv = np.concatenate(result_vals)
    return csr((np.concatenate((vv,vv)),
                (np.concatenate((rr,cc)),np.concatenate((cc,rr)))),
               shape=(m,m),dtype=np.int64)


def select_mapper_algorithm(
    cover: Sequence[Sequence[int] | np.ndarray], n_samples: int,
    *, stream_max_balls: int = 128, stream_max_memberships: int = 2048,
    stream_max_witnesses: int = 512, stream_witness_ratio: float = 4.0,
) -> Literal['csr', 'streaming']:
    """Choose a low-overhead exact graph builder from cover statistics.

    Python dictionary aggregation is effective on tiny, sparse covers;
    sparse SpGEMM generally wins as the cover or witness work grows.
    The thresholds are heuristic and callers can override the choice.
    """
    m = len(cover)
    if m > stream_max_balls:
        return 'csr'
    memberships = [np.unique(_indices(ids, n_samples)) for ids in cover]
    s = sum(len(a) for a in memberships)
    if s > stream_max_memberships:
        return 'csr'
    if not s:
        return 'streaming'
    degrees = np.bincount(np.concatenate(memberships), minlength=n_samples)
    t = sum(int(d) * (int(d) - 1) // 2 for d in degrees)
    return (
        'streaming'
        if t <= min(stream_max_witnesses, stream_witness_ratio * max(s, 1))
        else 'csr'
    )


def build_mapper_auto(
    cover: Sequence[Sequence[int] | np.ndarray], n_samples: int, *,
    output: Literal['csr','networkx'] = 'csr', row_block_size: int = 256,
    stream_witness_ratio: float = 4.0,
    algorithm: Literal['auto','csr','streaming'] = 'auto',
):
    """Build exact witness-count graph, preferring CSR for numerical work.

    Default output='csr' avoids NetworkX Python-object memory overhead.
    output='networkx' remains supported. 'auto' dispatches tiny sparse covers
    to streaming and larger covers to sparse SpGEMM. Pass algorithm='csr'
    or algorithm='streaming' to override the selection.
    The stream_witness_ratio option is retained as a secondary dispatch cap.
    """
    if not np.isfinite(stream_witness_ratio) or stream_witness_ratio < 0:
        raise ValueError('stream_witness_ratio must be finite and nonnegative')
    if algorithm not in ('auto', 'csr', 'streaming'):
        raise ValueError("algorithm must be 'auto', 'csr', or 'streaming'")
    if output not in ('csr', 'networkx'):
        raise ValueError("output must be 'csr' or 'networkx'")
    method = (
        select_mapper_algorithm(
            cover, n_samples, stream_witness_ratio=stream_witness_ratio
        )
        if algorithm == 'auto' else algorithm
    )
    if method == 'streaming':
        return build_mapper_streaming(
            iter(cover), n_samples, n_balls=len(cover), output=output
        )
    adj = build_mapper_csr(cover, n_samples, row_block_size=row_block_size)
    if output == 'csr':
        return adj
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(len(cover)))
    upper = adj.tocoo()
    graph.add_edges_from(
        (int(r), int(c), {'witness_count': int(w)})
        for r, c, w in zip(upper.row, upper.col, upper.data, strict=True)
        if r < c
    )
    return graph


def iter_cover_blocked(
    x: np.ndarray, landmarks: Sequence[int], eps: float, *, block_size: int = 16384,
):
    """Yield complete balls one at a time with bounded query workspace."""
    pts = _points(x,eps)
    lm = _indices(landmarks,len(pts),unique=True)
    if (
        isinstance(block_size,bool)
        or not isinstance(block_size,(int,np.integer))
        or block_size <= 0
    ):
        raise ValueError('block_size must be positive')
    for landmark in lm:
        members=[]
        for start in range(0,len(pts),block_size):
            block=pts[start:start+block_size]
            members.append(start+np.flatnonzero(_within(block,pts[landmark],eps)))
        yield np.concatenate(members) if members else np.empty(0,dtype=np.intp)


def compute_mapper_streaming(
    x: np.ndarray, eps: float, *, block_size: int = 16384,
    output: Literal['csr','networkx'] = 'csr',
):
    """Select greedy landmarks and build graph without retaining cover arrays.

    Exact Euclidean path. Suitable when peak memory matters more than repeated
    exhaustive distance computations; does not imply improved runtime.
    Returns (landmark indices, adjacency).
    """
    landmarks, _ = select_landmarks_uncovered(x,eps,block_size=block_size)
    return landmarks, build_mapper_streaming(
        iter_cover_blocked(x,landmarks,eps,block_size=block_size),
        len(x),n_balls=len(landmarks),output=output,
    )


def fixed_landmark_h0_from_cover(
    x: np.ndarray, landmarks: Sequence[int], radii: Sequence[float],
    cover: Sequence[Sequence[int] | np.ndarray],
) -> ComponentFiltration:
    """Fixed-landmark H0 filtration using only a complete maximum-radius cover.

    cover[j] must equal the exact ball at radius max(radii), using Euclidean
    float64 distance. Each membership is verified to reject false positives;
    completeness must come from the trusted caller/back-end (it cannot be
    certified from the returned candidates alone). With trustworthy complete
    memberships, additional distance evaluations are O(S) rather than O(n*m).
    """
    rr = np.asarray(radii, dtype=np.float64)
    if rr.ndim != 1 or not np.all(np.isfinite(rr)) or np.any(rr < 0):
        raise ValueError('radii must be a one-dimensional finite nonnegative array')
    eps = float(np.max(rr)) if len(rr) else 0.0
    pts = _points(x,eps)
    lm = _indices(landmarks,len(pts),unique=True)
    if len(cover)!=len(lm):
        raise ValueError('cover and landmark lengths differ')
    incidence: dict[int,list[tuple[float,int]]] = defaultdict(list)
    for j,ids in enumerate(cover):
        ii=np.unique(_indices(ids,len(pts)))
        if len(ii)==0:
            continue
        dd=np.linalg.norm(pts[ii]-pts[lm[j]],axis=1)
        if np.any(~np.isfinite(dd)) or np.any(dd>eps):
            raise ValueError('cover has membership outside max(radii)')
        for p,d in zip(ii,dd,strict=True):
            incidence[int(p)].append((float(d),j))
    edges: dict[tuple[int,int],float]={}
    for members in incidence.values():
        if len(members)<2:
            continue
        best= min(members)
        owner=best[1]
        for distance,j in members:
            if j==owner:
                continue
            a,b=sorted((owner,j))
            old=edges.get((a,b),np.inf)
            if distance<old:
                edges[(a,b)]=distance
    events=sorted(((r,a,b) for (a,b),r in edges.items()),key=lambda x:(x[0],x[1],x[2]))
    uf=_UnionFind(len(lm))
    counts=np.empty(len(rr),dtype=np.intp)
    merges=[]
    pos=0
    for k in np.argsort(rr,kind='stable'):
        while pos<len(events) and events[pos][0]<=rr[k]:
            event=events[pos]
            if uf.union(event[1],event[2]):
                merges.append(event)
            pos+=1
        counts[k]=uf.n_components
    return ComponentFiltration(rr,counts,tuple(merges))