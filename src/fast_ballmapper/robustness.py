"""Research prototype: exact per-ball deletion-budget connectivity certificates.

No geometry is required after a complete finite ball cover is supplied.
The solver uses SciPy/HiGHS MILP and assumes integer per-landmark budgets,
conservative omissions, and a fixed set of graph vertices.
"""

from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from collections.abc import Sequence
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def root(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def join(self, a, b):
        a = self.root(a)
        b = self.root(b)
        if a == b:
            return False
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1
        return True


@dataclass
class Result:
    robust: bool | None
    status: str
    witness_deletions: tuple[tuple[int, int], ...]
    landmark_partition: tuple[int, ...] | None
    elapsed_seconds: float
    n_patterns: int
    n_variables: int
    n_protected_groups: int
    protected_edges: int
    method: str


def _prepare(cover: Sequence[Sequence[int]], n_samples: int, budgets: Sequence[int]):
    m = len(cover)
    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, (int, np.integer))
        or n_samples < 0
        or len(budgets) != m
    ):
        raise ValueError("invalid n_samples or budgets")
    if any(
        isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer)) or b < 0
        for b in budgets
    ):
        raise ValueError("budgets must be nonnegative integers")
    inc = defaultdict(list)
    for i, row in enumerate(cover):
        entries = set()
        for raw in row:
            if (
                isinstance(raw, (bool, np.bool_))
                or not isinstance(raw, (int, np.integer))
                or not 0 <= int(raw) < n_samples
            ):
                raise ValueError("observation indices must be valid integers")
            entries.add(int(raw))
        for x in entries:
            inc[x].append(i)
    patterns = Counter(tuple(sorted(v)) for v in inc.values() if len(v) >= 2)
    base = DSU(m)
    for ids in patterns:
        for j in ids[1:]:
            base.join(ids[0], j)
    exact_components = defaultdict(list)
    for i in range(m):
        exact_components[base.root(i)].append(i)
    by_pattern = defaultdict(list)
    for x, ids in inc.items():
        if len(ids) >= 2:
            by_pattern[tuple(sorted(ids))].append(x)
    return patterns, inc, list(exact_components.values()), by_pattern


def _local_solver(
    patterns: Counter,
    idxs: list[int],
    budgets: list[int],
    by_pattern: dict,
    *,
    shortcut=True,
    time_limit=30.0,
    presolve=True,
    must_split: Sequence[int] | None = None,
):
    """Decide if some conservative omissions split an initially connected component."""
    start = perf_counter()
    if len(idxs) <= 1:
        return Result(True, "single landmark", (), None, 0, 0, 0, 1, 0, "trivial")
    remap = {j: k for k, j in enumerate(idxs)}
    pats = Counter(
        {
            tuple(remap[i] for i in ids): int(w)
            for ids, w in patterns.items()
            if ids[0] in remap
        }
    )
    # More cautious when potential patterns include outside-group IDs
    assert all(len(t) >= 2 and all(0 <= i < len(idxs) for i in t) for t in pats)
    m = len(idxs)
    targets = tuple(idxs if must_split is None else must_split)
    if len(targets) <= 1:
        return Result(
            True,
            "target already inseparable",
            (),
            None,
            perf_counter() - start,
            len(pats),
            0,
            m,
            0,
            "trivial",
        )
    if not set(targets).issubset(idxs):
        raise ValueError("target must lie in exact component")
    # Can isolate i by dropping all shared memberships at i.
    row_sizes = [sum(w for ids, w in pats.items() if i in ids) for i in range(m)]
    for i in range(m):
        if idxs[i] in targets and row_sizes[i] <= budgets[idxs[i]]:
            deletions = tuple(
                sorted(
                    (idxs[i], int(x))
                    for ids, obs in by_pattern.items()
                    if idxs[i] in ids
                    for x in obs
                )
            )
            return Result(
                False,
                "isolate landmark",
                deletions,
                (idxs[i],),
                perf_counter() - start,
                len(pats),
                0,
                m,
                0,
                "direct",
            )
    protected = DSU(m)
    protected_edges = 0
    if shortcut:
        # Count exact witnesses only when enumerating cliques is moderate.
        pairwork = sum(len(ids) * (len(ids) - 1) // 2 for ids, w in pats.items())
        if pairwork <= 200_000:
            witnesses = Counter()
            for ids, w in pats.items():
                for a, b in combinations(ids, 2):
                    witnesses[a, b] += w
            for (i, j), w in witnesses.items():
                if w > budgets[idxs[i]] + budgets[idxs[j]]:
                    protected.join(i, j)
                    protected_edges += 1
    rootmap = {
        r: k for k, r in enumerate(sorted(set(protected.root(i) for i in range(m))))
    }
    groups = [rootmap[protected.root(i)] for i in range(m)]
    ng = len(rootmap)
    target_groups = sorted(set(groups[remap[i]] for i in targets))
    if len(target_groups) == 1:
        return Result(
            True,
            "target protected",
            (),
            None,
            perf_counter() - start,
            len(pats),
            0,
            ng,
            protected_edges,
            "protected",
        )
    if ng == 1:
        return Result(
            True,
            "protected spanning graph",
            (),
            None,
            perf_counter() - start,
            len(pats),
            0,
            ng,
            protected_edges,
            "protected",
        )
    # Observations whose incident landmarks are already constrained to the
    # same side never need to be deleted and can be removed from this problem.
    relevant = [(p, w) for p, w in pats.items() if len({groups[i] for i in p}) > 1]
    if not relevant:
        raise AssertionError("inconsistent protected contraction")
    # Variables y_g (binary), t_p (integer 0..w) and d_ip (nonnegative).
    s = sum(len(p) for p, w in relevant)
    pcount = len(relevant)
    vcount = ng + pcount + s
    lo = np.zeros(vcount)
    hi = np.empty(vcount)
    hi[:ng] = 1
    hi[ng : ng + pcount] = [w for _, w in relevant]
    hi[ng + pcount :] = [w for p, w in relevant for _ in p]
    # Fix one target component to side zero; complement symmetry then
    # preserves feasibility of a target-splitting cut.
    anchor = target_groups[0]
    hi[anchor] = 0
    integrality = np.zeros(vcount, dtype=np.int32)
    integrality[: ng + pcount] = 1
    rows = []
    cols = []
    vals = []
    low = []
    high = []

    def put(entries, lb=-np.inf, ub=np.inf):
        r = len(low)
        for c, v in entries:
            rows.append(r)
            cols.append(int(c))
            vals.append(float(v))
        low.append(float(lb))
        high.append(float(ub))

    put([(i, 1) for i in target_groups if i != anchor], lb=1)
    budget_rows = [[] for i in range(m)]
    zpos = ng + pcount
    for pindex, (pat, w) in enumerate(relevant):
        ti = ng + pindex
        for i in pat:
            di = zpos
            zpos += 1
            budget_rows[i].append(di)
            # d >= t - w*y; d >= w*y - t
            put([(di, 1), (ti, -1), (groups[i], w)], lb=0)
            put([(di, 1), (ti, 1), (groups[i], -w)], lb=0)
    for i in range(m):
        if budget_rows[i]:
            put([(d, 1) for d in budget_rows[i]], ub=budgets[idxs[i]])
    assert zpos == vcount
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(len(low), vcount)).tocsr()
    objective = np.zeros(vcount)
    objective[ng + pcount :] = 1.0
    solution = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lo, hi),
        constraints=LinearConstraint(A, low, high),
        options={
            "time_limit": time_limit,
            "presolve": presolve,
            "mip_rel_gap": 0.0,
        },
    )
    elapsed = perf_counter() - start
    if solution.status == 2:
        return Result(
            True,
            "MILP infeasible: certified robust",
            (),
            None,
            elapsed,
            pcount,
            vcount,
            ng,
            protected_edges,
            "milp",
        )
    if solution.x is None or solution.status != 0:
        return Result(
            None,
            f"MILP status {solution.status}: {solution.message}",
            (),
            None,
            elapsed,
            pcount,
            vcount,
            ng,
            protected_edges,
            "milp",
        )
    y = [round(float(v)) for v in solution.x[:ng]]
    ts = [round(float(v)) for v in solution.x[ng : ng + pcount]]
    selected = tuple(idxs[i] for i, g in enumerate(groups) if y[g] == 1)
    # Return pattern-aggregate omission counts: (ball,pattern,number)
    cuts = []
    for (pat, w), t in zip(relevant, ts, strict=False):
        global_pat = tuple(idxs[i] for i in pat)
        samples = by_pattern[global_pat]
        assert len(samples) == w
        assigned_one = set(samples[:t])
        for i in pat:
            for x in samples:
                if y[groups[i]] != int(x in assigned_one):
                    cuts.append((idxs[i], int(x)))
    cuts.sort()
    return Result(
        False,
        "MILP witness found",
        tuple(cuts),
        selected,
        elapsed,
        pcount,
        vcount,
        ng,
        protected_edges,
        "milp",
    )


def certify_per_ball_budgets(
    cover, n_samples, budgets, *, shortcut=True, time_limit=30
):
    """Return exact robustness certification or an adversarial cut witness.

    Robust means no conservative deletion of at most budgets[i] memberships
    in each ball can increase the number of connected components (landmarks
    remain vertices even when a ball is emptied).
    An unknown status means MILP stopped without proof.
    For a graph initially disconnected, solve each component separately.
    """
    patterns, inc, components, by_pattern = _prepare(cover, n_samples, budgets)
    results = []
    for component in components:
        r = _local_solver(
            patterns,
            component,
            list(map(int, budgets)),
            by_pattern,
            shortcut=shortcut,
            time_limit=time_limit,
        )
        results.append(r)
    if not results:
        return Result(True, "empty", (), None, 0, 0, 0, 0, 0, "trivial")
    failed = next((r for r in results if r.robust is False), None)
    if failed is not None:
        if not check_disconnection_witness(
            cover, n_samples, budgets, failed.witness_deletions
        ):
            raise RuntimeError("optimizer witness failed independent validation")
        return failed
    unknown = next((r for r in results if r.robust is None), None)
    if unknown is not None:
        return unknown
    return Result(
        True,
        "all exact components robust",
        (),
        None,
        sum(r.elapsed_seconds for r in results),
        sum(r.n_patterns for r in results),
        sum(r.n_variables for r in results),
        sum(r.n_protected_groups for r in results),
        sum(r.protected_edges for r in results),
        "+".join(sorted(set(r.method for r in results))),
    )


@dataclass
class AtomResult:
    """Universal connectivity equivalence classes over all admissible deletions."""

    atoms: tuple[tuple[int, ...], ...]
    milp_calls: int
    status: str
    counterexample_cuts: tuple[tuple[tuple[int, int], ...], ...]


def robust_atoms(cover, n_samples, budgets, *, time_limit=20.0, shortcut=True):
    """Identify landmark pairs connected under every allowed deletion.

    Returns an exact universal partition if status='complete'. Each returned
    atom remains connected in every approximate graph obeying these budgets.
    The maximum number of graph components is at most len(atoms); this bound
    is not generally attained by one simultaneous deletion pattern.
    Returns incomplete/unknown rather than falsely claiming completion when
    an MILP times out.
    """
    patterns, inc, components, by_pattern = _prepare(cover, n_samples, budgets)
    queue = [tuple(c) for c in components]
    finished = []
    cuts = []
    calls = 0
    while queue:
        block = queue.pop()
        if len(block) <= 1:
            finished.append(block)
            continue
        comp = next(c for c in components if block[0] in c)
        candidate = _local_solver(
            patterns,
            comp,
            list(map(int, budgets)),
            by_pattern,
            shortcut=shortcut,
            time_limit=time_limit,
            must_split=block,
        )
        calls += 1
        if candidate.robust is None:
            return AtomResult(
                tuple(sorted(finished + queue + [block])),
                calls,
                "incomplete: MILP stopped without proof",
                tuple(cuts),
            )
        if candidate.robust:
            finished.append(block)
            continue
        side = set(candidate.landmark_partition)
        left = tuple(i for i in block if i not in side)
        right = tuple(i for i in block if i in side)
        assert left and right
        queue.extend((left, right))
        cuts.append(candidate.witness_deletions)
    return AtomResult(tuple(sorted(finished)), calls, "complete", tuple(cuts))


def check_disconnection_witness(cover, n_samples, budgets, deletions):
    """Independently verify that an explicit omission list is a valid attack."""
    original = [set(row) for row in cover]
    changed = [row.copy() for row in original]
    for i, x in deletions:
        if not 0 <= i < len(changed) or x not in changed[i]:
            return False
        changed[i].remove(x)
    if any(
        len(a - b) > int(limit)
        for a, b, limit in zip(original, changed, budgets, strict=True)
    ):
        return False

    def signature(rows):
        d = DSU(len(rows))
        by_point = defaultdict(list)
        for i, ids in enumerate(rows):
            for x in ids:
                if not 0 <= x < n_samples:
                    return None
                by_point[x].append(i)
        for ids in by_point.values():
            for j in ids[1:]:
                d.join(ids[0], j)
        return sum(d.root(i) == i for i in range(len(rows)))

    before = signature(original)
    after = signature(changed)
    return before is not None and after is not None and after > before
