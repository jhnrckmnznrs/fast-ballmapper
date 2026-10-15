"""Exact per-ball deletion robustness using geometric nested dissection and budget DP.

For each landmark i, its color y_i is 0/1. Each observation-pattern T
chooses a side z, and deleting discordant incidences costs one unit of the
corresponding ball budget. The dynamic program carries spent budgets on
landmarks active in elimination separators and a two-bit used-color mask.

The solver reports INCONCLUSIVE if a user-specified state/time cap is hit.
It never translates a search cutoff into a robustness certificate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from time import perf_counter
from typing import Any
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Certificate:
    robust: bool | None
    status: str
    deletions: tuple[tuple[int, int], ...]
    elapsed: float
    width: int
    states_examined: int
    peak_states: int
    patterns: int
    m: int
    method: str = 'geometric-separator-budget-dp'
    split: tuple[int, ...] | None = None


class Capped(Exception):
    pass


def _prepare(cover, n_samples, budgets):
    if not isinstance(n_samples, int) or n_samples < 0:
        raise ValueError('invalid sample count')
    m = len(cover)
    if len(budgets) != m or any(
        isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer)) or b < 0
        for b in budgets
    ):
        raise ValueError('invalid ball budgets')
    by_sample = defaultdict(list)
    for i, row in enumerate(cover):
        for x in set(row):
            if (isinstance(x, (bool, np.bool_))
                    or not isinstance(x, (int, np.integer))
                    or not 0 <= x < n_samples):
                raise ValueError('invalid observation id')
            by_sample[int(x)].append(i)
    by_pattern = defaultdict(list)
    for x, pat in by_sample.items():
        if len(pat) > 1:
            by_pattern[tuple(sorted(pat))].append(x)
    pats = {p:len(ids) for p,ids in by_pattern.items()}
    g = nx.Graph()
    g.add_nodes_from(range(m))
    for pat in pats:
        g.add_edges_from(combinations(pat, 2))
    return pats, by_pattern, g


def geometric_order(vertices, coords, eps, leaf=10):
    """Axis-aligned geometric separator nested dissection.

    All edges between opposite open halfspaces are longer than 2*eps.
    This is a valid *geometric* vertex separator for Euclidean overlap graphs.
    It is not claimed to be asymptotically optimal on every input.
    """
    if coords is None:
        raise ValueError('geometric ordering needs landmark coordinates')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] <= max(vertices, default=-1):
        raise ValueError('coordinates must cover all landmarks')
    if not np.all(np.isfinite(coords)):
        raise ValueError('coordinates must be finite')
    def split(vs):
        if len(vs) <= leaf:
            return sorted(vs)
        arr=np.asarray(vs, dtype=int)
        chosen=None
        for axis in np.argsort(-np.ptp(coords[arr], axis=0)):
            cuts=np.quantile(coords[arr,axis],[.35,.5,.65])
            for cut in cuts:
                left=[i for i in vs if coords[i,axis] < cut-eps]
                right=[i for i in vs if coords[i,axis] > cut+eps]
                sep=[i for i in vs if i not in set(left) and i not in set(right)]
                if left and right and max(len(left),len(right))<=.8*len(vs):
                    candidate=(len(sep),abs(len(left)-len(right)),left,right,sep)
                    if chosen is None or candidate[:2]<chosen[:2]:
                        chosen=candidate
        if chosen is None:
            return sorted(vs)
        _,_,left,right,sep=chosen
        return split(left)+split(right)+sorted(sep)
    return split(list(vertices))


def _min_fill_order(graph):
    adj={v:set(graph.neighbors(v)) for v in graph}
    order=[]
    while adj:
        v=min(
            adj,
            key=lambda x: (
                sum(1 for a, b in combinations(adj[x], 2) if b not in adj[a]),
                len(adj[x]), x,
            ),
        )
        around=adj.pop(v)
        for i in around:
            adj[i].discard(v)
            adj[i].update(around-{i})
        order.append(v)
    return order


def _decomposition(graph, order):
    pos={v:i for i,v in enumerate(order)}
    adj={v:set(graph.neighbors(v)) for v in graph}
    bags={}; parent={}; children=defaultdict(list)
    for v in order:
        neighborhood=tuple(sorted(adj[v], key=lambda u:pos[u]))
        bag=frozenset((v,)+neighborhood)
        bags[v]=bag
        parent[v]=neighborhood[0] if neighborhood else None
        for a,b in combinations(neighborhood,2):
            adj[a].add(b)
            adj[b].add(a)
        for a in neighborhood:
            adj[a].discard(v)
        del adj[v]
    for v,p in parent.items():
        if p is not None:
            children[p].append(v)
            if not (bags[v]-{v}) <= bags[p]:
                raise AssertionError('invalid elimination tree')
    return bags,parent,children


def _roots_connected_and_block(graph):
    return graph.number_of_nodes()>1 and nx.is_connected(graph)


def certify_separator_dp(
    cover, n_samples, budgets, *, coords=None, eps=None,
    ordering='geometry', max_states=1_000_000, time_limit=30.0,
    validate_witness=True,
):
    """Decide whether allowed missing ball memberships can disconnect the graph.

    Initially disconnected covers are not supported in this reference solver.
    Sets of 0/1 landmark labels and cost vectors are enumerated exactly.
    A valid attack is returned on robust=False; robust=True proves infeasibility.
    On resource limits, robust=None is returned instead.
    """
    start=perf_counter()
    patterns,by_pat,g=_prepare(cover,n_samples,budgets)
    m=g.number_of_nodes()
    if not m or m==1:
        return Certificate(
            True, 'at most one landmark', (), perf_counter()-start,
            0, 0, 0, len(patterns), m,
        )
    if not _roots_connected_and_block(g):
        raise ValueError('connected exact Ball Mapper graph required')
    if ordering=='geometry':
        if coords is None or eps is None:
            raise ValueError('need geometry for geometric ordering')
        coords=np.asarray(coords,dtype=float)
        if coords.shape[0] != m:
            raise ValueError('coords should have one row per ball')
        # Safe because every true overlap edge is at most 2eps.
        for i,j in g.edges:
            if np.linalg.norm(coords[i]-coords[j]) > 2*eps+1e-10*eps:
                raise ValueError(
                    'observed cover inconsistent with coordinate proximity'
                )
        order=geometric_order(range(m),coords,eps)
    elif ordering=='minfill':
        order=_min_fill_order(g)
    else:
        raise ValueError('ordering must be geometry or minfill')
    bags,parent,children=_decomposition(g,order)
    width=max(map(len,bags.values()))-1
    # Never enumerate 2**width assignments if they alone exceed the cap.
    if width >= 20 or 2**(width+1)>max_states:
        return Certificate(None,'width/state cap before enumeration',(),
                           perf_counter()-start,width,0,0,len(patterns),m)
    # Place each pattern at earliest-eliminated vertex among its landmark set.
    # Every such pattern is a clique of the graph, so that bag contains it.
    where=defaultdict(list)
    priority={v:i for i,v in enumerate(order)}
    for pat,w in patterns.items():
        node=min(pat,key=priority.__getitem__)
        assert set(pat)<=bags[node]
        where[node].append((pat,w))
    # For every edge whose witnesses exceed the summed budgets, its endpoints
    # must be assigned the same side. Enforce this locally in its edge bag.
    witnesses=Counter()
    for pat,w in patterns.items():
        for edge in combinations(pat,2):
            witnesses[edge]+=w
    protected={tuple(sorted(edge)) for edge,w in witnesses.items()
               if w>int(budgets[edge[0]])+int(budgets[edge[1]])}
    if protected:
        d=nx.Graph();d.add_nodes_from(range(m));d.add_edges_from(protected)
        if nx.is_connected(d):
            return Certificate(
                True, 'protected spanning graph', (), perf_counter()-start,
                width, 0, 0, len(patterns), m,
            )
    caps=[int(min(b,n_samples)) for b in budgets]
    examined=0;peak=0
    table={} # node -> {(separator labels, spent costs, color mask): trace}
    node_meta={}
    # trace: (previous_trace, child_node, child_state) or local pattern t
    try:
        for v in order:
            B=tuple(sorted(bags[v]))
            idx={u:j for j,u in enumerate(B)}
            S=tuple(sorted(bags[v]-{v}))
            Sidx=tuple(idx[u] for u in S)
            child_ids=sorted(children[v])
            links=[]
            for a,b in protected:
                if a in idx and b in idx:
                    links.append((idx[a],idx[b]))
            # A color mask records colors of forgotten vertices, not active ones.
            dp={}
            for y in product((0,1),repeat=len(B)):
                if 0 in idx and y[idx[0]] != 0:
                    continue  # global complement symmetry
                if any(y[i]!=y[j] for i,j in links):
                    continue
                key=(y,(0,)*len(B),0)
                dp[key]=None
            for child in child_ids:
                sep=tuple(sorted(bags[child]-{child}))
                sel=tuple(idx[u] for u in sep)
                # keyed by shared side assignment so joins avoid unrelated states
                child_buckets=defaultdict(list)
                for ck in table[child]:
                    child_buckets[ck[0]].append(ck)
                joined={}
                for (y,c,mask), trace in dp.items():
                    for ck in child_buckets.get(tuple(y[k] for k in sel),()):
                        cs=ck[1]
                        cost=list(c)
                        for j,old in zip(sel,cs):
                            cost[j]+=old
                        if any(cost[j]>caps[u] for j,u in enumerate(B)):
                            continue
                        state=(y,tuple(cost),mask|ck[2])
                        if state not in joined:
                            joined[state]=(trace,('child',child,ck))
                        examined+=1
                        if examined>max_states or perf_counter()-start>time_limit:
                            raise Capped
                dp=joined
                # Retain child messages for a constructive witness backtrace.
                if not dp:
                    break
            for pat,w in where[v]:
                if not dp: break
                positions=tuple(idx[u] for u in pat)
                next_dp={}
                for (y,c,mask),trace in dp.items():
                    # t of w identical observations assigned to color 1
                    lo=max([0]+[w-(caps[u]-c[j]) for u,j in zip(pat,positions) if y[j]])
                    hi=min([w]+[caps[u]-c[j] for u,j in zip(pat,positions) if not y[j]])
                    for t in range(lo,hi+1):
                        cost=list(c)
                        for u,j in zip(pat,positions):
                            cost[j]+=w-t if y[j] else t
                        if any(cost[j]>caps[u] for j,u in enumerate(B)):
                            continue
                        state=(y,tuple(cost),mask)
                        if state not in next_dp:
                            next_dp[state]=(trace,('pattern',pat,t))
                        examined+=1
                        if examined>max_states or perf_counter()-start>time_limit:
                            raise Capped
                dp=next_dp
            msg={}
            for (y,c,mask),trace in dp.items():
                if c[idx[v]]>caps[v]:continue
                key = (
                    tuple(y[j] for j in Sidx),
                    tuple(c[j] for j in Sidx),
                    mask | (1 << y[idx[v]]),
                )
                msg.setdefault(key, (trace, y[idx[v]], B))
            table[v]=msg
            node_meta[v]=(idx,)
            peak=max(peak,len(dp),len(msg))
            if examined>max_states or perf_counter()-start>time_limit:
                raise Capped
    except Capped:
        return Certificate(
            None, 'state/time cap', (), perf_counter()-start,
            width, examined, peak, len(patterns), m,
        )
    roots=[v for v in order if parent[v] is None]
    assert len(roots)==1
    root=roots[0]
    possible=[key for key in table[root] if key[2]==3]
    if not possible:
        return Certificate(
            True, 'DP proved no feasible split', (), perf_counter()-start,
            width, examined, peak, len(patterns), m,
        )
    # Each state retains the choices used to construct it. Unwind the chosen
    # message recursively; witness-pattern decisions specify how many identical
    # observations retain side 1 and therefore yield concrete incidence edits.
    y_assignment={}
    choices={}
    def reconstruct(node, state):
        trace, yi, _bag = table[node][state]
        y_assignment[node]=yi
        while trace is not None:
            trace, action = trace
            if action[0] == 'child':
                reconstruct(action[1], action[2])
            else:
                choices[action[1]]=action[2]
    reconstruct(root,possible[0])
    assert len(y_assignment)==m and set(choices)==set(patterns)
    cuts=[]
    for pat,t in choices.items():
        obs=sorted(by_pat[pat])
        for xindex,obs_id in enumerate(obs):
            z=int(xindex<t)
            cuts.extend((i,obs_id) for i in pat if y_assignment[i]!=z)
    cuts=tuple(sorted(cuts))
    if validate_witness:
        edits=Counter(i for i,x in cuts)
        assert all(edits[i]<=caps[i] for i in range(m))
        changed=[set(row) for row in cover]
        for i,x in cuts:
            assert x in changed[i]
            changed[i].remove(x)
        g2=nx.Graph();g2.add_nodes_from(range(m))
        byx=defaultdict(list)
        for i,row in enumerate(changed):
            for x in row:byx[x].append(i)
        for members in byx.values():
            g2.add_edges_from(combinations(members,2))
        assert not nx.is_connected(g2), 'bad DP witness'
    return Certificate(
        False, 'DP feasible and witness verified', cuts, perf_counter()-start,
        width, examined, peak, len(patterns), m,
        split=tuple(sorted(i for i, y in y_assignment.items() if y)),
    )