"""Practical exact connectivity certification: block decomposition + DP/MILP.

A connected overlap graph decomposes into biconnected blocks. Bridges are
solved by exact witness counts; non-bridge blocks choose low-width DP if the
observed min-fill decomposition is sufficiently small, or exact MILP.
An inconclusive DP is retried with MILP and is never called a certificate.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import networkx as nx
import numpy as np

from fast_ballmapper.separator_dp import (
    _prepare, _min_fill_order, _decomposition, certify_separator_dp,
)
from fast_ballmapper.robustness import (
    certify_per_ball_budgets, check_disconnection_witness,
)


@dataclass
class HybridResult:
    robust: bool | None
    status: str
    deletions: tuple[tuple[int,int],...]
    elapsed: float
    blocks: int
    bridge_blocks: int
    dp_blocks: int
    milp_blocks: int
    dp_fallbacks: int
    max_width: int
    methods: tuple[str,...]


def certify_hybrid(cover,n_samples,budgets,*,width_cutoff=3,
                   max_states=300000,time_limit=15.,coords=None,eps=None):
    start=perf_counter()
    pats,by_pattern,g=_prepare(cover,n_samples,budgets)
    m=len(cover)
    if m<2 or not pats or all(b==0 for b in budgets):
        return HybridResult(True,'trivial',(),perf_counter()-start,0,0,0,0,0,0,())
    # Existing components cannot acquire new edges under conservative errors.
    # The global statement is robust iff every original component is robust.
    components=sorted(
        (tuple(sorted(c)) for c in nx.connected_components(g)),
        key=lambda x: x[0],
    )
    blocks=[]
    for cc in components:
        if len(cc)>1:
            blocks += [tuple(sorted(block)) for block in
                       nx.biconnected_components(g.subgraph(cc))]
    bridge_count=0;dps=0;milps=0;fallbacks=0;max_width=0;methods=[]
    unknown=False
    for b in blocks:
        vertex_set=set(b)
        relevant={pat:ids for pat,ids in by_pattern.items() if set(pat)<=vertex_set}
        # By clique containment, every size>=2 pattern lies in one block.
        assert all(set(p)<=vertex_set for p in relevant)
        remap={v:i for i,v in enumerate(b)}
        local=[sorted(x for pat,ids in relevant.items() if v in pat for x in ids)
               for v in b]
        bs=[int(budgets[v]) for v in b]
        if len(b)==2:
            bridge_count+=1
            w=sum(len(ids) for ids in relevant.values())
            if w>bs[0]+bs[1]:
                methods.append('bridge-protected')
                continue
            common=sorted(x for ids in relevant.values() for x in ids)
            k=min(bs[0],w)
            deletions=tuple(sorted([(b[0],x) for x in common[:k]]+
                                   [(b[1],x) for x in common[k:]]))
            assert check_disconnection_witness(cover,n_samples,budgets,deletions)
            return HybridResult(False,'breakable bridge',deletions,
                    perf_counter()-start,len(blocks),bridge_count,dps,milps,
                    fallbacks,max_width,tuple(methods)+('bridge-cut',))
        subg=g.subgraph(b).copy()
        bags,_,_=_decomposition(subg,_min_fill_order(subg))
        width=max(map(len,bags.values()))-1
        max_width=max(max_width,width)
        if width<=width_cutoff:
            dps+=1
            result=certify_separator_dp(local,n_samples,bs,
                      ordering='minfill',time_limit=time_limit,
                      max_states=max_states)
            methods.append('DP-width-'+str(width))
            if result.robust is None:
                fallbacks+=1
                result=certify_per_ball_budgets(local,n_samples,bs,
                            time_limit=time_limit)
                milps+=1
                methods.append('DP->MILP')
        else:
            milps+=1
            result=certify_per_ball_budgets(local,n_samples,bs,
                        time_limit=time_limit)
            methods.append('MILP-width-'+str(width))
        if result.robust is False:
            orig = (
                result.deletions if hasattr(result, 'deletions')
                else result.witness_deletions
            )
            deletions=tuple(sorted((b[i],int(x)) for i,x in orig))
            if not check_disconnection_witness(cover,n_samples,budgets,deletions):
                raise AssertionError('invalid hybrid attack witness')
            return HybridResult(False,'verified attack',deletions,
                    perf_counter()-start,len(blocks),bridge_count,dps,milps,
                    fallbacks,max_width,tuple(methods))
        if result.robust is None:
            unknown=True
    return HybridResult(None if unknown else True,
            'solver limit' if unknown else 'all blocks robust',(),
            perf_counter()-start,len(blocks),bridge_count,dps,milps,
            fallbacks,max_width,tuple(methods))