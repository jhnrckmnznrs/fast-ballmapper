"""Controlled, seeded exact-certificate comparison. Inputs excluded from timings."""
from __future__ import annotations
import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np

from fast_ballmapper.budget_certificates import certify_connectivity
from fast_ballmapper.robustness import check_disconnection_witness

def generate(shape, *, eps=1.0, spacing=1.4, seed=0, witness_reps=1):
    """Exact separated Euclidean grid cover with pair-midpoint witnesses."""
    rng = np.random.default_rng(seed)
    d = len(shape)
    centers = np.array(list(np.ndindex(*shape)), dtype=float) * spacing
    centers += rng.normal(0, 0.005, size=centers.shape)
    tree = cKDTree(centers)
    pairs = sorted(tree.query_pairs(2 * eps * 0.97))
    observations = [centers]
    if pairs:
        midpoints = np.array([
            (centers[i] + centers[j]) / 2 + rng.normal(0, .003, d)
            for i, j in pairs for _ in range(witness_reps)
        ])
        observations.append(midpoints)
    observations.append(centers + rng.normal(0, .08, size=centers.shape))
    points = np.vstack(observations)
    if any(np.linalg.norm(centers[i] - centers[j]) <= eps
           for i, j in tree.query_pairs(2 * eps)):
        raise AssertionError('landmarks not epsilon-separated')
    patterns = tree.query_ball_point(points, r=eps)
    if not all(patterns):
        raise AssertionError('all observations must have at least one owner')
    cover = [[] for _ in range(len(centers))]
    for x, pat in enumerate(patterns):
        for i in pat:
            cover[i].append(x)
    return points, centers, cover


CASES = [
    ('path-120',(120,),1,1), ('path-240',(240,),1,1),
    ('strip-2x12',(12,2),1,1), ('strip-2x30',(30,2),1,1),
    ('strip-2x60',(60,2),1,1),
    ('grid-4x4',(4,4),1,1), ('grid-6x6',(6,6),1,1),
    ('grid-8x8',(8,8),1,1),
    ('grid-6x6-protected',(6,6),1,3),
    ('grid-4x4-double',(4,4),2,2),
    ('volume-3x3x3',(3,3,3),1,1),
    ('volume-4x4x3',(4,4,3),1,1),
]


def measure(cover,n,b,method,repeat,timeout):
    durations=[];statuses=[];rs=[]
    for _ in range(repeat):
        t=time.perf_counter()
        r=certify_connectivity(cover,b,n_samples=n,method=method,
                              width_cutoff=3,max_states=300000,
                              time_limit=timeout)
        durations.append(time.perf_counter()-t)
        statuses.append(r.status)
        rs.append(r.robust)
        if r.robust is False:
            assert check_disconnection_witness(cover,n,b,r.deletions)
    assert len(set(rs)) == 1, (method,rs)
    return {'result':rs[-1],'status':statuses[-1],
            'wall_seconds_median':round(statistics.median(durations),6),
            'wall_seconds_all':[round(v,6) for v in durations],
            'blocks':r.blocks,'max_width':r.max_width}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--seeds',type=int,default=2)
    p.add_argument('--timeout',type=float,default=3.)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',type=Path,default=Path('results.json'))
    a=p.parse_args()
    if a.repeats < 1 or a.seeds < 1 or a.timeout <= 0:
        p.error('repeats, seeds, and timeout must be positive')
    cases=CASES[:5] if a.quick else CASES
    out=[]
    for label,shape,budget,reps in cases:
        for seed in range(a.seeds):
            points,centers,cover=generate(shape,seed=20+seed,witness_reps=reps)
            n=len(points);b=[budget]*len(cover)
            row={'case':label,'seed':seed,'dimensions':len(shape),
                 'n':n,'m':len(cover),'S':sum(len(s) for s in cover),
                 'budget':budget,'repeated_witnesses':reps}
            for method in ('auto','milp'):
                row[method]=measure(cover,n,b,method,a.repeats,a.timeout)
            if (row['auto']['result'] is not None and row['milp']['result'] is not None
                    and row['auto']['result'] != row['milp']['result']):
                raise AssertionError(row)
            out.append(row)
            print(json.dumps(row),flush=True)
            a.output.parent.mkdir(parents=True,exist_ok=True)
            a.output.write_text(json.dumps({'environment':{'python':sys.version,
                'cpu_count':os.cpu_count(),'single_thread':True},'rows':out},indent=2)+'\n')
    with a.output.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case','seed','dimensions','n','m','S','budget',
            'repeated_witnesses','auto_result','milp_result','auto_seconds','milp_seconds',
            'auto_status','milp_status','blocks','max_width'])
        w.writeheader()
        for row in out:
            w.writerow({**{k:row[k] for k in ('case','seed','dimensions','n','m','S',
                             'budget','repeated_witnesses')},
                        'auto_result':row['auto']['result'],'milp_result':row['milp']['result'],
                        'auto_seconds':row['auto']['wall_seconds_median'],
                        'milp_seconds':row['milp']['wall_seconds_median'],
                        'auto_status':row['auto']['status'],
                        'milp_status':row['milp']['status'],
                        'blocks':row['auto']['blocks'],'max_width':row['auto']['max_width']})

if __name__=='__main__':main()