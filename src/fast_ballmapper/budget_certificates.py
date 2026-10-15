"""Sharp fixed-cover connectivity robustness under independent deletion budgets.

A robust result means no allowed *membership omissions* can increase the number
of connected components. False means a verifiable destructive attack exists;
None means the configured exact solver could not finish. Landmarks are fixed.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fast_ballmapper.hybrid import certify_hybrid
from fast_ballmapper.robustness import (
    certify_per_ball_budgets,
    check_disconnection_witness,
)
from fast_ballmapper.separator_dp import certify_separator_dp


@dataclass(frozen=True)
class ConnectivityCertificate:
    robust: bool | None
    status: str
    method: str
    deletions: tuple[tuple[int, int], ...]
    elapsed_seconds: float
    blocks: int = 0
    max_width: int = 0


def certify_connectivity(
    exact_cover: Sequence[Sequence[int]],
    budgets: Sequence[int],
    *,
    n_samples: int | None = None,
    method: str = "auto",
    width_cutoff: int = 3,
    max_states: int = 300_000,
    time_limit: float = 15.0,
) -> ConnectivityCertificate:
    """Attempt an exact, sharp certificate for a fixed complete cover.

    Uses block decomposition and a measured-width dispatch for method='auto'.
    method='milp' uses pattern-compressed integer optimization; method='dp'
    requires a connected original graph and tries min-fill DP alone.
    Every returned deletion witness is checked against the complete input.
    SciPy MILP uses floating point internally, so its infeasibility status is
    a solver certificate rather than a machine-verified integer proof.
    """
    if method not in {"auto", "milp", "dp"}:
        raise ValueError("method must be 'auto', 'milp', or 'dp'")
    if n_samples is None:
        n_samples = max((int(j) for row in exact_cover for j in row), default=-1) + 1
    if isinstance(n_samples, (bool, np.bool_)) or not isinstance(
        n_samples, (int, np.integer)
    ) or n_samples < 0:
        raise ValueError("n_samples must be a nonnegative integer")
    if len(budgets) != len(exact_cover) or any(
        isinstance(b, (bool, np.bool_))
        or not isinstance(b, (int, np.integer))
        or int(b) < 0
        for b in budgets
    ):
        raise ValueError("one nonnegative integer budget is required per ball")
    if not isinstance(max_states, int) or max_states < 1:
        raise ValueError("max_states must be positive")
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be positive and finite")
    if not isinstance(width_cutoff, int) or width_cutoff < 0:
        raise ValueError("width_cutoff must be nonnegative")

    if method == "auto":
        r = certify_hybrid(
            exact_cover, int(n_samples), budgets,
            width_cutoff=width_cutoff, max_states=max_states,
            time_limit=time_limit,
        )
        deletions = r.deletions
        status = r.status
        elapsed = r.elapsed
        blocks = r.blocks
        width = r.max_width
    elif method == "milp":
        r = certify_per_ball_budgets(
            exact_cover, int(n_samples), budgets, time_limit=time_limit
        )
        deletions = r.witness_deletions
        status = r.status
        elapsed = r.elapsed_seconds
        blocks = 0
        width = 0
    else:
        r = certify_separator_dp(
            exact_cover, int(n_samples), budgets, ordering="minfill",
            max_states=max_states, time_limit=time_limit,
        )
        deletions = r.deletions
        status = r.status
        elapsed = r.elapsed
        blocks = 0
        width = r.width
    if r.robust is False and not check_disconnection_witness(
        exact_cover, int(n_samples), budgets, deletions
    ):
        raise RuntimeError("solver returned an invalid disconnection witness")
    return ConnectivityCertificate(
        robust=r.robust, status=status, method=method,
        deletions=tuple(deletions), elapsed_seconds=float(elapsed),
        blocks=blocks, max_width=width,
    )