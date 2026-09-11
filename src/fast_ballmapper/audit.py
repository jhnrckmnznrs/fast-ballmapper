"""Auditing tools for exact and approximate Ball Mapper covers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fast_ballmapper.diagnostics import (
    BoundaryDiagnostics,
    compute_boundary_diagnostics,
)
from fast_ballmapper.graph import build_mapper

AuditMetric = Literal["euclidean", "cosine"]


@dataclass(frozen=True)
class BallMembershipAudit:
    """Per-ball membership agreement between two covers."""

    reference_sizes: np.ndarray
    approximate_sizes: np.ndarray
    intersection_sizes: np.ndarray
    false_positive_counts: np.ndarray
    false_negative_counts: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    jaccard: np.ndarray


@dataclass(frozen=True)
class EdgeWitnessChange:
    """Change in witness multiplicity for one Ball Mapper edge."""

    reference_count: int
    approximate_count: int
    retained_witnesses: int
    lost_witnesses: int
    added_witnesses: int
    change: int
    retained_fraction: float


@dataclass(frozen=True)
class ColorAudit:
    """Mean-color discrepancy and its cover-membership error bound."""

    reference_colors: np.ndarray
    approximate_colors: np.ndarray
    absolute_error: np.ndarray
    local_oscillation: np.ndarray
    theoretical_bound: np.ndarray
    mean_absolute_error: float
    root_mean_square_error: float
    max_absolute_error: float


@dataclass(frozen=True)
class ApproximationAudit:
    """Complete comparison of a reference and approximate fixed-landmark cover."""

    balls: BallMembershipAudit
    membership_true_positives: int
    membership_false_positives: int
    membership_false_negatives: int
    membership_precision: float
    membership_recall: float
    membership_jaccard: float
    macro_precision: float
    macro_recall: float
    macro_jaccard: float
    reference_edge_count: int
    approximate_edge_count: int
    edge_true_positives: int
    edge_false_positives: int
    edge_false_negatives: int
    edge_precision: float
    edge_recall: float
    edge_jaccard: float
    preserved_edges: tuple[tuple[int, int], ...]
    missing_edges: tuple[tuple[int, int], ...]
    added_edges: tuple[tuple[int, int], ...]
    witness_changes: dict[tuple[int, int], EdgeWitnessChange]
    boundary: BoundaryDiagnostics | None = None
    colors: ColorAudit | None = None


def compare_covers(
    reference_cover,
    approximate_cover,
    *,
    x: np.ndarray | None = None,
    landmarks=None,
    eps: float | None = None,
    metric: AuditMetric = "euclidean",
    deltas=None,
    values: np.ndarray | None = None,
) -> ApproximationAudit:
    """Compare an approximate cover against a fixed-landmark reference cover.

    Parameters
    ----------
    reference_cover, approximate_cover:
        Cover sequences whose entries correspond to the same landmarks in the
        same order. Duplicate point indices inside one cover element are
        ignored throughout the audit.
    x, landmarks, eps:
        Optional geometry used to attach :class:`BoundaryDiagnostics`. Either
        provide all three or none of them.
    metric, deltas:
        Passed to :func:`compute_boundary_diagnostics` when geometry is given.
    values:
        Optional scalar value per observation. When supplied, the report
        compares mean node colors and evaluates the sharp finite-set bound
        ``oscillation * (1 - min(precision, recall))`` for every ball.

    Notes
    -----
    The function assumes a *fixed landmark set*. If an approximate landmark
    selection procedure returns different landmarks, align or fix the
    landmarks first and then call :func:`compare_covers` on their covers.
    """
    if len(reference_cover) != len(approximate_cover):
        raise ValueError(
            "reference_cover and approximate_cover must have the same number "
            "of fixed-landmark cover elements."
        )

    reference_sets = [_as_index_set(indices) for indices in reference_cover]
    approximate_sets = [_as_index_set(indices) for indices in approximate_cover]

    ball_audit = _audit_ball_memberships(reference_sets, approximate_sets)

    tp = int(np.sum(ball_audit.intersection_sizes))
    fp = int(np.sum(ball_audit.false_positive_counts))
    fn = int(np.sum(ball_audit.false_negative_counts))
    membership_precision = _safe_ratio(tp, tp + fp)
    membership_recall = _safe_ratio(tp, tp + fn)
    membership_jaccard = _safe_ratio(tp, tp + fp + fn)

    reference_graph = build_mapper(reference_cover)
    approximate_graph = build_mapper(approximate_cover)
    reference_edges = {
        _edge_key(left, right) for left, right in reference_graph.edges()
    }
    approximate_edges = {
        _edge_key(left, right) for left, right in approximate_graph.edges()
    }

    preserved_edges = tuple(sorted(reference_edges & approximate_edges))
    missing_edges = tuple(sorted(reference_edges - approximate_edges))
    added_edges = tuple(sorted(approximate_edges - reference_edges))

    edge_tp = len(preserved_edges)
    edge_fp = len(added_edges)
    edge_fn = len(missing_edges)
    edge_precision = _safe_ratio(edge_tp, edge_tp + edge_fp)
    edge_recall = _safe_ratio(edge_tp, edge_tp + edge_fn)
    edge_jaccard = _safe_ratio(edge_tp, edge_tp + edge_fp + edge_fn)

    witness_changes: dict[tuple[int, int], EdgeWitnessChange] = {}
    for edge in sorted(reference_edges | approximate_edges):
        left, right = edge
        reference_witnesses = reference_sets[left] & reference_sets[right]
        approximate_witnesses = approximate_sets[left] & approximate_sets[right]
        retained_witnesses = len(reference_witnesses & approximate_witnesses)
        lost_witnesses = len(reference_witnesses - approximate_witnesses)
        added_witnesses = len(approximate_witnesses - reference_witnesses)
        reference_count = len(reference_witnesses)
        approximate_count = len(approximate_witnesses)
        witness_changes[edge] = EdgeWitnessChange(
            reference_count=reference_count,
            approximate_count=approximate_count,
            retained_witnesses=retained_witnesses,
            lost_witnesses=lost_witnesses,
            added_witnesses=added_witnesses,
            change=approximate_count - reference_count,
            retained_fraction=(
                retained_witnesses / reference_count
                if reference_count > 0
                else float("nan")
            ),
        )

    boundary = _optional_boundary_diagnostics(
        x=x,
        landmarks=landmarks,
        eps=eps,
        metric=metric,
        deltas=deltas,
    )
    colors = (
        _audit_mean_colors(reference_sets, approximate_sets, values)
        if values is not None
        else None
    )

    return ApproximationAudit(
        balls=ball_audit,
        membership_true_positives=tp,
        membership_false_positives=fp,
        membership_false_negatives=fn,
        membership_precision=membership_precision,
        membership_recall=membership_recall,
        membership_jaccard=membership_jaccard,
        macro_precision=(
            float(np.mean(ball_audit.precision)) if len(reference_sets) else 1.0
        ),
        macro_recall=float(np.mean(ball_audit.recall)) if len(reference_sets) else 1.0,
        macro_jaccard=(
            float(np.mean(ball_audit.jaccard)) if len(reference_sets) else 1.0
        ),
        reference_edge_count=len(reference_edges),
        approximate_edge_count=len(approximate_edges),
        edge_true_positives=edge_tp,
        edge_false_positives=edge_fp,
        edge_false_negatives=edge_fn,
        edge_precision=edge_precision,
        edge_recall=edge_recall,
        edge_jaccard=edge_jaccard,
        preserved_edges=preserved_edges,
        missing_edges=missing_edges,
        added_edges=added_edges,
        witness_changes=witness_changes,
        boundary=boundary,
        colors=colors,
    )


def _audit_ball_memberships(
    reference_sets: list[set[int]],
    approximate_sets: list[set[int]],
) -> BallMembershipAudit:
    n_balls = len(reference_sets)
    reference_sizes = np.empty(n_balls, dtype=np.intp)
    approximate_sizes = np.empty(n_balls, dtype=np.intp)
    intersection_sizes = np.empty(n_balls, dtype=np.intp)
    false_positive_counts = np.empty(n_balls, dtype=np.intp)
    false_negative_counts = np.empty(n_balls, dtype=np.intp)
    precision = np.empty(n_balls, dtype=float)
    recall = np.empty(n_balls, dtype=float)
    jaccard = np.empty(n_balls, dtype=float)

    for index, (reference, approximate) in enumerate(
        zip(reference_sets, approximate_sets, strict=True)
    ):
        intersection = len(reference & approximate)
        fp = len(approximate - reference)
        fn = len(reference - approximate)
        reference_sizes[index] = len(reference)
        approximate_sizes[index] = len(approximate)
        intersection_sizes[index] = intersection
        false_positive_counts[index] = fp
        false_negative_counts[index] = fn
        precision[index] = _safe_ratio(intersection, intersection + fp)
        recall[index] = _safe_ratio(intersection, intersection + fn)
        jaccard[index] = _safe_ratio(intersection, intersection + fp + fn)

    return BallMembershipAudit(
        reference_sizes=reference_sizes,
        approximate_sizes=approximate_sizes,
        intersection_sizes=intersection_sizes,
        false_positive_counts=false_positive_counts,
        false_negative_counts=false_negative_counts,
        precision=precision,
        recall=recall,
        jaccard=jaccard,
    )


def _audit_mean_colors(
    reference_sets: list[set[int]],
    approximate_sets: list[set[int]],
    values: np.ndarray,
) -> ColorAudit:
    scalar_values = np.asarray(values)
    if scalar_values.ndim != 1:
        raise ValueError("values must be a one-dimensional array of scalar colors.")

    all_indices = (
        set().union(*reference_sets, *approximate_sets) if reference_sets else set()
    )
    if all_indices:
        minimum = min(all_indices)
        maximum = max(all_indices)
        if minimum < 0 or maximum >= len(scalar_values):
            raise IndexError(
                "Cover point indices must be valid indices into values when "
                "color auditing is requested."
            )

    n_balls = len(reference_sets)
    reference_colors = np.full(n_balls, np.nan, dtype=float)
    approximate_colors = np.full(n_balls, np.nan, dtype=float)
    absolute_error = np.full(n_balls, np.nan, dtype=float)
    local_oscillation = np.zeros(n_balls, dtype=float)
    theoretical_bound = np.zeros(n_balls, dtype=float)

    for index, (reference, approximate) in enumerate(
        zip(reference_sets, approximate_sets, strict=True)
    ):
        if reference:
            reference_colors[index] = float(
                np.mean(scalar_values[np.fromiter(sorted(reference), dtype=np.intp)])
            )
        if approximate:
            approximate_colors[index] = float(
                np.mean(scalar_values[np.fromiter(sorted(approximate), dtype=np.intp)])
            )
        if reference and approximate:
            absolute_error[index] = abs(
                reference_colors[index] - approximate_colors[index]
            )

        union = reference | approximate
        if union:
            union_indices = np.fromiter(sorted(union), dtype=np.intp)
            union_values = scalar_values[union_indices].astype(float, copy=False)
            local_oscillation[index] = float(
                np.max(union_values) - np.min(union_values)
            )

        intersection = len(reference & approximate)
        precision = _safe_ratio(intersection, len(approximate))
        recall = _safe_ratio(intersection, len(reference))
        theoretical_bound[index] = local_oscillation[index] * (
            1.0 - min(precision, recall)
        )

    finite_errors = absolute_error[np.isfinite(absolute_error)]
    if finite_errors.size:
        mae = float(np.mean(finite_errors))
        rmse = float(np.sqrt(np.mean(np.square(finite_errors))))
        max_error = float(np.max(finite_errors))
    else:
        mae = rmse = max_error = float("nan")

    return ColorAudit(
        reference_colors=reference_colors,
        approximate_colors=approximate_colors,
        absolute_error=absolute_error,
        local_oscillation=local_oscillation,
        theoretical_bound=theoretical_bound,
        mean_absolute_error=mae,
        root_mean_square_error=rmse,
        max_absolute_error=max_error,
    )


def _optional_boundary_diagnostics(
    *,
    x: np.ndarray | None,
    landmarks,
    eps: float | None,
    metric: AuditMetric,
    deltas,
) -> BoundaryDiagnostics | None:
    provided = (x is not None, landmarks is not None, eps is not None)
    if any(provided) and not all(provided):
        raise ValueError(
            "x, landmarks, and eps must either all be supplied or all be omitted."
        )
    if not all(provided):
        return None
    return compute_boundary_diagnostics(
        x,
        landmarks,
        float(eps),
        metric=metric,
        deltas=deltas,
    )


def _as_index_set(indices) -> set[int]:
    result: set[int] = set()
    for index in indices:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Cover point indices must be integers.")
        value = int(index)
        if value < 0:
            raise IndexError("Cover point indices must be non-negative.")
        result.add(value)
    return result


def _edge_key(left: int, right: int) -> tuple[int, int]:
    left = int(left)
    right = int(right)
    return (left, right) if left < right else (right, left)


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return a score with the standard empty-set agreement convention."""
    return float(numerator / denominator) if denominator else 1.0
