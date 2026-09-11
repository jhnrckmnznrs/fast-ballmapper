"""Floating-point helpers for exact closed-ball membership.

Ball Mapper uses closed metric balls, ``d(x, l) <= eps``. Some range-query
APIs implement a strict threshold internally. For those APIs we query at the
smallest representable radius above ``eps`` and retain candidates whose
reported distance is strictly below that expanded threshold. This is
equivalent to an inclusive comparison at ``eps`` for finite floating-point
distances while avoiding an arbitrary numerical tolerance.
"""

from __future__ import annotations

import numpy as np


def closed_ball_radius(eps: float) -> float:
    """Return the smallest float64 radius strictly larger than ``eps``."""
    return float(np.nextafter(np.float64(eps), np.float64(np.inf)))


def faiss_l2_radius(eps: float) -> np.float32:
    """Return an outward-rounded squared-L2 threshold for FAISS.

    FAISS Euclidean indexes report squared L2 distances and use a strict
    ``distance < radius`` range predicate.  Advancing the float32 threshold by
    one representable value makes points whose reported squared distance equals
    the rounded value of ``eps**2`` eligible.
    """
    squared = np.float32(float(eps) * float(eps))
    return np.nextafter(squared, np.float32(np.inf))


def faiss_cosine_similarity_threshold(eps: float) -> np.float32:
    """Return an outward-rounded similarity threshold for cosine distance.

    ``d_cos(x, y) <= eps`` is equivalent to ``similarity >= 1 - eps`` for the
    normalized vectors used by FAISS.  FAISS inner-product range search uses a
    strict lower threshold, so we move it one float32 value toward ``-inf``.
    """
    similarity = np.float32(1.0 - float(eps))
    return np.nextafter(similarity, np.float32(-np.inf))
