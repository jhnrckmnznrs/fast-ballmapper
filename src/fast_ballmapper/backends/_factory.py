"""Backend construction and legacy-method compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from fast_ballmapper.backends._base import RangeQueryBackend
from fast_ballmapper.backends._ball_tree import BallTreeBackend
from fast_ballmapper.backends._brute_force import BruteForceBackend
from fast_ballmapper.backends._ckdtree import CKDTreeBackend
from fast_ballmapper.backends._configs import CuVSConfig, HnswlibConfig
from fast_ballmapper.backends._cuvs import CuVSBackend
from fast_ballmapper.backends._faiss import FaissRangeBackend
from fast_ballmapper.backends._hnswlib import HnswlibBackend
from fast_ballmapper.faiss import FaissConfig


BACKEND_ALIASES = {
    "balltree": "ball_tree",
    "brute": "brute_force",
    "bruteforce": "brute_force",
    "kd_tree": "ckdtree",
    "c_kd_tree": "ckdtree",
    "scipy_ckdtree": "ckdtree",
    "hnsw": "hnswlib",
    "cuvs_cagra": "cuvs",
    "cuvs_brute_force": "cuvs",
}


def normalize_backend_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("method must be a string.")
    key = name.lower().replace("-", "_")
    return BACKEND_ALIASES.get(key, key)


def make_backend(
    x: np.ndarray,
    method: str = "ball_tree",
    *,
    metric: str = "euclidean",
    leaf_size: int = 40,
    metric_kwargs: Mapping[str, Any] | None = None,
    faiss_config: FaissConfig | None = None,
    hnswlib_config: HnswlibConfig | None = None,
    cuvs_config: CuVSConfig | None = None,
) -> RangeQueryBackend:
    """Construct a fitted range-query backend.

    This is the preferred extension point for library users.  The high-level
    Ball Mapper functions also accept a pre-built backend object directly.
    """
    key = normalize_backend_name(method)
    metric_key = metric.lower()
    metric_kwargs = dict(metric_kwargs or {})

    if key == "ball_tree":
        if faiss_config or hnswlib_config or cuvs_config:
            raise ValueError(
                "Backend-specific configs do not match method='ball_tree'."
            )
        return BallTreeBackend(x, metric_key, leaf_size, metric_kwargs)

    if metric_kwargs:
        raise ValueError("metric_kwargs is supported only by method='ball_tree'.")
    if metric_key not in {"euclidean", "cosine"}:
        raise ValueError(
            f"For method={key!r}, metric must be 'euclidean' or 'cosine'."
        )

    if key == "brute_force":
        if faiss_config or hnswlib_config or cuvs_config:
            raise ValueError("Backend-specific configs do not match brute_force.")
        return BruteForceBackend(x, metric_key)  # type: ignore[arg-type]

    if key == "ckdtree":
        if metric_key != "euclidean":
            raise ValueError("method='ckdtree' currently supports only euclidean.")
        if faiss_config or hnswlib_config or cuvs_config:
            raise ValueError("Backend-specific configs do not match ckdtree.")
        return CKDTreeBackend(x, leaf_size=leaf_size)

    if key == "faiss":
        if hnswlib_config or cuvs_config:
            raise ValueError("Only faiss_config may be supplied with method='faiss'.")
        return FaissRangeBackend(x, metric_key, faiss_config)  # type: ignore[arg-type]

    if key == "hnswlib":
        if faiss_config or cuvs_config:
            raise ValueError("Only hnswlib_config may be used with method='hnswlib'.")
        return HnswlibBackend(x, metric_key, hnswlib_config)

    if key == "cuvs":
        if faiss_config or hnswlib_config:
            raise ValueError("Only cuvs_config may be used with method='cuvs'.")
        selected = cuvs_config
        original = method.lower().replace("-", "_")
        if selected is None and original == "cuvs_brute_force":
            selected = CuVSConfig(algorithm="brute_force", candidate_k=None)
        elif selected is None and original == "cuvs_cagra":
            selected = CuVSConfig(algorithm="cagra")
        return CuVSBackend(x, metric_key, selected)

    raise ValueError(
        "Unknown backend method. Expected one of: ball_tree, brute_force, "
        "ckdtree, faiss, hnswlib, cuvs, cuvs_brute_force, or cuvs_cagra."
    )
