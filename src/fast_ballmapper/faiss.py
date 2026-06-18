"""Configuration objects for the optional FAISS backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

Number = int | float
FaissQueryMode = Literal["auto", "range", "knn"]
FaissDevice = Literal["cpu", "gpu", "auto"]


@dataclass(frozen=True)
class FaissConfig:
    """Configuration for constructing and querying a FAISS index.

    Parameters
    ----------
    factory:
        A FAISS index-factory string, such as ``"Flat"``,
        ``"IVF256,Flat"``, or ``"HNSW32"``.
    construction_params:
        Parameters applied before training and adding vectors, such as
        ``efConstruction``.
    search_params:
        Parameters applied before queries, such as ``nprobe`` or
        ``efSearch``.
    train_size:
        Maximum number of observations used to train an index. If None,
        all observations are used.
    train_seed:
        Seed used when selecting a training subset.
    query_mode:
        ``"range"`` requires native FAISS range search. ``"knn"`` uses
        a candidate-limited k-nearest-neighbour search. ``"auto"`` tries
        native range search and falls back to k-nearest-neighbour search
        when ``candidate_k`` is provided.
    candidate_k:
        Number of candidates requested when ``query_mode="knn"`` or when
        automatic fallback is required.
    exact_verify:
        Recompute exact distances for candidates and remove candidates
        outside the requested ball.
    """

    factory: str = "Flat"
    construction_params: Mapping[str, Number] = field(default_factory=dict)
    search_params: Mapping[str, Number] = field(default_factory=dict)
    train_size: int | None = None
    train_seed: int = 42
    query_mode: FaissQueryMode = "auto"
    candidate_k: int | None = None
    exact_verify: bool = False

    # Device settings.
    device: FaissDevice = "cpu"
    gpu_device: int = 0
    gpu_fallback_to_cpu: bool = True
    gpu_use_float16: bool = False
    gpu_temp_memory: int | None = None
