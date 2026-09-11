"""Configuration objects for optional external range-query backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True)
class HnswlibConfig:
    """Configuration for the optional hnswlib backend."""

    m: int = 32
    ef_construction: int = 200
    ef_search: int = 128
    candidate_k: int = 512
    random_seed: int = 100
    num_threads: int = -1
    exact_verify: bool = True


@dataclass(frozen=True)
class CuVSConfig:
    """Configuration for NVIDIA cuVS backends.

    ``algorithm`` may be ``"brute_force"`` or ``"cagra"``.  cuVS exposes
    k-nearest-neighbour searches rather than a native Ball Mapper radius API in
    these Python interfaces, so ``candidate_k`` controls the candidate set.
    For ``brute_force`` a value of ``None`` requests all indexed points and is
    therefore exact relative to the float32 GPU representation.
    """

    algorithm: str = "cagra"
    candidate_k: int | None = 1024
    exact_verify: bool = True
    index_params: Mapping[str, Any] = field(default_factory=dict)
    search_params: Mapping[str, Any] = field(default_factory=dict)
