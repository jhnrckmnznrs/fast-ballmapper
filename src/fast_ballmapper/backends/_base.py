"""Common range-query backend interface.

The public Ball Mapper algorithms depend only on this small protocol.  Concrete
backends may use exact spatial trees, exhaustive reference searches, or
approximate ANN indices, but they all expose the same closed-ball operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class BackendMetadata:
    """Machine-readable description of a range-query backend."""

    name: str
    is_exact: bool
    metric: str
    dtype: str
    device: str
    supports_batch_queries: bool = True
    supports_distances_to_all: bool = False
    notes: str | None = None


@runtime_checkable
class RangeQueryBackend(Protocol):
    """Protocol implemented by all Ball Mapper neighborhood backends.

    A backend is fitted to one point cloud at construction time.  Queries are
    expressed by point indices rather than arbitrary external vectors because
    Ball Mapper landmarks are always observations from the indexed dataset.
    """

    metadata: BackendMetadata

    @property
    def n_samples(self) -> int:
        """Number of indexed observations."""
        ...

    def query_radius(
        self,
        point_indices: Sequence[int],
        eps: float,
    ) -> list[np.ndarray]:
        """Return closed-ball memberships for the requested indexed points."""
        ...

    def distances_to_all(self, point_index: int) -> np.ndarray:
        """Return distances from one point to every indexed observation.

        Backends that cannot provide this operation should raise
        ``NotImplementedError`` and advertise
        ``metadata.supports_distances_to_all=False``.
        """
        ...
