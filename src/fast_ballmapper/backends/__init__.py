"""Range-query backends for :mod:`fast_ballmapper`."""

from fast_ballmapper.backends._base import BackendMetadata, RangeQueryBackend
from fast_ballmapper.backends._ball_tree import BallTreeBackend
from fast_ballmapper.backends._brute_force import BruteForceBackend
from fast_ballmapper.backends._ckdtree import CKDTreeBackend
from fast_ballmapper.backends._configs import CuVSConfig, HnswlibConfig
from fast_ballmapper.backends._cuvs import CuVSBackend
from fast_ballmapper.backends._factory import make_backend
from fast_ballmapper.backends._faiss import (
    FaissFlatBackend,
    FaissHNSWBackend,
    FaissIVFBackend,
    FaissRangeBackend,
)
from fast_ballmapper.backends._hnswlib import HnswlibBackend

__all__ = [
    "BackendMetadata",
    "RangeQueryBackend",
    "BallTreeBackend",
    "BruteForceBackend",
    "CKDTreeBackend",
    "FaissRangeBackend",
    "FaissFlatBackend",
    "FaissIVFBackend",
    "FaissHNSWBackend",
    "HnswlibBackend",
    "HnswlibConfig",
    "CuVSBackend",
    "CuVSConfig",
    "make_backend",
]
