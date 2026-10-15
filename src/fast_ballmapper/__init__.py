"""Fast construction and analysis of Ball Mapper graphs."""

from importlib.metadata import PackageNotFoundError, version

from fast_ballmapper.backends import (
    BackendMetadata,
    BallTreeBackend,
    BruteForceBackend,
    CKDTreeBackend,
    CuVSBackend,
    CuVSConfig,
    FaissFlatBackend,
    FaissHNSWBackend,
    FaissIVFBackend,
    FaissRangeBackend,
    HnswlibBackend,
    HnswlibConfig,
    RangeQueryBackend,
    make_backend,
)
from fast_ballmapper.audit import (
    ApproximationAudit,
    BallMembershipAudit,
    ColorAudit,
    EdgeWitnessChange,
    compare_covers,
)
from fast_ballmapper.coloring import (
    color_by_density,
    color_by_entropy,
    color_by_function,
    color_by_mode,
    color_by_size,
)
from fast_ballmapper.diagnostics import (
    BoundaryDiagnostics,
    compute_boundary_diagnostics,
)
from fast_ballmapper.faiss import FaissConfig
from fast_ballmapper.graph import build_mapper, compute_edge_overlaps
from fast_ballmapper.landmarks import (
    build_cover,
    compute_landmarks,
    compute_landmarks_fps,
)
from fast_ballmapper.verified import (
    VerifiedSelection,
    build_cover_blocked,
    compute_landmarks_verified,
)
from fast_ballmapper.sparse import (
    ComponentCertificate,
    build_mapper_sparse,
    certify_component_preservation,
    cover_statistics,
)

from fast_ballmapper.optimized import (
    CompletionStats,
    ComponentFiltration,
    ball_mapper_components,
    build_cover_localized,
    build_mapper_auto,
    build_mapper_csr,
    build_mapper_streaming,
    compute_landmarks_uncovered,
    compute_mapper_streaming,
    fixed_landmark_h0,
    fixed_landmark_h0_from_cover,
    iter_cover_blocked,
    select_landmarks_uncovered,
)

try:
    __version__ = version("fast-ballmapper")
except PackageNotFoundError:  # pragma: no cover - source checkout fallback
    __version__ = "0.2.0"

__all__ = [
    "CompletionStats",
    "ComponentFiltration",
    "ball_mapper_components",
    "build_cover_localized",
    "build_mapper_auto",
    "build_mapper_csr",
    "build_mapper_streaming",
    "compute_landmarks_uncovered",
    "compute_mapper_streaming",
    "fixed_landmark_h0",
    "fixed_landmark_h0_from_cover",
    "iter_cover_blocked",
    "select_landmarks_uncovered",
    "VerifiedSelection",
    "ComponentCertificate",
    "build_cover_blocked",
    "build_mapper_sparse",
    "certify_component_preservation",
    "compute_landmarks_verified",
    "cover_statistics",
    "ApproximationAudit",
    "BackendMetadata",
    "BallMembershipAudit",
    "BallTreeBackend",
    "BoundaryDiagnostics",
    "BruteForceBackend",
    "CKDTreeBackend",
    "ColorAudit",
    "CuVSBackend",
    "CuVSConfig",
    "EdgeWitnessChange",
    "FaissConfig",
    "FaissFlatBackend",
    "FaissHNSWBackend",
    "FaissIVFBackend",
    "FaissRangeBackend",
    "HnswlibBackend",
    "HnswlibConfig",
    "RangeQueryBackend",
    "build_cover",
    "build_mapper",
    "color_by_density",
    "color_by_entropy",
    "color_by_function",
    "color_by_mode",
    "color_by_size",
    "compare_covers",
    "compute_boundary_diagnostics",
    "compute_edge_overlaps",
    "compute_landmarks",
    "compute_landmarks_fps",
    "make_backend",
]
