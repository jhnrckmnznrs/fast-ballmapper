import fast_ballmapper


def test_public_api_uses_snake_case():
    expected = {
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
    }

    assert set(fast_ballmapper.__all__) == expected
    assert not hasattr(fast_ballmapper, "computeLandmarks")
    assert not hasattr(fast_ballmapper, "buildMapper")
