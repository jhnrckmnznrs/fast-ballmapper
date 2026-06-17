import fast_ballmapper


def test_public_api_uses_snake_case():
    expected = {
        "build_mapper",
        "color_by_density",
        "color_by_entropy",
        "color_by_function",
        "color_by_mode",
        "color_by_size",
        "compute_edge_overlaps",
        "compute_landmarks",
        "compute_landmarks_fps",
    }
    assert set(fast_ballmapper.__all__) == expected
    assert not hasattr(fast_ballmapper, "computeLandmarks")
    assert not hasattr(fast_ballmapper, "buildMapper")
