import numpy as np
import pytest

from fast_ballmapper import BoundaryDiagnostics, compute_boundary_diagnostics


def test_boundary_diagnostics_reports_known_margins_and_bands():
    x = np.array([[0.0], [0.8], [1.0], [1.3], [2.0]])

    diagnostics = compute_boundary_diagnostics(
        x,
        landmarks=[0],
        eps=1.0,
        deltas=[0.0, 0.25, 0.5],
    )

    assert isinstance(diagnostics, BoundaryDiagnostics)
    np.testing.assert_array_equal(diagnostics.landmarks, [0])
    np.testing.assert_allclose(diagnostics.min_abs_margin, [0.0])
    np.testing.assert_allclose(diagnostics.inside_margin, [0.0])
    np.testing.assert_allclose(diagnostics.outside_margin, [0.3])
    np.testing.assert_array_equal(diagnostics.band_counts[0.0], [1])
    np.testing.assert_array_equal(diagnostics.band_counts[0.25], [2])
    np.testing.assert_array_equal(diagnostics.band_counts[0.5], [3])


def test_boundary_diagnostics_preserves_landmark_order():
    x = np.array([[0.0], [0.5], [1.0], [2.0]])

    diagnostics = compute_boundary_diagnostics(
        x,
        landmarks=[2, 0],
        eps=0.6,
    )

    np.testing.assert_array_equal(diagnostics.landmarks, [2, 0])
    np.testing.assert_allclose(diagnostics.min_abs_margin, [0.1, 0.1])


def test_boundary_diagnostics_reports_infinite_outside_margin_when_ball_covers_all():
    x = np.array([[0.0], [0.5], [1.0]])

    diagnostics = compute_boundary_diagnostics(x, landmarks=[1], eps=10.0)

    assert np.isinf(diagnostics.outside_margin[0])


def test_boundary_diagnostics_supports_cosine_distance():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    diagnostics = compute_boundary_diagnostics(
        x,
        landmarks=[0],
        eps=1.0,
        metric="cosine",
        deltas=[0.0],
    )

    np.testing.assert_allclose(diagnostics.min_abs_margin, [0.0], atol=1e-15)
    np.testing.assert_array_equal(diagnostics.band_counts[0.0], [1])


def test_boundary_diagnostics_rejects_negative_band_width():
    with pytest.raises(ValueError, match="non-negative"):
        compute_boundary_diagnostics(
            np.array([[0.0], [1.0]]),
            landmarks=[0],
            eps=1.0,
            deltas=[-0.1],
        )
