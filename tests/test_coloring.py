import numpy as np

from fast_ballmapper import (
    color_by_density,
    color_by_entropy,
    color_by_function,
    color_by_mode,
    color_by_size,
)


def test_color_helpers():
    cover = [np.array([0, 1]), np.array([2]), np.array([], dtype=int)]
    values = np.array([1.0, 3.0, 9.0])
    labels = np.array([0, 0, 1])

    np.testing.assert_allclose(
        color_by_function(values, cover), [2.0, 9.0, np.nan], equal_nan=True
    )
    assert color_by_mode(labels, cover).tolist() == [0, 1, -1]
    np.testing.assert_allclose(
        color_by_entropy(labels, cover), [0.0, 0.0, 0.0], atol=1e-9
    )
    np.testing.assert_array_equal(color_by_size(cover), [2, 1, 0])
    np.testing.assert_allclose(color_by_density(cover), [1.0, 0.5, 0.0])
