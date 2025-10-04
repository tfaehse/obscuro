"""Tests for CPU-based ROI-blurring helpers."""

from __future__ import annotations

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Optional deps                                                               #
# --------------------------------------------------------------------------- #

cv2 = pytest.importorskip("cv2")

from anonymizer.io.blur_rois import (  # noqa: E402  (import after pytest.importorskip)
    blur_rois,
    convert_relative_to_absolute_rois,
)

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def sample_image() -> np.ndarray:
    """Return a 100x200 BGR image with a random rectangle ROI at (20,10,40,30)."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=42)
    img[10:40, 20:60] = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    return img


@pytest.fixture(params=["gaussian", "pixelate"])
def blur_type(request: pytest.FixtureRequest) -> str:
    return str(request.param)


# --------------------------------------------------------------------------- #
# Helper                                                                       #
# --------------------------------------------------------------------------- #


def _roi_slice(roi: tuple[int, int, int, int]) -> tuple[slice, slice]:
    x, y, w, h = roi
    return slice(y, y + h), slice(x, x + w)


# --------------------------------------------------------------------------- #
# Tests: blur_rois                                                            #
# --------------------------------------------------------------------------- #


def test_blur_rois_modifies_only_roi(sample_image: np.ndarray, blur_type: str) -> None:
    """Inside-ROI pixels must change; outside must stay identical."""
    roi = (20, 10, 40, 30)

    blurred = blur_rois(sample_image, [roi], blur_type=blur_type, blur_strength=11)

    rs = _roi_slice(roi)
    # Outside ROI unchanged (choose a pixel far away for certainty)
    assert np.array_equal(sample_image[0, 0], blurred[0, 0])
    # Inside ROI changed
    assert not np.array_equal(sample_image[rs], blurred[rs])


def test_blur_rois_blackout(sample_image: np.ndarray) -> None:
    roi = (20, 10, 40, 30)
    blurred = blur_rois(sample_image, [roi], blur_type="blackout")

    rs = _roi_slice(roi)
    assert np.all(blurred[rs] == 0)  # ROI turned black
    assert np.array_equal(sample_image[99, 199], blurred[99, 199])  # rest unchanged


def test_blur_rois_handles_out_of_bounds(sample_image: np.ndarray) -> None:
    """A too-large ROI should be clipped without crashing."""
    huge_roi = (-10, -10, 400, 300)
    out = blur_rois(sample_image, [huge_roi], blur_type="gaussian", blur_strength=7)

    assert out.shape == sample_image.shape  # dimensions preserved


# --------------------------------------------------------------------------- #
# Tests: convert_relative_to_absolute_rois                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("rel_roi", "expected"),
    [
        # width = 200, height = 100 → scale x independently from y
        ((0.1, 0.1, 0.3, 0.3), (20, 10, 40, 20)),
        # whole image
        ((0.0, 0.0, 1.0, 1.0), (0, 0, 200, 100)),
        # zero-area ROI should be discarded
        ((0.5, 0.5, 0.5, 0.6), None),
    ],
)
def test_convert_relative_to_absolute_rois(
    rel_roi: tuple[float, float, float, float],
    expected: tuple[int, int, int, int] | None,
) -> None:
    img_shape = (100, 200, 3)

    result = convert_relative_to_absolute_rois([rel_roi], img_shape)

    if expected is None:
        assert result == []
    else:
        assert result == [expected]
