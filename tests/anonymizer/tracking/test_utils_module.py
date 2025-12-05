import numpy as np
import pytest

from anonymizer.tracking import utils


def test_prepare_frame_rgb_from_gray():
    gray = np.zeros((4, 4), dtype=np.uint8)
    prepared = utils.prepare_frame_rgb(gray)
    assert prepared.shape == (4, 4, 3)
    assert prepared.dtype == np.uint8


def test_prepare_frame_rgb_from_bgra():
    bgra = np.zeros((2, 3, 4), dtype=np.uint8)
    prepared = utils.prepare_frame_rgb(bgra)
    assert prepared.shape == (2, 3, 3)


def test_prepare_frame_rgb_handles_none_and_dtype_and_non_contiguous():
    assert utils.prepare_frame_rgb(None) is None
    float_frame = np.ones((2, 2, 5), dtype=np.float32) * 300.0
    prepared = utils.prepare_frame_rgb(float_frame)
    assert prepared.dtype == np.uint8
    assert prepared.shape == (2, 2, 3)  # channels > 4 get trimmed
    non_contig = np.ones((2, 2, 3), dtype=np.uint8).transpose(2, 0, 1)
    prepared2 = utils.prepare_frame_rgb(non_contig.transpose(1, 2, 0))
    assert prepared2.flags["C_CONTIGUOUS"]


def test_clamp_bbox_and_crop_patch():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    clamped = utils.clamp_bbox((8, 8, 5, 5), frame.shape)
    assert clamped == (8, 8, 2, 2)
    patch = utils.crop_patch(frame, (8, 8, 5, 5))
    assert patch.shape == (2, 2, 3)
    assert utils.crop_patch(frame, (20, 20, 5, 5)) is None
    assert utils.clamp_bbox((0, 0, -1, -1), frame.shape) is None


def test_cosine_similarities():
    history = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    current = np.array([1.0, 0.0])
    sims = utils.cosine_similarities(history, current)
    assert sims.shape == (2,)
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] == pytest.approx(0.0)
    assert utils.cosine_similarities([], current).size == 0
