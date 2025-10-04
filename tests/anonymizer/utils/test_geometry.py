import numpy as np
import pytest

from anonymizer.utils.geometry import iou_tlwh, nms_tlwh, tlwh_to_xyxy


def test_tlwh_to_xyxy_conversion():
    box = np.array([10.0, 20.0, 30.0, 40.0])
    converted = tlwh_to_xyxy(box)
    assert np.allclose(converted, np.array([10.0, 20.0, 40.0, 60.0]))


def test_iou_tlwh_overlap():
    box_a = np.array([10.0, 10.0, 20.0, 20.0])
    box_b = np.array([15.0, 15.0, 20.0, 20.0])
    iou = iou_tlwh(box_a, box_b)
    # Intersection is 15x15 (area 225); each box area is 400 → union 575
    assert iou == pytest.approx(225.0 / 575.0)


def test_iou_tlwh_disjoint():
    box_a = np.array([0.0, 0.0, 10.0, 10.0])
    box_b = np.array([20.0, 20.0, 5.0, 5.0])
    assert iou_tlwh(box_a, box_b) == 0.0


def test_nms_tlwh_suppresses_lower_scores():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 10.0, 10.0],
            [20.0, 20.0, 5.0, 5.0],
        ]
    )
    scores = np.array([0.9, 0.8, 0.5])
    keep = nms_tlwh(boxes, scores, iou_threshold=0.5)
    # First two overlap > 0.5, keep highest score (index 0). Third is far and kept.
    assert set(keep) == {0, 2}


def test_nms_tlwh_empty_input():
    boxes = np.empty((0, 4))
    scores = np.empty((0,))
    assert nms_tlwh(boxes, scores, 0.5) == []
