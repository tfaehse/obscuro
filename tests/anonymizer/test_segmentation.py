import numpy as np

from anonymizer.segmentation import decode_yolo_masks


class TestSegmentation:
    def test_decode_yolo_masks_empty(self):
        masks = decode_yolo_masks(None, None, np.array([]), {}, 640)
        assert masks == []

        masks = decode_yolo_masks(np.array([]), np.array([]), np.array([]), {}, 640)
        assert masks == []

    def test_decode_yolo_masks_invalid_inputs(self):
        # Invalid proto dimensions
        proto = np.zeros((10, 10), dtype=np.float32)  # 2D
        coeffs = np.zeros((1, 32), dtype=np.float32)
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        meta = {"original_shape": (100, 100)}

        # 2D proto is reshaped if square, but here 10x10 -> sqrt(10) not int
        # Wait, 10x10 -> 100 elements. sqrt(10) is ~3.16.
        # Let's use valid square 2D proto
        proto = np.zeros((32, 160, 160), dtype=np.float32)

        # Mismatch mask dim
        coeffs = np.zeros((1, 10), dtype=np.float32)  # 10 vs 32
        masks = decode_yolo_masks(coeffs, proto, boxes, meta, 640)
        assert masks == [None]

    def test_decode_yolo_masks_basic(self):
        mask_dim = 32
        proto_h, proto_w = 160, 160
        imgsz = 640

        proto = np.random.rand(mask_dim, proto_h, proto_w).astype(np.float32)
        coeffs = np.random.rand(1, mask_dim).astype(np.float32)
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        meta = {
            "original_shape": (640, 640),
            "scale": (1.0, 1.0),
            "pad": (0.0, 0.0),
            "offset": (0.0, 0.0),
        }

        masks = decode_yolo_masks(coeffs, proto, boxes, meta, imgsz)

        assert len(masks) == 1
        assert masks[0] is not None
        assert "mask" in masks[0]
        assert "x1" in masks[0]
        assert "y1" in masks[0]

        mask = masks[0]["mask"]
        assert mask.ndim == 2
        assert mask.dtype == bool

        # Check coordinates
        x1, y1 = masks[0]["x1"], masks[0]["y1"]
        assert x1 >= 10
        assert y1 >= 10

    def test_decode_yolo_masks_with_padding(self):
        mask_dim = 32
        proto_h, proto_w = 160, 160
        imgsz = 640

        proto = np.random.rand(mask_dim, proto_h, proto_w).astype(np.float32)
        coeffs = np.random.rand(1, mask_dim).astype(np.float32)
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        # Simulate letterbox padding
        meta = {
            "original_shape": (320, 640),  # Half height
            "scale": (1.0, 1.0),  # Assuming scale 1 for simplicity in this mock
            "pad": (0.0, 160.0),  # Padded top/bottom
            "offset": (0.0, 0.0),
        }

        masks = decode_yolo_masks(coeffs, proto, boxes, meta, imgsz)

        assert len(masks) == 1
        assert masks[0] is not None
