from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from anonymizer.detection.masks import MaskManager


class TestMaskManager:
    @pytest.fixture
    def mask_manager(self):
        manager = MaskManager(imgsz=640)
        yield manager
        manager.clear_mask_cache()

    def test_initialization(self, mask_manager):
        assert mask_manager.imgsz == 640
        assert Path(mask_manager._mask_cache_dir).exists()
        assert Path(mask_manager._mask_proto_db).exists()
        assert mask_manager._next_mask_id == 0

    def test_store_and_get_mask_proto(self, mask_manager):
        frame_id = 1
        tile_id = 0
        proto_slice = np.random.rand(32, 160, 160).astype(np.float32)
        meta = {
            "scale": (1.0, 1.0),
            "pad": (0.0, 0.0),
            "original_shape": (640, 640),
            "tile_offset": (0.0, 0.0),
            "global_shape": (640, 640),
        }

        mask_manager.store_mask_proto(frame_id, tile_id, proto_slice, meta)

        entry = mask_manager.get_mask_proto_entry(frame_id, tile_id)
        assert entry is not None
        assert entry["imgsz"] == 640

        # Check that proto is loaded on demand
        assert "proto" in entry
        loaded_proto = entry["proto"]
        assert loaded_proto.shape == (32, 160, 160)
        assert loaded_proto.dtype == np.float16  # Stored as float16

    def test_store_mask_proto_idempotent(self, mask_manager):
        frame_id = 1
        tile_id = 0
        proto_slice = np.zeros((32, 160, 160), dtype=np.float32)
        meta = {}

        mask_manager.store_mask_proto(frame_id, tile_id, proto_slice, meta)
        entry1 = mask_manager.get_mask_proto_entry(frame_id, tile_id)

        # Store again should do nothing
        mask_manager.store_mask_proto(frame_id, tile_id, proto_slice, meta)
        entry2 = mask_manager.get_mask_proto_entry(frame_id, tile_id)

        assert entry1 == entry2

    def test_release_mask_proto(self, mask_manager):
        frame_id = 1
        tile_id = 0
        proto_slice = np.zeros((32, 160, 160), dtype=np.float32)
        meta = {}

        mask_manager.store_mask_proto(frame_id, tile_id, proto_slice, meta)
        assert mask_manager.get_mask_proto_entry(frame_id, tile_id) is not None

        mask_manager.release_mask_proto(frame_id, tile_id)
        assert mask_manager.get_mask_proto_entry(frame_id, tile_id) is None

    def test_release_mask_proto_all_tiles(self, mask_manager):
        frame_id = 1
        proto_slice = np.zeros((32, 160, 160), dtype=np.float32)
        meta = {}

        mask_manager.store_mask_proto(frame_id, 0, proto_slice, meta)
        mask_manager.store_mask_proto(frame_id, 1, proto_slice, meta)

        mask_manager.release_mask_proto(frame_id)  # Release all tiles for frame_id
        assert mask_manager.get_mask_proto_entry(frame_id, 0) is None
        assert mask_manager.get_mask_proto_entry(frame_id, 1) is None

    def test_finalize_proto_index_creates_index(self, mask_manager):
        mask_manager.finalize_proto_index()
        rows = mask_manager._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_mask_proto_frame_tile'"
        ).fetchall()
        assert rows

    def test_register_and_get_mask_payload(self, mask_manager):
        payload = {"format": "binary", "data": b"123"}
        mask_id = mask_manager.register_mask_payload(payload)
        assert mask_id == 0
        assert mask_manager.get_mask_payload(mask_id) == payload

        mask_id2 = mask_manager.register_mask_payload(payload)
        assert mask_id2 == 1

    def test_decode_mask_from_payload_binary(self, mask_manager):
        mask_data = np.ones((10, 10), dtype=bool)
        payload = {"format": "binary", "data": mask_data, "size": (10, 10)}
        row = {}

        decoded = mask_manager.decode_mask_from_payload(payload, row)
        assert decoded is not None
        assert np.array_equal(decoded["mask"], mask_data)
        assert decoded["x1"] == 0
        assert decoded["y1"] == 0

    def test_decode_mask_from_payload_invalid(self, mask_manager):
        assert mask_manager.decode_mask_from_payload({}, {}) is None
        assert mask_manager.decode_mask_from_payload({"format": "unknown"}, {}) is None

    @patch("anonymizer.detection.masks.decode_yolo_masks")
    def test_decode_mask_from_payload_coeff(self, mock_decode, mask_manager):
        frame_id = 1
        tile_id = 0
        proto_slice = np.zeros((32, 160, 160), dtype=np.float32)
        meta = {"scale": (1.0, 1.0), "pad": (0.0, 0.0), "original_shape": (640, 640)}
        mask_manager.store_mask_proto(frame_id, tile_id, proto_slice, meta)

        coeffs = np.zeros((1, 32), dtype=np.float32)
        payload = {
            "format": "coeff",
            "frame": frame_id,
            "tile_id": tile_id,
            "coeffs": coeffs.tobytes(),
            "num_coeffs": 32,
            "dtype": "float32",
            "box": [10.0, 10.0, 50.0, 50.0],
        }
        row = {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}

        mock_decode.return_value = [{"mask": np.ones((40, 40), dtype=bool), "x1": 10, "y1": 10}]

        decoded = mask_manager.decode_mask_from_payload(payload, row)

        assert decoded is not None
        mock_decode.assert_called_once()
        args, _ = mock_decode.call_args
        assert args[0].shape == (1, 32)  # coeffs
        assert args[1].shape == (32, 160, 160)  # proto
        assert args[2].shape == (1, 4)  # boxes

    def test_decode_masks_for_rows(self, mask_manager):
        # Test binary mask decoding in batch
        mask_data = np.ones((10, 10), dtype=bool)
        payload = {"format": "binary", "data": mask_data, "size": (10, 10)}
        rows = [{"mask": payload, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]

        results = mask_manager.decode_masks_for_rows(rows)
        assert len(results) == 1
        assert results[0] is not None
        assert np.array_equal(results[0]["mask"], mask_data)

    def test_decode_masks_for_rows_coeff_ingredients(self, mask_manager):
        # This tests the complex merging logic
        frame_id = 1
        tile_id = 0
        proto_slice = np.zeros((32, 160, 160), dtype=np.float32)
        meta = {"scale": (1.0, 1.0), "pad": (0.0, 0.0), "original_shape": (640, 640)}
        mask_manager.store_mask_proto(frame_id, tile_id, proto_slice, meta)

        coeffs = np.zeros((1, 32), dtype=np.float32)
        ingredient = {
            "frame": frame_id,
            "tile_id": tile_id,
            "coeffs": coeffs.tobytes(),
            "num_coeffs": 32,
            "dtype": "float32",
            "box": [10.0, 10.0, 50.0, 50.0],
        }
        payload = {"format": "coeff_ingredients", "ingredients": [ingredient]}
        rows = [{"mask": payload, "x1": 10, "y1": 10, "x2": 50, "y2": 50}]

        with patch("anonymizer.detection.masks.decode_yolo_masks") as mock_decode:
            mock_decode.return_value = [{"mask": np.ones((40, 40), dtype=bool), "x1": 10, "y1": 10}]

            results = mask_manager.decode_masks_for_rows(rows)

            assert len(results) == 1
            assert results[0] is not None
            mock_decode.assert_called_once()
