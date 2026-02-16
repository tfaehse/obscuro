from __future__ import annotations

import contextlib
import sqlite3
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from anonymizer.paths import get_temp_dir
from anonymizer.segmentation import decode_yolo_masks


class MaskManager:
    """Manages mask prototypes, caching, and decoding."""

    class ProtoRecord(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        frame_id: int
        tile_id: int
        dtype: str
        ndim: int
        shape0: int
        shape1: int
        shape2: int
        shape3: int
        proto: bytes
        scale_x: float
        scale_y: float
        pad_x: float
        pad_y: float
        original_h: int
        original_w: int
        offset_x: float
        offset_y: float
        global_h: int
        global_w: int
        scale_up: float
        imgsz: int

    def __init__(self, imgsz: int):
        self.imgsz = imgsz
        self._mask_cache_dir = Path(tempfile.mkdtemp(prefix="obscuro_proto_", dir=get_temp_dir()))
        self._mask_proto_meta: dict[tuple[int, int], dict[str, Any]] = {}
        self._mask_proto_db = self._mask_cache_dir / "mask_proto.sqlite3"
        self._conn = sqlite3.connect(self._mask_proto_db, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=OFF")
        self._conn.execute("PRAGMA synchronous=OFF")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA cache_size=10000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mask_proto (
                frame_id INTEGER NOT NULL,
                tile_id INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                ndim INTEGER NOT NULL,
                shape0 INTEGER NOT NULL,
                shape1 INTEGER NOT NULL,
                shape2 INTEGER NOT NULL,
                shape3 INTEGER NOT NULL,
                proto BLOB NOT NULL,
                scale_x REAL NOT NULL,
                scale_y REAL NOT NULL,
                pad_x REAL NOT NULL,
                pad_y REAL NOT NULL,
                original_h INTEGER NOT NULL,
                original_w INTEGER NOT NULL,
                offset_x REAL NOT NULL,
                offset_y REAL NOT NULL,
                global_h INTEGER NOT NULL,
                global_w INTEGER NOT NULL,
                scale_up REAL NOT NULL,
                imgsz INTEGER NOT NULL
            )
            """
        )
        self._conn.execute("BEGIN")
        self._index_ready = False
        self._mask_store: dict[int, dict[str, Any]] = {}
        self._next_mask_id = 0

    def finalize_proto_index(self) -> None:
        if self._index_ready:
            return
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute("COMMIT")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mask_proto_frame_tile "
                "ON mask_proto(frame_id, tile_id)"
            )
            self._conn.execute("COMMIT")
        self._index_ready = True

    def store_mask_proto(
        self, frame_id: int, tile_id: int, proto_slice: np.ndarray, meta: dict[str, Any]
    ) -> None:
        key = (frame_id, tile_id)
        if key in self._mask_proto_meta:
            return
        self._mask_cache_dir.mkdir(parents=True, exist_ok=True)
        proto_arr = np.asarray(proto_slice, dtype=np.float16)
        proto_arr = np.ascontiguousarray(proto_arr)
        ndim = int(proto_arr.ndim)
        shape = proto_arr.shape + (1,) * (4 - ndim) if ndim < 4 else proto_arr.shape[:4]
        record = self.ProtoRecord(
            frame_id=int(frame_id),
            tile_id=int(tile_id),
            dtype=str(proto_arr.dtype),
            ndim=ndim,
            shape0=int(shape[0]),
            shape1=int(shape[1]),
            shape2=int(shape[2]) if len(shape) > 2 else 1,
            shape3=int(shape[3]) if len(shape) > 3 else 1,
            proto=proto_arr.tobytes(order="C"),
            scale_x=float(meta.get("scale", (1.0, 1.0))[0]),
            scale_y=float(meta.get("scale", (1.0, 1.0))[1]),
            pad_x=float(meta.get("pad", (0.0, 0.0))[0]),
            pad_y=float(meta.get("pad", (0.0, 0.0))[1]),
            original_h=int(meta.get("original_shape", (0, 0))[0]),
            original_w=int(meta.get("original_shape", (0, 0))[1]),
            offset_x=float(meta.get("tile_offset", (0.0, 0.0))[0]),
            offset_y=float(meta.get("tile_offset", (0.0, 0.0))[1]),
            global_h=int(meta.get("global_shape", (0, 0))[0]),
            global_w=int(meta.get("global_shape", (0, 0))[1]),
            scale_up=float(meta.get("scale_up", 1.0)),
            imgsz=int(meta.get("imgsz", self.imgsz)),
        )
        try:
            self._conn.execute(
                """
                INSERT INTO mask_proto (
                    frame_id, tile_id, dtype, ndim, shape0, shape1, shape2, shape3, proto,
                    scale_x, scale_y, pad_x, pad_y, original_h, original_w,
                    offset_x, offset_y, global_h, global_w, scale_up, imgsz
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.frame_id,
                    record.tile_id,
                    record.dtype,
                    record.ndim,
                    record.shape0,
                    record.shape1,
                    record.shape2,
                    record.shape3,
                    record.proto,
                    record.scale_x,
                    record.scale_y,
                    record.pad_x,
                    record.pad_y,
                    record.original_h,
                    record.original_w,
                    record.offset_x,
                    record.offset_y,
                    record.global_h,
                    record.global_w,
                    record.scale_up,
                    record.imgsz,
                ),
            )
        except sqlite3.Error as exc:
            raise RuntimeError(
                "Mask proto storage failed; clear temp storage, disable masks, or "
                "process a shorter video."
            ) from exc
        self._mask_proto_meta[key] = {
            "scale": (record.scale_x, record.scale_y),
            "pad": (record.pad_x, record.pad_y),
            "original_shape": (record.original_h, record.original_w),
            "offset": (record.offset_x, record.offset_y),
            "global_shape": (record.global_h, record.global_w),
            "scale_up": record.scale_up,
            "imgsz": record.imgsz,
        }

    def get_mask_proto_entry(self, frame_id: int, tile_id: int = 0) -> dict[str, Any] | None:
        key = (frame_id, tile_id)
        entry = self._mask_proto_meta.get(key)
        if entry is None:
            return None
        if "proto" not in entry:
            row = self._conn.execute(
                "SELECT dtype, ndim, shape0, shape1, shape2, shape3, proto "
                "FROM mask_proto WHERE frame_id = ? AND tile_id = ?",
                (frame_id, tile_id),
            ).fetchone()
            if row is None:
                return None
            dtype, ndim, s0, s1, s2, s3, proto_bytes = row
            shape = (int(s0), int(s1), int(s2), int(s3))[: int(ndim)]
            arr = np.frombuffer(proto_bytes, dtype=np.dtype(dtype)).reshape(shape)
            entry["proto"] = arr
        return entry

    def release_mask_proto(self, frame_id: int, tile_id: int | None = None) -> None:
        if tile_id is None:
            keys = [k for k in self._mask_proto_meta if k[0] == frame_id]
        else:
            keys = [(frame_id, tile_id)]
        for key in keys:
            entry = self._mask_proto_meta.pop(key, None)
            if not entry:
                continue
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute(
                    "DELETE FROM mask_proto WHERE frame_id = ? AND tile_id = ?",
                    (int(key[0]), int(key[1])),
                )

    def clear_mask_cache(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute("COMMIT")
        for frame_id, tile_id in list(self._mask_proto_meta.keys()):
            self.release_mask_proto(frame_id, tile_id)
        self._mask_store.clear()
        self._next_mask_id = 0
        with contextlib.suppress(sqlite3.Error):
            self._conn.close()
        with contextlib.suppress(FileNotFoundError):
            self._mask_proto_db.unlink()
        with contextlib.suppress(OSError):
            self._mask_cache_dir.rmdir()

    def register_mask_payload(self, payload: dict[str, Any]) -> int:
        mask_id = self._next_mask_id
        self._next_mask_id += 1
        self._mask_store[mask_id] = payload
        return mask_id

    def get_mask_payload(self, mask_id: int) -> dict[str, Any] | None:
        return self._mask_store.get(int(mask_id))

    def decode_mask_from_payload(
        self,
        payload: dict[str, Any],
        row: dict[str, Any],
        frame_shape: tuple[int, int] | None = None,
    ) -> dict[str, Any] | None:
        if isinstance(payload, int):
            resolved = self.get_mask_payload(payload)
            if resolved is None:
                return None
            payload = resolved
        if not isinstance(payload, dict):
            return None
        fmt = payload.get("format")
        if fmt == "binary":
            data = payload.get("data")
            size = payload.get("size")
            if data is None or size is None:
                return None
            mask = np.asarray(data, dtype=bool)
            try:
                h, w = map(int, size)
                mask = mask.reshape((h, w))
            except (ValueError, TypeError):
                return None
            return {"mask": mask, "x1": 0, "y1": 0}
        if fmt not in {"coeff", "coeff_ingredients"}:
            return None
        frame_id = int(payload.get("frame", -1))
        tile_id = int(payload.get("tile_id", payload.get("frame", 0)))
        entry = self.get_mask_proto_entry(frame_id, tile_id)
        if entry is None:
            return None
        proto = entry.get("proto")
        if proto is None:
            return None
        ingredients = (
            payload.get("ingredients")
            if payload.get("format") == "coeff_ingredients"
            else [payload]
        )
        coeff_list: list[np.ndarray] = []
        box_list: list[list[float]] = []
        for ingredient in ingredients or []:
            coeff_bytes = ingredient.get("coeffs")
            if not isinstance(coeff_bytes, bytes | bytearray):
                continue
            dtype = ingredient.get("dtype", "float16")
            np_dtype = np.float16 if dtype == "float16" else np.float32
            num_coeffs = int(ingredient.get("num_coeffs", proto.shape[0]))
            coeff_arr = (
                np.frombuffer(coeff_bytes, dtype=np_dtype, count=num_coeffs)
                .reshape(1, -1)
                .astype(np.float32)
            )
            box = ingredient.get("box") or [
                float(row.get("x1", 0.0)),
                float(row.get("y1", 0.0)),
                float(row.get("x2", 0.0)),
                float(row.get("y2", 0.0)),
            ]
            if len(box) != 4:
                continue
            coeff_list.append(coeff_arr[0])
            box_list.append([float(v) for v in box])
        if not coeff_list:
            return None
        coeffs = np.stack(coeff_list, axis=0)
        boxes = np.asarray(box_list, dtype=np.float32)
        meta = {
            "pad": entry.get("pad", (0.0, 0.0)),
            "scale": entry.get("scale", (1.0, 1.0)),
            "original_shape": entry.get("original_shape", (0, 0)),
            "offset": entry.get("offset", (0.0, 0.0)),
            "global_shape": entry.get("global_shape", (0, 0)),
            "scale_up": entry.get("scale_up", 1.0),
        }
        masks = decode_yolo_masks(coeffs, proto.astype(np.float32), boxes, meta, entry["imgsz"])
        return masks[0] if masks else None

    def decode_masks_for_rows(
        self,
        rows: Sequence[dict[str, Any]],
        frame_shape: tuple[int, int] | None = None,
    ) -> list[dict[str, Any] | None]:
        results: list[dict[str, Any] | None] = [None] * len(rows)

        def _merge_regions(
            base: dict[str, Any] | None, new: dict[str, Any] | None
        ) -> dict[str, Any] | None:
            if base is None:
                return new
            if new is None:
                return base
            mask_a = np.asarray(base.get("mask"), dtype=bool)
            mask_b = np.asarray(new.get("mask"), dtype=bool)
            if mask_a.ndim != 2 or mask_b.ndim != 2 or mask_a.size == 0 or mask_b.size == 0:
                return base
            xa, ya = int(base.get("x1", 0)), int(base.get("y1", 0))
            xb, yb = int(new.get("x1", 0)), int(new.get("y1", 0))
            x1 = min(xa, xb)
            y1 = min(ya, yb)
            x2 = max(xa + mask_a.shape[1], xb + mask_b.shape[1])
            y2 = max(ya + mask_a.shape[0], yb + mask_b.shape[0])
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if width == 0 or height == 0:
                return base
            combined = np.zeros((height, width), dtype=bool)
            combined[ya - y1 : ya - y1 + mask_a.shape[0], xa - x1 : xa - x1 + mask_a.shape[1]] |= (
                mask_a
            )
            combined[yb - y1 : yb - y1 + mask_b.shape[0], xb - x1 : xb - x1 + mask_b.shape[1]] |= (
                mask_b
            )
            return {"mask": combined, "x1": x1, "y1": y1}

        grouped: dict[
            tuple[
                int,
                int,
                tuple[float, ...],
                tuple[float, ...],
                tuple[int, ...],
                tuple[float, ...],
                int,
                tuple[int, ...],
                float,
            ],
            list[tuple[int, dict[str, Any]]],
        ] = {}

        for idx, row in enumerate(rows):
            payload = row.get("mask")
            if isinstance(payload, int):
                payload = self.get_mask_payload(payload)
            if not isinstance(payload, dict):
                continue
            fmt = payload.get("format")
            if fmt == "binary":
                decoded = self.decode_mask_from_payload(payload, row, frame_shape)
                results[idx] = decoded
                continue
            if fmt == "coeff":
                items = [payload]
            elif fmt == "coeff_ingredients":
                items = payload.get("ingredients") or []
            else:
                continue
            for ingredient in items:
                if not isinstance(ingredient, dict):
                    continue
                frame_id = int(ingredient.get("frame", -1))
                tile_id = int(ingredient.get("tile_id", 0))
                coeff_bytes = ingredient.get("coeffs")
                if not isinstance(coeff_bytes, bytes | bytearray):
                    continue
                meta = ingredient.get("meta", {}) or {}
                pad = tuple(float(v) for v in meta.get("pad", (0.0, 0.0)))
                scale = tuple(float(v) for v in meta.get("scale", (1.0, 1.0)))
                original_shape = tuple(int(v) for v in meta.get("original_shape", (0, 0)))
                if frame_shape and (original_shape == (0, 0) or original_shape == (0,)):
                    original_shape = (frame_shape[0], frame_shape[1])
                offset = tuple(float(v) for v in meta.get("offset", (0.0, 0.0)))
                global_shape = tuple(int(v) for v in meta.get("global_shape", (0, 0)))
                scale_up = float(meta.get("scale_up", 1.0))
                imgsz = int(meta.get("imgsz", self.imgsz))
                key = (
                    frame_id,
                    tile_id,
                    pad,
                    scale,
                    original_shape,
                    offset,
                    imgsz,
                    global_shape,
                    scale_up,
                )
                grouped.setdefault(key, []).append((idx, ingredient))

        for key, items in grouped.items():
            frame_id, tile_id, pad, scale, original_shape, offset, imgsz, global_shape, scale_up = (
                key
            )
            entry = self.get_mask_proto_entry(frame_id, tile_id)
            if entry is None:
                continue
            proto = entry.get("proto")
            if proto is None:
                continue
            coeff_list: list[np.ndarray] = []
            box_list: list[list[float]] = []
            owners: list[int] = []
            for idx, ingredient in items:
                coeff_bytes = ingredient.get("coeffs")
                dtype = str(ingredient.get("dtype", "float16")).lower()
                np_dtype = np.float16 if dtype == "float16" else np.float32
                num_coeffs = int(ingredient.get("num_coeffs", proto.shape[0]))
                coeff_arr = np.frombuffer(coeff_bytes, dtype=np_dtype, count=num_coeffs).astype(
                    np.float32
                )
                box = ingredient.get("box") or [
                    float(rows[idx].get("x1", 0.0)),
                    float(rows[idx].get("y1", 0.0)),
                    float(rows[idx].get("x2", 0.0)),
                    float(rows[idx].get("y2", 0.0)),
                ]
                if len(box) != 4:
                    continue
                coeff_list.append(coeff_arr)
                box_list.append([float(v) for v in box])
                owners.append(idx)
            if not coeff_list:
                continue
            coeffs = np.stack(coeff_list, axis=0)
            boxes = np.asarray(box_list, dtype=np.float32)
            meta = {
                "pad": pad,
                "scale": scale,
                "original_shape": original_shape,
                "offset": offset,
                "global_shape": global_shape,
            }
            masks = decode_yolo_masks(
                coeffs,
                proto.astype(np.float32),
                boxes,
                meta,
                imgsz,
            )
            for owner_idx, mask in zip(owners, masks, strict=False):
                if mask is not None and scale_up != 1.0:
                    m_data = mask["mask"]
                    if m_data.size > 0:
                        h, w = m_data.shape
                        new_h, new_w = round(h * scale_up), round(w * scale_up)
                        if new_h > 0 and new_w > 0:
                            m_resized = cv2.resize(
                                m_data.astype(np.uint8),
                                (new_w, new_h),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                            mask["mask"] = m_resized
                            mask["x1"] = round(mask["x1"] * scale_up)
                            mask["y1"] = round(mask["y1"] * scale_up)
                results[owner_idx] = _merge_regions(results[owner_idx], mask)

        return results
