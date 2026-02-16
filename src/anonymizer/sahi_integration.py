from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from PIL import Image
from sahi.models.base import DetectionModel
from sahi.postprocess.combine import NMMPostprocess
from sahi.predict import PredictionResult
from sahi.prediction import ObjectPrediction
from sahi.slicing import slice_image

from anonymizer.constants import DEFAULT_CATEGORY_MAPPING
from anonymizer.detection.core import BaseDetector
from anonymizer.detection.utils import preprocess_image


class SahiOnnxDetectionModel(DetectionModel):
    """SAHI DetectionModel adapter that reuses an existing Detector ONNX session."""

    def __init__(
        self,
        detector: BaseDetector,
        category_mapping: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        self.detector = detector
        self._input_name: str | None = None
        self._last_dataframe: pl.DataFrame | None = None
        self._last_meta: list[dict[str, Any]] | None = None

        super().__init__(
            model_path=None,
            model=detector.session,
            load_at_init=False,
            confidence_threshold=0.0,
            **kwargs,
        )
        mapping = dict(DEFAULT_CATEGORY_MAPPING)
        if category_mapping:
            mapping.update({str(k): str(v) for k, v in category_mapping.items()})
        self.category_mapping = mapping
        self.set_model(detector.session)
        # Tile batch size for batched tiled inference; can be set by user.
        self.tile_batch_size: int = 1

    @property
    def last_dataframe(self) -> pl.DataFrame | None:
        return self._last_dataframe

    def load_model(self) -> None:  # pragma: no cover - Loader delegates to detector session
        self.set_model(self.detector.session)

    def set_model(self, model: Any):
        self.model = model
        inputs = getattr(self.model, "get_inputs", lambda: [])()
        if not inputs:
            raise ValueError("ONNX session has no input tensors configured")
        self._input_name = inputs[0].name

    def perform_inference(self, image: np.ndarray):
        if self.model is None or self._input_name is None:
            raise RuntimeError("SAHI model is not initialised")

        input_tensor, meta = preprocess_image(image, self.detector.imgsz)
        model = self.model
        input_name = self._input_name
        if model is None or input_name is None:
            raise RuntimeError("SAHI model is not initialised")
        outputs = model.run(None, {input_name: input_tensor})

        self._last_meta = [meta]
        dataframe = self.detector._postprocess(outputs, [meta], frame_ids=[0])  # pylint: disable=protected-access
        self._last_dataframe = dataframe
        self._original_predictions = dataframe

    def _predict_full_frame(self, image: np.ndarray) -> list[ObjectPrediction]:
        """Run a single full-frame inference and return SAHI object predictions."""
        self.perform_inference(image)
        self._create_object_prediction_list_from_original_predictions(
            shift_amount_list=[[0, 0]],
            full_shape_list=[[image.shape[0], image.shape[1]]],
        )
        if not self._object_prediction_list_per_image:
            return []
        return list(self._object_prediction_list_per_image[0])

    def _run_sliced_predictions(
        self,
        image: np.ndarray,
        *,
        slice_height: int,
        slice_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
    ) -> list[ObjectPrediction]:
        """Run tiled inference and return raw object predictions for each slice."""
        slice_result = slice_image(
            image=Image.fromarray(image.astype(np.uint8)),
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        slices = slice_result.sliced_image_list
        if not slices:
            self._original_predictions = self._empty_dataframe()
            self._object_prediction_list_per_image = [[]]
            return []

        tile_bs = max(1, int(self.tile_batch_size))
        all_object_predictions: list[ObjectPrediction] = []
        model = self.model
        input_name = self._input_name
        if model is None or input_name is None:
            raise RuntimeError("SAHI model is not initialised")
        for i in range(0, len(slices), tile_bs):
            chunk = slices[i : i + tile_bs]
            tiles = []
            offsets = []
            metas = []
            for tile in chunk:
                tile_image = np.asarray(tile.image)
                starting_pixel = tile.starting_pixel
                pre, meta = preprocess_image(tile_image, self.detector.imgsz)
                tiles.append(pre[0])
                metas.append(meta)
                offsets.append((starting_pixel[0], starting_pixel[1]))
            batch_tensor = np.stack(tiles, axis=0)
            outputs = model.run(None, {input_name: batch_tensor})
            frame_ids = list(range(len(metas)))
            df_chunk = self.detector._postprocess(outputs, metas, frame_ids=frame_ids)
            for j in range(len(chunk)):
                tile_df = df_chunk.filter(pl.col("frame") == j)
                if tile_df.is_empty():
                    continue
                self._original_predictions = tile_df
                self._last_meta = [metas[j]]
                self._create_object_prediction_list_from_original_predictions(
                    shift_amount_list=[[offsets[j][0], offsets[j][1]]],
                    full_shape_list=[[image.shape[0], image.shape[1]]],
                )
                if self._object_prediction_list_per_image:
                    all_object_predictions.extend(self._object_prediction_list_per_image[0])
        return all_object_predictions

    def predict_sliced_batched(
        self,
        image: np.ndarray,
        *,
        slice_height: int,
        slice_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
        postprocess_type: str,
        postprocess_match_threshold: float,
        postprocess_class_agnostic: bool,
        verbose: int = 0,
    ) -> PredictionResult:
        """
        Run SAHI batched tiled inference on the input image.
        This is the preferred entrypoint for Detector to call for efficient tile batching.
        """
        all_object_predictions = self._run_sliced_predictions(
            image,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        if not all_object_predictions:
            return PredictionResult([], image)

        pp = NMMPostprocess(
            match_threshold=float(postprocess_match_threshold),
            class_agnostic=bool(postprocess_class_agnostic),
        )
        merged_predictions = pp(all_object_predictions)
        self._object_prediction_list_per_image = [merged_predictions]
        self._last_dataframe = None
        return PredictionResult(
            object_prediction_list=merged_predictions,
            image=image,
        )

    def predict_fused(
        self,
        image: np.ndarray,
        *,
        slice_height: int,
        slice_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
        postprocess_type: str,
        postprocess_match_threshold: float,
        postprocess_class_agnostic: bool,
        verbose: int = 0,
    ) -> PredictionResult:
        """
        Run a full-frame inference along with sliced inference and fuse the detections.
        """
        full_predictions = self._predict_full_frame(image)
        sliced_predictions = self._run_sliced_predictions(
            image,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        combined_predictions = full_predictions + sliced_predictions
        if not combined_predictions:
            self._object_prediction_list_per_image = [[]]
            self._last_dataframe = self._empty_dataframe()
            return PredictionResult([], image)

        pp = NMMPostprocess(
            match_threshold=float(postprocess_match_threshold),
            class_agnostic=bool(postprocess_class_agnostic),
        )
        merged_predictions = pp(combined_predictions)
        self._object_prediction_list_per_image = [merged_predictions]
        self._last_dataframe = None
        return PredictionResult(
            object_prediction_list=merged_predictions,
            image=image,
        )

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = None,
        full_shape_list: list[list[int]] | None = None,
    ):
        dataframe = self._ensure_dataframe()
        if dataframe.is_empty():
            self._object_prediction_list_per_image = [[]]
            return

        # Determine per-frame ordering so we can pair shifts and tiles correctly.
        if "frame" in dataframe.columns:
            frame_indices = sorted(
                {int(value) for value in dataframe.get_column("frame").to_list()}
            )
        else:
            frame_indices = [0]

        def _normalize_shift(raw_shift: list[int] | None) -> list[int]:
            if not raw_shift or len(raw_shift) != 2:
                return [0, 0]
            return [int(raw_shift[0]), int(raw_shift[1])]

        def _normalize_shape(raw_shape: list[int] | None) -> list[int]:
            if not raw_shape or len(raw_shape) != 2:
                return [0, 0]
            return [int(raw_shape[0]), int(raw_shape[1])]

        fallback_shape = None
        if self._last_meta and "original_shape" in self._last_meta[0]:
            fallback_shape = list(self._last_meta[0]["original_shape"])
        if fallback_shape is None:
            height_series = dataframe.get_column("frame_height")
            width_series = dataframe.get_column("frame_width")
            frame_height = int(height_series[0]) if len(height_series) else 0
            frame_width = int(width_series[0]) if len(width_series) else 0
            fallback_shape = [frame_height, frame_width]

        object_prediction_list_per_image: list[list[ObjectPrediction]] = []

        for ordinal, frame_id in enumerate(frame_indices):
            raw_shift = (
                shift_amount_list[ordinal]
                if shift_amount_list and ordinal < len(shift_amount_list)
                else None
            )
            raw_shape = (
                full_shape_list[ordinal]
                if full_shape_list and ordinal < len(full_shape_list)
                else fallback_shape
            )

            shift = _normalize_shift(raw_shift)
            full_shape = _normalize_shape(raw_shape)

            frame_df = (
                dataframe.filter(pl.col("frame") == frame_id)
                if "frame" in dataframe.columns
                else dataframe
            )

            predictions: list[ObjectPrediction] = []
            for row in frame_df.iter_rows(named=True):
                bbox = [
                    int(round(float(row["x1"]))),
                    int(round(float(row["y1"]))),
                    int(round(float(row["x2"]))),
                    int(round(float(row["y2"]))),
                ]
                category_id = int(row["object_class"])
                score = float(row["confidence"])
                category_mapping = self.category_mapping or {}
                category_name = category_mapping.get(str(category_id), str(category_id))

                predictions.append(
                    ObjectPrediction(
                        bbox=bbox,
                        score=score,
                        category_id=category_id,
                        category_name=category_name,
                        shift_amount=shift,
                        full_shape=full_shape,
                    )
                )

            object_prediction_list_per_image.append(predictions)

        self._object_prediction_list_per_image = object_prediction_list_per_image

    def _ensure_dataframe(self) -> pl.DataFrame:
        if self._original_predictions is None:
            return self._empty_dataframe()

        if isinstance(self._original_predictions, pl.DataFrame):
            return self._original_predictions
        raise TypeError("Unexpected prediction payload type from detector")

    @staticmethod
    def _empty_dataframe() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "frame": [],
                "x1": [],
                "y1": [],
                "x2": [],
                "y2": [],
                "confidence": [],
                "object_class": [],
                "frame_width": [],
                "frame_height": [],
                "is_confident": [],
            }
        )
