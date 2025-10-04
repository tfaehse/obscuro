"""Tests for anonymizer.io.video utilities using PyAV for decoding."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

# Skip all tests in this module if PyAV is not available
av = pytest.importorskip("av")

from anonymizer.io.video import (  # noqa: E402  (import after pytest.importorskip)
    VideoProcessor,
    blur_video_av,
    get_video_info,
    iter_frames,
)

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #


def _make_dummy_video(
    path: Path,
    *,
    num_frames: int = 5,
    width: int = 64,
    height: int = 48,
    fps: int = 24,
    colour_step: int = 40,
) -> None:
    """
    Create a minimal synthetic video for testing purposes.

    Each frame is filled with a flat colour whose intensity depends on the
    frame index (easy to recognise frames later).
    """
    container = av.open(str(path), mode="w")

    # `mpeg4` is almost always available in CI - `libx264`/`hevc` might not be.
    stream = container.add_stream("mpeg4", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for idx in range(num_frames):
        rgb = np.full((height, width, 3), (idx * colour_step) % 255, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame = frame.reformat(format="yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()


@pytest.fixture()
def dummy_video(tmp_path: Path) -> Path:
    """Return the path to a fresh dummy video file."""
    vid_path = tmp_path / "dummy.mp4"
    _make_dummy_video(vid_path, num_frames=5)
    return vid_path


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_iter_frames_yields_expected_frames(dummy_video: Path) -> None:
    frames = list(iter_frames(dummy_video))
    assert len(frames) == 5

    # Frames are (frame_number, ndarray) tuples
    for idx, (fnum, arr) in enumerate(frames):
        assert fnum == idx
        assert arr.shape == (48, 64, 3)
        assert arr.dtype == np.uint8


def test_get_video_info_returns_reasonable_metadata(dummy_video: Path) -> None:
    info = get_video_info(dummy_video)

    expected_keys = {
        "fps",
        "duration",
        "width",
        "height",
        "frame_count",
        "codec",
        "pixel_format",
    }
    assert expected_keys.issubset(info)

    assert info["width"] == 64
    assert info["height"] == 48
    assert info["frame_count"] == 5
    assert info["pixel_format"] is None or isinstance(info["pixel_format"], str)

    # fps may differ slightly due to time_base rounding
    assert math.isclose(info["fps"], 24, rel_tol=0.05, abs_tol=0.1)
    assert info["duration"] > 0


def test_blur_video_av_identity_roundtrip(dummy_video: Path, tmp_path: Path) -> None:
    """Run blur_video_av with an identity function and compare outputs."""

    def identity(frame: np.ndarray, _idx: int) -> np.ndarray:
        return frame

    out_path = tmp_path / "roundtrip.mp4"
    blur_video_av(dummy_video, out_path, blur_func=identity, codec="mpeg4")

    in_frames = [arr for _, arr in iter_frames(dummy_video)]
    out_frames = [arr for _, arr in iter_frames(out_path)]

    assert len(in_frames) == len(out_frames) == 5

    # Allow small differences due to lossy compression
    for in_f, out_f in zip(in_frames, out_frames, strict=False):
        rmse = np.sqrt(((in_f.astype(np.int16) - out_f.astype(np.int16)) ** 2).mean())
        assert rmse < 5


def test_video_processor_get_frame_at(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    _make_dummy_video(src, num_frames=3)

    dst = tmp_path / "processed.mp4"

    # Simple processor that flips frames horizontally
    def flipper(frame: np.ndarray, _idx: int) -> np.ndarray:
        return np.flip(frame, axis=1)

    processor = VideoProcessor()
    processor.process_video(src, dst, frame_processor=flipper, codec="mpeg4")

    # The first frame in the source and processed videos should be mirror images
    first_src = processor.get_frame_at(src, 0)
    first_dst = processor.get_frame_at(dst, 0)

    assert first_src is not None and first_dst is not None
    assert np.array_equal(first_src[:, ::-1, :], first_dst)
