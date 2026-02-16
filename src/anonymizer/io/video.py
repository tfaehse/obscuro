"""High-performance video I/O utilities built on PyAV with optional prefetching."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from queue import Queue
from typing import Any

import av
import numpy as np

_DEFAULT_PREFETCH = 4
_ENCODE_PREFETCH = 4


def _iter_pyav_frames(path: str | Path):
    video_path = str(path)
    with av.open(video_path, options={"threads": "auto"}) as container:
        stream = container.streams.video[0]
        if hasattr(stream, "thread_type"):
            with contextlib.suppress(Exception):
                stream.thread_type = "AUTO"

        for frame_number, frame in enumerate(container.decode(stream)):
            yield (
                frame_number,
                frame.pts,
                frame.time_base,
                frame.to_ndarray(format="rgb24"),
            )


def iter_frames(
    path: str | Path, *, prefetch: int = _DEFAULT_PREFETCH
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_number, frame ndarray) pairs from the video using PyAV."""
    for frame_idx, _, _, array in iter_frames_with_metadata(path, prefetch=prefetch):
        yield frame_idx, array


def iter_frames_with_metadata(
    path: str | Path, *, prefetch: int = _DEFAULT_PREFETCH
) -> Iterator[tuple[int, int | None, Any, np.ndarray]]:
    if prefetch <= 1:
        yield from _iter_pyav_frames(path)
        return

    sentinel = object()
    queue: Queue = Queue(maxsize=max(1, prefetch))

    def producer():
        for item in _iter_pyav_frames(path):
            queue.put(item)
        queue.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    while True:
        item = queue.get()
        if item is sentinel:
            break
        yield item


def get_video_info(path: str | Path) -> dict:
    """Return basic metadata about a video file."""
    video_path = str(path)
    with av.open(video_path, options={"threads": "auto"}) as container:
        stream = container.streams.video[0]
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = 0.0
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        frame_count = stream.frames if stream.frames else 0
        if frame_count == 0 and fps > 0 and duration > 0:
            frame_count = int(fps * duration)
        info = {
            "fps": fps,
            "duration": duration,
            "width": stream.width,
            "height": stream.height,
            "frame_count": frame_count,
            "codec": stream.codec_context.codec.name if stream.codec_context else "",
            "pixel_format": stream.codec_context.pix_fmt if stream.codec_context else None,
        }
        return info


def blur_video_av(
    input_path: str | Path,
    output_path: str | Path,
    blur_func: Callable[[np.ndarray, int], np.ndarray],
    codec: str = "h264",
    quality: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> None:
    """Process a video frame-by-frame using PyAV."""

    input_path = str(input_path)
    output_path = str(output_path)
    in_container = av.open(input_path, options={"threads": "auto"})
    out_container = av.open(output_path, "w")

    try:
        v_in = in_container.streams.video[0]
        if v_in.frames:
            total_frames = int(v_in.frames)
        elif v_in.average_rate and v_in.duration and v_in.time_base:
            total_frames = int(v_in.average_rate * v_in.duration * v_in.time_base)
        else:
            total_frames = 0

        if v_in.average_rate:
            fps_rate = v_in.average_rate
            fps = float(fps_rate)
        else:
            fps = 0.0
            fps_rate = Fraction(30, 1)
        if fps <= 0:
            fps = 30.0
            fps_rate = Fraction(30, 1)

        v_out = out_container.add_stream(codec_name=codec, rate=fps_rate)
        v_out.width = v_in.width
        v_out.height = v_in.height
        v_out.pix_fmt = "yuv420p"
        v_out.bit_rate = v_in.bit_rate
        output_time_base = Fraction(1, round(fps))
        v_out.time_base = output_time_base
        if hasattr(v_in, "rate"):
            v_out.rate = v_in.rate
        elif hasattr(v_in, "average_rate"):
            v_out.rate = v_in.average_rate

        audio_streams = {}
        audio_in_streams = tuple(in_container.streams.audio)
        for stream in audio_in_streams:
            audio_out = out_container.add_stream(stream.codec_context.name)
            audio_streams[stream.index] = audio_out

        if hasattr(v_out, "thread_type"):
            with contextlib.suppress(Exception):
                v_out.thread_type = "AUTO"
        if hasattr(v_out, "codec_context") and hasattr(v_out.codec_context, "options"):
            with contextlib.suppress(Exception):
                v_out.codec_context.options["threads"] = "auto"
            if quality is not None:
                with contextlib.suppress(Exception):
                    v_out.codec_context.options["crf"] = str(int(quality))

        frame_number = 0
        for frame_idx, _, _, img_cpu in iter_frames_with_metadata(
            input_path, prefetch=_ENCODE_PREFETCH
        ):
            processed_img = blur_func(img_cpu, frame_idx)
            if not isinstance(processed_img, np.ndarray):
                raise ValueError("Blur function must return numpy array")
            out_frame = av.VideoFrame.from_ndarray(processed_img, format="rgb24")
            out_frame.time_base = output_time_base
            out_frame.pts = frame_number
            for encoded in v_out.encode(out_frame):
                out_container.mux(encoded)
            frame_number += 1
            if progress_callback and total_frames > 0:
                progress_callback(
                    frame_number,
                    total_frames,
                    f"Processing frame {frame_number}/{total_frames}",
                )

        for encoded in v_out.encode():
            out_container.mux(encoded)

        if audio_streams:
            for packet in in_container.demux(audio_in_streams):
                if packet.stream.index in audio_streams:
                    packet.stream = audio_streams[packet.stream.index]
                    out_container.mux(packet)
    finally:
        in_container.close()
        out_container.close()


class VideoProcessor:
    """High-level video processor for CPU processing."""

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        frame_processor: Callable[[np.ndarray, int], np.ndarray],
        codec: str = "h264",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        blur_video_av(
            input_path=input_path,
            output_path=output_path,
            blur_func=frame_processor,
            codec=codec,
            progress_callback=progress_callback,
        )

    def get_frame_at(self, path: str | Path, frame_number: int) -> np.ndarray | None:
        if frame_number < 0:
            return None
        video_path = str(path)
        with av.open(video_path, options={"threads": "auto"}) as container:
            stream = container.streams.video[0]
            if hasattr(stream, "thread_type"):
                with contextlib.suppress(Exception):
                    stream.thread_type = "AUTO"
            for idx, frame in enumerate(container.decode(stream)):
                if idx == frame_number:
                    return frame.to_ndarray(format="bgr24")
                if idx > frame_number:
                    break
        return None


def iter_frame_batches(
    path: str | Path,
    batch_size: int,
    *,
    prefetch: int = _DEFAULT_PREFETCH,
) -> Iterator[list[tuple[int, np.ndarray]]]:
    """Yield batches of (frame_index, frame ndarray) pairs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[tuple[int, np.ndarray]] = []
    for frame_idx, _, _, array in iter_frames_with_metadata(path, prefetch=prefetch):
        batch.append((frame_idx, array))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
