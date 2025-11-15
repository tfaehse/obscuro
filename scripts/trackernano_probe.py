#!/usr/bin/env python3
"""
Ad-hoc TrackerNano sandbox to reproduce reshape failures with synthetic frames.

Usage (from repo root):
    python scripts/trackernano_probe.py --bbox 1115 833 12 8 --backend trackernano
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from anonymizer.tracking.hybrid import _create_visual_tracker


def build_frame(width: int, height: int, channels: int, noise: bool) -> np.ndarray:
    if channels == 1:
        base = np.zeros((height, width), dtype=np.uint8)
    else:
        base = np.zeros((height, width, channels), dtype=np.uint8)
    if noise:
        base[:] = np.random.randint(0, 255, size=base.shape, dtype=np.uint8)
    else:
        base[:] = 127
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic TrackerNano repro harness")
    parser.add_argument("--backend", default="trackernano", help="visual tracker backend to use")
    parser.add_argument("--frame-width", type=int, default=2560)
    parser.add_argument("--frame-height", type=int, default=1440)
    parser.add_argument(
        "--channels", type=int, default=3, choices=(1, 3, 4), help="number of frame channels"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        default=(500, 500, 50, 20),
        help="initial bbox (pixels)",
    )
    parser.add_argument(
        "--shift",
        nargs=2,
        type=int,
        metavar=("DX", "DY"),
        default=(0, 0),
        help="translation applied to bbox before update() to mimic drift",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help="fill frames with random noise instead of flat gray (exercise resize path)",
    )
    args = parser.parse_args()

    bbox = tuple(args.bbox)  # type: ignore[assignment]
    dx, dy = (int(v) for v in args.shift)

    frame_a = build_frame(args.frame_width, args.frame_height, args.channels, args.noise)
    frame_b = frame_a.copy()

    tracker = _create_visual_tracker(args.backend)
    if tracker is None:
        print(f"Backend {args.backend!r} is unavailable in this OpenCV build.", file=sys.stderr)
        return 1

    print(
        f"Frame shape={frame_a.shape}, dtype={frame_a.dtype}, contiguous={frame_a.flags['C_CONTIGUOUS']}"
    )
    print(f"Init bbox={bbox}")

    try:
        tracker.init(frame_a, bbox)
        print("tracker.init() succeeded")
    except cv2.error as exc:
        print(f"tracker.init() failed: {exc}")
        return 2

    shifted_bbox = (bbox[0] + dx, bbox[1] + dy, bbox[2], bbox[3])
    print(f"Update frame shape={frame_b.shape}")
    print(f"Expected bbox after shift={shifted_bbox}")

    try:
        ok, new_bbox = tracker.update(frame_b)
    except cv2.error as exc:
        print(f"tracker.update() raised: {exc}")
        return 3

    print(f"tracker.update() returned ok={ok}, bbox={new_bbox}")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
