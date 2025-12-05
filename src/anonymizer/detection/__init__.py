from .core import BaseDetector, FrameDetector
from .sahi import SahiDetector


def Detector(*args, **kwargs):
    """
    Detection module providing object detection with ONNX models.

    Public API:
    - Detector: Factory function for creating detectors (preserves backwards compatibility)
    - FrameDetector: Base detector for frame-by-frame inference
    - SahiDetector: Detector with SAHI (Slicing Aided Hyper Inference) support
    """
    use_sahi = kwargs.pop("use_sahi", False)
    overlap = kwargs.pop("sahi_overlap_ratio", 0.2)
    if use_sahi:
        kwargs["sahi_overlap_ratio"] = overlap
        return SahiDetector(*args, **kwargs)
    return FrameDetector(*args, **kwargs)


__all__ = ["BaseDetector", "Detector", "FrameDetector", "SahiDetector"]
