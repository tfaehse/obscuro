from .core import BaseDetector
from .sahi import SahiDetector


def Detector(*args, **kwargs):
    """
    Detection module providing object detection with ONNX models.

    Public API:
    - Detector: Factory function for creating detectors
    - SahiDetector: Detector with SAHI (Slicing Aided Hyper Inference) support
    """
    return SahiDetector(*args, **kwargs)


__all__ = ["BaseDetector", "Detector", "SahiDetector"]
