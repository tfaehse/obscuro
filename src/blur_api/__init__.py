"""
API server for dashcam anonymization service.

This module provides a FastAPI-based REST API for anonymizing
dashcam videos and images.
"""

from .serve import start_server

__all__ = ["start_server"]
