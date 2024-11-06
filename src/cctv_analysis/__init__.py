"""CCTV Analysis Package

A package for analyzing CCTV footage with multi-camera tracking capabilities.
"""

__version__ = "0.1.0"

from cctv_analysis.analysis import CCTVAnalysis
from cctv_analysis.camera_processor import CameraProcessor
from cctv_analysis.person_tracker import PersonTracker

__all__ = [
    "CCTVAnalysis",
    "CameraProcessor",
    "PersonTracker",
]
