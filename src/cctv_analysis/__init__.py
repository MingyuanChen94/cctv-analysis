from .camera_processor import CameraProcessor
from .demographic_analyzer import DemographicAnalyzer
from .person_detector import PersonDetector
from .person_tracker import PersonTracker
from .utils import (
    VideoWriter,
    calculate_metrics,
    load_config,
    prepare_report,
    save_results,
    setup_logging,
    visualize_metrics,
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "CameraProcessor",
    "PersonDetector",
    "PersonTracker",
    "DemographicAnalyzer",
    "setup_logging",
    "load_config",
    "save_results",
    "VideoWriter",
    "calculate_metrics",
    "visualize_metrics",
    "prepare_report",
]
