from typing import NewType, Tuple, Dict, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# Type definitions for core components
BoundingBox = NewType('BoundingBox', Tuple[int, int, int, int])  # x1, y1, x2, y2
Detection = NewType('Detection', Tuple[BoundingBox, float])  # bbox, confidence
TrackID = NewType('TrackID', int)
CameraID = NewType('CameraID', int)
GlobalID = NewType('GlobalID', int)

@dataclass
class Frame:
    """Representation of a video frame with metadata."""
    image: np.ndarray
    timestamp: datetime
    camera_id: CameraID
    frame_number: int

@dataclass
class Track:
    """Track information for a detected person."""
    track_id: TrackID
    bbox: BoundingBox
    feature: np.ndarray
    confidence: float
    last_seen: datetime
    history: List[Tuple[BoundingBox, datetime]]

@dataclass
class GlobalTrack:
    """Global track information across multiple cameras."""
    global_id: GlobalID
    camera_tracks: Dict[CameraID, TrackID]
    first_seen: Dict[CameraID, datetime]
    last_seen: Dict[CameraID, datetime]
    features: Dict[CameraID, np.ndarray]

@dataclass
class DemographicInfo:
    """Demographic information for a tracked person."""
    gender: str
    age_group: str
    confidence: Dict[str, float]

@dataclass
class TrackingResult:
    """Complete tracking result for a frame."""
    frame: Frame
    detections: List[Detection]
    tracks: List[Track]
    global_tracks: List[GlobalTrack]
    demographics: Dict[TrackID, DemographicInfo]

# Custom exception types
class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class DetectionError(ModelError):
    """Exception raised for errors in the detection process."""
    pass

class TrackingError(ModelError):
    """Exception raised for errors in the tracking process."""
    pass

class ReIDError(ModelError):
    """Exception raised for errors in the re-identification process."""
    pass

class ConfigError(Exception):
    """Exception raised for configuration-related errors."""
    pass

# Enumerated types
from enum import Enum, auto

class DetectorBackend(Enum):
    """Supported detection model backends."""
    YOLOV8 = auto()
    YOLOX = auto()
    DETR = auto()

class ReIDBackend(Enum):
    """Supported ReID model backends."""
    OSNET = auto()
    TORCHREID = auto()

class TrackingMode(Enum):
    """Available tracking modes."""
    SINGLE_CAMERA = auto()
    MULTI_CAMERA = auto()

class FeatureType(Enum):
    """Types of features used for tracking."""
    APPEARANCE = auto()
    MOTION = auto()
    HYBRID = auto()

class AgeGroup(Enum):
    """Age group categories."""
    CHILD = "child"         # 0-12
    TEENAGER = "teenager"   # 13-19
    YOUNG_ADULT = "young_adult"  # 20-39
    ADULT = "adult"        # 40-59
    SENIOR = "senior"      # 60+
    UNKNOWN = "unknown"

class Gender(Enum):
    """Gender categories."""
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"
