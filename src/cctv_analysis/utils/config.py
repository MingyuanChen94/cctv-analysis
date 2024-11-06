from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

# Define root project directory
PROJECT_ROOT = Path(__file__).parents[3]  # Go up 3 levels from utils/config.py

@dataclass
class DetectorConfig:
    """YOLOX detector configuration"""
    exp_file: str = "yolox_l"
    weights: Path = PROJECT_ROOT / "models" / "detector" / "yolox_l.pth"
    confidence_threshold: float = 0.5
    input_size: Tuple[int, int] = (608, 1088)  # (height, width)

@dataclass
class TrackerConfig:
    """ByteTracker configuration"""
    track_buffer: int = 30
    track_thresh: float = 0.5
    match_thresh: float = 0.5

@dataclass
class ReIDConfig:
    """ReID model configuration"""
    model: str = 'osnet_x1_0'
    weights: Path = PROJECT_ROOT / "models" / "reid" / "osnet_x1_0_market.pth"
    input_size: Tuple[int, int] = (256, 128)  # (height, width)

@dataclass
class ProcessingConfig:
    """Video processing configuration"""
    batch_size: int = 4
    device: str = 'cuda'
    data_dir: Path = PROJECT_ROOT / "data" / "videos"
    output_dir: Path = PROJECT_ROOT / "output"
    log_dir: Path = PROJECT_ROOT / "logs"

@dataclass
class MatchingConfig:
    """Person matching configuration"""
    max_cosine_distance: float = 0.3
    max_time_difference: int = 300  # maximum time difference in seconds

@dataclass
class Config:
    """Main configuration class"""
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    reid: ReIDConfig = ReIDConfig()
    processing: ProcessingConfig = ProcessingConfig()
    matching: MatchingConfig = MatchingConfig()
    
    def __post_init__(self):
        """Ensure all required directories exist"""
        # Create required directories
        for dir_path in [
            self.processing.data_dir,
            self.processing.output_dir,
            self.processing.log_dir,
            self.detector.weights.parent,
            self.reid.weights.parent
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def validate(self):
        """Validate configuration settings"""
        # Check if model files exist
        if not self.detector.weights.exists():
            raise FileNotFoundError(
                f"Detector weights not found at: {self.detector.weights}"
            )
        if not self.reid.weights.exists():
            raise FileNotFoundError(
                f"ReID model weights not found at: {self.reid.weights}"
            )
