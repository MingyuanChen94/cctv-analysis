import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model paths and parameters."""
    detector_model: str
    reid_model: str
    gender_model: str
    age_model: str
    
    detector_confidence: float
    matching_threshold: float
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ModelConfig':
        return cls(
            detector_model=config['models']['detector']['path'],
            reid_model=config['models']['reid']['path'],
            gender_model=config['models']['demographics']['gender_path'],
            age_model=config['models']['demographics']['age_path'],
            detector_confidence=config['models']['detector']['confidence_threshold'],
            matching_threshold=config['models']['reid']['matching_threshold']
        )

@dataclass
class TrackingConfig:
    """Configuration for tracking parameters."""
    max_age: int
    min_hits: int
    max_time_gap: int  # seconds
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TrackingConfig':
        return cls(
            max_age=config['tracking']['max_age'],
            min_hits=config['tracking']['min_hits'],
