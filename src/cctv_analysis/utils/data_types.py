from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json

@dataclass
class PersonDetection:
    """Represents a single person detection in a frame"""
    track_id: int
    frame_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    timestamp: datetime
    confidence: float
    reid_features: Optional[np.ndarray] = None

@dataclass
class PersonMatch:
    """Represents a match between detections across cameras"""
    track_id_cam1: int
    track_id_cam2: int
    first_appearance_cam1: datetime
    first_appearance_cam2: datetime
    similarity_score: float
    transition_time: float  # time difference in seconds

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'track_id_cam1': self.track_id_cam1,
            'track_id_cam2': self.track_id_cam2,
            'first_appearance_cam1': self.first_appearance_cam1.isoformat(),
            'first_appearance_cam2': self.first_appearance_cam2.isoformat(),
            'similarity_score': float(self.similarity_score),
            'transition_time': float(self.transition_time)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PersonMatch':
        """Create instance from dictionary"""
        return cls(
            track_id_cam1=data['track_id_cam1'],
            track_id_cam2=data['track_id_cam2'],
            first_appearance_cam1=datetime.fromisoformat(data['first_appearance_cam1']),
            first_appearance_cam2=datetime.fromisoformat(data['first_appearance_cam2']),
            similarity_score=data['similarity_score'],
            transition_time=data['transition_time']
        )

class DetectionDatabase:
    """Manages detections from multiple cameras"""
    def __init__(self):
        self.detections: Dict[int, List[PersonDetection]] = {1: [], 2: []}
        self.matches: List[PersonMatch] = []
    
    def add_detection(self, camera_id: int, detection: PersonDetection):
        """Add a detection for a specific camera"""
        self.detections[camera_id].append(detection)
    
    def add_match(self, match: PersonMatch):
        """Add a match between cameras"""
        self.matches.append(match)
    
    def get_camera_detections(self, camera_id: int) -> List[PersonDetection]:
        """Get all detections for a specific camera"""
        return self.detections[camera_id]
    
    def get_all_matches(self) -> List[PersonMatch]:
        """Get all matches between cameras"""
        return self.matches

    def save(self, output_dir: Path):
        """Save database to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detections for each camera
        for camera_id in self.detections:
            detections = self.detections[camera_id]
            if not detections:
                continue
                
            # Save detections to NPZ file
            detection_file = output_dir / f"camera_{camera_id}_detections.npz"
            np.savez_compressed(
                detection_file,
                track_ids=np.array([d.track_id for d in detections]),
                frame_ids=np.array([d.frame_id for d in detections]),
                bboxes=np.array([d.bbox for d in detections]),
                timestamps=np.array([d.timestamp.timestamp() for d in detections]),
                confidences=np.array([d.confidence for d in detections]),
                reid_features=np.array([d.reid_features for d in detections])
            )
        
        # Save matches to JSON file
        if self.matches:
            matches_file = output_dir / "camera_matches.json"
            matches_data = [match.to_dict() for match in self.matches]
            with open(matches_file, 'w') as f:
                json.dump(matches_data, f, indent=2)
    
    @classmethod
    def load(cls, input_dir: Path) -> 'DetectionDatabase':
        """Load database from files"""
        db = cls()
        input_dir = Path(input_dir)
        
        # Load detections for each camera
        for camera_id in [1, 2]:
            detection_file = input_dir / f"camera_{camera_id}_detections.npz"
            if not detection_file.exists():
                continue
                
            data = np.load(detection_file)
            for i in range(len(data['track_ids'])):
                detection = PersonDetection(
                    track_id=int(data['track_ids'][i]),
                    frame_id=int(data['frame_ids'][i]),
                    bbox=data['bboxes'][i],
                    timestamp=datetime.fromtimestamp(data['timestamps'][i]),
                    confidence=float(data['confidences'][i]),
                    reid_features=data['reid_features'][i]
                )
                db.add_detection(camera_id, detection)
        
        # Load matches
        matches_file = input_dir / "camera_matches.json"
        if matches_file.exists():
            with open(matches_file, 'r') as f:
                matches_data = json.load(f)
                for match_data in matches_data:
                    match = PersonMatch.from_dict(match_data)
                    db.add_match(match)
        
        return db
