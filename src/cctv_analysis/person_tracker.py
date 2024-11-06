import torch
import numpy as np
from typing import List, Tuple
from datetime import datetime, timedelta

from cctv_analysis.utils.config import Config
from cctv_analysis.utils.models import ModelManager
from cctv_analysis.utils.data_types import PersonDetection
from cctv_analysis.utils.logger import setup_logger

class PersonTracker:
    """Handles person detection, tracking, and re-identification"""
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("PersonTracker")
        self.model_manager = ModelManager(config)
    
    def process_batch(self, batch: torch.Tensor,
                     frame_ids: List[int],
                     fps: float) -> List[PersonDetection]:
        """Process a batch of frames for detection and tracking"""
        detections = []
        
        # Run detection
        outputs = self.model_manager.detect(batch)
        
        # Process each frame's detections
        for frame_idx, output in zip(frame_ids, outputs):
            if output is None:
                continue
            
            # Get timestamp
            timestamp = self._frame_id_to_timestamp(frame_idx, fps)
            
            # Run tracking
            online_targets = self.model_manager.track(
                output,
                self.config.detector.input_size
            )
            
            # Process tracked objects
            for t in online_targets:
                bbox = self._tlwh_to_xyxy(t.tlwh)
                
                # Extract ReID features
                reid_features = self.model_manager.extract_reid_features(
                    batch[frame_ids.index(frame_idx)],
                    bbox
                )
                
                detection = PersonDetection(
                    track_id=t.track_id,
                    frame_id=frame_idx,
                    bbox=bbox,
                    timestamp=timestamp,
                    confidence=t.score,
                    reid_features=reid_features.cpu().numpy()
                )
                detections.append(detection)
        
        return detections

    def match_tracks(self, tracks1: List[PersonDetection],
                    tracks2: List[PersonDetection]) -> List[Tuple[int, int, float]]:
        """Match tracks between two cameras based on ReID features"""
        if not tracks1 or not tracks2:
            return []
        
        # Group tracks by ID
        tracks1_dict = self._group_tracks_by_id(tracks1)
        tracks2_dict = self._group_tracks_by_id(tracks2)
        
        matches = []
        for track_id1, detections1 in tracks1_dict.items():
            features1 = self._get_average_features(detections1)
            earliest_time1 = min(d.timestamp for d in detections1)
            
            for track_id2, detections2 in tracks2_dict.items():
                features2 = self._get_average_features(detections2)
                earliest_time2 = min(d.timestamp for d in detections2)
                
                # Check time difference
                time_diff = abs((earliest_time2 - earliest_time1).total_seconds())
                if time_diff > self.config.matching.max_time_difference:
                    continue
                
                # Calculate similarity
                similarity = self._compute_cosine_similarity(features1, features2)
                
                # Check if match criteria are met
                if similarity > 1 - self.config.matching.max_cosine_distance:
                    matches.append((track_id1, track_id2, similarity))
        
        return matches

    def _frame_id_to_timestamp(self, frame_id: int, fps: float) -> datetime:
        """Convert frame ID to timestamp"""
        seconds = frame_id / fps
        return datetime.now().replace(microsecond=0) + timedelta(seconds=seconds)

    def _tlwh_to_xyxy(self, bbox: np.ndarray) -> np.ndarray:
        """Convert bbox from [top, left, width, height] to [x1, y1, x2, y2]"""
        return np.array([
            bbox[0],
            bbox[1],
            bbox[0] + bbox[2],
            bbox[1] + bbox[3]
        ])

    def _group_tracks_by_id(self, tracks: List[PersonDetection]) -> dict:
        """Group track detections by track ID"""
        grouped = {}
        for track in tracks:
            if track.track_id not in grouped:
                grouped[track.track_id] = []
            grouped[track.track_id].append(track)
        return grouped

    def _get_average_features(self, detections: List[PersonDetection]) -> np.ndarray:
        """Calculate average ReID features for a track"""
        features = [d.reid_features for d in detections]
        return np.mean(features, axis=0)

    def _compute_cosine_similarity(self, features1: np.ndarray,
                                 features2: np.ndarray) -> float:
        """Compute cosine similarity between feature vectors"""
        return np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2)
        )
