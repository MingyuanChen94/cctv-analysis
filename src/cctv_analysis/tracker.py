from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import deque
import torch
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrackState:
    """Track state information"""
    Tentative = 1  # Track is being initialized
    Confirmed = 2  # Track is confirmed after min_hits
    Deleted = 3    # Track is marked for deletion

class Track:
    """Class to store information about each person track."""
    
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], 
                 feature: Optional[torch.Tensor] = None):
        """
        Initialize a new track.
        
        Args:
            track_id: Unique identifier for this track
            bbox: Initial bounding box (x1, y1, x2, y2)
            feature: ReID feature vector for the person
        """
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature
        self.state = TrackState.Tentative
        self.hits = 1  # Number of detections assigned to this track
        self.missed_frames = 0  # Number of frames since last detection
        self.age = 1  # Track age in frames
        self.features_history = deque(maxlen=10)
        if feature is not None:
            self.features_history.append(feature)
        
        # Motion prediction using Kalman filter
        self.kalman = self._initialize_kalman()
        self._update_kalman_state(bbox)
        
    def _initialize_kalman(self) -> KalmanFilter:
        """Initialize Kalman filter for tracking."""
        # State: [x, y, w, h, vx, vy, vw, vh]
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix
        kf.F = np.eye(8)
        kf.F[0, 4] = 1  # x' = x + vx
        kf.F[1, 5] = 1  # y' = y + vy
        kf.F[2, 6] = 1  # w' = w + vw
        kf.F[3, 7] = 1  # h' = h + vh
        
        # Measurement matrix (we only observe position and size)
        kf.H = np.eye(4, 8)
        
        # Measurement noise
        kf.R = np.eye(4) * 10
        
        # Process noise
        kf.Q = np.eye(8)
        kf.Q[4:, 4:] *= 0.1  # Lower noise for velocity components
        
        # Initial state covariance
        kf.P *= 1000
        
        return kf
    
    def _update_kalman_state(self, bbox: Tuple[int, int, int, int]):
        """Update Kalman filter state with new bbox."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        self.kalman.x = np.array([[center_x], [center_y], [w], [h], 
                                 [0], [0], [0], [0]])
    
    def predict(self) -> Tuple[int, int, int, int]:
        """
        Predict next position using Kalman filter.
        
        Returns:
            Predicted bounding box (x1, y1, x2, y2)
        """
        self.kalman.predict()
        center_x = self.kalman.x[0, 0]
        center_y = self.kalman.x[1, 0]
        w = self.kalman.x[2, 0]
        h = self.kalman.x[3, 0]
        
        x1 = center_x - w/2
        y1 = center_y - h/2
        x2 = center_x + w/2
        y2 = center_y + h/2
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def update(self, bbox: Tuple[int, int, int, int], 
               feature: Optional[torch.Tensor] = None):
        """
        Update track with new detection.
        
        Args:
            bbox: New bounding box
            feature: New ReID feature
        """
        self.bbox = bbox
        if feature is not None:
            self.feature = feature
            self.features_history.append(feature)
        
        self.hits += 1
        self.missed_frames = 0
        self.age += 1
        
        # Update Kalman filter
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        measurement = np.array([[center_x], [center_y], [w], [h]])
        self.kalman.update(measurement)
        
        # Update track state
        if self.state == TrackState.Tentative and self.hits >= 3:
            self.state = TrackState.Confirmed

class PersonTracker:
    """Multi-person tracker using Kalman filtering and ReID features."""
    
    def __init__(self, reid_model, max_age: int = 30, min_hits: int = 3, 
                 match_threshold: float = 0.5):
        """
        Initialize tracker.
        
        Args:
            reid_model: ReID model for feature extraction
            max_age: Maximum frames to keep track alive without detection
            min_hits: Minimum detection hits to confirm track
            match_threshold: Threshold for detection-to-track matching
        """
        self.reid_model = reid_model
        self.max_age = max_age
        self.min_hits = min_hits
        self.match_threshold = match_threshold
        
        self.tracks: List[Track] = []
        self.track_id_count = 0
        
    def update(self, frame: np.ndarray, 
              detections: List[Tuple[Tuple[int, int, int, int], float]]) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            frame: Current video frame
            detections: List of detections (bbox, confidence)
            
        Returns:
            List of active tracks
        """
        # Extract features for new detections
        detection_features = []
        if detections:
            patches = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2), _ in detections]
            detection_features = self.reid_model.extract_features(patches)
        
        # Predict new locations of tracks
        predicted_tracks = []
        for track in self.tracks:
            predicted_bbox = track.predict()
            predicted_tracks.append((track, predicted_bbox))
        
        # Match detections to predicted tracks
        matches, unmatched_tracks, unmatched_detections = \
            self._match_detections_to_tracks(predicted_tracks, detections, detection_features)
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det_bbox, _ = detections[det_idx]
            det_feature = detection_features[det_idx]
            track.update(det_bbox, det_feature)
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].missed_frames += 1
        
        # Initialize new tracks
        for det_idx in unmatched_detections:
            self._initiate_track(detections[det_idx], detection_features[det_idx])
        
        # Update track states and remove dead tracks
        self.tracks = [track for track in self.tracks 
                      if track.state != TrackState.Deleted]
        
        # Mark tracks for deletion
        for track in self.tracks:
            if track.missed_frames > self.max_age:
                track.state = TrackState.Deleted
        
        # Return active tracks
        return [track for track in self.tracks 
                if track.state == TrackState.Confirmed]
    
    def _match_detections_to_tracks(self, predicted_tracks: List[Tuple[Track, Tuple]], 
                                  detections: List[Tuple[Tuple[int, int, int, int], float]],
                                  detection_features: torch.Tensor) -> Tuple[List[Tuple[int, int]], 
                                                                          List[int], 
                                                                          List[int]]:
        """
        Match detections to predicted track locations.
        
        Uses a combination of IoU and feature similarity for matching.
        
        Args:
            predicted_tracks: List of (track, predicted_bbox) pairs
            detections: List of (bbox, confidence) pairs
            detection_features: Features for each detection
            
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        if not predicted_tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(predicted_tracks), len(detections)))
        for i, (_, pred_bbox) in enumerate(predicted_tracks):
            for j, (det_bbox, _) in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(pred_bbox, det_bbox)
        
        # Calculate feature similarity matrix
        similarity_matrix = np.zeros((len(predicted_tracks), len(detections)))
        track_features = torch.stack([track.feature for track, _ in predicted_tracks])
        similarities = self.reid_model.compute_similarity(track_features, detection_features)
        similarity_matrix = similarities.cpu().numpy()
        
        # Combine IoU and feature similarity
        cost_matrix = -(0.7 * iou_matrix + 0.3 * similarity_matrix)
        
        # Apply Hungarian algorithm
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches using threshold
        matches = []
        unmatched_tracks = set(range(len(predicted_tracks)))
        unmatched_detections = set(range(len(detections)))
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] > -self.match_threshold:
                continue
            matches.append((track_idx, det_idx))
            unmatched_tracks.remove(track_idx)
            unmatched_detections.remove(det_idx)
        
        return matches, list(unmatched_tracks), list(unmatched_detections)
    
    def _initiate_track(self, detection: Tuple[Tuple[int, int, int, int], float],
                       feature: torch.Tensor):
        """Initialize a new track from detection."""
        bbox, _ = detection
        new_track = Track(self.track_id_count, bbox, feature)
        self.tracks.append(new_track)
        self.track_id_count += 1
    
    @staticmethod
    def _calculate_iou(bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        # Calculate areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
