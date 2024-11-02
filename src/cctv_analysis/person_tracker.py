import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine


class PersonTracker:
    """
    Track persons across frames and match between cameras using feature similarity
    and temporal information.
    """

    def __init__(self, config: dict):
        """
        Initialize the person tracker.

        Args:
            config (dict): Configuration dictionary containing:
                - max_disappeared: Maximum number of frames before track is deleted
                - max_distance: Maximum feature distance for matching
                - min_confidence: Minimum confidence for track initialization
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.max_disappeared = config.get("max_disappeared", 30)
        self.max_distance = config.get("max_distance", 0.6)
        self.min_confidence = config.get("min_confidence", 0.5)

        self.next_track_id = 0
        self.tracks = {}  # Dictionary to store active tracks

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections.

        Args:
            detections (List[Dict]): List of detections from person detector

        Returns:
            List[Dict]: Updated tracks with track IDs
        """
        # Filter detections by confidence
        valid_detections = [
            d for d in detections if d["confidence"] >= self.min_confidence
        ]

        # If no valid detections, update disappeared counters
        if not valid_detections:
            return self._handle_disappeared_tracks()

        # If no existing tracks, initialize new tracks for all detections
        if not self.tracks:
            return self._initialize_new_tracks(valid_detections)

        # Match detections with existing tracks
        (
            matched_tracks,
            unmatched_detections,
            unmatched_tracks,
        ) = self._match_detections_to_tracks(valid_detections)

        # Update matched tracks
        for track_id, detection_idx in matched_tracks:
            self._update_track(track_id, valid_detections[detection_idx])

        # Handle unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id]["disappeared"] += 1

        # Create new tracks for unmatched detections
        for idx in unmatched_detections:
            self._create_new_track(valid_detections[idx])

        # Remove tracks that have disappeared for too long
        self._remove_disappeared_tracks()

        # Return current tracks
        return self._get_active_tracks()

    def match_features(self, features1: np.ndarray, features2: np.ndarray) -> bool:
        """
        Match two feature vectors to determine if they belong to the same person.

        Args:
            features1 (np.ndarray): First feature vector
            features2 (np.ndarray): Second feature vector

        Returns:
            bool: True if features match, False otherwise
        """
        if features1 is None or features2 is None:
            return False

        try:
            distance = cosine(features1, features2)
            return distance <= self.max_distance
        except Exception as e:
            self.logger.error(f"Error matching features: {str(e)}")
            return False

    def _match_detections_to_tracks(self, detections: List[Dict]) -> tuple:
        """
        Match detections to existing tracks using feature similarity.

        Returns:
            tuple: (matched_tracks, unmatched_detections, unmatched_tracks)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(self.tracks.keys())

        # Compute cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track_id in enumerate(self.tracks):
            track = self.tracks[track_id]
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = cosine(track["features"], detection["features"])

        # Apply Hungarian algorithm for optimal assignment
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        # Filter matches by distance threshold
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        for track_idx, det_idx in zip(track_indices, detection_indices):
            if cost_matrix[track_idx, det_idx] <= self.max_distance:
                track_id = list(self.tracks.keys())[track_idx]
                matched_tracks.append((track_id, det_idx))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track_id)

        return matched_tracks, unmatched_detections, unmatched_tracks

    def _initialize_new_tracks(self, detections: List[Dict]) -> List[Dict]:
        """Initialize new tracks for all detections."""
        for detection in detections:
            self._create_new_track(detection)
        return self._get_active_tracks()

    def _create_new_track(self, detection: Dict):
        """Create a new track from detection."""
        self.tracks[self.next_track_id] = {
            "bbox": detection["bbox"],
            "features": detection["features"],
            "confidence": detection["confidence"],
            "disappeared": 0,
        }
        self.next_track_id += 1

    def _update_track(self, track_id: int, detection: Dict):
        """Update existing track with new detection."""
        self.tracks[track_id].update(
            {
                "bbox": detection["bbox"],
                "features": detection["features"],
                "confidence": detection["confidence"],
                "disappeared": 0,
            }
        )

    def _handle_disappeared_tracks(self) -> List[Dict]:
        """Update disappeared counter for all tracks."""
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["disappeared"] += 1
        self._remove_disappeared_tracks()
        return self._get_active_tracks()

    def _remove_disappeared_tracks(self):
        """Remove tracks that have disappeared for too long."""
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]["disappeared"] > self.max_disappeared:
                del self.tracks[track_id]

    def _get_active_tracks(self) -> List[Dict]:
        """Get list of active tracks."""
        return [
            {"track_id": track_id, **track_info}
            for track_id, track_info in self.tracks.items()
            if track_info["disappeared"] == 0
        ]
