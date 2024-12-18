from typing import List, Dict, Tuple, Set
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PersonMatch:
    """Class to store person matching information across cameras."""
    global_id: int
    camera_ids: Set[int]
    track_ids: Dict[int, int]  # camera_id -> track_id
    last_seen: Dict[int, datetime]  # camera_id -> timestamp
    features: Dict[int, torch.Tensor]  # camera_id -> feature vector

class CameraPersonMatcher:
    """Match person identities across multiple cameras."""
    
    def __init__(self, reid_model, matching_threshold: float = 0.7, 
                 max_time_gap: timedelta = timedelta(minutes=5)):
        """
        Initialize the cross-camera matcher.
        
        Args:
            reid_model: ReID model for feature comparison
            matching_threshold: Threshold for feature similarity
            max_time_gap: Maximum time gap for matching across cameras
        """
        self.reid_model = reid_model
        self.matching_threshold = matching_threshold
        self.max_time_gap = max_time_gap
        
        self.global_id_counter = 0
        self.person_matches: List[PersonMatch] = []
        self.camera_tracks: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        
    def update(self, camera_id: int, track_id: int, feature: torch.Tensor, 
               timestamp: datetime) -> int:
        """
        Update matcher with new track information.
        
        Args:
            camera_id: ID of the camera
            track_id: ID of the track in this camera
            feature: ReID feature vector
            timestamp: Current timestamp
            
        Returns:
            Global person ID
        """
        # Check if track already exists in this camera
        if track_id in self.camera_tracks[camera_id]:
            # Update existing track's feature
            self.camera_tracks[camera_id][track_id] = feature
            
            # Find and update existing match
            for match in self.person_matches:
                if camera_id in match.track_ids and match.track_ids[camera_id] == track_id:
                    match.features[camera_id] = feature
                    match.last_seen[camera_id] = timestamp
                    return match.global_id
        
        # If track is new, try to match with existing persons
        best_match = None
        best_similarity = -1
        
        feature = feature.unsqueeze(0)  # Add batch dimension
        
        for match in self.person_matches:
            # Skip if person was recently seen in this camera
            if camera_id in match.last_seen:
                time_diff = timestamp - match.last_seen[camera_id]
                if time_diff < self.max_time_gap:
                    continue
            
            # Compare features with all cameras where person was seen
            for cam_id, feat in match.features.items():
                if cam_id != camera_id:
                    similarity = self.reid_model.compute_similarity(feature, feat.unsqueeze(0))
                    if similarity.item() > best_similarity:
                        best_similarity = similarity.item()
                        best_match = match
        
        # If good match found, update existing person
        if best_match is not None and best_similarity > self.matching_threshold:
            best_match.camera_ids.add(camera_id)
            best_match.track_ids[camera_id] = track_id
            best_match.last_seen[camera_id] = timestamp
            best_match.features[camera_id] = feature.squeeze(0)
            self.camera_tracks[camera_id][track_id] = feature.squeeze(0)
            return best_match.global_id
        
        # If no match found, create new person
        new_match = PersonMatch(
            global_id=self.global_id_counter,
            camera_ids={camera_id},
            track_ids={camera_id: track_id},
            last_seen={camera_id: timestamp},
            features={camera_id: feature.squeeze(0)}
        )
        self.person_matches.append(new_match)
        self.camera_tracks[camera_id][track_id] = feature.squeeze(0)
        self.global_id_counter += 1
        
        return new_match.global_id
    
    def get_person_cameras(self, global_id: int) -> Set[int]:
        """Get all cameras where a person has been seen."""
        for match in self.person_matches:
            if match.global_id == global_id:
                return match.camera_ids
        return set()
    
    def get_person_trajectory(self, global_id: int) -> List[Tuple[int, datetime]]:
        """Get the trajectory of a person across cameras."""
        for match in self.person_matches:
            if match.global_id == global_id:
                trajectory = [(cam_id, timestamp) 
                            for cam_id, timestamp in match.last_seen.items()]
                return sorted(trajectory, key=lambda x: x[1])
        return []
    
    def clean_old_matches(self, current_time: datetime):
        """Remove old matches based on time threshold."""
        self.person_matches = [
            match for match in self.person_matches
            if any((current_time - timestamp) <= self.max_time_gap 
                  for timestamp in match.last_seen.values())
        ]
