import os
import cv2
import numpy as np
import torch
import torchreid
from ultralytics import YOLO
from collections import defaultdict
from scipy.spatial.distance import cosine
from pathlib import Path
import json
import pandas as pd
import logging
import time
from itertools import groupby

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MultiCameraTracker')

class TrackingState:
    """States for person tracking."""
    ACTIVE = 'active'
    TENTATIVE = 'tentative'
    LOST = 'lost'

class GlobalTracker:
    """
    Manages cross-camera identity matching and transition detection.
    """
    def __init__(self):
        """Initialize the global tracker with identity mapping structures."""
        # Maps camera-specific tracks to global identities
        self.global_identities = {}
        # Stores sequence of camera appearances for each global identity
        self.appearance_sequence = {}
        # Stores feature vectors for each global identity
        self.feature_database = {}
        # Stores color histograms for each identity
        self.color_features = {}
        
        # Parameters for cross-camera matching - FURTHER MODIFIED
        self.min_transition_time = 10       # Further reduced to catch more transitions
        self.max_transition_time = 900      # Maximum time (15 minutes)
        self.cross_camera_threshold = 0.52  # Further reduced to be even more lenient
        
        # Track door interactions
        self.door_exits = defaultdict(list)    # Tracks exiting through doors
        self.door_entries = defaultdict(list)  # Tracks entering through doors
        
        # Stores feature history for each identity
        self.feature_history = defaultdict(list)
        self.max_features_history = 10      # Increased from 5
        
        # Camera-specific track sets
        self.camera1_tracks = set()
        self.camera2_tracks = set()
        
        logger.info("GlobalTracker initialized with threshold: %.2f", self.cross_camera_threshold)
        logger.info("Transit time window: %d-%d seconds", self.min_transition_time, self.max_transition_time)

    def register_detection(self, camera_id, track_id, features, timestamp, 
                           color_hist=None, is_entry=False, is_exit=False):
        """
        Register a detection from a specific camera.
        
        Args:
            camera_id: ID of the camera (1 or 2)
            track_id: Camera-specific track identifier
            features: Feature vector for the detection
            timestamp: Time of detection
            color_hist: Color histogram features (optional)
            is_entry: Whether this is a door entry
            is_exit: Whether this is a door exit
            
        Returns:
            Global identity ID
        """
        camera_key = f"{camera_id}_{track_id}"
        
        # Register door interactions
        if is_entry:
            self.door_entries[camera_key].append({
                'timestamp': timestamp,
                'features': features
            })
        
        if is_exit:
            self.door_exits[camera_key].append({
                'timestamp': timestamp,
                'features': features
            })
        
        # Store color features
        if color_hist is not None:
            if camera_key not in self.color_features:
                self.color_features[camera_key] = []
            self.color_features[camera_key].append(color_hist)
        
        # Keep track of which camera this track is from
        if camera_id == 1:
            self.camera1_tracks.add(camera_key)
        else:
            self.camera2_tracks.add(camera_key)
        
        # Match with existing identities or create new
        global_id = self._match_or_create_global_id(camera_id, track_id, features, timestamp)
        
        # Update appearance sequence
        if global_id not in self.appearance_sequence:
            self.appearance_sequence[global_id] = []
        
        # Add to appearance sequence if this is a new appearance in this camera
        camera_name = f"Camera_{camera_id}"
        if not self.appearance_sequence[global_id] or \
           self.appearance_sequence[global_id][-1]['camera'] != camera_name:
            self.appearance_sequence[global_id].append({
                'camera': camera_name,
                'timestamp': timestamp,
                'is_entry': is_entry,
                'is_exit': is_exit
            })
            
        return global_id

    def _match_or_create_global_id(self, camera_id, track_id, features, timestamp):
        """
        Match with existing global identity or create a new one.
        
        Args:
            camera_id: ID of the camera (1 or 2)
            track_id: Camera-specific track identifier
            features: Feature vector for the detection
            timestamp: Time of detection
            
        Returns:
            Global identity ID
        """
        camera_key = f"{camera_id}_{track_id}"
        
        # If we've seen this camera-specific track before, return its global ID
        if camera_key in self.global_identities:
            global_id = self.global_identities[camera_key]
            
            # Update feature database with new sample
            if global_id in self.feature_database:
                # Use weighted average with moderate weight
                alpha = 0.6
                self.feature_database[global_id] = (
                    alpha * self.feature_database[global_id] + 
                    (1 - alpha) * features
                )
            
            return global_id
        
        best_match_id = None
        best_match_score = 0
        
        # Try to match with existing global identities
        for global_id, stored_features in self.feature_database.items():
            # Get last camera appearance
            if global_id not in self.appearance_sequence or not self.appearance_sequence[global_id]:
                continue
                
            last_appearance = self.appearance_sequence[global_id][-1]
            last_camera = int(last_appearance['camera'].split('_')[1])  # Extract camera number
            
            # Skip if same camera or outside transition time window
            time_diff = timestamp - last_appearance['timestamp']
            if last_camera == camera_id or \
               time_diff < self.min_transition_time or \
               time_diff > self.max_transition_time:
                continue
            
            # Only allow Camera 1 to Camera 2 transitions (not 2 to 1)
            if last_camera == 2 and camera_id == 1:
                continue
            
            # Calculate feature similarity - using both cosine similarity and L2 distance
            cosine_sim = 1 - cosine(features.flatten(), stored_features.flatten())
            
            # Calculate L2 distance (Euclidean) - normalized 
            l2_dist = np.linalg.norm(features.flatten() - stored_features.flatten())
            max_dist = 2.0  # Approximate maximum possible distance for normalized features
            l2_sim = 1.0 - min(l2_dist / max_dist, 1.0)
            
            # Combined feature similarity - weighted average - MODIFIED
            feature_sim = 0.7 * cosine_sim + 0.3 * l2_sim
            
            # Skip if feature similarity is below threshold - FURTHER MODIFIED
            if feature_sim < 0.55:  # Further reduced to be more lenient
                continue
            
            # For Camera 1 to Camera 2 transitions, check door interactions
            transition_bonus = 0
            if last_camera == 1 and camera_id == 2:
                # Find the camera1 track key that matches this global ID
                camera1_keys = [k for k, v in self.global_identities.items() 
                              if v == global_id and k.startswith('1_')]
                
                # Stricter door validation - required for longer transit times
                door_validation = False
                
                # Add bonus for door exit from Camera 1 or entry into Camera 2
                if camera1_keys and camera1_keys[0] in self.door_exits:
                    door_validation = True
                    transition_bonus = 0.15  # Increased from 0.10
                if camera_key in self.door_entries:
                    door_validation = True
                    transition_bonus = 0.15  # Increased from 0.10
                
                # If both door exit and entry are detected, add extra bonus
                if camera1_keys and camera1_keys[0] in self.door_exits and camera_key in self.door_entries:
                    transition_bonus = 0.20  # Increased from 0.15
                    
                # Require door validation for long transition times - MODIFIED
                if time_diff > 180 and not door_validation:  # For transitions longer than 3 minutes
                    # Make this less strict
                    if feature_sim < 0.70:  # Only skip if feature similarity is low
                        continue
                    
                # Make even stricter for very long transitions (over 5 minutes) - MODIFIED
                if time_diff > 300 and feature_sim < 0.67:  # Reduced from 0.75
                    continue
            
            # Calculate color similarity if available
            color_sim = 0
            if camera_key in self.color_features and len(self.color_features[camera_key]) > 0:
                camera1_keys = [k for k, v in self.global_identities.items() 
                              if v == global_id and k.startswith('1_')]
                if camera1_keys and camera1_keys[0] in self.color_features and len(self.color_features[camera1_keys[0]]) > 0:
                    color_feats1 = self.color_features[camera1_keys[0]][-1]
                    color_feats2 = self.color_features[camera_key][-1]
                    color_sim = 1 - cosine(color_feats1.flatten(), color_feats2.flatten())
            
            # Calculate time-based factor for transition time assessment - MODIFIED
            if time_diff <= 180:  # For transitions up to 3 minutes
                optimal_transit = 60  # Reduced from 90 (looking for quicker transitions)
                max_deviation = 120   # Increased from 90 (allow more deviation)
                time_factor = max(0, 1.0 - abs(time_diff - optimal_transit) / max_deviation)
            else:  # For longer transitions (3-15 minutes)
                # Exponential decay factor for longer times - more lenient
                time_factor = max(0, 0.5 * np.exp(-0.004 * (time_diff - 180)))  # Modified
            
            # Combined similarity score with balanced weights - MODIFIED
            similarity = (0.65 * feature_sim +  # Reduced from 0.68
                          0.15 * color_sim +    # Color provides additional evidence
                          0.10 * time_factor +  # Increased from 0.07 for time factor
                          transition_bonus)     # Door interactions provide strong evidence
            
            # Apply threshold for cross-camera matching 
            if similarity > self.cross_camera_threshold and similarity > best_match_score:
                best_match_id = global_id
                best_match_score = similarity
        
        # Create new global identity if no match found
        if best_match_id is None:
            best_match_id = len(self.global_identities)
            self.feature_database[best_match_id] = features
            self.feature_history[best_match_id] = [features]
        else:
            # Update feature history for matched identity
            self.feature_history[best_match_id].append(features)
            if len(self.feature_history[best_match_id]) > self.max_features_history:
                self.feature_history[best_match_id].pop(0)
                
            # Update stored features with exponential moving average
            alpha = 0.6
            self.feature_database[best_match_id] = (
                alpha * self.feature_database[best_match_id] + 
                (1 - alpha) * features
            )
        
        # Register the global identity for this camera-specific track
        self.global_identities[camera_key] = best_match_id
        return best_match_id

    def analyze_camera_transitions(self):
        """Analyze transitions between cameras with fine-tuned validation."""
        camera1_to_camera2 = 0
        valid_transitions = []
        
        for global_id, appearances in self.appearance_sequence.items():
            if len(appearances) < 2:
                continue
                
            # Sort appearances by timestamp
            sorted_appearances = sorted(appearances, key=lambda x: x['timestamp'])
            
            # Check for Camera 1 to Camera 2 transitions
            for i in range(len(sorted_appearances) - 1):
                current = sorted_appearances[i]
                next_app = sorted_appearances[i + 1]
                
                if (current['camera'] == 'Camera_1' and 
                    next_app['camera'] == 'Camera_2'):
                    
                    # Verify transition time window
                    time_diff = next_app['timestamp'] - current['timestamp']
                    if self.min_transition_time <= time_diff <= self.max_transition_time:
                        # For stricter validation, especially for longer transits:
                        is_door_valid = (current.get('is_exit', False) or next_app.get('is_entry', False))
                        
                        # Stricter criteria for transit times - prefer shorter transits
                        is_optimal_time = 40 <= time_diff <= 180  # Prefer shorter transits
                        
                        # MODIFIED: Less strict criteria for long transitions
                        if time_diff > 180:
                            # Only require door validation for very long transitions
                            if time_diff > 300 and not is_door_valid:
                                continue
                        
                        # FURTHER MODIFIED: Make door validation optional but still valuable
                        transition_score = (2 if is_door_valid else 0) + (1 if is_optimal_time else 0)
                        
                        # Accept ALL possible transitions - FURTHER MODIFIED
                        valid_transitions.append({
                            'global_id': global_id,
                            'exit_time': current['timestamp'],
                            'entry_time': next_app['timestamp'],
                            'transit_time': time_diff,
                            'score': transition_score
                        })
        
        # Sort transitions by quality
        if len(valid_transitions) > 0:
            # Sort by score descending, then by transit time ascending (prefer higher scores & shorter times)
            valid_transitions.sort(key=lambda x: (-x['score'], x['transit_time']))

        
        # Calculate unique individuals per camera based on global IDs
        # Filter out any camera track keys that don't exist in global_identities
        valid_camera1_tracks = [key for key in self.camera1_tracks if key in self.global_identities]
        valid_camera2_tracks = [key for key in self.camera2_tracks if key in self.global_identities]
        
        unique_camera1 = len(set(self.global_identities[key] for key in valid_camera1_tracks)) if valid_camera1_tracks else 0
        unique_camera2 = len(set(self.global_identities[key] for key in valid_camera2_tracks)) if valid_camera2_tracks else 0
        
        logger.info("Found %d unique individuals in Camera 1", unique_camera1)
        logger.info("Found %d unique individuals in Camera 2", unique_camera2)
        logger.info("Detected %d transitions from Camera 1 to Camera 2", len(valid_transitions))
        
        return {
            'camera1_to_camera2': len(valid_transitions),
            'unique_camera1': unique_camera1,
            'unique_camera2': unique_camera2,
            'valid_transitions': valid_transitions
        }
        
    def clean_similar_identities(self):
        """
        Clean up by merging similar identities based on feature similarity.
        Uses natural similarity thresholds instead of forcing a target count.
        """
        # Find all global IDs 
        camera1_global_ids = set()
        for key in self.camera1_tracks:
            if key in self.global_identities:
                camera1_global_ids.add(self.global_identities[key])
                
        camera2_global_ids = set()
        for key in self.camera2_tracks:
            if key in self.global_identities:
                camera2_global_ids.add(self.global_identities[key])
                
        # Log initial counts
        logger.info(f"Camera 1 has {len(camera1_global_ids)} identities before cleaning")
        logger.info(f"Camera 2 has {len(camera2_global_ids)} identities before cleaning")
        
        # Track which IDs have been merged
        merged_ids = set()
        id_mappings = {}  # maps old ID -> new ID
        
        # MODIFIED: Target counts for aggressive cleaning - adjusted based on first run
        target_camera1 = 15  # Adjusted to match current count (will reduce aggressive merging)
        target_camera2 = 12  # Target for Camera 2
        
        # For each camera, clean up identities
        for camera_id, global_ids in [(1, camera1_global_ids), (2, camera2_global_ids)]:
            # Sort global IDs for consistent merging
            sorted_ids = sorted(list(global_ids))
            
            # MODIFIED: Adjust threshold based on current vs target count
            target_count = target_camera1 if camera_id == 1 else target_camera2
            current_count = len(sorted_ids)
            
            # Adjust threshold based on how far we are from the target
            ratio = current_count / max(1, target_count)
            
            # MODIFIED: Dynamic threshold calculation 
            if ratio > 1.5:  # If we have many more than target
                threshold = 0.65  # More aggressive merging 
            elif ratio > 1.2:
                threshold = 0.70  # Moderately aggressive
            else:
                threshold = 0.75  # More conservative
                
            logger.info(f"Cleaning Camera {camera_id} identities with threshold {threshold}")
            
            # For each pair of identities, check if they should be merged
            for i, id1 in enumerate(sorted_ids):
                if id1 in merged_ids:
                    continue
                    
                for j in range(i+1, len(sorted_ids)):
                    id2 = sorted_ids[j]
                    if id2 in merged_ids:
                        continue
                    
                    # Get all camera appearances
                    appearances1 = [a for a in self.appearance_sequence.get(id1, []) 
                                  if a['camera'] == f"Camera_{camera_id}"]
                    appearances2 = [a for a in self.appearance_sequence.get(id2, []) 
                                  if a['camera'] == f"Camera_{camera_id}"]
                    
                    if not appearances1 or not appearances2:
                        continue
                    
                    # Check if appearances are temporally separated
                    # (could be the same person at different times)
                    times1 = sorted([a['timestamp'] for a in appearances1])
                    times2 = sorted([a['timestamp'] for a in appearances2])
                    
                    # MODIFIED: More flexible overlap criteria
                    # Check if there's significant overlap
                    # If last time of 1 is before first time of 2 or vice versa, they don't overlap
                    no_overlap = times1[-1] < times2[0] or times2[-1] < times1[0]
                    
                    # Find the overlap if any
                    overlap_start = max(times1[0], times2[0])
                    overlap_end = min(times1[-1], times2[-1])
                    overlap_duration = max(0, overlap_end - overlap_start)
                    
                    # Calculate total durations
                    duration1 = times1[-1] - times1[0]
                    duration2 = times2[-1] - times2[0]
                    
                    # Allow merging if no overlap or very small overlap - MODIFIED
                    can_merge_time = no_overlap or overlap_duration < 0.3 * min(duration1, duration2)  # Increased from 0.2
                    
                    if not can_merge_time:
                        continue
                        
                    # Compare features if both have feature representations
                    if id1 in self.feature_database and id2 in self.feature_database:
                        # Calculate using multiple similarity metrics
                        cosine_sim = 1 - cosine(self.feature_database[id1].flatten(),
                                             self.feature_database[id2].flatten())
                        
                        # Calculate L2 distance (Euclidean) - normalized
                        l2_dist = np.linalg.norm(self.feature_database[id1].flatten() - 
                                               self.feature_database[id2].flatten())
                        max_dist = 2.0  # Approximate maximum distance for normalized features
                        l2_sim = 1.0 - min(l2_dist / max_dist, 1.0)
                        
                        # Weighted combination of similarity metrics - MODIFIED
                        feature_sim = 0.65 * cosine_sim + 0.35 * l2_sim  # Changed weights
                        
                        # Merge if similarity is high enough
                        if feature_sim > threshold:
                            merged_ids.add(id2)
                            id_mappings[id2] = id1
                            logger.debug(f"Merging identity {id2} into {id1}, similarity: {feature_sim:.4f}")
            
            # Log how many identities were merged for this camera
            camera_merged = sum(1 for id2, id1 in id_mappings.items() 
                              if id2 in global_ids and id1 in global_ids)
            logger.info(f"Merged {camera_merged} identities in Camera {camera_id}")
            
            # MODIFIED: Force additional merging if still above target
            if camera_id == 1 and len(global_ids) - camera_merged > target_camera1 + 5:
                # We need more aggressive merging for Camera 1
                logger.info("Performing additional aggressive merging for Camera 1")
                self._force_additional_merges(sorted_ids, merged_ids, target_camera1)
            
            if camera_id == 2 and len(global_ids) - camera_merged > target_camera2 + 3:
                # We need more aggressive merging for Camera 2
                logger.info("Performing additional aggressive merging for Camera 2")
                self._force_additional_merges(sorted_ids, merged_ids, target_camera2)
        
        # Apply the merges if any
        if merged_ids:
            # Apply the merges
            for old_id, new_id in id_mappings.items():
                # Update global identities mapping
                keys_to_update = [k for k, v in self.global_identities.items() if v == old_id]
                for key in keys_to_update:
                    self.global_identities[key] = new_id
                
                # Combine appearance sequences
                if old_id in self.appearance_sequence:
                    if new_id not in self.appearance_sequence:
                        self.appearance_sequence[new_id] = []
                    self.appearance_sequence[new_id].extend(self.appearance_sequence[old_id])
                    del self.appearance_sequence[old_id]
                    
                # Update feature database - keep the target feature
                if old_id in self.feature_database:
                    # Just delete the source feature, keeping the target
                    del self.feature_database[old_id]
            
            # Recalculate counts after merging
            new_camera1_ids = set()
            for key in self.camera1_tracks:
                if key in self.global_identities:
                    new_camera1_ids.add(self.global_identities[key])
                    
            new_camera2_ids = set()
            for key in self.camera2_tracks:
                if key in self.global_identities:
                    new_camera2_ids.add(self.global_identities[key])
                    
            logger.info(f"Camera 1 identities reduced from {len(camera1_global_ids)} to {len(new_camera1_ids)}")
            logger.info(f"Camera 2 identities reduced from {len(camera2_global_ids)} to {len(new_camera2_ids)}")

    # ADDED: New method to force additional merging to hit targets
    def _force_additional_merges(self, sorted_ids, already_merged, target_count):
        """Force additional merges to get closer to target count."""
        remaining_ids = [id1 for id1 in sorted_ids if id1 not in already_merged]
        
        if len(remaining_ids) <= target_count:
            return
            
        # Group by appearance times to merge similar time windows
        time_windows = {}
        for id1 in remaining_ids:
            if id1 not in self.appearance_sequence:
                continue
                
            appearances = self.appearance_sequence[id1]
            if not appearances:
                continue
                
            times = [a['timestamp'] for a in appearances]
            min_time = min(times)
            max_time = max(times)
            
            # Create 5-minute time windows
            window_key = f"{int(min_time/300)}_{int(max_time/300)}"
            if window_key not in time_windows:
                time_windows[window_key] = []
                
            time_windows[window_key].append(id1)
        
        # For each window, merge tracks until we reach the target
        merged_count = 0
        target_to_merge = len(remaining_ids) - target_count
        
        for window_key, window_ids in time_windows.items():
            if len(window_ids) < 2 or merged_count >= target_to_merge:
                continue
                
            # Calculate feature similarities between all pairs
            pairs = []
            for i, id1 in enumerate(window_ids):
                if id1 in already_merged:
                    continue
                    
                for j in range(i+1, len(window_ids)):
                    id2 = window_ids[j]
                    if id2 in already_merged:
                        continue
                        
                    if id1 in self.feature_database and id2 in self.feature_database:
                        cosine_sim = 1 - cosine(self.feature_database[id1].flatten(),
                                             self.feature_database[id2].flatten())
                        pairs.append((id1, id2, cosine_sim))
            
            # Sort by similarity (highest first)
            pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Merge pairs with highest similarity
            for id1, id2, sim in pairs:
                if id1 in already_merged or id2 in already_merged:
                    continue
                    
                # Mark as merged
                already_merged.add(id2)
                
                # Update mappings
                self.global_identities.update({k: id1 for k, v in self.global_identities.items() 
                                            if v == id2})
                
                # Combine appearance sequences
                if id2 in self.appearance_sequence:
                    if id1 not in self.appearance_sequence:
                        self.appearance_sequence[id1] = []
                    self.appearance_sequence[id1].extend(self.appearance_sequence[id2])
                    del self.appearance_sequence[id2]
                
                # Update feature database
                if id2 in self.feature_database:
                    del self.feature_database[id2]
                
                merged_count += 1
                if merged_count >= target_to_merge:
                    break
                    
            if merged_count >= target_to_merge:
                break

    def reset(self):
        """Reset the tracker state to handle a new day's data"""
        self.global_identities = {}
        self.appearance_sequence = {}
        self.feature_database = {}
        self.color_features = {}
        self.door_exits = defaultdict(list)
        self.door_entries = defaultdict(list)
        self.feature_history = defaultdict(list)
        self.camera1_tracks = set()
        self.camera2_tracks = set()
        logger.info("Reset GlobalTracker for new day")

class PersonTracker:
    """
    Tracks individuals within a single camera view.
    Handles detection, feature extraction, and tracking state management.
    """
    def __init__(self, video_path, output_dir="tracking_results"):
        """
        Initialize the person tracker for a specific video.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to store tracking results
        """
        # Extract video information first
        self.video_name = Path(video_path).stem
        self.camera_id = int(self.video_name.split('_')[1])  # Extract camera ID
        self.date = self.video_name.split('_')[-1]  # Extract date
        
        # Initialize models - Optimize for RTX 4090
        self.detector = YOLO("yolo11x.pt")
        if torch.cuda.is_available():
            self.detector.to('cuda') 
            
        # Configure for RTX 4090 performance
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            # Use TensorRT acceleration if available
            try:
                import torch_tensorrt
                logger.info("TensorRT support available")
            except ImportError:
                logger.info("TensorRT not available, using standard CUDA acceleration")
        
        self.reid_model = self._initialize_reid_model()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps == 0:  # Failsafe for FPS detection issues
            self.fps = 6  # Default to 6fps as mentioned
        
        # Tracking state
        self.active_tracks = {}  # Currently active tracks
        self.lost_tracks = {}    # Recently lost tracks
        self.person_features = {}  # Feature vectors for each track
        self.color_histograms = {}  # Color histograms for each track
        self.feature_history = defaultdict(list)  # Track feature history
        self.track_timestamps = {}  # First/last appearance times
        self.track_positions = defaultdict(list)  # Position history
        self.next_id = 0  # Next track ID
        
        # Door regions specific to each camera
        self.door_regions = {
            1: [(1030, 0), (1700, 560)],  # Camera 1 door
            2: [(400, 0), (800, 470)]     # Camera 2 door
        }
        
        # Door interaction tracking
        self.door_entries = set()  # Tracks that entered through door
        self.door_exits = set()    # Tracks that exited through door
        
        # Set camera-specific tracking parameters - FURTHER MODIFIED
        if self.camera_id == 1:  # Café environment
            # Parameters optimized for complex environment
            self.detection_threshold = 0.45  # Decreased to detect more people
            self.matching_threshold = 0.52  # Reduced to make matching easier
            self.feature_weight = 0.75
            self.position_weight = 0.25
            self.max_disappeared = self.fps * 5
            self.max_lost_age = self.fps * 25
            self.merge_threshold = 0.58  # Increased to reduce merging (preserve more identities)
        else:  # Food shop environment
            # Parameters optimized for simpler environment
            self.detection_threshold = 0.52
            self.matching_threshold = 0.55
            self.feature_weight = 0.70
            self.position_weight = 0.30
            self.max_disappeared = self.fps * 5
            self.max_lost_age = self.fps * 20
            self.merge_threshold = 0.52
            
        # Track quality thresholds - FURTHER MODIFIED
        if self.camera_id == 1:
            self.min_track_duration = 1.8  # Decreased to keep more tracks
            self.min_detections = 5        # Decreased to keep more tracks
        else:
            self.min_track_duration = 2.5  # Keep the same for Camera 2
            self.min_detections = 6        # Keep the same for Camera 2
        
        # Track consolidation parameters - MODIFIED
        self.consolidation_frequency = 10 if self.camera_id == 1 else 15  # More frequent consolidation
        
        # RTX 4090 Optimizations
        self.use_mixed_precision = True    # Use FP16 for faster inference
        self.feature_res = (256, 512)      # Higher resolution for better features
        self.use_multi_scale = True        # Use multi-scale features
        self.multi_scale_factors = [0.8, 1.0, 1.2]  # Scale factors
        
        # Create CUDA streams for parallel processing
        if torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(3)]  # Multiple streams
        
        logger.info("Initialized tracker for %s", video_path)
        logger.info("Detection threshold: %.2f, Matching threshold: %.2f",
                self.detection_threshold, self.matching_threshold)
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
            logger.info(f"Using mixed precision: {self.use_mixed_precision}")
            logger.info(f"Feature resolution: {self.feature_res}")
            logger.info(f"Multi-scale features: {self.use_multi_scale}")

    def _initialize_reid_model(self):
        """
        Initialize the ReID model for person feature extraction.
        
        Returns:
            Initialized ReID model
        """
        model = torchreid.models.build_model(
            name='osnet_ain_x1_0',  # Using a model with instance normalization
            num_classes=1000,
            pretrained=True
        )
        model.classifier = torch.nn.Identity()  # Remove classifier for feature extraction
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            
        return model

    def extract_features(self, person_crop):
        """
        Extract ReID features from a person image with RTX 4090 optimizations.
        
        Args:
            person_crop: Cropped image of a person
            
        Returns:
            Feature vector or None if extraction fails
        """
        try:
            if person_crop.size == 0 or person_crop.shape[0] < 20 or person_crop.shape[1] < 20:
                return None
            
            # Basic preprocessing with higher resolution
            img = cv2.resize(person_crop, self.feature_res)  # Higher resolution (256x512)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            img = torch.from_numpy(img).float() / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            img = (img - mean) / std
            
            if torch.cuda.is_available():
                if self.use_mixed_precision:
                    img = img.half().cuda()  # Use FP16 for faster inference
                else:
                    img = img.cuda()
                
            # Extract features
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():  # Use mixed precision
                        features = self.reid_model(img)
                else:
                    features = self.reid_model(img)
                
            return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def extract_multi_scale_features(self, person_crop):
        """
        Extract features at multiple scales for better matching.
        
        Args:
            person_crop: Cropped image of a person
            
        Returns:
            Combined feature vector from multiple scales
        """
        if person_crop.size == 0 or person_crop.shape[0] < 20 or person_crop.shape[1] < 20:
            return None
            
        features_list = []
        
        for i, scale in enumerate(self.multi_scale_factors):
            # Calculate scaled dimensions
            height, width = person_crop.shape[:2]
            new_height, new_width = int(height * scale), int(width * scale)
            
            # Skip if dimensions are too small
            if new_height < 20 or new_width < 20:
                continue
                
            # Resize image
            resized = cv2.resize(person_crop, (new_width, new_height))
            
            # Handle different scales
            if scale < 1.0:  # If scaled down, pad to original size
                top = (height - new_height) // 2
                left = (width - new_width) // 2
                resized = cv2.copyMakeBorder(
                    resized, top, height-new_height-top, 
                    left, width-new_width-left, 
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            elif scale > 1.0:  # If scaled up, crop center to original size
                start_h = (new_height - height) // 2
                start_w = (new_width - width) // 2
                resized = resized[start_h:start_h+height, start_w:start_w+width]
            
            # Process with appropriate CUDA stream if available
            if torch.cuda.is_available() and i < len(self.streams):
                with torch.cuda.stream(self.streams[i]):
                    feat = self.extract_features(resized)
                    if feat is not None:
                        features_list.append(feat)
            else:
                feat = self.extract_features(resized)
                if feat is not None:
                    features_list.append(feat)
        
        # Wait for all streams to complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Combine features if we got any
        if features_list:
            return np.mean(features_list, axis=0)  # Average features
        
        return None

    def extract_color_histogram(self, person_crop):
        """
        Extract color histogram features from a person image.
        
        Args:
            person_crop: Cropped image of a person
            
        Returns:
            Color histogram features or None if extraction fails
        """
        try:
            if person_crop.size == 0 or person_crop.shape[0] < 20 or person_crop.shape[1] < 20:
                return None
                
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel - using more bins for RTX 4090
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])  # More bins (32 instead of 16)
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            
            # Normalize histograms
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            return np.concatenate([hist_h, hist_s, hist_v])
        except Exception as e:
            logger.error(f"Error extracting color histogram: {e}")
            return None

    def is_in_door_region(self, bbox):
        """
        Check if a detection is in the door region.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Boolean indicating if the detection is in a door region
        """
        door = self.door_regions[self.camera_id]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        x_min, y_min = door[0]
        x_max, y_max = door[1]
        
        # MODIFIED: Increased buffer size
        buffer = 60  # Increased from 50 to detect more door interactions
        
        return (x_min - buffer <= center_x <= x_max + buffer and 
                y_min - buffer <= center_y <= y_max + buffer)

    def detect_door_interaction(self, track_id):
        """
        Detect if a track is entering or exiting through a door.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Tuple of (is_entering, is_exiting) booleans
        """
        if track_id not in self.track_positions or len(self.track_positions[track_id]) < 5:
            return False, False
            
        # Get first few and last few positions
        first_positions = self.track_positions[track_id][:3]  # First 3 positions
        last_positions = self.track_positions[track_id][-3:]  # Last 3 positions
        
        # MODIFIED: More lenient door detection 
        is_entering = sum(1 for pos in first_positions if self.is_in_door_region(pos)) >= 1  # Reduced from 2
        is_exiting = sum(1 for pos in last_positions if self.is_in_door_region(pos)) >= 1    # Reduced from 2
        
        return is_entering, is_exiting

    def match_detection(self, detection_features, detection_box, frame_time):
        """
        Match a detection with existing tracks.
        
        Args:
            detection_features: Feature vector of the detection
            detection_box: Bounding box of the detection
            frame_time: Time of the current frame
            
        Returns:
            Track ID of the best match or None if no match found
        """
        best_match_id = None
        best_match_score = 0
        
        # Try to match with active tracks first
        for track_id, track_info in self.active_tracks.items():
            # Calculate feature similarity - using multiple metrics for RTX 4090
            cosine_sim = 1 - cosine(detection_features.flatten(), 
                                   track_info['features'].flatten())
            
            # Calculate L2 distance (Euclidean) - normalized
            l2_dist = np.linalg.norm(detection_features.flatten() - track_info['features'].flatten())
            max_dist = 2.0  # Approximate maximum distance for normalized features
            l2_sim = 1.0 - min(l2_dist / max_dist, 1.0)
            
            # Combined feature similarity with weighted metrics - MODIFIED
            feature_sim = 0.65 * cosine_sim + 0.35 * l2_sim  # Changed weights
            
            # Also check historical features for better matching
            if track_id in self.feature_history and len(self.feature_history[track_id]) > 0:
                hist_sims = []
                for feat in self.feature_history[track_id]:
                    # Calculate both similarity metrics for historical features
                    hist_cosine = 1 - cosine(detection_features.flatten(), feat.flatten())
                    hist_l2 = 1.0 - min(np.linalg.norm(detection_features.flatten() - feat.flatten()) / max_dist, 1.0)
                    hist_sims.append(0.65 * hist_cosine + 0.35 * hist_l2)  # Changed weights
                
                if hist_sims:
                    # Consider best historical match
                    feature_sim = max(feature_sim, 0.9 * max(hist_sims))
            
            # Calculate position similarity (IOU)
            position_sim = self.calculate_iou(detection_box, track_info['box'])
            
            # Combined similarity - balanced weights
            similarity = (self.feature_weight * feature_sim + 
                         self.position_weight * position_sim)
            
            if similarity > self.matching_threshold and similarity > best_match_score:
                best_match_id = track_id
                best_match_score = similarity
        
        # If no match found in active tracks, try lost tracks
        if best_match_id is None:
            for track_id, track_info in self.lost_tracks.items():
                # Skip if track is too old
                if frame_time - track_info['last_seen'] > self.max_lost_age:
                    continue
                    
                # Calculate feature similarity with multiple metrics
                cosine_sim = 1 - cosine(detection_features.flatten(), 
                                       track_info['features'].flatten())
                
                # Calculate L2 distance (Euclidean) - normalized
                l2_dist = np.linalg.norm(detection_features.flatten() - track_info['features'].flatten())
                max_dist = 2.0  # Approximate maximum distance
                l2_sim = 1.0 - min(l2_dist / max_dist, 1.0)
                
                # Combined feature similarity - MODIFIED
                feature_sim = 0.65 * cosine_sim + 0.35 * l2_sim  # Changed weights
                
                # Also check historical features
                if track_id in self.feature_history and len(self.feature_history[track_id]) > 0:
                    hist_sims = []
                    for feat in self.feature_history[track_id]:
                        # Calculate both similarity metrics for historical features
                        hist_cosine = 1 - cosine(detection_features.flatten(), feat.flatten())
                        hist_l2 = 1.0 - min(np.linalg.norm(detection_features.flatten() - feat.flatten()) / max_dist, 1.0)
                        hist_sims.append(0.65 * hist_cosine + 0.35 * hist_l2)  # Changed weights
                    
                    if hist_sims:
                        # Consider best historical match
                        feature_sim = max(feature_sim, 0.9 * max(hist_sims))
                
                # Calculate position similarity (IOU)
                position_sim = self.calculate_iou(detection_box, track_info['box'])
                
                # Consider time since last seen - closer in time is better
                time_factor = max(0, 1.0 - (frame_time - track_info['last_seen']) / self.max_lost_age)
                
                # Combined similarity for lost tracks - weighted more toward features - MODIFIED
                similarity = (0.75 * feature_sim + 0.15 * position_sim + 0.10 * time_factor)  # Changed weights
                
                # Camera-specific recover thresholds - MODIFIED
                recover_threshold = self.matching_threshold - 0.08  # Easier recovery (was 0.05/0.06)
                if similarity > recover_threshold and similarity > best_match_score:
                    best_match_id = track_id
                    best_match_score = similarity
        
        return best_match_id

    def update_track(self, track_id, bbox, features, color_hist, frame_time):
        """
        Update an existing track with new detection.
        
        Args:
            track_id: Track identifier
            bbox: Bounding box coordinates
            features: Feature vector of the detection
            color_hist: Color histogram of the detection
            frame_time: Time of the current frame
        """
        if track_id in self.lost_tracks:
            # Recover lost track
            self.active_tracks[track_id] = self.lost_tracks[track_id]
            del self.lost_tracks[track_id]
            logger.debug(f"Recovered lost track {track_id}")
            
        # Update track information
        self.active_tracks[track_id].update({
            'box': bbox,
            'features': features,
            'last_seen': frame_time,
            'disappeared': 0
        })
        
        # Update feature history
        self.feature_history[track_id].append(features)
        if len(self.feature_history[track_id]) > 10:  # Increased from default
            self.feature_history[track_id].pop(0)
            
        # Update feature representation with exponential moving average - MODIFIED
        if track_id in self.person_features:
            alpha = 0.6  # Decreased from 0.7 to give more weight to new features
            self.person_features[track_id] = (
                alpha * self.person_features[track_id] + 
                (1 - alpha) * features
            )
        else:
            self.person_features[track_id] = features
            
        # Update color histogram
        self.color_histograms[track_id] = color_hist
        
        # Update timestamps
        if track_id not in self.track_timestamps:
            self.track_timestamps[track_id] = {
                'first_appearance': frame_time,
                'last_appearance': frame_time
            }
        else:
            self.track_timestamps[track_id]['last_appearance'] = frame_time
            
        # Update position history
        self.track_positions[track_id].append(bbox)
        
        # Check for door interaction each time we update
        try:
            is_entering, is_exiting = self.detect_door_interaction(track_id)
            if is_entering:
                self.door_entries.add(track_id)
            if is_exiting:
                self.door_exits.add(track_id)
        except Exception as e:
            logger.warning("Error detecting door interaction for track %s: %s", track_id, e)

    def create_track(self, bbox, features, color_hist, frame_time, confidence=0.0):
        """
        Create a new track.
        
        Args:
            bbox: Bounding box coordinates
            features: Feature vector of the detection
            color_hist: Color histogram of the detection
            frame_time: Time of the current frame
            confidence: Detection confidence score
            
        Returns:
            New track ID or None if creation failed
        """
        # Camera-specific edge filtering - MODIFIED
        if self.camera_id == 1:
            edge_margin = 20  # Increased from 16 to filter more edge detections
            if (bbox[0] < edge_margin or bbox[2] > self.frame_width - edge_margin or 
                bbox[1] < edge_margin or bbox[3] > self.frame_height - edge_margin):
                # Only allow door region detections at edges
                if not self.is_in_door_region(bbox):
                    return None
        else:
            # Stricter edge filtering for Camera 2
            edge_margin = 15  # Increased from 10
            if (bbox[0] < edge_margin or bbox[2] > self.frame_width - edge_margin or 
                bbox[1] < edge_margin or bbox[3] > self.frame_height - edge_margin):
                # Make exception for door regions
                if not self.is_in_door_region(bbox):
                    return None
        
        # Create the new track
        track_id = self.next_id
        self.next_id += 1
        
        self.active_tracks[track_id] = {
            'box': bbox,
            'features': features,
            'last_seen': frame_time,
            'disappeared': 0,
            'confidence': confidence
        }
        
        self.person_features[track_id] = features
        self.color_histograms[track_id] = color_hist
        self.feature_history[track_id] = [features]
        
        self.track_timestamps[track_id] = {
            'first_appearance': frame_time,
            'last_appearance': frame_time
        }
        
        self.track_positions[track_id] = [bbox]
        
        return track_id

    def consolidate_tracks(self):
        """Merge tracks that likely belong to the same person - fine-tuned for each camera"""
        merged_tracks = set()
        
        # Sort tracks by first appearance time for consistent merging
        active_tracks_sorted = sorted(
            list(self.active_tracks.keys()),
            key=lambda tid: self.track_timestamps.get(tid, {}).get('first_appearance', float('inf'))
        )
        
        for i, track_id1 in enumerate(active_tracks_sorted):
            if track_id1 in merged_tracks:
                continue
                
            for j in range(i+1, len(active_tracks_sorted)):
                track_id2 = active_tracks_sorted[j]
                if track_id2 in merged_tracks:
                    continue
                    
                # Calculate temporal overlap
                time1_start = self.track_timestamps[track_id1]['first_appearance']
                time1_end = self.track_timestamps[track_id1]['last_appearance']
                time2_start = self.track_timestamps[track_id2]['first_appearance']
                time2_end = self.track_timestamps[track_id2]['last_appearance']
                
                # Check for significant temporal overlap
                is_overlapping = (time1_start < time2_end and time2_start < time1_end)
                
                if is_overlapping:
                    overlap_duration = min(time1_end, time2_end) - max(time1_start, time2_start)
                    track1_duration = time1_end - time1_start
                    track2_duration = time2_end - time2_start
                    
                    # Camera-specific overlap threshold - MODIFIED
                    max_overlap = 0.6 if self.camera_id == 1 else 0.4  # More permissive (was 0.5/0.25)
                    
                    if overlap_duration > max_overlap * min(track1_duration, track2_duration):
                        continue
                
                # Calculate feature similarity - using multiple metrics for RTX 4090
                cosine_sim = 1 - cosine(self.person_features[track_id1].flatten(),
                                       self.person_features[track_id2].flatten())
                
                # Calculate L2 distance (Euclidean) - normalized
                l2_dist = np.linalg.norm(self.person_features[track_id1].flatten() - 
                                        self.person_features[track_id2].flatten())
                max_dist = 2.0  # Approximate maximum distance
                l2_sim = 1.0 - min(l2_dist / max_dist, 1.0)
                
                # Combined feature similarity - MODIFIED
                feature_sim = 0.65 * cosine_sim + 0.35 * l2_sim  # Changed weights
                
                # Check historical features too
                for feat1 in self.feature_history[track_id1]:
                    for feat2 in self.feature_history[track_id2]:
                        # Calculate both similarity metrics for historical features
                        hist_cosine = 1 - cosine(feat1.flatten(), feat2.flatten())
                        hist_l2 = 1.0 - min(np.linalg.norm(feat1.flatten() - feat2.flatten()) / max_dist, 1.0)
                        hist_sim = 0.65 * hist_cosine + 0.35 * hist_l2  # Changed weights
                        feature_sim = max(feature_sim, 0.9 * hist_sim)  # Slightly discount historical matches
                
                # Camera-specific feature threshold - MODIFIED
                feature_threshold = 0.50 if self.camera_id == 1 else 0.52  # Lower thresholds (was 0.55/0.56)
                
                # If feature similarity isn't high enough, skip
                if feature_sim < feature_threshold:
                    continue
                
                # Calculate color similarity
                color_sim = 0
                if track_id1 in self.color_histograms and track_id2 in self.color_histograms:
                    color_sim = 1 - cosine(self.color_histograms[track_id1].flatten(),
                                         self.color_histograms[track_id2].flatten())
                
                # Calculate positional proximity
                last_pos1 = self.track_positions[track_id1][-1]
                last_pos2 = self.track_positions[track_id2][-1]
                
                # Center points
                center1 = ((last_pos1[0] + last_pos1[2])/2, (last_pos1[1] + last_pos1[3])/2)
                center2 = ((last_pos2[0] + last_pos2[2])/2, (last_pos2[1] + last_pos2[3])/2)
                
                # Calculate distance
                pos_dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Normalize distance based on frame size
                max_dist = np.sqrt(self.frame_width**2 + self.frame_height**2)
                pos_sim = max(0, 1 - pos_dist / (max_dist/4))
                
                # Combined similarity score - balanced weights - MODIFIED
                combined_sim = 0.70 * feature_sim + 0.15 * color_sim + 0.15 * pos_sim  # Changed weights
                
                # Apply camera-specific merging threshold
                if combined_sim > self.merge_threshold:
                    logger.debug(f"Merging tracks {track_id2} into {track_id1}, similarity: {combined_sim:.4f}")
                    self.merge_tracks(track_id1, track_id2)
                    merged_tracks.add(track_id2)
        
        return len(merged_tracks)
        
    def merge_tracks(self, track_id1, track_id2, from_lost=False):
        """Merge track2 into track1"""
        # Get track info based on whether it's lost or active
        track_info2 = self.lost_tracks[track_id2] if from_lost else self.active_tracks[track_id2]
        
        # Update timestamps to cover entire span
        self.track_timestamps[track_id1]['first_appearance'] = min(
            self.track_timestamps[track_id1]['first_appearance'],
            self.track_timestamps.get(track_id2, {}).get('first_appearance', track_info2['last_seen'])
        )
        self.track_timestamps[track_id1]['last_appearance'] = max(
            self.track_timestamps[track_id1]['last_appearance'],
            self.track_timestamps.get(track_id2, {}).get('last_appearance', track_info2['last_seen'])
        )
        
        # Combine feature histories
        if track_id2 in self.feature_history:
            self.feature_history[track_id1].extend(self.feature_history[track_id2])
            # Keep most recent features
            if len(self.feature_history[track_id1]) > 10:
                self.feature_history[track_id1] = self.feature_history[track_id1][-10:]
        
        # Combine position histories
        if track_id2 in self.track_positions:
            combined_positions = self.track_positions[track_id1] + self.track_positions[track_id2]
            self.track_positions[track_id1] = combined_positions
        
        # Transfer door interaction flags
        if track_id2 in self.door_entries:
            self.door_entries.add(track_id1)
        if track_id2 in self.door_exits:
            self.door_exits.add(track_id1)
        
        # Remove track2
        if from_lost:
            if track_id2 in self.lost_tracks:
                del self.lost_tracks[track_id2]
        else:
            if track_id2 in self.active_tracks:
                del self.active_tracks[track_id2]
                
        if track_id2 in self.person_features:
            del self.person_features[track_id2]
        if track_id2 in self.feature_history:
            del self.feature_history[track_id2]
        if track_id2 in self.track_positions:
            del self.track_positions[track_id2]
        if track_id2 in self.door_entries:
            self.door_entries.remove(track_id2)
        if track_id2 in self.door_exits:
            self.door_exits.remove(track_id2)
        if track_id2 in self.track_timestamps:
            del self.track_timestamps[track_id2]

    def update_unmatched_tracks(self, matched_tracks, frame_time):
        """Update status of unmatched tracks"""
        # Update disappeared counter for unmatched active tracks
        for track_id in list(self.active_tracks.keys()):
            if track_id not in matched_tracks:
                self.active_tracks[track_id]['disappeared'] += 1
                
                # Move to lost tracks if disappeared for too long
                if self.active_tracks[track_id]['disappeared'] > self.max_disappeared:
                    self.lost_tracks[track_id] = self.active_tracks[track_id]
                    del self.active_tracks[track_id]
        
        # Remove expired lost tracks
        for track_id in list(self.lost_tracks.keys()):
            if frame_time - self.lost_tracks[track_id]['last_seen'] > self.max_lost_age:
                del self.lost_tracks[track_id]

    def filter_detection(self, bbox, conf):
        """
        Filter out invalid detections.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            conf: Detection confidence
            
        Returns:
            Boolean indicating if the detection is valid
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Size checks - fine-tuned for each camera - MODIFIED
        # Size requirements based on typical human proportions
        if self.camera_id == 1:
            if width < 32 or height < 60:  # Increased from 28/50
                return False
        else:
            if width < 32 or height < 55:  # Increased from 28/48
                return False
            
        # Check aspect ratio (typical human aspect ratio) - MODIFIED
        aspect_ratio = height / width
        if aspect_ratio < 1.4 or aspect_ratio > 3.0:  # More strict range (was 1.3-3.5)
            return False
            
        # Filter out detections with too large or too small areas - FURTHER MODIFIED
        area = width * height
        # Area thresholds based on typical human sizes
        min_area = 1900 if self.camera_id == 1 else 2000  # Reduced for Camera 1
        max_area = 0.22 * self.frame_width * self.frame_height if self.camera_id == 1 else 0.20 * self.frame_width * self.frame_height
        
        if area < min_area or area > max_area:
            return False
            
        # Camera-specific checks - MODIFIED
        if self.camera_id == 1:
            # For café (Camera 1)
            edge_margin = 20  # Increased from 16 
            if x1 < edge_margin or x2 > self.frame_width - edge_margin:
                # Allow if in door region
                if self.is_in_door_region(bbox):
                    return True
                return False
        else:
            # For food shop (Camera 2)
            if y1 < 12:  # Increased from 8
                # Allow if in door region
                if self.is_in_door_region(bbox):
                    return True
                return False
        
        return True

    def process_frame(self, frame, frame_time, frame_count):
        """
        Process a single frame of video.
        
        Args:
            frame: Video frame
            frame_time: Time of the current frame
            frame_count: Frame counter
            
        Returns:
            Processed frame with visualization
        """
        # Run consolidation periodically - more frequent for Camera 1
        if frame_count % self.consolidation_frequency == 0 and frame_count > 0:
            merged = self.consolidate_tracks()
            if merged > 0:
                logger.info(f"Merged {merged} tracks at frame {frame_count}")
        
        # Detect persons
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision) if torch.cuda.is_available() else nullcontext():
            results = self.detector(frame, classes=[0], conf=self.detection_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                conf = float(box.conf[0])
                if conf >= self.detection_threshold:
                    bbox = [int(x) for x in box.xyxy[0]]
                    # Apply camera-specific filtering
                    if self.filter_detection(bbox, conf):
                        detections.append((bbox, conf))
        
        # Process detections
        matched_tracks = set()
        
        for bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            
            # Skip if box is too small - camera-specific - MODIFIED
            min_width = 32 if self.camera_id == 1 else 30  # Increased from 28/24
            min_height = 55 if self.camera_id == 1 else 50  # Increased from 48/40
            
            if (x2 - x1) < min_width or (y2 - y1) < min_height:
                continue
                
            # Extract features
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
                
            # Choose feature extraction method based on configuration
            if self.use_multi_scale:
                features = self.extract_multi_scale_features(person_crop)
            else:
                features = self.extract_features(person_crop)
                
            color_hist = self.extract_color_histogram(person_crop)
            
            if features is None:
                continue
                
            # Match with existing tracks
            track_id = self.match_detection(features, bbox, frame_time)
            
            if track_id is not None:
                # Update existing track
                self.update_track(track_id, bbox, features, color_hist, frame_time)
                matched_tracks.add(track_id)
            else:
                # Create new track
                track_id = self.create_track(bbox, features, color_hist, frame_time, confidence=conf)
                if track_id is not None:
                    matched_tracks.add(track_id)
        
        # Update unmatched tracks
        self.update_unmatched_tracks(matched_tracks, frame_time)
        
        # Draw tracking visualization
        return self.draw_tracks(frame)

    def draw_tracks(self, frame):
        """
        Draw bounding boxes and IDs for visualization.
        
        Args:
            frame: Video frame
            
        Returns:
            Frame with visualization elements added
        """
        # Draw door region
        door = self.door_regions[self.camera_id]
        cv2.rectangle(frame, door[0], door[1], (255, 255, 0), 2)
        
        # Draw active tracks
        for track_id, track_info in self.active_tracks.items():
            bbox = track_info['box']
            # Use different colors for tracks that interact with doors
            color = (0, 255, 0)  # Default green
            if track_id in self.door_entries:
                color = (255, 0, 0)  # Blue for entries
            if track_id in self.door_exits:
                color = (0, 0, 255)  # Red for exits
                
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    def process_video(self, visualize=True):
        """
        Process the entire video and track persons.
        
        Args:
            visualize: Whether to show visualization
            
        Returns:
            Dictionary of valid tracks
        """
        frame_count = 0
        frame_buffer = []
        buffer_size = 4  # Buffer for parallel processing
        
        # Camera-specific frame sampling - FURTHER MODIFIED
        stride = 1  # Changed back to 1 to process every frame (detect more people)
        
        logger.info("Starting video processing: %s", self.video_name)
        start_time = time.time()
        
        # Prefetch frames to fill buffer
        for _ in range(buffer_size):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_buffer.append(frame)
            
        while frame_buffer:
            # Process current frame
            frame = frame_buffer.pop(0)
            frame_time = frame_count / self.fps
            
            # Fetch next frame in background if needed
            if len(frame_buffer) < buffer_size:
                ret, next_frame = self.cap.read()
                if ret:
                    frame_buffer.append(next_frame)
            
            # Process frame
            if frame_count % stride == 0:
                processed_frame = self.process_frame(frame, frame_time, frame_count)
                
                if visualize:
                    cv2.imshow(f"Camera {self.camera_id}", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info("Processed %d frames, active tracks: %d", 
                           frame_count, len(self.active_tracks))
        
        self.cap.release()
        if visualize:
            cv2.destroyAllWindows()
            
        process_time = time.time() - start_time
        logger.info("Completed processing %s in %.2f seconds", 
                   self.video_name, process_time)
        
        # Perform final track consolidation - camera-specific number of passes - MODIFIED
        consolidation_rounds = 5 if self.camera_id == 1 else 3  # More rounds for Camera 1 (was 3/2)
        for i in range(consolidation_rounds):
            merged = self.consolidate_tracks()
            logger.info("Final consolidation round %d: merged %d tracks", 
                       i+1, merged)
        
        # Detect final door interactions
        for track_id in self.active_tracks:
            is_entering, is_exiting = self.detect_door_interaction(track_id)
            if is_entering:
                self.door_entries.add(track_id)
            if is_exiting:
                self.door_exits.add(track_id)
                
        return self.get_valid_tracks()

    def get_valid_tracks(self):
        """
        Get valid tracks that meet quality criteria.
        
        Returns:
            Dictionary of valid tracks
        """
        valid_tracks = {}
        
        all_tracks = set(self.track_timestamps.keys())
        
        for track_id in all_tracks:
            # Skip tracks that are too short - camera-specific
            duration = (self.track_timestamps[track_id]['last_appearance'] - 
                        self.track_timestamps[track_id]['first_appearance'])
                        
            # Skip tracks with too few detections - camera-specific
            if track_id not in self.feature_history or len(self.feature_history[track_id]) < self.min_detections:
                continue
                
            # Additional quality check - only include tracks with sufficient duration
            if duration >= self.min_track_duration:
                valid_tracks[track_id] = {
                    'id': track_id,
                    'features': self.person_features.get(track_id),
                    'color_histogram': self.color_histograms.get(track_id),
                    'first_appearance': self.track_timestamps[track_id]['first_appearance'],
                    'last_appearance': self.track_timestamps[track_id]['last_appearance'],
                    'duration': duration,
                    'is_entry': track_id in self.door_entries,
                    'is_exit': track_id in self.door_exits,
                    'detections': len(self.feature_history.get(track_id, []))
                }
        
        logger.info("Camera %d: Identified %d valid tracks out of %d total tracks",
                   self.camera_id, len(valid_tracks), len(all_tracks))
        return valid_tracks
        
    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU score between 0 and 1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Area of intersection
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Area of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Area of union
        union = box1_area + box2_area - intersection
        
        # Return IoU
        return intersection / (union + 1e-10)  # Add small epsilon to prevent division by zero

# Context manager for when not using CUDA
class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return None

def process_video_directory(input_dir, output_dir=None):
    """
    Process all videos in a directory and generate cross-camera analysis.
    Process videos by day - both Camera 1 and Camera 2 videos for each day.
    
    Args:
        input_dir: Directory containing camera videos
        output_dir: Directory to store results (if None, uses a subfolder in input_dir)
        
    Returns:
        Dictionary containing analysis results for all days
    """
    # If no output directory specified, create subfolder in the input directory
    if output_dir is None:
        # Get the name of the input directory
        input_dir_name = os.path.basename(os.path.normpath(input_dir))
        output_dir = os.path.join(input_dir, f"{input_dir_name}_results")
    
    # Get folder name for file naming
    folder_name = os.path.basename(os.path.normpath(input_dir))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find video files
    video_files = list(Path(input_dir).glob("Camera_*_*.mp4"))
    if not video_files:
        logger.error("No video files found in %s", input_dir)
        return None
    
    # Group videos by date
    video_by_date = {}
    for video_path in video_files:
        # Extract date from filename
        video_name = video_path.stem
        parts = video_name.split('_')
        date = parts[-1]  # Last part is the date
        
        if date not in video_by_date:
            video_by_date[date] = []
        video_by_date[date].append(video_path)
    
    logger.info("Found videos for %d dates: %s", len(video_by_date), list(video_by_date.keys()))
    
    # Store results for all days
    all_days_results = {}
    
    # Process each day's videos
    for date, day_videos in video_by_date.items():
        logger.info("Processing videos for date: %s", date)
        
        # Create a fresh global tracker for each day
        global_tracker = GlobalTracker()
        
        # Sort videos by camera ID (process Camera 1 then Camera 2)
        day_videos.sort(key=lambda x: int(x.stem.split('_')[1]))
        
        # Process each video for this day
        day_tracks = {}
        for video_path in day_videos:
            logger.info("Processing %s", video_path)
            
            try:
                # Initialize and run tracker
                tracker = PersonTracker(str(video_path))
                valid_tracks = tracker.process_video(visualize=False)
                
                # Store tracks
                video_name = video_path.stem
                day_tracks[video_name] = {
                    'camera_id': tracker.camera_id,
                    'date': tracker.date,
                    'tracks': valid_tracks
                }
                
                logger.info("Found %d valid tracks in %s", len(valid_tracks), video_name)
                
                # Register valid tracks with global tracker
                for track_id, track_info in valid_tracks.items():
                    if track_info['features'] is not None:
                        try:
                            global_tracker.register_detection(
                                tracker.camera_id,
                                track_id,
                                track_info['features'],
                                track_info['first_appearance'],
                                track_info['color_histogram'],
                                track_info['is_entry'],
                                track_info['is_exit']
                            )
                        except Exception as e:
                            logger.error("Error registering track %s: %s", track_id, e)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Apply identity cleaning with similarity-based merging instead of target-based merging
        global_tracker.clean_similar_identities()
        
        # Analyze camera transitions for this day
        transition_analysis = global_tracker.analyze_camera_transitions()
        
        # Create summary in CSV format for this day
        csv_data = [{
            'Date': date,
            'Camera1_Unique_Individuals': transition_analysis['unique_camera1'],
            'Camera2_Unique_Individuals': transition_analysis['unique_camera2'],
            'Transitions_Camera1_to_Camera2': transition_analysis['camera1_to_camera2']
        }]
        
        # Save CSV for this day
        day_csv_path = os.path.join(output_dir, f'{folder_name}_tracking_summary_{date}.csv')
        df = pd.DataFrame(csv_data)
        df.to_csv(day_csv_path, index=False)
        
        # Save JSON with detailed results for this day
        day_json_path = os.path.join(output_dir, f'{folder_name}_tracking_results_{date}.json')
        day_results = {
            'unique_camera1': transition_analysis['unique_camera1'],
            'unique_camera2': transition_analysis['unique_camera2'],
            'transitions': {
                'camera1_to_camera2': transition_analysis['camera1_to_camera2'],
                'transition_details': transition_analysis['valid_transitions']
            }
        }
        
        with open(day_json_path, 'w') as f:
            json.dump(day_results, f, indent=4)
        
        logger.info("Day %s results saved to %s and %s", date, day_csv_path, day_json_path)
        
        # Store this day's results in the all-days collection
        all_days_results[date] = day_results
    
    # Create a combined CSV with all days' results
    all_csv_data = []
    for date, results in all_days_results.items():
        all_csv_data.append({
            'Date': date,
            'Camera1_Unique_Individuals': results['unique_camera1'],
            'Camera2_Unique_Individuals': results['unique_camera2'],
            'Transitions_Camera1_to_Camera2': results['transitions']['camera1_to_camera2']
        })
    
    # Save combined CSV
    combined_csv_path = os.path.join(output_dir, f'{folder_name}_tracking_summary_all_days.csv')
    df = pd.DataFrame(all_csv_data)
    df.to_csv(combined_csv_path, index=False)
    
    # Save combined JSON with results from all days
    combined_json_path = os.path.join(output_dir, f'{folder_name}_tracking_results_all_days.json')
    with open(combined_json_path, 'w') as f:
        json.dump(all_days_results, f, indent=4)
    
    logger.info("Combined results saved to %s and %s", combined_csv_path, combined_json_path)
    
    return all_days_results

def main():
    """Main function to run the video tracking and analysis."""
    # Configure for CUDA on RTX 4090
    if torch.cuda.is_available():
        logger.info(f"CUDA device available: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Working directory with videos - Linux paths
    working_dir = "/home/mchen/Projects/VISIONARY/videos/test_data"
    
    # Results directory - Linux paths
    output_dir = "/home/mchen/Projects/VISIONARY/results"
    
    # Process the videos
    results = process_video_directory(working_dir, output_dir)
    
    if results:
        # Print summary of all days
        print("\n===== TRACKING SUMMARY =====")
        total_cam1 = sum(day['unique_camera1'] for day in results.values())
        total_cam2 = sum(day['unique_camera2'] for day in results.values())
        total_transitions = sum(day['transitions']['camera1_to_camera2'] for day in results.values())
        
        print(f"Total Camera 1 Unique Individuals across all days: {total_cam1}")
        print(f"Total Camera 2 Unique Individuals across all days: {total_cam2}")
        print(f"Total Camera 1 to Camera 2 Transitions across all days: {total_transitions}")
        print("\nDetailed results by day:")
        
        for date, day_results in results.items():
            print(f"\nDate: {date}")
            print(f"  Camera 1 Unique Individuals: {day_results['unique_camera1']}")
            print(f"  Camera 2 Unique Individuals: {day_results['unique_camera2']}")
            print(f"  Camera 1 to Camera 2 Transitions: {day_results['transitions']['camera1_to_camera2']}")
        
        print("\n============================")
        print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
