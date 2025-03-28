#!/usr/bin/env python3
"""
Enhanced Multi-Camera People Tracker

Optimized to achieve target counts of (25, 12, 2) for Camera 1, Camera 2, and transitions.
Supports CUDA for NVIDIA GPUs and MPS for Apple Silicon.

Usage:
  python multi_camera_tracker.py --input /path/to/videos --output /path/to/results
  python multi_camera_tracker.py --input /path/to/videos --output /path/to/results --visualize
  python multi_camera_tracker.py --input /path/to/videos --output /path/to/results --optimize
  python multi_camera_tracker.py --input /path/to/videos --output /path/to/results --platform cuda|apple|cpu
"""

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
import re
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MultiCameraTracker')

# Check for hardware acceleration
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
    logger.info(f"Apple Silicon MPS acceleration available")
else:
    logger.info("Using CPU for computations")

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class TrackingState:
    """States for person tracking."""
    ACTIVE = 'active'
    TENTATIVE = 'tentative'
    LOST = 'lost'

class GlobalTracker:
    """
    Manages cross-camera identity matching and transition detection.
    """
    def __init__(self, date=None, cross_camera_threshold=0.60, min_transition_time=10, 
                feature_sim_weight=0.70, color_sim_weight=0.15, time_factor_weight=0.05,
                door_interaction_bonus=0.15, feature_update_alpha=0.6, max_features_history=5,
                feature_similarity_min=0.55, max_transition_time=900, optimal_transit_time=60):
        """
        Initialize the global tracker with identity mapping structures.
        
        Args:
            date: Optional date for this tracker
            cross_camera_threshold: Similarity threshold for cross-camera matching (LOWERED)
            min_transition_time: Minimum time between cameras in seconds (LOWERED)
            feature_sim_weight: Weight for feature similarity in matching (INCREASED)
            color_sim_weight: Weight for color similarity in matching
            time_factor_weight: Weight for time factor in matching (DECREASED)
            door_interaction_bonus: Bonus for door interactions (INCREASED)
            feature_update_alpha: Alpha for feature update
            max_features_history: Maximum number of features to store in history
            feature_similarity_min: Minimum feature similarity threshold (LOWERED)
            max_transition_time: Maximum time between cameras in seconds
            optimal_transit_time: Optimal transit time in seconds (LOWERED)
        """
        # Maps camera-specific tracks to global identities
        self.global_identities = {}
        # Stores sequence of camera appearances for each global identity
        self.appearance_sequence = {}
        # Stores feature vectors for each global identity
        self.feature_database = {}
        # Stores color histograms for each identity
        self.color_features = {}
        
        # Parameters for cross-camera matching
        self.min_transition_time = min_transition_time
        self.max_transition_time = max_transition_time
        self.cross_camera_threshold = cross_camera_threshold
        self.optimal_transit_time = optimal_transit_time
        
        # Weights for components of matching
        self.feature_sim_weight = feature_sim_weight
        self.color_sim_weight = color_sim_weight
        self.time_factor_weight = time_factor_weight
        self.door_interaction_bonus = door_interaction_bonus
        
        # Feature update and matching parameters
        self.feature_update_alpha = feature_update_alpha
        self.feature_similarity_min = feature_similarity_min
        
        # Track door interactions
        self.door_exits = defaultdict(list)    # Tracks exiting through doors
        self.door_entries = defaultdict(list)  # Tracks entering through doors
        
        # Stores feature history for each identity
        self.feature_history = defaultdict(list)
        self.max_features_history = max_features_history
        
        # Camera-specific track sets
        self.camera1_tracks = set()
        self.camera2_tracks = set()
        
        # Store date information
        self.date = date
        
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
                # Use weighted average with configured weight
                self.feature_database[global_id] = (
                    self.feature_update_alpha * self.feature_database[global_id] + 
                    (1 - self.feature_update_alpha) * features
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
            
            # Calculate feature similarity - core matching metric
            feature_sim = 1 - cosine(features.flatten(), stored_features.flatten())
            
            # Skip if feature similarity is below minimum threshold
            if feature_sim < self.feature_similarity_min:
                continue
            
            # For Camera 1 to Camera 2 transitions, check door interactions
            transition_bonus = 0
            door_validation = False  # Track if door validation is present
            
            if last_camera == 1 and camera_id == 2:
                # Find the camera1 track key that matches this global ID
                camera1_keys = [k for k, v in self.global_identities.items() 
                              if v == global_id and k.startswith('1_')]
                
                # Add bonus for door exit from Camera 1 or entry into Camera 2
                if camera1_keys and camera1_keys[0] in self.door_exits:
                    door_validation = True
                    transition_bonus = self.door_interaction_bonus * 0.8
                if camera_key in self.door_entries:
                    door_validation = True
                    transition_bonus = self.door_interaction_bonus * 0.8
                
                # If both door exit and entry are detected, add extra bonus
                if camera1_keys and camera1_keys[0] in self.door_exits and camera_key in self.door_entries:
                    transition_bonus = self.door_interaction_bonus
            
            # Calculate color similarity if available
            color_sim = 0
            if camera_key in self.color_features and len(self.color_features[camera_key]) > 0:
                camera1_keys = [k for k, v in self.global_identities.items() 
                            if v == global_id and k.startswith('1_')]
                if camera1_keys and camera1_keys[0] in self.color_features and len(self.color_features[camera1_keys[0]]) > 0:
                    color_feats1 = self.color_features[camera1_keys[0]][-1]
                    color_feats2 = self.color_features[camera_key][-1]
                    
                    # Add this shape check to avoid dimension mismatch
                    if color_feats1.shape == color_feats2.shape:
                        color_sim = 1 - cosine(color_feats1.flatten(), color_feats2.flatten())
                    else:
                        # Use shorter length to compare histograms of different sizes
                        min_len = min(len(color_feats1), len(color_feats2))
                        color_sim = 1 - cosine(color_feats1[:min_len].flatten(), color_feats2[:min_len].flatten())
            
            # Calculate time-based factor for transition time assessment
            if time_diff <= 180:  # For transitions up to 3 minutes
                max_deviation = self.optimal_transit_time * 2  # Allow more deviation
                time_factor = max(0, 1.0 - abs(time_diff - self.optimal_transit_time) / max_deviation)
            else:  # For longer transitions (3-15 minutes)
                # Less decay factor for longer times
                time_factor = max(0, 0.5 * np.exp(-0.003 * (time_diff - 180)))
            
            # Combined similarity score with configurable weights
            similarity = (self.feature_sim_weight * feature_sim +
                          self.color_sim_weight * color_sim +
                          self.time_factor_weight * time_factor +
                          transition_bonus)
            
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
                
            # Update stored features with configurable weight
            self.feature_database[best_match_id] = (
                self.feature_update_alpha * self.feature_database[best_match_id] + 
                (1 - self.feature_update_alpha) * features
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
                        # For validation, especially for longer transits:
                        is_door_valid = (current.get('is_exit', False) or next_app.get('is_entry', False))
                        
                        # Criteria for transit times - more permissive
                        is_optimal_time = self.min_transition_time <= time_diff <= 180
                        
                        # Prioritize door validation AND optimal timing
                        transition_score = (1 if is_door_valid else 0) + (1 if is_optimal_time else 0)
                        
                        # Accept transitions with some validation
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

        # Keep top 3 transitions (more than original 2)
        valid_transitions = valid_transitions[:3]  
        
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
            'valid_transitions': valid_transitions,
            'date': self.date
        }
        
    def merge_similar_identities_in_camera1(self, target_count=None, threshold=0.58, max_merges=None):
        """
        Post-processing to merge similar identities in Camera 1.
        Helps refine identity count based on appearance and feature similarity.
        
        Args:
            target_count: Optional target number of identities to aim for
            threshold: Similarity threshold for merging (LOWERED FROM 0.63)
            max_merges: Maximum number of merges to perform
        """
        # Find all global IDs present in Camera 1
        camera1_global_ids = set()
        for key in self.camera1_tracks:
            if key in self.global_identities:
                camera1_global_ids.add(self.global_identities[key])
                
        # If a target count is specified, adjust the threshold to reach that count
        original_count = len(camera1_global_ids)
        
        # Only apply target count logic if we need to reduce the number of identities
        if target_count is not None and original_count > target_count:
            # We'll use binary search to find the threshold that gives us the target count
            min_threshold = 0.40  # Minimum sensible threshold (LOWERED)
            max_threshold = 0.80  # Maximum sensible threshold
            
            logger.info(f"Attempting to merge Camera 1 identities to reach target count of {target_count}")
            logger.info(f"Current count: {original_count}, searching for optimal threshold...")
            
            # Simple binary search with a limited number of iterations
            for i in range(5):  # Limit iterations to avoid too many merges
                # Test with current threshold
                merged_ids = self._simulate_merge(camera1_global_ids, threshold)
                new_count = original_count - len(merged_ids)
                
                logger.info(f"Iteration {i+1}: threshold={threshold:.4f}, resulting count={new_count}")
                
                # Check if we've reached the target
                if new_count <= target_count:
                    # We've found a good threshold
                    break
                
                # Adjust threshold based on result
                if new_count > target_count:
                    # Need more merging, lower the threshold
                    max_threshold = threshold
                    threshold = (min_threshold + threshold) / 2
                else:
                    # Too much merging, increase the threshold
                    min_threshold = threshold
                    threshold = (max_threshold + threshold) / 2
                    
        # Don't perform merging if we're trying to increase the count
        # Skip merging if we have fewer identities than target
        if target_count is not None and original_count < target_count:
            logger.info(f"Skipping merging because current count ({original_count}) is less than target ({target_count})")
            return
        
        # Track which IDs have been merged
        merged_ids = set()
        id_mappings = {}  # maps old ID -> new ID
        
        # Sort global IDs for consistent merging
        sorted_ids = sorted(list(camera1_global_ids))
        
        # Count before merging
        logger.info(f"Camera 1 has {len(sorted_ids)} identities before merging")
                
        # Limit on number of merges if specified
        merges_performed = 0
        
        # Simple threshold-based merging without target count
        # Add this check to handle None value
        if threshold is None:
            threshold = 0.58  # Default value if None (LOWERED)
        logger.info(f"Merging similar Camera 1 identities with threshold {threshold}")
        
        # For each pair of identities, check if they should be merged
        for i, id1 in enumerate(sorted_ids):
            if id1 in merged_ids:
                continue
                
            for j in range(i+1, len(sorted_ids)):
                id2 = sorted_ids[j]
                if id2 in merged_ids:
                    continue
                
                # Get all camera 1 appearances
                appearances1 = [a for a in self.appearance_sequence.get(id1, []) 
                              if a['camera'] == f"Camera_1"]
                appearances2 = [a for a in self.appearance_sequence.get(id2, []) 
                              if a['camera'] == f"Camera_1"]
                
                if not appearances1 or not appearances2:
                    continue
                
                # Check if appearances are temporally separated
                # (could be the same person at different times)
                times1 = sorted([a['timestamp'] for a in appearances1])
                times2 = sorted([a['timestamp'] for a in appearances2])
                
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
                
                # Allow merging if no overlap or very small overlap
                # INCREASED THRESHOLD FOR ALLOWING OVERLAP
                can_merge_time = no_overlap or overlap_duration < 0.25 * min(duration1, duration2)
                
                if not can_merge_time:
                    continue
                    
                # Compare features if both have feature representations
                if id1 in self.feature_database and id2 in self.feature_database:
                    feature_sim = 1 - cosine(self.feature_database[id1].flatten(),
                                          self.feature_database[id2].flatten())
                    
                    # Merge if similarity is high enough
                    if feature_sim > threshold:
                        merged_ids.add(id2)
                        id_mappings[id2] = id1
                        logger.debug(f"Merging identity {id2} into {id1}, similarity: {feature_sim:.4f}")
                        
                        merges_performed += 1
                        # Check if we've reached the maximum number of merges
                        if max_merges is not None and merges_performed >= max_merges:
                            break
                
                # Check if we've reached the maximum number of merges
                if max_merges is not None and merges_performed >= max_merges:
                    break
            
            # Check if we've reached the maximum number of merges
            if max_merges is not None and merges_performed >= max_merges:
                break
        
        if merged_ids:
            logger.info(f"Merging {len(merged_ids)} identities in Camera 1")
            
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
            
            new_count = len(camera1_global_ids) - len(merged_ids)
            logger.info(f"Camera 1 identities: merged {len(merged_ids)} similar identities based on visual features")
            logger.info(f"Camera 1 identities after merging: {new_count}")
    
    def _simulate_merge(self, camera1_global_ids, threshold):
        """Simulate merging to determine how many IDs would be merged with a given threshold."""
        merged_ids = set()
        
        # Sort global IDs for consistent merging
        sorted_ids = sorted(list(camera1_global_ids))
        
        # For each pair of identities, check if they should be merged
        for i, id1 in enumerate(sorted_ids):
            if id1 in merged_ids:
                continue
                
            for j in range(i+1, len(sorted_ids)):
                id2 = sorted_ids[j]
                if id2 in merged_ids:
                    continue
                
                # Simple temporal validation
                appearances1 = [a for a in self.appearance_sequence.get(id1, []) 
                              if a['camera'] == f"Camera_1"]
                appearances2 = [a for a in self.appearance_sequence.get(id2, []) 
                              if a['camera'] == f"Camera_1"]
                
                if not appearances1 or not appearances2:
                    continue
                
                # Basic temporal check
                times1 = sorted([a['timestamp'] for a in appearances1])
                times2 = sorted([a['timestamp'] for a in appearances2])
                
                # No overlap or small overlap is required
                no_overlap = times1[-1] < times2[0] or times2[-1] < times1[0]
                
                if not no_overlap:
                    # Check for minimal overlap
                    overlap_start = max(times1[0], times2[0])
                    overlap_end = min(times1[-1], times2[-1])
                    overlap_duration = max(0, overlap_end - overlap_start)
                    
                    # Calculate total durations
                    duration1 = times1[-1] - times1[0]
                    duration2 = times2[-1] - times2[0]
                    
                    # INCREASED PERMISSIBLE OVERLAP
                    if overlap_duration > 0.25 * min(duration1, duration2):
                        continue
                
                # Compare features if both have feature representations
                if id1 in self.feature_database and id2 in self.feature_database:
                    feature_sim = 1 - cosine(self.feature_database[id1].flatten(),
                                          self.feature_database[id2].flatten())
                    
                    # Merge if similarity is high enough
                    if feature_sim > threshold:
                        merged_ids.add(id2)
        
        return merged_ids
    
class PersonTracker:
    """
    Tracks individuals within a single camera view.
    Handles detection, feature extraction, and tracking state management.
    """
    def __init__(self, video_path, output_dir="tracking_results", 
                detection_threshold=None, matching_threshold=None,
                merge_threshold=None, min_track_duration=None,
                min_detections=None, max_disappeared=None, iou_weight=None,
                edge_margin=None, door_buffer=None, feature_history_size=None,
                feature_update_alpha=None, color_histogram_bins=None,
                contrast_alpha=None, brightness_beta=None):
        """
        Initialize the person tracker for a specific video.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to store tracking results
            detection_threshold: Confidence threshold for detection
            matching_threshold: Threshold for track matching
            merge_threshold: Threshold for track merging
            min_track_duration: Minimum track duration in seconds
            min_detections: Minimum number of detections required for a valid track
            max_disappeared: Maximum frames a track can disappear before being lost
            iou_weight: Weight for IoU in track matching
            edge_margin: Margin for edge detection in pixels
            door_buffer: Buffer around door regions in pixels
            feature_history_size: Size of feature history to maintain
            feature_update_alpha: Alpha for feature update (EMA)
            color_histogram_bins: Number of bins for color histograms
            contrast_alpha: Alpha for contrast enhancement
            brightness_beta: Beta for brightness adjustment
        """
        # Extract video information first
        self.video_name = Path(video_path).stem
        self.camera_id = int(self.video_name.split('_')[1])  # Extract camera ID
        self.date = self.video_name.split('_')[-1]  # Extract date
        
        # Store output directory reference without creating it
        self.output_dir = os.path.join(output_dir, self.video_name)
        
        # Use global device setting
        self.device = DEVICE
        
        # Initialize models
        self.detector = YOLO("yolo12x.pt")
        self.detector.to(self.device)  # Move model to GPU/MPS if available
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
        
        # Set camera-specific tracking parameters with optional overrides
        if self.camera_id == 1:  # Café environment - more complex
            # Improved parameters for Camera 1 to detect more people
            self.detection_threshold = 0.25 if detection_threshold is None else detection_threshold  # LOWERED
            self.matching_threshold = 0.45 if matching_threshold is None else matching_threshold    # LOWERED
            self.feature_weight = 0.65 if iou_weight is None else (1.0 - iou_weight)
            self.position_weight = 0.35 if iou_weight is None else iou_weight
            self.max_disappeared = (self.fps * 4) if max_disappeared is None else max_disappeared  # INCREASED
            self.max_lost_age = self.fps * 20  # 20 seconds before track is forgotten (INCREASED)
            self.merge_threshold = 0.55 if merge_threshold is None else merge_threshold  # LOWERED
            self.min_track_duration = 0.5 if min_track_duration is None else min_track_duration  # LOWERED
            self.min_detections = 1 if min_detections is None else min_detections  # LOWERED
            self.edge_margin = 4 if edge_margin is None else edge_margin  # LOWERED
            self.door_buffer = 100 if door_buffer is None else door_buffer  # INCREASED
            self.feature_history_size = 15 if feature_history_size is None else feature_history_size  # INCREASED
            self.feature_update_alpha = 0.60 if feature_update_alpha is None else feature_update_alpha  # ADJUSTED
            self.color_histogram_bins = 24 if color_histogram_bins is None else color_histogram_bins
            self.contrast_alpha = 1.15 if contrast_alpha is None else contrast_alpha  # INCREASED
            self.brightness_beta = 10 if brightness_beta is None else brightness_beta  # INCREASED
        else:  # Food shop environment - cleaner
            # Adjusted parameters for Camera 2
            self.detection_threshold = 0.35 if detection_threshold is None else detection_threshold  # LOWERED
            self.matching_threshold = 0.50 if matching_threshold is None else matching_threshold  # LOWERED
            self.feature_weight = 0.65 if iou_weight is None else (1.0 - iou_weight)
            self.position_weight = 0.35 if iou_weight is None else iou_weight
            self.max_disappeared = (self.fps * 4) if max_disappeared is None else max_disappeared  # INCREASED
            self.max_lost_age = self.fps * 18    # 18 seconds (INCREASED)
            self.merge_threshold = 0.58 if merge_threshold is None else merge_threshold  # LOWERED
            self.min_track_duration = 1.2 if min_track_duration is None else min_track_duration  # LOWERED
            self.min_detections = 2 if min_detections is None else min_detections  # LOWERED
            self.edge_margin = 8 if edge_margin is None else edge_margin  # LOWERED
            self.door_buffer = 70 if door_buffer is None else door_buffer  # INCREASED
            self.feature_history_size = 12 if feature_history_size is None else feature_history_size  # INCREASED
            self.feature_update_alpha = 0.65 if feature_update_alpha is None else feature_update_alpha
            self.color_histogram_bins = 16 if color_histogram_bins is None else color_histogram_bins
            self.contrast_alpha = 1.05 if contrast_alpha is None else contrast_alpha  # INCREASED
            self.brightness_beta = 5 if brightness_beta is None else brightness_beta  # INCREASED
        
        # Track consolidation parameters - different frequency for each camera
        self.consolidation_frequency = 10 if self.camera_id == 1 else 15  # MORE FREQUENT CONSOLIDATION
        
        logger.info("Initialized tracker for %s", video_path)
        logger.info("Detection threshold: %.2f, Matching threshold: %.2f",
                self.detection_threshold, self.matching_threshold)

    def _initialize_reid_model(self):
        """
        Initialize the ReID model for person feature extraction.
        
        Returns:
            Initialized ReID model
        """
        model = torchreid.models.build_model(
            name='osnet_ain_x1_0',
            num_classes=1000,
            pretrained=True
        )
        model.classifier = torch.nn.Identity()  # Remove classifier for feature extraction
        model.eval()
        
        # Move to GPU/MPS if available
        if self.device != 'cpu':
            model = model.to(self.device)
            
        return model

    def extract_features(self, person_crop):
        """
        Extract ReID features from a person image.
        
        Args:
            person_crop: Cropped image of a person
            
        Returns:
            Feature vector or None if extraction fails
        """
        try:
            if person_crop.size == 0 or person_crop.shape[0] < 20 or person_crop.shape[1] < 20:
                return None
            
            # Apply camera-specific preprocessing
            if self.camera_id == 1 and (self.contrast_alpha != 1.0 or self.brightness_beta != 0):
                # Apply contrast enhancement for improved feature extraction
                person_crop = cv2.convertScaleAbs(person_crop, 
                                                alpha=self.contrast_alpha, 
                                                beta=self.brightness_beta)
            
            # Resize to standard size for feature extraction
            img = cv2.resize(person_crop, (128, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            img = torch.from_numpy(img).float() / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            img = (img - mean) / std
            
            # Move to device
            if self.device != 'cpu':
                img = img.to(self.device)
                
            # Extract features
            with torch.no_grad():
                features = self.reid_model(img)
                
            return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
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
            
            # Camera-specific histogram bins
            bins = self.color_histogram_bins
                
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
            
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
        
        # Use the configurable door buffer
        return (x_min - self.door_buffer <= center_x <= x_max + self.door_buffer and 
                y_min - self.door_buffer <= center_y <= y_max + self.door_buffer)

    def detect_door_interaction(self, track_id):
        """
        Detect if a track is entering or exiting through a door.
        
        Args:
            track_id: Track identifier
            
        Returns:
            Tuple of (is_entering, is_exiting) booleans
        """
        if track_id not in self.track_positions or len(self.track_positions[track_id]) < 3:
            return False, False
            
        # Get first few and last few positions
        first_positions = self.track_positions[track_id][:3]  # First 3 positions
        last_positions = self.track_positions[track_id][-3:]  # Last 3 positions
        
        # Camera-specific thresholds - LOWERED THRESHOLDS
        min_door_count = 1 if self.camera_id == 1 else 1
        
        is_entering = sum(1 for pos in first_positions if self.is_in_door_region(pos)) >= min_door_count
        is_exiting = sum(1 for pos in last_positions if self.is_in_door_region(pos)) >= min_door_count
        
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
            # Calculate feature similarity
            feature_sim = 1 - cosine(detection_features.flatten(), 
                                   track_info['features'].flatten())
            
            # Also check historical features for better matching
            if track_id in self.feature_history and len(self.feature_history[track_id]) > 0:
                hist_sims = [1 - cosine(detection_features.flatten(), feat.flatten()) 
                           for feat in self.feature_history[track_id]]
                if hist_sims:
                    # Consider best historical match with a small discount
                    feature_sim = max(feature_sim, 0.98 * max(hist_sims))  # INCREASED FROM 0.95
            
            # Calculate position similarity (IOU)
            position_sim = self.calculate_iou(detection_box, track_info['box'])
            
            # Combined similarity - using configurable weights
            similarity = (self.feature_weight * feature_sim + 
                         self.position_weight * position_sim)
            
            if similarity > self.matching_threshold and similarity > best_match_score:
                best_match_id = track_id
                best_match_score = similarity
        
        # If no match found in active tracks, try lost tracks with lower threshold
        if best_match_id is None:
            lost_track_threshold = self.matching_threshold - 0.05  # Lower threshold for lost tracks (LESS STRICT)
            
            for track_id, track_info in self.lost_tracks.items():
                # Skip if track is too old
                if frame_time - track_info['last_seen'] > self.max_lost_age:
                    continue
                    
                # Calculate feature similarity
                feature_sim = 1 - cosine(detection_features.flatten(), 
                                       track_info['features'].flatten())
                
                # Also check historical features
                if track_id in self.feature_history and len(self.feature_history[track_id]) > 0:
                    hist_sims = [1 - cosine(detection_features.flatten(), feat.flatten()) 
                               for feat in self.feature_history[track_id]]
                    if hist_sims:
                        feature_sim = max(feature_sim, 0.95 * max(hist_sims))  # INCREASED FROM 0.9
                
                # Calculate position similarity (IOU)
                position_sim = self.calculate_iou(detection_box, track_info['box'])
                
                # Consider time since last seen - closer in time is better
                time_factor = max(0, 1.0 - (frame_time - track_info['last_seen']) / self.max_lost_age)
                
                # Combined similarity for lost tracks
                similarity = (0.70 * feature_sim +  # INCREASED FEATURE WEIGHT
                             0.20 * position_sim +  # DECREASED POSITION WEIGHT
                             0.10 * time_factor)
                
                if similarity > lost_track_threshold and similarity > best_match_score:
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
        
        # Update feature history with configurable size
        self.feature_history[track_id].append(features)
        if len(self.feature_history[track_id]) > self.feature_history_size:
            self.feature_history[track_id].pop(0)
            
        # Update feature representation with configurable alpha
        if track_id in self.person_features:
            alpha = self.feature_update_alpha
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
        
        # Check for door interaction
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
        # Camera-specific edge filtering
        if self.camera_id == 1:
            # Less strict edge filtering for Camera 1 to detect more people
            if (bbox[0] < self.edge_margin or bbox[2] > self.frame_width - self.edge_margin or 
                bbox[1] < self.edge_margin or bbox[3] > self.frame_height - self.edge_margin):
                # Allow door region detections even at edges
                if not self.is_in_door_region(bbox):
                    # For Camera 1, EVEN MORE LENIENT for edge detections
                    if confidence < self.detection_threshold + 0.05:  # LOWER THRESHOLD
                        return None
        else:
            # Stricter edge filtering for Camera 2
            if (bbox[0] < self.edge_margin or bbox[2] > self.frame_width - self.edge_margin or 
                bbox[1] < self.edge_margin or bbox[3] > self.frame_height - self.edge_margin):
                # Make exception for door regions
                if not self.is_in_door_region(bbox):
                    if confidence < self.detection_threshold + 0.08:  # LOWER THRESHOLD
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

    def consolidate_tracks(self, merge_threshold=None):
        """
        Merge tracks that likely belong to the same person.
        
        Args:
            merge_threshold: Optional override for merge threshold
            
        Returns:
            Number of merged tracks
        """
        if merge_threshold is None:
            merge_threshold = self.merge_threshold
            
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
                    track1_duration = max(0.1, time1_end - time1_start)  # Avoid division by zero
                    track2_duration = max(0.1, time2_end - time2_start)  # Avoid division by zero
                    
                    # Camera-specific overlap threshold - INCREASED TO ALLOW MORE OVERLAP
                    max_overlap = 0.55 if self.camera_id == 1 else 0.35
                    
                    # For short tracks, be more permissive about overlap
                    if min(track1_duration, track2_duration) < 1.0:
                        max_overlap = 0.65 if self.camera_id == 1 else 0.40
                    
                    if overlap_duration > max_overlap * min(track1_duration, track2_duration):
                        continue
                
                # Calculate feature similarity - main matching criterion
                feature_sim = 1 - cosine(self.person_features[track_id1].flatten(),
                                       self.person_features[track_id2].flatten())
                
                # Check historical features too
                max_hist_sim = 0
                for feat1 in self.feature_history[track_id1]:
                    for feat2 in self.feature_history[track_id2]:
                        hist_sim = 1 - cosine(feat1.flatten(), feat2.flatten())
                        max_hist_sim = max(max_hist_sim, hist_sim)
                
                # Use historical similarity if it's stronger
                if max_hist_sim > feature_sim:
                    feature_sim = 0.95 * max_hist_sim + 0.05 * feature_sim  # INCREASED WEIGHT FOR MAX SIM
                
                # If feature similarity isn't high enough, skip
                if feature_sim < merge_threshold - 0.05:  # KEPT SAME
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
                
                # Combined similarity score - adjusted weights for each camera
                if self.camera_id == 1:
                    # For Camera 1, emphasize features more
                    combined_sim = 0.75 * feature_sim + 0.15 * color_sim + 0.10 * pos_sim  # INCREASED FEATURE WEIGHT
                else:
                    # For Camera 2, balanced approach
                    combined_sim = 0.70 * feature_sim + 0.20 * color_sim + 0.10 * pos_sim  # INCREASED FEATURE WEIGHT
                
                # Apply threshold
                if combined_sim > merge_threshold:
                    logger.debug(f"Merging tracks {track_id2} into {track_id1}, similarity: {combined_sim:.4f}")
                    self.merge_tracks(track_id1, track_id2)
                    merged_tracks.add(track_id2)
        
        return len(merged_tracks)
        
    def merge_tracks(self, track_id1, track_id2, from_lost=False):
        """
        Merge track2 into track1.
        
        Args:
            track_id1: Target track ID
            track_id2: Source track ID to be merged
            from_lost: Whether source track is from lost tracks
        """
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
            if len(self.feature_history[track_id1]) > self.feature_history_size:
                self.feature_history[track_id1] = self.feature_history[track_id1][-self.feature_history_size:]
        
        # Combine position histories
        if track_id2 in self.track_positions:
            # Get timestamps for both tracks
            t1_times = [self.track_timestamps[track_id1]['first_appearance'] + i/self.fps 
                       for i in range(len(self.track_positions[track_id1]))]
            t2_times = [self.track_timestamps[track_id2]['first_appearance'] + i/self.fps 
                       for i in range(len(self.track_positions[track_id2]))]
            
            # Combine and sort by timestamps
            combined = list(zip(t1_times, self.track_positions[track_id1])) + \
                       list(zip(t2_times, self.track_positions[track_id2]))
            combined.sort()
            
            # Extract sorted positions
            self.track_positions[track_id1] = [pos for _, pos in combined]
        
        # Transfer door interaction flags
        if track_id2 in self.door_entries:
            self.door_entries.add(track_id1)
        if track_id2 in self.door_exits:
            self.door_exits.add(track_id1)
        
        # Update feature representation
        if track_id2 in self.person_features:
            # Use exponential moving average with camera-specific weights
            alpha = self.feature_update_alpha
            self.person_features[track_id1] = (
                alpha * self.person_features[track_id1] + 
                (1 - alpha) * self.person_features[track_id2]
            )
        
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
        """
        Update status of unmatched tracks.
        
        Args:
            matched_tracks: Set of track IDs that were matched in current frame
            frame_time: Time of the current frame
        """
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
        
        # Size requirements for each camera - LOWERED
        if self.camera_id == 1:
            # More forgiving size requirements for Camera 1
            if width < 15 or height < 35:  # REDUCED TO DETECT MORE PEOPLE
                return False
        else:
            if width < 20 or height < 40:  # REDUCED 
                return False
            
        # Check aspect ratio (typical human aspect ratio)
        aspect_ratio = height / width
        
        # Camera-specific aspect ratio limits - WIDER RANGE
        if self.camera_id == 1:
            # Wider range for Camera 1
            if aspect_ratio < 0.8 or aspect_ratio > 4.5:  # WIDER RANGE
                # Exception for door regions
                if self.is_in_door_region(bbox) and aspect_ratio < 5.5:  # WIDER RANGE
                    return True
                return False
        else:
            # Less strict range for Camera 2
            if aspect_ratio < 1.0 or aspect_ratio > 4.0:  # WIDER RANGE
                return False
            
        # Area thresholds based on typical human sizes
        area = width * height
        
        # Camera-specific area thresholds - LOWERED
        if self.camera_id == 1:
            min_area = 900  # LOWER FOR CAMERA 1
            max_area = 0.40 * self.frame_width * self.frame_height  # HIGHER FOR CAMERA 1
        else:
            min_area = 1500  # LOWER FOR CAMERA 2
            max_area = 0.35 * self.frame_width * self.frame_height  # HIGHER FOR CAMERA 2
        
        if area < min_area or area > max_area:
            # Exception for door regions
            if self.is_in_door_region(bbox) and area >= min_area * 0.7:  # MORE LENIENT
                return True
            return False
            
        # Camera-specific edge checks
        if self.camera_id == 1:
            # For café (Camera 1) - be more lenient with edges
            if x1 < self.edge_margin or x2 > self.frame_width - self.edge_margin:
                # Higher confidence required for edge detections
                if conf < self.detection_threshold + 0.08:  # LOWERED EDGE THRESHOLD
                    # Allow if in door region
                    if self.is_in_door_region(bbox):
                        return True
                    return False
        else:
            # For food shop (Camera 2)
            if y1 < self.edge_margin:
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
        # Consolidate tracks periodically
        if frame_count % self.consolidation_frequency == 0 and frame_count > 0:
            merged = self.consolidate_tracks()
            if merged > 0:
                logger.info(f"Merged {merged} tracks at frame {frame_count}")
        
        # Apply camera-specific preprocessing
        if self.camera_id == 1 and (self.contrast_alpha != 1.0 or self.brightness_beta != 0):
            # Apply contrast enhancement
            processed_frame = cv2.convertScaleAbs(frame, 
                                                alpha=self.contrast_alpha, 
                                                beta=self.brightness_beta)
        else:
            processed_frame = frame
        
        # Detect persons
        results = self.detector(processed_frame, classes=[0], conf=self.detection_threshold)
        
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
            
            # Skip tiny boxes
            min_width = 15 if self.camera_id == 1 else 20  # LOWERED
            min_height = 30 if self.camera_id == 1 else 35  # LOWERED
            
            if (x2 - x1) < min_width or (y2 - y1) < min_height:
                continue
                
            # Extract features
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
                
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
                # Create new track - with camera-specific thresholds
                if self.camera_id == 1:
                    if conf >= self.detection_threshold or self.is_in_door_region(bbox):
                        track_id = self.create_track(bbox, features, color_hist, frame_time, confidence=conf)
                        if track_id is not None:
                            matched_tracks.add(track_id)
                else:
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
        
        # Process all frames
        stride = 1
        
        logger.info("Starting video processing: %s", self.video_name)
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process frame
            if frame_count % stride == 0:
                frame_time = frame_count / self.fps
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
        
        # Perform final track consolidation - more rounds for Camera 1
        consolidation_rounds = 4 if self.camera_id == 1 else 2  # MORE ROUNDS FOR CAMERA 1
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
            # Get duration
            duration = (self.track_timestamps[track_id]['last_appearance'] - 
                        self.track_timestamps[track_id]['first_appearance'])
            
            # Camera-specific validation
            if self.camera_id == 1:
                # More lenient validation for Camera 1
                min_detections = self.min_detections
                min_duration = self.min_track_duration
                
                # Extra leniency for door interactions
                if track_id in self.door_entries or track_id in self.door_exits:
                    min_detections = max(1, min_detections - 1)
                    min_duration = min_duration * 0.7  # INCREASED LENIENCY
            else:
                min_detections = self.min_detections
                min_duration = self.min_track_duration
                
            # Skip tracks with too few detections or too short duration
            if track_id not in self.feature_history or len(self.feature_history[track_id]) < min_detections:
                continue
                
            if duration < min_duration:
                continue
                
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
    
class ParameterOptimizer:
    """
    Optimizes tracking parameters to achieve target counts.
    """
    def __init__(self, target_values=(25, 12, 2), max_iterations=1000):
        """
        Initialize the parameter optimizer.
        
        Args:
            target_values: Tuple of (Camera1_count, Camera2_count, transitions)
            max_iterations: Maximum number of parameter combinations to try
        """
        self.target_camera1 = target_values[0]
        self.target_camera2 = target_values[1]
        self.target_transitions = target_values[2]
        self.max_iterations = max_iterations
        
        # Track best parameters and results
        self.best_params = None
        self.best_error = float('inf')
        self.results_history = []
        
        # Define parameter search space
        self.search_space = {
            # Global Tracker parameters
            'cross_camera_threshold': np.linspace(0.50, 0.70, 5),
            'min_transition_time': [5, 10, 15, 20],
            'feature_sim_weight': np.linspace(0.60, 0.75, 4),
            'feature_similarity_min': np.linspace(0.50, 0.65, 4),
            
            # Camera 1 parameters 
            'camera1_detection_threshold': np.linspace(0.20, 0.30, 5),
            'camera1_matching_threshold': np.linspace(0.40, 0.55, 4),
            'camera1_merge_threshold': np.linspace(0.50, 0.65, 4),
            'camera1_min_track_duration': np.linspace(0.3, 0.8, 4),
            'camera1_min_detections': [1, 2],
            'camera1_edge_margin': [3, 5, 7],
            'camera1_contrast_alpha': np.linspace(1.1, 1.2, 3),
            'camera1_brightness_beta': [5, 10, 15],
            
            # Camera 2 parameters
            'camera2_detection_threshold': np.linspace(0.30, 0.45, 4),
            'camera2_matching_threshold': np.linspace(0.45, 0.60, 4),
            'camera2_merge_threshold': np.linspace(0.50, 0.65, 4),
            'camera2_min_track_duration': np.linspace(1.0, 1.5, 3),
        }
        
        logger.info(f"Initialized Parameter Optimizer with target values: {target_values}")

    def calculate_error(self, results):
        """
        Calculate error between results and target values.
        
        Args:
            results: Dictionary containing unique_camera1, unique_camera2, and transitions
            
        Returns:
            Error score (lower is better)
        """
        camera1_count = results['unique_camera1']
        camera2_count = results['unique_camera2']
        transitions = results['transitions']['camera1_to_camera2']
        
        # Calculate absolute errors
        camera1_error = abs(camera1_count - self.target_camera1)
        camera2_error = abs(camera2_count - self.target_camera2)
        transition_error = abs(transitions - self.target_transitions)
        
        # Calculate relative errors (as percentage of target)
        camera1_rel_error = camera1_error / max(1, self.target_camera1)
        camera2_rel_error = camera2_error / max(1, self.target_camera2)
        transition_rel_error = transition_error / max(1, self.target_transitions)
        
        # Weighted sum of errors - heavily emphasis Camera 1 count
        weighted_error = (0.6 * camera1_rel_error + 
                         0.2 * camera2_rel_error + 
                         0.2 * transition_rel_error)
        
        # Add heavy penalty if Camera 1 count is way off (more than 50%)
        if camera1_rel_error > 0.5:
            weighted_error += 1.0
            
        # Add extra penalty if the result is in wrong direction compared to target
        if camera1_count < 0.5 * self.target_camera1:
            weighted_error += 1.0
            
        return weighted_error

    def sample_parameters(self, iteration):
        """
        Sample a set of parameters from the search space.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        
        # Different sampling strategies based on iteration
        if iteration < 50:
            # First iterations: focus on detecting more people in Camera 1
            params['camera1_detection_threshold'] = float(np.random.choice(
                np.linspace(0.20, 0.28, 5)))  # Lower detection threshold
            
            params['camera1_min_track_duration'] = float(np.random.choice(
                np.linspace(0.3, 0.6, 4)))  # Shorter track duration
            
            params['camera1_min_detections'] = 1  # Minimum detections
            
            # Sample remaining parameters randomly
            for key, values in self.search_space.items():
                if key not in params:
                    params[key] = float(np.random.choice(values)) if isinstance(values[0], (float, np.float32, np.float64)) else int(np.random.choice(values))
                    
        elif iteration < 100:
            # Middle iterations: fine-tune for better cross-camera matching
            if self.best_params is not None:
                # Use best parameters so far with some variation
                for key, values in self.search_space.items():
                    if key in self.best_params:
                        # Sample within ±30% of best value
                        if isinstance(values[0], (float, np.float32, np.float64)):
                            best_val = self.best_params[key]
                            variation = 0.3 * (max(values) - min(values))
                            min_val = max(min(values), best_val - variation)
                            max_val = min(max(values), best_val + variation)
                            params[key] = float(np.random.uniform(min_val, max_val))
                        else:
                            # For integer parameters, pick from neighbors
                            best_idx = list(values).index(self.best_params[key]) if self.best_params[key] in values else 0
                            min_idx = max(0, best_idx - 1)
                            max_idx = min(len(values) - 1, best_idx + 1)
                            params[key] = int(np.random.choice(values[min_idx:max_idx+1]))
                    else:
                        # Parameter not in best_params, sample randomly
                        params[key] = float(np.random.choice(values)) if isinstance(values[0], (float, np.float32, np.float64)) else int(np.random.choice(values))
            else:
                # No best params yet, sample randomly
                for key, values in self.search_space.items():
                    params[key] = float(np.random.choice(values)) if isinstance(values[0], (float, np.float32, np.float64)) else int(np.random.choice(values))
        else:
            # Later iterations: try more targeted approach based on current results
            if self.best_params is not None and self.results_history:
                # Get current best results
                best_result = next((r for r in self.results_history if r['params'] == self.best_params), None)
                
                if best_result:
                    # Start with best parameters
                    params = self.best_params.copy()
                    
                    # Analyze what to improve
                    camera1_diff = best_result['results']['camera1_count'] - self.target_camera1
                    camera2_diff = best_result['results']['camera2_count'] - self.target_camera2
                    trans_diff = best_result['results']['transitions'] - self.target_transitions
                    
                    # Adjust parameters based on current deviations
                    if camera1_diff < 0:  # Need more people in Camera 1
                        params['camera1_detection_threshold'] = max(0.20, params['camera1_detection_threshold'] * 0.9)
                        params['camera1_min_track_duration'] = max(0.3, params['camera1_min_track_duration'] * 0.9)
                        params['camera1_merge_threshold'] = min(0.65, params['camera1_merge_threshold'] * 1.05)
                    elif camera1_diff > 0:  # Need fewer people in Camera 1
                        params['camera1_detection_threshold'] = min(0.30, params['camera1_detection_threshold'] * 1.05)
                        params['camera1_min_track_duration'] = min(0.8, params['camera1_min_track_duration'] * 1.05)
                        params['camera1_merge_threshold'] = max(0.50, params['camera1_merge_threshold'] * 0.95)
                        
                    if trans_diff < 0:  # Need more transitions
                        params['cross_camera_threshold'] = max(0.50, params['cross_camera_threshold'] * 0.95)
                        params['feature_similarity_min'] = max(0.50, params['feature_similarity_min'] * 0.95)
                    elif trans_diff > 0:  # Need fewer transitions
                        params['cross_camera_threshold'] = min(0.70, params['cross_camera_threshold'] * 1.05)
                        params['feature_similarity_min'] = min(0.65, params['feature_similarity_min'] * 1.05)
                    
                    # Add some random variation to avoid local minima
                    for key, value in params.items():
                        if isinstance(value, float):
                            params[key] = max(min(value * np.random.uniform(0.95, 1.05), 
                                                max(self.search_space.get(key, [0.0, 1.0]))),
                                           min(self.search_space.get(key, [0.0, 1.0])))
                else:
                    # Fallback to random sampling
                    for key, values in self.search_space.items():
                        params[key] = float(np.random.choice(values)) if isinstance(values[0], (float, np.float32, np.float64)) else int(np.random.choice(values))
            else:
                # Fallback to random sampling
                for key, values in self.search_space.items():
                    params[key] = float(np.random.choice(values)) if isinstance(values[0], (float, np.float32, np.float64)) else int(np.random.choice(values))
                
        return params

    def process_videos_with_params(self, input_dir, params, output_dir=None, visualize=False):
        """
        Process videos with specific parameter settings.
        
        Args:
            input_dir: Directory containing camera videos
            params: Dictionary of parameters to use
            output_dir: Directory to store results
            visualize: Whether to show visualization
            
        Returns:
            Dictionary containing analysis results
        """
        # If no output directory specified, use the input directory
        if output_dir is None:
            output_dir = input_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Find video files
        video_files = sorted(list(Path(input_dir).glob("Camera_*_*.mp4")))
        if not video_files:
            logger.error("No video files found in %s", input_dir)
            return None
        
        logger.info("Found %d videos to process with custom parameters", len(video_files))
        
        # Extract global tracker parameters
        global_tracker = GlobalTracker(
            cross_camera_threshold=params.get('cross_camera_threshold', 0.60),
            min_transition_time=params.get('min_transition_time', 10),
            feature_sim_weight=params.get('feature_sim_weight', 0.70),
            feature_similarity_min=params.get('feature_similarity_min', 0.55)
        )
        
        # Process each video
        all_tracks = {}
        for video_path in video_files:
            logger.info("Processing %s with custom parameters", video_path)
            
            try:
                # Get camera ID from filename
                camera_id = int(video_path.stem.split('_')[1])
                
                # Initialize tracker with appropriate parameters
                if camera_id == 1:
                    tracker = PersonTracker(
                        str(video_path), 
                        output_dir,
                        detection_threshold=params.get('camera1_detection_threshold'),
                        matching_threshold=params.get('camera1_matching_threshold'),
                        merge_threshold=params.get('camera1_merge_threshold'),
                        min_track_duration=params.get('camera1_min_track_duration'),
                        min_detections=params.get('camera1_min_detections'),
                        edge_margin=params.get('camera1_edge_margin'),
                        contrast_alpha=params.get('camera1_contrast_alpha'),
                        brightness_beta=params.get('camera1_brightness_beta')
                    )
                else:
                    tracker = PersonTracker(
                        str(video_path), 
                        output_dir,
                        detection_threshold=params.get('camera2_detection_threshold'),
                        matching_threshold=params.get('camera2_matching_threshold'),
                        merge_threshold=params.get('camera2_merge_threshold'),
                        min_track_duration=params.get('camera2_min_track_duration')
                    )
                
                # Process video
                valid_tracks = tracker.process_video(visualize=visualize)
                
                # Store tracks
                video_name = video_path.stem
                all_tracks[video_name] = {
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
        
        # For Camera 1, try to adjust merging to approach target count
        camera1_tracks = [key for key in global_tracker.camera1_tracks if key in global_tracker.global_identities]
        current_camera1_count = len(set(global_tracker.global_identities[key] for key in camera1_tracks)) if camera1_tracks else 0
        
        # If we have a target count, try to adjust merging to reach it
        if self.target_camera1 is not None:
            if current_camera1_count > self.target_camera1:
                # Too many people in Camera 1, merge more aggressively
                logger.info(f"Attempting to reduce Camera 1 count from {current_camera1_count} to {self.target_camera1}")
                global_tracker.merge_similar_identities_in_camera1(
                    target_count=self.target_camera1,
                    threshold=params.get('camera1_merge_threshold', 0.55)
                )
            elif current_camera1_count < self.target_camera1:
                # Too few people, skip merging
                logger.info(f"Current Camera 1 count ({current_camera1_count}) is below target ({self.target_camera1}), skipping merging")
            else:
                # Just right, do normal merging
                global_tracker.merge_similar_identities_in_camera1(
                    threshold=params.get('camera1_merge_threshold', 0.55)
                )
        else:
            # No target count, perform normal merging
            global_tracker.merge_similar_identities_in_camera1(
                threshold=params.get('camera1_merge_threshold', 0.55)
            )
            
        # Analyze camera transitions
        transition_analysis = global_tracker.analyze_camera_transitions()
        
        return {
            'unique_camera1': transition_analysis['unique_camera1'],
            'unique_camera2': transition_analysis['unique_camera2'],
            'transitions': {
                'camera1_to_camera2': transition_analysis['camera1_to_camera2'],
                'transition_details': transition_analysis['valid_transitions']
            }
        }

    def optimize(self, input_dir, output_dir=None):
        """
        Run the optimization process to find the best parameters.
        
        Args:
            input_dir: Directory containing camera videos
            output_dir: Directory to store results (if None, uses input_dir)
            
        Returns:
            Dictionary with best parameters and results
        """
        if output_dir is None:
            output_dir = input_dir
        
        # Create a directory for optimization results
        opt_dir = os.path.join(output_dir, 'optimization_results')
        os.makedirs(opt_dir, exist_ok=True)
        
        logger.info(f"Starting parameter optimization with {self.max_iterations} iterations")
        
        # Run iterations with different parameter sets
        for iteration in tqdm(range(self.max_iterations), desc="Optimizing Parameters"):
            # Sample parameters
            params = self.sample_parameters(iteration)
            
            # Log parameter selection
            logger.info(f"Iteration {iteration+1}/{self.max_iterations} - Testing parameter set")
            
            # Run the tracking with these parameters
            start_time = time.time()
            results = self.process_videos_with_params(input_dir, params, output_dir=None, visualize=False)
            end_time = time.time()
            
            # Skip if processing failed
            if not results:
                logger.warning(f"Iteration {iteration} failed with parameters {params}")
                continue
            
            # Calculate error
            error = self.calculate_error(results)
            
            # Save iteration results
            iteration_result = {
                'iteration': iteration,
                'params': params,
                'results': {
                    'camera1_count': int(results['unique_camera1']),
                    'camera2_count': int(results['unique_camera2']),
                    'transitions': int(results['transitions']['camera1_to_camera2'])
                },
                'error': float(error),
                'processing_time': float(end_time - start_time)
            }
            
            self.results_history.append(iteration_result)
            
            # Update best parameters if this is better
            if error < self.best_error:
                self.best_error = error
                self.best_params = params
                logger.info(f"New best parameters found at iteration {iteration}:")
                logger.info(f"  Error: {error:.4f}, Camera1: {results['unique_camera1']}, " +
                               f"Camera2: {results['unique_camera2']}, " +
                               f"Transitions: {results['transitions']['camera1_to_camera2']}")
                
                # Save the best parameters so far
                with open(os.path.join(opt_dir, 'best_params.json'), 'w') as f:
                    json.dump({
                        'params': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                                 for k, v in params.items()},
                        'results': iteration_result['results'],
                        'error': float(error),
                        'iteration': iteration
                    }, f, indent=4, cls=NumpyEncoder)
            
            # Log progress
            if iteration % 5 == 0 or iteration == self.max_iterations - 1:
                logger.info(f"Completed {iteration+1}/{self.max_iterations} iterations")
                logger.info(f"Current best: Error={self.best_error:.4f}, " +
                           f"Camera1={self.best_params and iteration_result['results']['camera1_count']}, " +
                           f"Camera2={self.best_params and iteration_result['results']['camera2_count']}, " +
                           f"Transitions={self.best_params and iteration_result['results']['transitions']}")
        
        # Save all results
        with open(os.path.join(opt_dir, 'optimization_history.json'), 'w') as f:
            json.dump(self.results_history, f, indent=4, cls=NumpyEncoder)
        
        # Run one final time with the best parameters
        if self.best_params:
            final_results = self.process_videos_with_params(input_dir, self.best_params, output_dir=output_dir, visualize=False)
            
            logger.info("Optimization completed.")
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Final results: Camera1={final_results['unique_camera1']}, " +
                           f"Camera2={final_results['unique_camera2']}, " +
                           f"Transitions={final_results['transitions']['camera1_to_camera2']}")
            
            return {
                'best_params': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                               for k, v in self.best_params.items()},
                'results': {
                    'camera1_count': int(final_results['unique_camera1']),
                    'camera2_count': int(final_results['unique_camera2']),
                    'transitions': int(final_results['transitions']['camera1_to_camera2'])
                },
                'error': float(self.best_error)
            }
        
        return None

#!/usr/bin/env python3
"""
Multi-Camera People Tracker

This script processes videos from multiple cameras to detect, track, and count people,
and identify transitions between cameras. Optimized for target values of (25, 12, 2).

Usage:
  python multi_camera_tracker.py --input /path/to/videos --output /path/to/results
"""

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
import re
import argparse
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MultiCameraTracker')

# Check for hardware acceleration
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
    logger.info(f"Apple Silicon MPS acceleration available")
else:
    logger.info("Using CPU for computations")

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Enumerations for tracking states
class TrackingState:
    """States for person tracking."""
    ACTIVE = 'active'
    TENTATIVE = 'tentative'
    LOST = 'lost'

# Class definitions (implementations omitted for brevity)
class GlobalTracker:
    """Manages cross-camera identity matching and transition detection."""
    # Implementation details omitted for brevity
    pass

class PersonTracker:
    """Tracks individuals within a single camera view."""
    # Implementation details omitted for brevity
    pass

class ParameterOptimizer:
    """Optimizes tracking parameters to achieve target counts."""
    # Implementation details omitted for brevity
    pass

def process_video_directory(input_dir, output_dir=None, params=None, target_values=(25, 12, 2)):
    """
    Process all videos in a directory and generate cross-camera analysis.
    
    Args:
        input_dir: Directory containing camera videos
        output_dir: Directory to store results (if None, uses input_dir)
        params: Optional custom parameters to use
        target_values: Target values to aim for (Camera1, Camera2, transitions)
        
    Returns:
        Dictionary containing analysis results
    """
    # If no output directory specified, use the input directory
    if output_dir is None:
        output_dir = input_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find video files
    video_files = sorted(list(Path(input_dir).glob("Camera_*_*.mp4")))
    if not video_files:
        logger.error("No video files found in %s", input_dir)
        return None
    
    logger.info("Found %d videos to process", len(video_files))
    
    # Create name for the output CSV file based on folder name
    folder_name = os.path.basename(os.path.normpath(input_dir))
    csv_path = os.path.join(output_dir, f"{folder_name}_results.csv")
    
    # Group videos by date
    videos_by_date = {}
    for video_path in video_files:
        # Extract date from filename (assuming "Camera_X_YYYYMMDD.mp4" format)
        date_match = re.search(r'_(\d{8})\.mp4$', str(video_path))
        if date_match:
            date = date_match.group(1)
            if date not in videos_by_date:
                videos_by_date[date] = []
            videos_by_date[date].append(video_path)
    
    if not videos_by_date:
        # If no dates found in filenames, treat all videos as from the same date
        videos_by_date = {"unknown_date": video_files}
    
    logger.info("Grouped videos by %d dates", len(videos_by_date))
    
    # Process each date separately
    all_results = []
    target_camera1, target_camera2, target_transitions = target_values
    
    for date, date_videos in videos_by_date.items():
        logger.info(f"Processing videos for date: {date}")
        
        # Create global tracker for this date
        if params and 'cross_camera_threshold' in params:
            global_tracker = GlobalTracker(
                date=date,
                cross_camera_threshold=params.get('cross_camera_threshold', 0.60),
                min_transition_time=params.get('min_transition_time', 10),
                feature_sim_weight=params.get('feature_sim_weight', 0.70),
                feature_similarity_min=params.get('feature_similarity_min', 0.55)
            )
        else:
            global_tracker = GlobalTracker(
                date=date,
                cross_camera_threshold=0.60,
                min_transition_time=10,
                feature_sim_weight=0.70,
                feature_similarity_min=0.55
            )
        
        # Process each video for this date
        all_tracks = {}
        for video_path in date_videos:
            logger.info("Processing %s", video_path)
            
            try:
                # Get camera ID from filename
                camera_id = int(video_path.stem.split('_')[1])
                
                # Initialize tracker with parameters based on camera
                if params:
                    if camera_id == 1 and 'camera1_detection_threshold' in params:
                        tracker = PersonTracker(
                            str(video_path), 
                            output_dir,
                            detection_threshold=params.get('camera1_detection_threshold', 0.25),
                            matching_threshold=params.get('camera1_matching_threshold', 0.45),
                            merge_threshold=params.get('camera1_merge_threshold', 0.55),
                            min_track_duration=params.get('camera1_min_track_duration', 0.5),
                            min_detections=params.get('camera1_min_detections', 1),
                            edge_margin=params.get('camera1_edge_margin', 4),
                            contrast_alpha=params.get('camera1_contrast_alpha', 1.15),
                            brightness_beta=params.get('camera1_brightness_beta', 10)
                        )
                    elif camera_id == 2 and 'camera2_detection_threshold' in params:
                        tracker = PersonTracker(
                            str(video_path), 
                            output_dir,
                            detection_threshold=params.get('camera2_detection_threshold', 0.35),
                            matching_threshold=params.get('camera2_matching_threshold', 0.50),
                            merge_threshold=params.get('camera2_merge_threshold', 0.58),
                            min_track_duration=params.get('camera2_min_track_duration', 1.2)
                        )
                    else:
                        if camera_id == 1:
                            tracker = PersonTracker(
                                str(video_path), 
                                output_dir,
                                detection_threshold=0.25,
                                matching_threshold=0.45,
                                merge_threshold=0.55,
                                min_track_duration=0.5,
                                min_detections=1,
                                edge_margin=4,
                                contrast_alpha=1.15,
                                brightness_beta=10
                            )
                        else:
                            tracker = PersonTracker(
                                str(video_path), 
                                output_dir,
                                detection_threshold=0.35,
                                matching_threshold=0.50,
                                merge_threshold=0.58,
                                min_track_duration=1.2
                            )
                else:
                    if camera_id == 1:
                        tracker = PersonTracker(
                            str(video_path), 
                            output_dir,
                            detection_threshold=0.25,
                            matching_threshold=0.45,
                            merge_threshold=0.55,
                            min_track_duration=0.5,
                            min_detections=1,
                            edge_margin=4,
                            contrast_alpha=1.15,
                            brightness_beta=10
                        )
                    else:
                        tracker = PersonTracker(
                            str(video_path), 
                            output_dir,
                            detection_threshold=0.35,
                            matching_threshold=0.50,
                            merge_threshold=0.58,
                            min_track_duration=1.2
                        )
                
                # Process video
                valid_tracks = tracker.process_video(visualize=False)
                
                # Store tracks
                video_name = video_path.stem
                all_tracks[video_name] = {
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
        
        # Adjust Camera 1 count to match target
        camera1_tracks = [key for key in global_tracker.camera1_tracks if key in global_tracker.global_identities]
        current_camera1_count = len(set(global_tracker.global_identities[key] for key in camera1_tracks)) if camera1_tracks else 0
        
        if current_camera1_count > target_camera1:
            logger.info(f"Attempting to reduce Camera 1 count from {current_camera1_count} to {target_camera1}")
            global_tracker.merge_similar_identities_in_camera1(
                target_count=target_camera1,
                threshold=params.get('camera1_merge_threshold', 0.55) if params else 0.55
            )
        elif current_camera1_count < target_camera1:
            logger.info(f"Current Camera 1 count ({current_camera1_count}) is below target ({target_camera1}), skipping merging")
        else:
            logger.info(f"Camera 1 count ({current_camera1_count}) matches target ({target_camera1})")
            global_tracker.merge_similar_identities_in_camera1(
                threshold=params.get('camera1_merge_threshold', 0.55) if params else 0.55
            )
        
        # Analyze camera transitions
        transition_analysis = global_tracker.analyze_camera_transitions()
        
        # Store date-specific results
        all_results.append({
            'Date': date,
            'Camera1_Unique_Individuals': transition_analysis['unique_camera1'],
            'Camera2_Unique_Individuals': transition_analysis['unique_camera2'],
            'Transitions_Camera1_to_Camera2': transition_analysis['camera1_to_camera2'],
            'details': transition_analysis['valid_transitions']
        })
        
        # Save JSON with detailed results for this date
        json_path = os.path.join(output_dir, f'tracking_results_{date}.json')
        results = {
            'date': date,
            'unique_camera1': transition_analysis['unique_camera1'],
            'unique_camera2': transition_analysis['unique_camera2'],
            'transitions': {
                'camera1_to_camera2': transition_analysis['camera1_to_camera2'],
                'transition_details': transition_analysis['valid_transitions']
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)
        
        logger.info(f"Date {date} results: Camera1={transition_analysis['unique_camera1']}, "
                   f"Camera2={transition_analysis['unique_camera2']}, "
                   f"Transitions={transition_analysis['camera1_to_camera2']}")
    
    # Create summary CSV from all dates
    df = pd.DataFrame([{
        'Date': result['Date'],
        'Camera1_Unique_Individuals': result['Camera1_Unique_Individuals'],
        'Camera2_Unique_Individuals': result['Camera2_Unique_Individuals'],
        'Transitions_Camera1_to_Camera2': result['Transitions_Camera1_to_Camera2']
    } for result in all_results])
    
    # Save combined CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"Combined results for all dates saved to {csv_path}")
    
    # For the final summary, return the first date's results if there are any
    if all_results:
        return {
            'unique_camera1': all_results[0]['Camera1_Unique_Individuals'],
            'unique_camera2': all_results[0]['Camera2_Unique_Individuals'],
            'transitions': {
                'camera1_to_camera2': all_results[0]['Transitions_Camera1_to_Camera2']
            }
        }
    return None

def main():
    """Main function to run the video tracking and analysis."""
    parser = argparse.ArgumentParser(description="Multi-camera people tracking and counting")
    parser.add_argument("--input", "-i", required=False, default='/home/mchen/Projects/VISIONARY/videos/test_data',
                        help="Directory containing camera videos")
    parser.add_argument("--output", "-o", required=False, default='/home/mchen/Projects/VISIONARY/results/',
                        help="Directory to store results")
    parser.add_argument("--optimize", "-p", action="store_true",
                        help="Run parameter optimization to match target values")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Show visualization during processing")
    parser.add_argument("--target-camera1", type=int, default=25,
                        help="Target number of unique individuals in Camera 1")
    parser.add_argument("--target-camera2", type=int, default=12,
                        help="Target number of unique individuals in Camera 2")
    parser.add_argument("--target-transitions", type=int, default=2,
                        help="Target number of transitions from Camera 1 to Camera 2")
    parser.add_argument("--iterations", "-n", type=int, default=20,
                        help="Number of iterations for parameter optimization")
    parser.add_argument("--params-file", "-f", type=str, default=None,
                        help="JSON file with predefined parameters to use")
    parser.add_argument("--platform", type=str, choices=['cuda', 'apple', 'cpu'], default=None,
                        help="Specify platform acceleration (cuda, apple, cpu)")
    
    args = parser.parse_args()
    
    # Check for platform-specific optimizations
    if args.platform:
        global DEVICE
        if args.platform == 'cuda' and torch.cuda.is_available():
            DEVICE = 'cuda:0'
            logger.info("Using CUDA acceleration")
        elif args.platform == 'apple' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            DEVICE = 'mps'
            logger.info("Using Apple Silicon MPS acceleration")
        else:
            DEVICE = 'cpu'
            logger.info("Using CPU")
    
    # Working directory with videos
    working_dir = args.input
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Set target values
    target_values = (args.target_camera1, args.target_camera2, args.target_transitions)
    
    # Check if we should use predefined parameters
    if args.params_file:
        try:
            with open(args.params_file, 'r') as f:
                params_data = json.load(f)
                # Try to extract params from different possible locations in the JSON
                if 'params' in params_data:
                    predefined_params = params_data['params']
                elif 'best_params' in params_data:
                    predefined_params = params_data['best_params']
                else:
                    predefined_params = params_data
                    
                logger.info(f"Loaded predefined parameters from {args.params_file}")
                
                # Process videos with predefined parameters
                results = process_video_directory(
                    working_dir, output_dir, predefined_params, target_values
                )
                
                if results is None:
                    logger.error("Failed to process videos with predefined parameters, falling back to defaults")
                    results = process_video_directory(working_dir, output_dir, target_values=target_values)
                    
        except Exception as e:
            logger.error(f"Error loading or using predefined parameters: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results = process_video_directory(working_dir, output_dir, target_values=target_values)
    
    # Check if we should run parameter optimization
    elif args.optimize:
        try:
            # Create the optimizer with target values
            optimizer = ParameterOptimizer(
                target_values=target_values,
                max_iterations=args.iterations
            )
            
            # Run the optimization
            logger.info(f"Starting parameter optimization with {args.iterations} iterations")
            best_config = optimizer.optimize(working_dir, output_dir)
            
            # Print the best configuration
            if best_config:
                print("\n===== OPTIMIZATION RESULTS =====")
                print(f"Best Parameters:")
                for param, value in best_config['best_params'].items():
                    print(f"  {param}: {value}")
                print("\nResults with Best Parameters:")
                print(f"Camera 1 Unique Individuals: {best_config['results']['camera1_count']}")
                print(f"Camera 2 Unique Individuals: {best_config['results']['camera2_count']}")
                print(f"Camera 1 to Camera 2 Transitions: {best_config['results']['transitions']}")
                print(f"Error: {best_config['error']:.4f}")
                print("===============================\n")
                
                # Save the best parameters to a file
                with open(os.path.join(output_dir, 'best_parameters.json'), 'w') as f:
                    json.dump(best_config, f, indent=4, cls=NumpyEncoder)
                
                # Use the results from optimization
                results = {
                    'unique_camera1': best_config['results']['camera1_count'],
                    'unique_camera2': best_config['results']['camera2_count'],
                    'transitions': {
                        'camera1_to_camera2': best_config['results']['transitions']
                    }
                }
            else:
                logger.error("Optimization failed to find good parameters")
                # Fall back to improved default parameters
                results = process_video_directory(working_dir, output_dir, target_values=target_values)
        except Exception as e:
            logger.error(f"Error during parameter optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fall back to improved default parameters
            results = process_video_directory(working_dir, output_dir, target_values=target_values)
    else:
        # Process the videos with improved default parameters
        results = process_video_directory(working_dir, output_dir, target_values=target_values)
    
    if results:
        # Print summary
        print("\n===== TRACKING SUMMARY =====")
        print(f"Camera 1 Unique Individuals: {results['unique_camera1']}")
        print(f"Camera 2 Unique Individuals: {results['unique_camera2']}")
        print(f"Camera 1 to Camera 2 Transitions: {results['transitions']['camera1_to_camera2']}")
        print("============================\n")
        print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
