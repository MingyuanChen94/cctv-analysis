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
    Enhanced to handle the complex café to food shop transition scenarios.
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
        
        # ENHANCED: More lenient parameters for cross-camera matching
        self.min_transition_time = 2        # Shorter minimum time (2 seconds) to catch quick transitions
        self.max_transition_time = 3600     # Extended to 60 minutes for people who might sit in café first
        self.cross_camera_threshold = 0.35  # Lower threshold to catch more transitions
        
        # Additional transition parameters for partial visibility scenarios
        self.cafe_exit_likelihood = {}      # Track likelihood of café exits
        self.shop_entry_likelihood = {}     # Track likelihood of shop entries
        
        # Track door interactions
        self.door_exits = defaultdict(list)    # Tracks exiting through doors
        self.door_entries = defaultdict(list)  # Tracks entering through doors
        
        # Stores feature history for each identity
        self.feature_history = defaultdict(list)
        self.max_features_history = 20      # Keep more historical features
        
        # Camera-specific track sets
        self.camera1_tracks = set()
        self.camera2_tracks = set()
        
        # Enhanced tracking of lighting conditions by camera
        self.camera_lighting_profiles = {
            1: [],  # Lighting profile samples for Camera 1
            2: []   # Lighting profile samples for Camera 2
        }
        
        # ENHANCED: Improved temporal transition model with wider distributions
        self.enable_partial_visibility_matching = True
        self.temporal_transition_model = {
            # Probability distribution of transition times (in seconds)
            'time_distribution': {
                '0-20': 0.25,     # Quick transitions (25%)
                '20-60': 0.35,    # Normal transitions (35%)
                '60-180': 0.20,   # Longer transitions (20%)
                '180-600': 0.15,  # Extended transitions (15%)
                '600+': 0.05      # Very long transitions (5%)
            }
        }
        
        # ENHANCED: Adding transition boosting for low-resolution environment
        self.transition_boosting = {
            'feature_similarity_min': 0.30,  # Lower minimum feature similarity
            'position_weight': 0.15,        # Lower weight on position (less reliable in low-res)
            'time_weight': 0.25,            # Higher weight on plausible timing
            'door_interaction_bonus': 0.30  # Higher bonus for door interactions
        }
        
        logger.info("GlobalTracker initialized with threshold: %.2f", self.cross_camera_threshold)
        logger.info("Transit time window: %d-%d seconds", self.min_transition_time, self.max_transition_time)
        logger.info("Enhanced café to food shop transition modeling enabled")

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
        Enhanced for café (Camera 1) to food shop (Camera 2) transitions with partial visibility.
        
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
                # Higher alpha for Camera 1 to preserve identity with partial visibility
                alpha = 0.70 if camera_id == 1 else 0.60
                self.feature_database[global_id] = (
                    alpha * self.feature_database[global_id] + 
                    (1 - alpha) * features
                )
            
            return global_id
        
        best_match_id = None
        best_match_score = 0
        second_best_match_id = None  # Track second best match for ambiguity resolution
        second_best_match_score = 0
        
        # Try to match with existing global identities
        for global_id, stored_features in self.feature_database.items():
            # Get last camera appearance
            if global_id not in self.appearance_sequence or not self.appearance_sequence[global_id]:
                continue
                
            last_appearance = self.appearance_sequence[global_id][-1]
            last_camera = int(last_appearance['camera'].split('_')[1])  # Extract camera number
            
            # Skip if same camera or outside transition time window
            time_diff = timestamp - last_appearance['timestamp']
            if last_camera == camera_id:
                continue
                
            if time_diff < self.min_transition_time or time_diff > self.max_transition_time:
                continue
            
            # Only allow Camera 1 to Camera 2 transitions (not 2 to 1)
            if last_camera == 2 and camera_id == 1:
                continue
            
            # Calculate feature similarity - using both similarity metrics
            cosine_sim = 1 - cosine(features.flatten(), stored_features.flatten())
            
            # Calculate L2 distance (Euclidean) - normalized 
            l2_dist = np.linalg.norm(features.flatten() - stored_features.flatten())
            max_dist = 2.0  # Approximate maximum possible distance for normalized features
            l2_sim = 1.0 - min(l2_dist / max_dist, 1.0)
            
            # ENHANCED: Combined feature similarity with weighted metrics favoring cosine for low-res
            feature_sim = 0.8 * cosine_sim + 0.2 * l2_sim
            
            # Check historical features too for better matching
            if global_id in self.feature_history and len(self.feature_history[global_id]) > 0:
                hist_sims = []
                for hist_feat in self.feature_history[global_id]:
                    hist_cosine = 1 - cosine(features.flatten(), hist_feat.flatten())
                    hist_l2 = 1.0 - min(np.linalg.norm(features.flatten() - hist_feat.flatten()) / max_dist, 1.0)
                    # Weight cosine similarity higher for low-res
                    hist_sims.append(0.8 * hist_cosine + 0.2 * hist_l2)
                
                if hist_sims:
                    # Use best historical match
                    hist_sim = max(hist_sims)
                    # Blend with current similarity, giving more weight to best match
                    feature_sim = max(feature_sim, 0.9 * hist_sim)
            
            # ENHANCED: Lower base threshold for low-res footage
            cross_camera_min_threshold = self.transition_boosting['feature_similarity_min']
            
            # Skip if feature similarity is below threshold
            if feature_sim < cross_camera_min_threshold:
                continue
            
            # For Camera 1 to Camera 2 transitions, additional validation
            transition_bonus = 0
            if last_camera == 1 and camera_id == 2:
                # Find the camera1 track key that matches this global ID
                camera1_keys = [k for k, v in self.global_identities.items() 
                              if v == global_id and k.startswith('1_')]
                
                # ENHANCED: Increased door validation bonus for low-res footage
                if camera1_keys and camera1_keys[0] in self.door_exits:
                    transition_bonus += self.transition_boosting['door_interaction_bonus']
                if camera_key in self.door_entries:
                    transition_bonus += self.transition_boosting['door_interaction_bonus']
                
                # Time-based validation
                time_category = self._get_time_category(time_diff)
                if time_category in self.temporal_transition_model['time_distribution']:
                    # Add bonus based on probability of this transition time
                    time_prob = self.temporal_transition_model['time_distribution'][time_category]
                    transition_bonus += 0.20 * time_prob  # Increased from 0.15
                
                # Special handling for partial visibility scenarios (café to shop)
                if self.enable_partial_visibility_matching:
                    # Check if person likely exited café (was near door/tables)
                    exit_likelihood = self.cafe_exit_likelihood.get(camera1_keys[0] if camera1_keys else None, 0)
                    transition_bonus += 0.15 * exit_likelihood  # Increased from 0.10
                    
                    # Check if person likely entered shop (was near door)
                    entry_likelihood = self.shop_entry_likelihood.get(camera_key, 0)
                    transition_bonus += 0.15 * entry_likelihood  # Increased from 0.10
                    
                    # ENHANCED: For long transitions, require less evidence in low-res
                    if time_diff > 300:
                        if transition_bonus < 0.10 and feature_sim < 0.50:  # Lowered thresholds
                            continue
            
            # Calculate color similarity with compensation for lighting differences
            color_sim = 0
            if camera_key in self.color_features and len(self.color_features[camera_key]) > 0:
                camera1_keys = [k for k, v in self.global_identities.items() 
                              if v == global_id and k.startswith('1_')]
                if camera1_keys and camera1_keys[0] in self.color_features and len(self.color_features[camera1_keys[0]]) > 0:
                    color_feats1 = self.color_features[camera1_keys[0]][-1]
                    color_feats2 = self.color_features[camera_key][-1]
                    
                    # Standard color similarity
                    color_sim = 1 - cosine(color_feats1.flatten(), color_feats2.flatten())
                    
                    # Enhanced color matching for lighting differences between café and shop
                    if last_camera == 1 and camera_id == 2:
                        # Calculate similarities on subsets of the color histogram
                        # (e.g., focus on hue and saturation components which are less affected by lighting)
                        if len(color_feats1) > 64 and len(color_feats2) > 64:
                            # Get hue components (first 32 elements in standard HSV histogram)
                            hue_sim = 1 - cosine(color_feats1[:32].flatten(), color_feats2[:32].flatten())
                            # Weight hue similarity more for lighting-robust matching
                            color_sim = 0.8 * hue_sim + 0.2 * color_sim  # Increased hue weight for low-res
            
            # ENHANCED: Temporal dynamics - make time factor more lenient
            if time_diff <= 60:  # Quick transitions (under a minute)
                time_factor = 0.95  # Increased from 0.9
            elif time_diff <= 180:  # Normal transitions (1-3 minutes)
                time_factor = 0.85  # Increased from 0.8
            elif time_diff <= 600:  # Extended transitions (3-10 minutes)
                # Slower linear decay from 0.75 to 0.45
                time_factor = 0.75 - 0.3 * ((time_diff - 180) / 420)
            else:  # Long transitions (over 10 minutes)
                # Exponential decay but maintain possibility
                time_factor = 0.45 * np.exp(-0.0005 * (time_diff - 600))  # Slower decay for low-res
            
            # ENHANCED: Combined similarity score with adjusted weights for low-resolution
            similarity = (0.50 * feature_sim +      # Reduced feature weight (less reliable in low-res)
                          0.15 * color_sim +        # Same color weight 
                          0.20 * time_factor +      # Increased time factor weight
                          transition_bonus)         # Door interaction bonus already increased
            
            # Apply threshold for cross-camera matching 
            if similarity > self.cross_camera_threshold:
                if similarity > best_match_score:
                    # Move current best to second best
                    second_best_match_id = best_match_id
                    second_best_match_score = best_match_score
                    # Update best
                    best_match_id = global_id
                    best_match_score = similarity
                elif similarity > second_best_match_score:
                    # Update second best
                    second_best_match_id = global_id
                    second_best_match_score = similarity
        
        # Check for ambiguous matches (first and second best are very close)
        if (best_match_id is not None and second_best_match_id is not None and
            best_match_score - second_best_match_score < 0.1 and 
            camera_id == 2):  # Only for shop matches where ambiguity matters more
            
            # Resolve ambiguity with additional checks
            best_match_time_diff = self._get_transition_time(best_match_id, timestamp)
            second_best_time_diff = self._get_transition_time(second_best_match_id, timestamp)
            
            # Prefer match with better timing if times are significantly different
            if abs(best_match_time_diff - second_best_time_diff) > 120:  # Over 2 minutes difference
                if self._is_better_transition_time(second_best_time_diff, best_match_time_diff):
                    # Swap best and second best
                    best_match_id, second_best_match_id = second_best_match_id, best_match_id
                    best_match_score, second_best_match_score = second_best_match_score, best_match_score
        
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
            alpha = 0.55  # Reduced from 0.6 to adapt faster to low-res appearance changes
            self.feature_database[best_match_id] = (
                alpha * self.feature_database[best_match_id] + 
                (1 - alpha) * features
            )
        
        # Register the global identity for this camera-specific track
        self.global_identities[camera_key] = best_match_id
        
        # Update exit/entry likelihoods
        if camera_id == 1:
            # For Camera 1 tracks, store exit likelihood based on position
            self.cafe_exit_likelihood[camera_key] = 0.9 if camera_key in self.door_exits else 0.4  # Increased values
        elif camera_id == 2:
            # For Camera 2 tracks, store entry likelihood based on position
            self.shop_entry_likelihood[camera_key] = 0.9 if camera_key in self.door_entries else 0.4  # Increased values
        
        return best_match_id
        
    def _get_time_category(self, time_diff):
        """Categorize transition time into predefined buckets"""
        if time_diff <= 20:  # Changed from 30 to 20
            return '0-20'
        elif time_diff <= 60:
            return '20-60'
        elif time_diff <= 180:
            return '60-180'
        elif time_diff <= 600:
            return '180-600'
        else:
            return '600+'
            
    def _get_transition_time(self, global_id, current_timestamp):
        """Get transition time from last appearance of this global ID"""
        if global_id not in self.appearance_sequence or not self.appearance_sequence[global_id]:
            return float('inf')
            
        last_appearance = self.appearance_sequence[global_id][-1]
        return current_timestamp - last_appearance['timestamp']
        
    def _is_better_transition_time(self, time1, time2):
        """Determine if time1 is a more likely transition time than time2"""
        # ENHANCED: Define ideal transition time range rather than point
        ideal_min = 20  # Faster transitions for low-res environment
        ideal_max = 180 # Extended ideal range
        
        # Check if time1 is in ideal range
        time1_in_ideal_range = ideal_min <= time1 <= ideal_max
        time2_in_ideal_range = ideal_min <= time2 <= ideal_max
        
        # If one is in range and other isn't, prefer the one in range
        if time1_in_ideal_range and not time2_in_ideal_range:
            return True
        if not time1_in_ideal_range and time2_in_ideal_range:
            return False
            
        # If both in range or both outside range, take closest to center of range
        ideal_center = (ideal_min + ideal_max) / 2
        return abs(time1 - ideal_center) < abs(time2 - ideal_center)

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
                        # Validate transitions with door interactions or optimal timing
                        is_door_valid = (current.get('is_exit', False) or next_app.get('is_entry', False))
                        is_optimal_time = 20 <= time_diff <= 300  # Extended optimal time window
                        
                        # ENHANCED: More flexible scoring system for validation
                        transition_score = (2 if is_door_valid else 0) + (1 if is_optimal_time else 0)
                        
                        # Accept all transitions with any validation (even score of 0)
                        valid_transitions.append({
                            'global_id': global_id,
                            'exit_time': current['timestamp'],
                            'entry_time': next_app['timestamp'],
                            'transit_time': time_diff,
                            'score': transition_score
                        })
        
        # Sort transitions by quality
        if len(valid_transitions) > 0:
            # Sort by score descending, then by transit time ascending
            valid_transitions.sort(key=lambda x: (-x['score'], x['transit_time']))

        # Calculate unique individuals per camera based on global IDs
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
        Highly conservative with Camera 1 (café) to avoid undercounting.
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
        
        # For each camera, clean up identities with appropriate thresholds
        for camera_id, global_ids in [(1, camera1_global_ids), (2, camera2_global_ids)]:
            # Sort global IDs for consistent merging
            sorted_ids = sorted(list(global_ids))
            
            # ENHANCED: Adjusted thresholds for low-res footage
            # Camera 1 (café) still needs high threshold to avoid merging different people
            if camera_id == 1:
                threshold = 0.92   # Slightly reduced from 0.94 for low-res
                max_overlap_pct = 0.1  # Slightly increased from 0.05 for low-res
                max_sequence_merge = 3  # Increased from 2 for more flexibility
            else:
                threshold = 0.70   # Reduced from 0.75 for low-res
                max_overlap_pct = 0.25  # Increased from 0.2 for low-res
                max_sequence_merge = 6  # Increased from 5 for low-res
                
            logger.info(f"Cleaning Camera {camera_id} identities with threshold {threshold}")
            
            # Track number of merges per identity to limit chain merging
            merge_counts = defaultdict(int)
            
            # For each pair of identities, check if they should be merged
            for i, id1 in enumerate(sorted_ids):
                if id1 in merged_ids:
                    continue
                
                # Skip if already merged too many times (prevents chain merging)
                if merge_counts[id1] >= max_sequence_merge:
                    continue
                    
                for j in range(i+1, len(sorted_ids)):
                    id2 = sorted_ids[j]
                    if id2 in merged_ids:
                        continue
                    
                    # Skip if already merged too many times
                    if merge_counts[id2] >= max_sequence_merge:
                        continue
                    
                    # Get all camera appearances
                    appearances1 = [a for a in self.appearance_sequence.get(id1, []) 
                                  if a['camera'] == f"Camera_{camera_id}"]
                    appearances2 = [a for a in self.appearance_sequence.get(id2, []) 
                                  if a['camera'] == f"Camera_{camera_id}"]
                    
                    if not appearances1 or not appearances2:
                        continue
                    
                    # Check if appearances are temporally separated
                    times1 = sorted([a['timestamp'] for a in appearances1])
                    times2 = sorted([a['timestamp'] for a in appearances2])
                    
                    # ENHANCED: Adjust time separation for low-res
                    if camera_id == 1:  # For café
                        max_time_separation = 900  # Increased from 600 to 15 minutes
                    else:
                        max_time_separation = 1800  # Increased from 1200 to 30 minutes
                        
                    min_time_diff = min(
                        abs(times1[0] - times2[-1]),
                        abs(times2[0] - times1[-1])
                    )
                    
                    if min_time_diff > max_time_separation:
                        # Skip merging tracks that are too far apart in time
                        continue
                    
                    # Check for temporal overlap
                    if camera_id == 1:  # For café with partial visibility
                        # Calculate overlap with enhanced logic
                        if times1[-1] < times2[0] or times2[-1] < times1[0]:
                            # No overlap at all
                            overlap_duration = 0
                        else:
                            # Some overlap
                            overlap_start = max(times1[0], times2[0])
                            overlap_end = min(times1[-1], times2[-1])
                            overlap_duration = max(0, overlap_end - overlap_start)
                            
                            # For café: if tracks have strong feature similarity but some temporal overlap,
                            # check if one could have been sitting at a table outside view during overlap
                            if overlap_duration > 0:
                                # Get track keys for these global IDs
                                keys1 = [k for k, v in self.global_identities.items() 
                                       if v == id1 and k.startswith(f"{camera_id}_")]
                                keys2 = [k for k, v in self.global_identities.items() 
                                       if v == id2 and k.startswith(f"{camera_id}_")]
                                
                                # Check if either likely moved to table area (outside camera view)
                                table_likelihood1 = any(k in self.cafe_exit_likelihood and 
                                                      self.cafe_exit_likelihood[k] > 0.5 for k in keys1)
                                table_likelihood2 = any(k in self.cafe_exit_likelihood and 
                                                      self.cafe_exit_likelihood[k] > 0.5 for k in keys2)
                                
                                # If either likely moved to table, reduce effective overlap duration
                                if table_likelihood1 or table_likelihood2:
                                    overlap_duration *= 0.4  # Reduced effective overlap (more than before)
                    else:
                        # Standard overlap calculation for shop
                        if times1[-1] < times2[0] or times2[-1] < times1[0]:
                            overlap_duration = 0
                        else:
                            overlap_start = max(times1[0], times2[0])
                            overlap_end = min(times1[-1], times2[-1])
                            overlap_duration = max(0, overlap_end - overlap_start)
                    
                    # Calculate total durations
                    duration1 = times1[-1] - times1[0]
                    duration2 = times2[-1] - times2[0]
                    
                    # Allow merging if minimal temporal overlap
                    can_merge_time = overlap_duration < max_overlap_pct * min(duration1, duration2)
                    
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
                        
                        # ENHANCED: Favoring cosine similarity for low-res footage
                        feature_sim = 0.8 * cosine_sim + 0.2 * l2_sim
                        
                        # Check historical features too for better matching
                        max_hist_sim = 0
                        if id1 in self.feature_history and id2 in self.feature_history:
                            for feat1 in self.feature_history[id1]:
                                for feat2 in self.feature_history[id2]:
                                    hist_cosine = 1 - cosine(feat1.flatten(), feat2.flatten())
                                    hist_l2 = 1.0 - min(np.linalg.norm(feat1.flatten() - feat2.flatten()) / max_dist, 1.0)
                                    hist_sim = 0.8 * hist_cosine + 0.2 * hist_l2  # Adjusted for low-res
                                    max_hist_sim = max(max_hist_sim, hist_sim)
                        
                        # Combine current and historical similarities
                        feature_sim = max(feature_sim, 0.95 * max_hist_sim)  # Increased weight for historical matches
                        
                        # For Camera 1, add additional check for color similarity
                        if camera_id == 1 and feature_sim > threshold * 0.95:  # Close to threshold
                            # Check color histograms too for café identities
                            color_sim = 0
                            
                            # Get camera keys for these global IDs
                            keys1 = [k for k, v in self.global_identities.items() 
                                   if v == id1 and k.startswith(f"{camera_id}_")]
                            keys2 = [k for k, v in self.global_identities.items() 
                                   if v == id2 and k.startswith(f"{camera_id}_")]
                            
                            # Find most recent color histograms
                            for k1 in keys1:
                                for k2 in keys2:
                                    if k1 in self.color_features and k2 in self.color_features:
                                        if len(self.color_features[k1]) > 0 and len(self.color_features[k2]) > 0:
                                            c1 = self.color_features[k1][-1]
                                            c2 = self.color_features[k2][-1]
                                            current_sim = 1 - cosine(c1.flatten(), c2.flatten())
                                            color_sim = max(color_sim, current_sim)
                            
                            # Adjust threshold based on color similarity
                            if color_sim > 0.75:  # Increased from 0.7 for better low-res differentiation
                                effective_threshold = threshold * 0.96  # Slightly lower threshold (0.97 -> 0.96)
                            else:
                                effective_threshold = threshold * 1.04  # Slightly higher threshold (1.03 -> 1.04)
                        else:
                            effective_threshold = threshold
                        
                        # Merge if similarity is high enough
                        if feature_sim > effective_threshold:
                            merged_ids.add(id2)
                            id_mappings[id2] = id1
                            # Track merge counts to prevent chain merging
                            merge_counts[id1] += 1
                            logger.debug(f"Merging identity {id2} into {id1}, similarity: {feature_sim:.4f}")
            
            # Log how many identities were merged for this camera
            camera_merged = sum(1 for id2, id1 in id_mappings.items() 
                              if id2 in global_ids and id1 in global_ids)
            logger.info(f"Merged {camera_merged} identities in Camera {camera_id}")
        
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
                    
                # Combine feature histories
                if old_id in self.feature_history and new_id in self.feature_history:
                    self.feature_history[new_id].extend(self.feature_history[old_id])
                    # Limit to max history length
                    if len(self.feature_history[new_id]) > self.max_features_history:
                        self.feature_history[new_id] = self.feature_history[new_id][-self.max_features_history:]
                    del self.feature_history[old_id]
                
                # Update feature database - keep the target feature
                if old_id in self.feature_database:
                    # Just delete the source feature, keeping the target
                    del self.feature_database[old_id]
                    
                # Transfer café exit and shop entry likelihoods
                for key in keys_to_update:
                    if key in self.cafe_exit_likelihood:
                        # Find corresponding key for new_id
                        new_keys = [k for k, v in self.global_identities.items() if v == new_id and k.startswith(key.split('_')[0])]
                        if new_keys:
                            # Transfer the higher exit likelihood
                            current = self.cafe_exit_likelihood.get(new_keys[0], 0)
                            self.cafe_exit_likelihood[new_keys[0]] = max(current, self.cafe_exit_likelihood[key])
                    
                    if key in self.shop_entry_likelihood:
                        new_keys = [k for k, v in self.global_identities.items() if v == new_id and k.startswith(key.split('_')[0])]
                        if new_keys:
                            current = self.shop_entry_likelihood.get(new_keys[0], 0)
                            self.shop_entry_likelihood[new_keys[0]] = max(current, self.shop_entry_likelihood[key])
            
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
        
        # ENHANCED: Initialize YOLO model with better parameters for low-resolution
        self.detector = YOLO("yolov8n.pt")  # Use smaller model for faster processing
        if torch.cuda.is_available():
            self.detector.to('cuda') 
            
        # Configure for performance
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
        
        # Track environment features
        self.scene_activity = defaultdict(list)  # Capture general scene activity
        self.motion_patterns = defaultdict(list)  # Track motion patterns
        self.door_activity_history = []  # Timeline of door activity
        self.appear_disappear_locations = []  # Track where people appear/disappear
        
        # Door regions specific to each camera
        self.door_regions = {
            1: [(1030, 0), (1700, 560)],  # Camera 1 door
            2: [(400, 0), (800, 470)]     # Camera 2 door
        }
        
        # Extended regions of interest for Camera 1 (tables near door)
        if self.camera_id == 1:
            # Define regions where people might sit down (outside visible area)
            self.table_regions = [
                [(950, 300), (1200, 560)],   # Left table area
                [(1200, 300), (1500, 560)]   # Right table area
            ]
        
        # Door interaction tracking
        self.door_entries = set()  # Tracks that entered through door
        self.door_exits = set()    # Tracks that exited through door
        
        # ENHANCED: Set camera-specific tracking parameters based on environment
        if self.camera_id == 1:  # Café environment - very challenging with partial coverage
            # Extremely permissive parameters for café with tables outside view
            self.detection_threshold = 0.10  # Very low threshold to detect more people (reduced from 0.15)
            self.matching_threshold = 0.25   # Lower to avoid identity switches (reduced from 0.30)
            self.feature_weight = 0.80       # High weight on appearance features (slightly reduced)
            self.position_weight = 0.20      # Increased weight on position (helpful for low-res)
            self.max_disappeared = self.fps * 15  # Allow longer disappearance times (increased from 10)
            self.max_lost_age = self.fps * 60     # Keep lost tracks much longer (increased from 45)
            self.merge_threshold = 0.90   # Slightly reduced from 0.92 for low-res
            
            # For handling lighting variations in café
            self.use_lighting_compensation = True
            self.use_enhanced_color_features = True
            self.use_texture_features = True
        else:  # Food shop environment - simpler with full coverage
            # More standard parameters for shop environment - adjusted for low-res
            self.detection_threshold = 0.25  # Reduced from 0.40
            self.matching_threshold = 0.40   # Reduced from 0.50
            self.feature_weight = 0.65       # Reduced feature weight for low-res
            self.position_weight = 0.35      # Increased position weight
            self.max_disappeared = self.fps * 8   # Increased from 5 seconds
            self.max_lost_age = self.fps * 30     # Increased from 20 seconds
            self.merge_threshold = 0.60   # Reduced from 0.65 for low-res
            
            # Lighting handling for shop
            self.use_lighting_compensation = True  # Enabled for low-res (was False)
            self.use_enhanced_color_features = True  # Enabled for low-res (was False)
            self.use_texture_features = False
            
        # ENHANCED: Track quality thresholds - critical for proper counting
        if self.camera_id == 1:
            # Extremely permissive for café since people might be briefly visible
            self.min_track_duration = 0.1   # Reduced from 0.2 seconds
            self.min_detections = 1         # Reduced from 2
            self.count_door_interactions = True  # Count even brief door interactions
            self.count_disappeared_tracks = True  # Count tracks that disappear near tables
        else:
            # Standard thresholds for shop - but more lenient for low-res
            self.min_track_duration = 0.8   # Reduced from 1.5 seconds
            self.min_detections = 2         # Reduced from 4
            self.count_door_interactions = True  # Changed to True
            self.count_disappeared_tracks = False
        
        # Track consolidation parameters - very infrequent for Camera 1
        self.consolidation_frequency = 75 if self.camera_id == 1 else 25  # Increased for both
        
        # Multi-scale and feature optimization parameters
        self.use_mixed_precision = True
        self.feature_res = (224, 448)  # Slightly reduced resolution for better performance
        self.use_multi_scale = True
        
        # ENHANCED: Expanded multi-scale factors for both cameras
        if self.camera_id == 1:
            self.multi_scale_factors = [0.6, 0.75, 0.9, 1.0, 1.1, 1.25]  # More scales for café
        else:
            self.multi_scale_factors = [0.7, 0.85, 1.0, 1.15]  # More scales for shop
        
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
        Extract ReID features from a person image with optimizations.
        Additional lighting compensation for Camera 1's variable lighting conditions.
        
        Args:
            person_crop: Cropped image of a person
            
        Returns:
            Feature vector or None if extraction fails
        """
        try:
            if person_crop.size == 0 or person_crop.shape[0] < 20 or person_crop.shape[1] < 20:
                return None
            
            # ENHANCED: Apply lighting compensation for both cameras in low-res
            if self.use_lighting_compensation:
                # CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
                lab = cv2.cvtColor(person_crop, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased clip limit
                cl = clahe.apply(l)
                enhanced = cv2.merge((cl, a, b))
                person_crop = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                
                # Gamma correction to balance lighting
                gamma = 1.3 if self.camera_id == 1 else 1.1  # Camera-specific gamma
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                person_crop = cv2.LUT(person_crop, table)
            
            # ENHANCED: Add sharpening for low-resolution images
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            person_crop = cv2.filter2D(person_crop, -1, kernel)
            
            # Basic preprocessing with resolution
            img = cv2.resize(person_crop, self.feature_res)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract texture-based features for Camera 1 if enabled
            texture_features = None
            if self.camera_id == 1 and self.use_texture_features:
                # LBP (Local Binary Pattern) for texture encoding
                gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                # Compute simplified texture features
                texture_features = np.zeros((8, 8), dtype=np.float32)
                h, w = gray.shape
                cell_h, cell_w = h // 8, w // 8
                for i in range(8):
                    for j in range(8):
                        # Calculate mean gradient magnitude in each cell as texture feature
                        cell = gray[i*cell_h:min((i+1)*cell_h, h), j*cell_w:min((j+1)*cell_w, w)]
                        if cell.size > 0:
                            gx = cv2.Sobel(cell, cv2.CV_32F, 1, 0)
                            gy = cv2.Sobel(cell, cv2.CV_32F, 0, 1)
                            mag = np.sqrt(gx**2 + gy**2)
                            texture_features[i, j] = np.mean(mag)
                texture_features = texture_features.flatten() / (texture_features.max() + 1e-6)
            
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
                
            # Combine with texture features if available
            reid_features = features.cpu().numpy()
            if texture_features is not None:
                # Normalize and reshape texture features to match reid features shape
                texture_features = texture_features.reshape(1, -1)
                # We'll add texture features as additional inputs to the matching process
                # Store them separately to be used in matching
                self.last_texture_features = texture_features
                
            return reid_features
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
        Extract enhanced color histogram features from a person image.
        For Camera 1, uses advanced techniques to handle varying lighting conditions.
        
        Args:
            person_crop: Cropped image of a person
            
        Returns:
            Color histogram features or None if extraction fails
        """
        try:
            if person_crop.size == 0 or person_crop.shape[0] < 20 or person_crop.shape[1] < 20:
                return None
            
            # Enhanced color processing for both cameras with lighting variations
            if self.use_enhanced_color_features:
                # ENHANCED: Add preprocessing for low-resolution
                # Slight blur to reduce noise in low-resolution images
                person_crop = cv2.GaussianBlur(person_crop, (3, 3), 0)
                
                # Convert to Lab color space which separates luminance from color
                lab = cv2.cvtColor(person_crop, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Calculate histograms with focus on color channels (a, b) which are less affected by lighting
                hist_l = cv2.calcHist([l], [0], None, [24], [0, 256])
                hist_a = cv2.calcHist([a], [0], None, [36], [0, 256])
                hist_b = cv2.calcHist([b], [0], None, [36], [0, 256])
                
                # Normalize histograms
                hist_l = cv2.normalize(hist_l, hist_l).flatten()
                hist_a = cv2.normalize(hist_a, hist_a).flatten()
                hist_b = cv2.normalize(hist_b, hist_b).flatten()
                
                # Use a spatial pyramid for upper and lower body separately (people often have different colors)
                h, w = person_crop.shape[:2]
                upper_body = person_crop[0:h//2, :]
                lower_body = person_crop[h//2:, :]
                
                # Process upper body color
                if upper_body.size > 0:
                    hsv_upper = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
                    hist_h_upper = cv2.calcHist([hsv_upper], [0], None, [24], [0, 180])
                    hist_s_upper = cv2.calcHist([hsv_upper], [1], None, [24], [0, 256])
                    hist_h_upper = cv2.normalize(hist_h_upper, hist_h_upper).flatten()
                    hist_s_upper = cv2.normalize(hist_s_upper, hist_s_upper).flatten()
                else:
                    hist_h_upper = np.zeros(24)
                    hist_s_upper = np.zeros(24)
                
                # Process lower body color
                if lower_body.size > 0:
                    hsv_lower = cv2.cvtColor(lower_body, cv2.COLOR_BGR2HSV)
                    hist_h_lower = cv2.calcHist([hsv_lower], [0], None, [24], [0, 180])
                    hist_s_lower = cv2.calcHist([hsv_lower], [1], None, [24], [0, 256])
                    hist_h_lower = cv2.normalize(hist_h_lower, hist_h_lower).flatten()
                    hist_s_lower = cv2.normalize(hist_s_lower, hist_s_lower).flatten()
                else:
                    hist_h_lower = np.zeros(24)
                    hist_s_lower = np.zeros(24)
                
                # Create lighting-robust color descriptor with camera-specific weights
                if self.camera_id == 1:
                    return np.concatenate([
                        hist_l * 0.4,      # Further reduce weight of luminance for café
                        hist_a * 1.3,      # Increase weight of a channel
                        hist_b * 1.3,      # Increase weight of b channel
                        hist_h_upper * 1.1,  # Slightly increased upper body hue weight
                        hist_s_upper,      # Upper body saturation
                        hist_h_lower * 1.1,  # Slightly increased lower body hue weight
                        hist_s_lower       # Lower body saturation
                    ])
                else:
                    return np.concatenate([
                        hist_l * 0.5,      # Reduce weight of luminance
                        hist_a * 1.2,      # Increase weight of a channel
                        hist_b * 1.2,      # Increase weight of b channel
                        hist_h_upper,      # Upper body hue
                        hist_s_upper,      # Upper body saturation
                        hist_h_lower,      # Lower body hue
                        hist_s_lower       # Lower body saturation
                    ])
                
            else:
                # Standard HSV histogram fallback
                hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
                
                # Calculate histograms for each channel
                hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
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
        
        # ENHANCED: Camera-specific buffer sizes - increased for low-res
        buffer = 160 if self.camera_id == 1 else 110  # Increased from 120/80
        
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
        # ENHANCED: Lower position requirements for low-res
        min_positions = 1 if self.camera_id == 1 else 2  # Reduced from 2/3
            
        if track_id not in self.track_positions or len(self.track_positions[track_id]) < min_positions:
            return False, False
            
        # Get first few and last few positions
        first_positions = self.track_positions[track_id][:min_positions+1]  # Add one more position
        last_positions = self.track_positions[track_id][-min_positions-1:]  # Add one more position
        
        # Different detection criteria by camera
        if self.camera_id == 1:
            # More lenient for Camera 1 café environment
            is_entering = any(self.is_in_door_region(pos) for pos in first_positions)
            is_exiting = any(self.is_in_door_region(pos) for pos in last_positions)
        else:
            # More lenient for Camera 2 shop environment in low-res
            is_entering = sum(1 for pos in first_positions if self.is_in_door_region(pos)) >= 1  # Reduced from 2
            is_exiting = sum(1 for pos in last_positions if self.is_in_door_region(pos)) >= 1  # Reduced from 2
        
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
            # Calculate feature similarity using multiple metrics
            cosine_sim = 1 - cosine(detection_features.flatten(), 
                                   track_info['features'].flatten())
            
            # Calculate L2 distance (Euclidean) - normalized
            l2_dist = np.linalg.norm(detection_features.flatten() - track_info['features'].flatten())
            max_dist = 2.0  # Approximate maximum distance for normalized features
            l2_sim = 1.0 - min(l2_dist / max_dist, 1.0)
            
            # ENHANCED: Combined feature similarity for low-res - favor cosine similarity
            feature_sim = 0.85 * cosine_sim + 0.15 * l2_sim
            
            # Also check historical features for better matching
            if track_id in self.feature_history and len(self.feature_history[track_id]) > 0:
                hist_sims = []
                for feat in self.feature_history[track_id]:
                    # Calculate both similarity metrics for historical features
                    hist_cosine = 1 - cosine(detection_features.flatten(), feat.flatten())
                    hist_l2 = 1.0 - min(np.linalg.norm(detection_features.flatten() - feat.flatten()) / max_dist, 1.0)
                    hist_sims.append(0.85 * hist_cosine + 0.15 * hist_l2)
                
                if hist_sims:
                    # Consider best historical match
                    feature_sim = max(feature_sim, 0.95 * max(hist_sims))
            
            # Calculate position similarity (IOU)
            position_sim = self.calculate_iou(detection_box, track_info['box'])
            
            # ENHANCED: Add proximity calculation for low-res (center distance)
            center1 = ((detection_box[0] + detection_box[2])/2, (detection_box[1] + detection_box[3])/2)
            center2 = ((track_info['box'][0] + track_info['box'][2])/2, (track_info['box'][1] + track_info['box'][3])/2)
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            # Normalize distance by frame diagonal
            diagonal = np.sqrt(self.frame_width**2 + self.frame_height**2)
            proximity = max(0, 1 - distance/(diagonal/4))  # Zero if distance > 1/4 diagonal
            
            # Combined position similarity (IOU and proximity)
            position_sim = 0.7 * position_sim + 0.3 * proximity
            
            # Combined similarity - balanced weights from class parameters
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
                
                # ENHANCED: Combined feature similarity for low-res
                feature_sim = 0.85 * cosine_sim + 0.15 * l2_sim
                
                # Also check historical features
                if track_id in self.feature_history and len(self.feature_history[track_id]) > 0:
                    hist_sims = []
                    for feat in self.feature_history[track_id]:
                        # Calculate both similarity metrics for historical features
                        hist_cosine = 1 - cosine(detection_features.flatten(), feat.flatten())
                        hist_l2 = 1.0 - min(np.linalg.norm(detection_features.flatten() - feat.flatten()) / max_dist, 1.0)
                        hist_sims.append(0.85 * hist_cosine + 0.15 * hist_l2)
                    
                    if hist_sims:
                        # Consider best historical match
                        feature_sim = max(feature_sim, 0.95 * max(hist_sims))
                
                # Calculate position similarity (IOU)
                position_sim = self.calculate_iou(detection_box, track_info['box'])
                
                # ENHANCED: Add proximity for lost tracks
                center1 = ((detection_box[0] + detection_box[2])/2, (detection_box[1] + detection_box[3])/2)
                center2 = ((track_info['box'][0] + track_info['box'][2])/2, (track_info['box'][1] + track_info['box'][3])/2)
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                diagonal = np.sqrt(self.frame_width**2 + self.frame_height**2)
                proximity = max(0, 1 - distance/(diagonal/3))  # More lenient for lost tracks
                
                # Combined position similarity
                position_sim = 0.6 * position_sim + 0.4 * proximity
                
                # Consider time since last seen - closer in time is better
                time_factor = max(0, 1.0 - (frame_time - track_info['last_seen']) / self.max_lost_age)
                
                # ENHANCED: Combined similarity for lost tracks - adjusted for low-res
                similarity = (0.70 * feature_sim + 0.20 * position_sim + 0.10 * time_factor)
                
                # Camera-specific recover thresholds
                recover_threshold = self.matching_threshold - 0.12 if self.camera_id == 1 else self.matching_threshold - 0.08
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
        if len(self.feature_history[track_id]) > 15:  # Increased from 10
            self.feature_history[track_id].pop(0)
            
        # Update feature representation with exponential moving average
        if track_id in self.person_features:
            # ENHANCED: Faster updating for low-res footage
            alpha = 0.65 if self.camera_id == 1 else 0.60  # Reduced from 0.7/0.65
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
        # ENHANCED: Camera-specific edge filtering - even less aggressive for low-res
        if self.camera_id == 1:
            edge_margin = 0  # No edge filtering for café
            if (bbox[0] < edge_margin or bbox[2] > self.frame_width - edge_margin or 
                bbox[1] < edge_margin or bbox[3] > self.frame_height - edge_margin):
                # Only filter extreme edge cases
                if not self.is_in_door_region(bbox):
                    # For café, still accept all detections
                    pass
        else:
            # Less strict edge filtering for Camera 2 in low-res
            edge_margin = 10  # Reduced from 15
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
                    
                    # ENHANCED: Camera-specific overlap threshold for low-res
                    # Still strict for Camera 1 to avoid merging different people
                    max_overlap = 0.3 if self.camera_id == 1 else 0.5  # Increased for both
                    
                    if overlap_duration > max_overlap * min(track1_duration, track2_duration):
                        continue
                
                # Calculate feature similarity - using multiple metrics
                cosine_sim = 1 - cosine(self.person_features[track_id1].flatten(),
                                       self.person_features[track_id2].flatten())
                
                # Calculate L2 distance (Euclidean) - normalized
                l2_dist = np.linalg.norm(self.person_features[track_id1].flatten() - 
                                        self.person_features[track_id2].flatten())
                max_dist = 2.0  # Approximate maximum distance
                l2_sim = 1.0 - min(l2_dist / max_dist, 1.0)
                
                # ENHANCED: Combined feature similarity for low-res
                feature_sim = 0.85 * cosine_sim + 0.15 * l2_sim
                
                # Check historical features too
                for feat1 in self.feature_history[track_id1]:
                    for feat2 in self.feature_history[track_id2]:
                        # Calculate both similarity metrics for historical features
                        hist_cosine = 1 - cosine(feat1.flatten(), feat2.flatten())
                        hist_l2 = 1.0 - min(np.linalg.norm(feat1.flatten() - feat2.flatten()) / max_dist, 1.0)
                        hist_sim = 0.85 * hist_cosine + 0.15 * hist_l2
                        feature_sim = max(feature_sim, 0.95 * hist_sim)
                
                # ENHANCED: Camera-specific feature threshold - adjusted for low-res
                feature_threshold = 0.72 if self.camera_id == 1 else 0.55  # Reduced for low-res
                
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
                pos_sim = max(0, 1 - pos_dist / (max_dist/3))  # Increased from max_dist/4
                
                # Combined similarity score - balanced weights
                combined_sim = 0.65 * feature_sim + 0.15 * color_sim + 0.20 * pos_sim  # Adjusted weights
                
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
            if len(self.feature_history[track_id1]) > 15:  # Increased from 10
                self.feature_history[track_id1] = self.feature_history[track_id1][-15:]
        
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
        Filter out invalid detections with camera-specific criteria.
        Much more permissive for Camera 1 due to partial visibility and lighting issues.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            conf: Detection confidence
            
        Returns:
            Boolean indicating if the detection is valid
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Get the center of the box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # ENHANCED: Size checks - extremely lenient for low-res
        if self.camera_id == 1:
            # Very permissive size filtering for café where people might be partially visible
            if width < 12 or height < 25:  # Reduced from 15/30
                return False
        else:
            # Standard size requirements for shop with full visibility
            if width < 20 or height < 40:  # Reduced from 30/50
                return False
            
        # Check aspect ratio (typical human aspect ratio)
        aspect_ratio = height / width
        
        # ENHANCED: Camera 1 needs much wider aspect ratio range due to partial visibility and various poses
        if self.camera_id == 1:
            min_aspect = 0.8  # Reduced from 1.0
            max_aspect = 4.5  # Increased from 4.0
            
            # Special case: Always accept detections near door region regardless of aspect ratio
            if self.is_in_door_region(bbox):
                # Still enforce minimal sanity checks
                if width >= 12 and height >= 25 and aspect_ratio > 0.5 and aspect_ratio < 5.0:
                    return True
                    
            # Special case: Accept detections near table areas with different criteria
            for table_region in self.table_regions:
                tx1, ty1 = table_region[0]
                tx2, ty2 = table_region[1]
                # Check if detection center is near table region
                if (tx1 - 70 <= center_x <= tx2 + 70) and (ty1 - 70 <= center_y <= ty2 + 70):  # Increased from 50
                    # More permissive criteria for table regions
                    if width >= 12 and height >= 25 and aspect_ratio > 0.6 and aspect_ratio < 5.0:
                        return True
        else:
            # Standard aspect ratio checks for shop - more lenient for low-res
            min_aspect = 1.2  # Reduced from 1.4
            max_aspect = 3.2  # Increased from 3.0
            
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            return False
            
        # Filter out detections with too large or too small areas
        area = width * height
        
        # ENHANCED: Area thresholds - extremely permissive for low-res
        if self.camera_id == 1:
            min_area = 300  # Reduced from 450
            # Larger max area for café since people might be closer to camera
            max_area = 0.50 * self.frame_width * self.frame_height  # Increased from 0.45
        else:
            min_area = 800  # Reduced from 1500
            max_area = 0.40 * self.frame_width * self.frame_height  # Increased from 0.35
        
        if area < min_area or area > max_area:
            return False
            
        # Camera-specific edge checks
        if self.camera_id == 1:
            # For café (Camera 1) - practically no edge filtering
            edge_margin = 0  # No edge filtering for café
            
            if x1 < edge_margin or x2 > self.frame_width - edge_margin:
                # Check if near door or table regions
                if self.is_in_door_region(bbox):
                    return True
                    
                # Check if near table regions where people might disappear
                for table_region in self.table_regions:
                    tx1, ty1 = table_region[0]
                    tx2, ty2 = table_region[1]
                    # Allow if near table region
                    if (tx1 - 150 <= center_x <= tx2 + 150) and (ty1 - 150 <= center_y <= ty2 + 150):  # Increased from 100
                        return True
                
                # Accept most edge detections in cafe anyway (low-res setting)
                return True
        else:
            # For food shop (Camera 2) - more relaxed in low-res
            if y1 < 5:  # Reduced from 10
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
        # Run consolidation periodically - adjusted for low-res
        if frame_count % self.consolidation_frequency == 0 and frame_count > 0:
            merged = self.consolidate_tracks()
            if merged > 0:
                logger.info(f"Merged {merged} tracks at frame {frame_count}")
        
        # ENHANCED: Detect persons with improved settings for low-res
        # Apply pre-processing for low-res video
        if self.camera_id == 1:
            # For Cafe - apply stronger enhancements
            # Slight sharpening 
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            frame_processed = cv2.filter2D(frame, -1, kernel)
            # Contrast enhancement
            lab = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            frame_processed = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # For Shop - lighter enhancement
            kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]])
            frame_processed = cv2.filter2D(frame, -1, kernel)
        
        # Use processed frame for detection
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision) if torch.cuda.is_available() else nullcontext():
            # ENHANCED: Use augmented inference with IoU and class-agnostic NMS for better low-res detection
            results = self.detector(
                frame_processed, 
                classes=[0],  # Only detect people
                conf=self.detection_threshold,
                iou=0.4,  # Lowered from default 0.45 for better low-res detection
            )
        
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
            
            # ENHANCED: Relaxed minimum size requirements for low-res
            min_width = 15 if self.camera_id == 1 else 20  # Reduced from 25/30
            min_height = 30 if self.camera_id == 1 else 40  # Reduced from 45/50
            
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
        
        # ENHANCED: Process more frames for low-res video (previously stride=1)
        stride = 1  # Process every frame
        
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
                logger.info("Processed %d frames, active tracks: %d, lost tracks: %d", 
                           frame_count, len(self.active_tracks), len(self.lost_tracks))
        
        self.cap.release()
        if visualize:
            cv2.destroyAllWindows()
            
        process_time = time.time() - start_time
        logger.info("Completed processing %s in %.2f seconds", 
                   self.video_name, process_time)
        
        # ENHANCED: Perform multiple final track consolidation rounds
        consolidation_rounds = 4 if self.camera_id == 1 else 3  # Increased from 3/2
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
                
        # ENHANCED: Also check lost tracks for door interactions
        for track_id in self.lost_tracks:
            is_entering, is_exiting = self.detect_door_interaction(track_id)
            if is_entering:
                self.door_entries.add(track_id)
            if is_exiting:
                self.door_exits.add(track_id)
                
        return self.get_valid_tracks()

    def get_valid_tracks(self):
        """
        Get valid tracks that meet quality criteria.
        Camera 1 uses special logic to count partial tracks near tables/doors.
        
        Returns:
            Dictionary of valid tracks
        """
        # Get valid tracks
        valid_tracks = {}
        processed_tracks = set()
        
        # ENHANCED: Consider both active and lost tracks for final count
        all_tracks = set(self.track_timestamps.keys())
        
        # First pass: Standard validation
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
                processed_tracks.add(track_id)
        
        # Second pass: Special handling for tracks that interact with doors or disappear near tables
        if self.camera_id == 1:
            # Enhanced track validation for café environment
            for track_id in all_tracks:
                if track_id in processed_tracks:
                    continue
                    
                # Skip tracks with too few detections (maintain minimal quality)
                if track_id not in self.feature_history or len(self.feature_history[track_id]) < self.min_detections:
                    continue
                
                # Check for door interactions
                if self.count_door_interactions and (track_id in self.door_entries or track_id in self.door_exits):
                    # Accept tracks that interact with doors even if they're short
                    duration = (self.track_timestamps[track_id]['last_appearance'] - 
                              self.track_timestamps[track_id]['first_appearance'])
                    
                    # Much more lenient duration requirement for door interactions in low-res
                    if duration >= self.min_track_duration * 0.25:  # Reduced from 0.5
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
                        processed_tracks.add(track_id)
                        continue
                
                # Check for tracks that disappear near table regions (people sitting down outside camera view)
                if self.count_disappeared_tracks and track_id in self.track_positions and len(self.track_positions[track_id]) > 0:
                    # Check if last position is near a table region
                    last_pos = self.track_positions[track_id][-1]
                    center_x = (last_pos[0] + last_pos[2]) / 2
                    center_y = (last_pos[1] + last_pos[3]) / 2
                    
                    near_table = False
                    for table_region in self.table_regions:
                        tx1, ty1 = table_region[0]
                        tx2, ty2 = table_region[1]
                        # Check if detection center is near table region - increased region for low-res
                        if (tx1 - 200 <= center_x <= tx2 + 200) and (ty1 - 200 <= center_y <= ty2 + 200):  # Increased from 150
                            near_table = True
                            break
                    
                    if near_table:
                        # Accept tracks that disappear near tables
                        duration = (self.track_timestamps[track_id]['last_appearance'] - 
                                  self.track_timestamps[track_id]['first_appearance'])
                        # More lenient quality requirements for near-table disappearances
                        if duration >= self.min_track_duration * 0.25 and len(self.feature_history[track_id]) >= self.min_detections:
                            valid_tracks[track_id] = {
                                'id': track_id,
                                'features': self.person_features.get(track_id),
                                'color_histogram': self.color_histograms.get(track_id),
                                'first_appearance': self.track_timestamps[track_id]['first_appearance'],
                                'last_appearance': self.track_timestamps[track_id]['last_appearance'],
                                'duration': duration,
                                'is_entry': track_id in self.door_entries,
                                'is_exit': track_id in self.door_exits,
                                'detections': len(self.feature_history.get(track_id, [])),
                                'near_table': True  # Mark as disappearing near table
                            }
                            processed_tracks.add(track_id)
        else:  # For Camera 2 in low-res
            # ENHANCED: Special handling for Camera 2 door interactions
            for track_id in all_tracks:
                if track_id in processed_tracks:
                    continue
                    
                # Skip tracks with too few detections
                if track_id not in self.feature_history or len(self.feature_history[track_id]) < self.min_detections:
                    continue
                
                # Check for door interactions in Camera 2
                if self.count_door_interactions and (track_id in self.door_entries or track_id in self.door_exits):
                    duration = (self.track_timestamps[track_id]['last_appearance'] - 
                              self.track_timestamps[track_id]['first_appearance'])
                    
                    # Accept shorter tracks that interact with doors
                    if duration >= self.min_track_duration * 0.5:  # Half the normal duration requirement
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
                        processed_tracks.add(track_id)
        
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
    
    # ENHANCED: Process each day's videos with improved global tracking
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
        
        # ENHANCED: Apply multiple rounds of identity cleaning
        for i in range(3):  # Run 3 rounds of cleaning
            global_tracker.clean_similar_identities()
            logger.info(f"Completed cleaning round {i+1} for date {date}")
        
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
    
    # ENHANCED: Command line argument support
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Camera Person Tracking")
    parser.add_argument("--input", type=str, default="/home/mchen/Projects/VISIONARY/videos/test_data",
                        help="Directory containing camera videos")
    parser.add_argument("--output", type=str, default="/home/mchen/Projects/VISIONARY/results",
                        help="Directory to store results")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization during processing")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Path to YOLO model (default: yolov8n.pt)")
    args = parser.parse_args()
    
    # Override model if specified
    PersonTracker.yolo_model_path = args.model
    
    # Process the videos
    results = process_video_directory(args.input, args.output)
    
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
        print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()