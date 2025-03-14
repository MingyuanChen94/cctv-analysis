import os
import cv2
import numpy as np
import torch
import torchreid
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
from scipy.spatial.distance import cdist, cosine
from pathlib import Path
import json
import pandas as pd
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MultiCameraTracker')

class TrackingState:
    ACTIVE = 'active'
    TENTATIVE = 'tentative'
    LOST = 'lost'

class GlobalTracker:
    """
    Manages cross-camera identity matching and transition detection
    """
    def __init__(self):
        # Maps camera-specific tracks to global identities
        self.global_identities = {}
        # Stores sequence of camera appearances for each global identity
        self.appearance_sequence = {}
        # Stores feature vectors for each global identity
        self.feature_database = {}
        # Stores color histograms for each identity
        self.color_features = {}
        
        # Parameters for cross-camera matching
        self.min_transition_time = 30   # Minimum time (seconds) for camera transition
        self.max_transition_time = 300  # Maximum time (seconds) for camera transition
        self.cross_camera_threshold = 0.65  # Similarity threshold for cross-camera matching
        
        # Track door interactions
        self.door_exits = defaultdict(list)  # Tracks exiting through doors
        self.door_entries = defaultdict(list)  # Tracks entering through doors
        
        # Stores feature history for each identity
        self.feature_history = defaultdict(list)
        self.max_features_history = 5  # Max features to store per identity
        
        # Camera-specific track counts
        self.camera1_tracks = set()
        self.camera2_tracks = set()

    def register_detection(self, camera_id, track_id, features, timestamp, 
                           color_hist=None, is_entry=False, is_exit=False):
        """
        Register a detection from a specific camera
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
        """Match with existing global identity or create a new one"""
        camera_key = f"{camera_id}_{track_id}"
        
        # If we've seen this camera-specific track before, return its global ID
        if camera_key in self.global_identities:
            return self.global_identities[camera_key]
        
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
            # This is based on the physical layout described - people move from café to food shop
            if last_camera == 2 and camera_id == 1:
                continue
                
            # For a potential transition from Camera 1 to Camera 2,
            # check if person exited from Camera 1 and is entering Camera 2
            transition_bonus = 0
            if last_camera == 1 and camera_id == 2:
                # Find the camera1 track key that matches this global ID
                camera1_keys = [k for k, v in self.global_identities.items() 
                              if v == global_id and k.startswith('1_')]
                
                if camera1_keys and camera1_keys[0] in self.door_exits and camera_key in self.door_entries:
                    transition_bonus = 0.2
            
            # Calculate feature similarity
            feature_sim = 1 - cosine(features.flatten(), stored_features.flatten())
            
            # Calculate color similarity if available
            color_sim = 0
            if camera_key in self.color_features:
                camera1_keys = [k for k, v in self.global_identities.items() 
                              if v == global_id and k.startswith('1_')]
                if camera1_keys and camera1_keys[0] in self.color_features:
                    color_feats1 = self.color_features[camera1_keys[0]][-1]
                    color_feats2 = self.color_features[camera_key][-1]
                    color_sim = 1 - cosine(color_feats1.flatten(), color_feats2.flatten())
            
            # Calculate time-based penalty (transitions too fast or too slow are less likely)
            # Optimal transition time is around 2 minutes
            time_factor = max(0, 1.0 - abs(time_diff - 120) / 120)
            
            # Combined similarity score
            similarity = (0.5 * feature_sim + 
                          0.2 * color_sim + 
                          0.2 * time_factor +
                          transition_bonus)
            
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
            alpha = 0.7  # Weight for historical features
            self.feature_database[best_match_id] = (
                alpha * self.feature_database[best_match_id] + 
                (1 - alpha) * features
            )
        
        # Register the global identity for this camera-specific track
        self.global_identities[camera_key] = best_match_id
        return best_match_id

    def analyze_camera_transitions(self):
        """Analyze transitions between cameras"""
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
                        # Validate that current appearance either exited through door
                        # or next appearance entered through door
                        is_valid_transition = (current.get('is_exit', False) or 
                                             next_app.get('is_entry', False))
                        
                        if is_valid_transition:
                            camera1_to_camera2 += 1
                            valid_transitions.append({
                                'global_id': global_id,
                                'exit_time': current['timestamp'],
                                'entry_time': next_app['timestamp'],
                                'transit_time': time_diff
                            })
        
        # Calculate unique individuals per camera based on global IDs
        unique_camera1 = len(set(self.global_identities[key] for key in self.camera1_tracks))
        unique_camera2 = len(set(self.global_identities[key] for key in self.camera2_tracks))
        
        return {
            'camera1_to_camera2': camera1_to_camera2,
            'unique_camera1': unique_camera1,
            'unique_camera2': unique_camera2,
            'valid_transitions': valid_transitions
        }

class PersonTracker:
    """
    Tracks individuals within a single camera view
    """
    def __init__(self, video_path, output_dir="tracking_results"):
        self.video_name = Path(video_path).stem
        self.camera_id = int(self.video_name.split('_')[1])  # Extract camera ID
        self.date = self.video_name.split('_')[-1]  # Extract date
        
        # Create output directories
        self.output_dir = os.path.join(output_dir, self.video_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize models
        self.detector = YOLO("yolov8x.pt")
        self.reid_model = self._initialize_reid_model()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
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
        
        # Tracking parameters - adjusted for better results
        if self.camera_id == 1:  # Café environment (more complex)
            self.detection_threshold = 0.5
            self.matching_threshold = 0.75
            self.feature_weight = 0.7
            self.position_weight = 0.3
            self.max_disappeared = self.fps * 3  # 3 seconds
            self.max_lost_age = self.fps * 30    # 30 seconds
            # More aggressive track merging for Camera 1
            self.merge_threshold = 0.65
        else:  # Food shop environment (cleaner)
            self.detection_threshold = 0.4
            self.matching_threshold = 0.7
            self.feature_weight = 0.65
            self.position_weight = 0.35
            self.max_disappeared = self.fps * 2  # 2 seconds
            self.max_lost_age = self.fps * 20    # 20 seconds
            # Less aggressive merging for Camera 2
            self.merge_threshold = 0.7
            
        # Track quality thresholds
        self.min_track_duration = 1.0  # Minimum 1 second
        self.min_detections = 3        # Minimum 3 detections
        
        # Track consolidation parameters
        self.consolidation_frequency = 30  # Frames between consolidation
        
        logger.info(f"Initialized tracker for {video_path}")

    def _initialize_reid_model(self):
        """Initialize the ReID model"""
        model = torchreid.models.build_model(
            name='osnet_ain_x1_0',
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
        """Extract ReID features from a person image"""
        try:
            if person_crop.size == 0 or person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
                return None
                
            # Preprocess image
            img = cv2.resize(person_crop, (128, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            img = torch.from_numpy(img).float() / 255.0
            img = img.permute(2, 0, 1).unsqueeze(0)
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            img = (img - mean) / std
            
            if torch.cuda.is_available():
                img = img.cuda()
                
            # Extract features
            with torch.no_grad():
                features = self.reid_model(img)
                
            return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def extract_color_histogram(self, person_crop):
        """Extract color histogram features"""
        try:
            if person_crop.size == 0:
                return None
                
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            
            # Normalize histograms
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()
            
            return np.concatenate([hist_h, hist_s, hist_v])
        except Exception as e:
            logger.error(f"Error extracting color histogram: {e}")
            return None

    def is_in_door_region(self, bbox):
        """Check if a detection is in the door region"""
        door = self.door_regions[self.camera_id]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        x_min, y_min = door[0]
        x_max, y_max = door[1]
        
        buffer = 50  # Add buffer around door region
        
        return (x_min - buffer <= center_x <= x_max + buffer and 
                y_min - buffer <= center_y <= y_max + buffer)

    def detect_door_interaction(self, track_id):
        """Detect if a track is entering or exiting through a door"""
        if track_id not in self.track_positions or len(self.track_positions[track_id]) < 3:
            return False, False
            
        # Check first and last positions
        first_pos = self.track_positions[track_id][0]
        last_pos = self.track_positions[track_id][-1]
        
        # Consider track entering if first position is in door region
        is_entering = self.is_in_door_region(first_pos)
        
        # Consider track exiting if last position is in door region
        is_exiting = self.is_in_door_region(last_pos)
        
        return is_entering, is_exiting

    def match_detection(self, detection_features, detection_box, frame_time):
        """Match a detection with existing tracks"""
        best_match_id = None
        best_match_score = 0
        
        # Try to match with active tracks first
        for track_id, track_info in self.active_tracks.items():
            # Calculate feature similarity
            feature_sim = 1 - cosine(detection_features.flatten(), 
                                   track_info['features'].flatten())
            
            # Calculate position similarity (IOU)
            position_sim = self.calculate_iou(detection_box, track_info['box'])
            
            # Combined similarity
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
                    
                # Calculate feature similarity
                feature_sim = 1 - cosine(detection_features.flatten(), 
                                       track_info['features'].flatten())
                
                # Calculate position similarity (IOU)
                position_sim = self.calculate_iou(detection_box, track_info['box'])
                
                # Combined similarity but with higher threshold for lost tracks
                similarity = (self.feature_weight * feature_sim + 
                             self.position_weight * position_sim)
                
                # Need higher confidence to recover lost track
                if similarity > self.matching_threshold + 0.05 and similarity > best_match_score:
                    best_match_id = track_id
                    best_match_score = similarity
        
        return best_match_id

    def update_track(self, track_id, bbox, features, color_hist, frame_time):
        """Update an existing track with new detection"""
        if track_id in self.lost_tracks:
            # Recover lost track
            self.active_tracks[track_id] = self.lost_tracks[track_id]
            del self.lost_tracks[track_id]
            
        # Update track information
        self.active_tracks[track_id].update({
            'box': bbox,
            'features': features,
            'last_seen': frame_time,
            'disappeared': 0
        })
        
        # Update feature history
        self.feature_history[track_id].append(features)
        if len(self.feature_history[track_id]) > 10:  # Keep last 10 features
            self.feature_history[track_id].pop(0)
            
        # Update feature representation with exponential moving average
        if track_id in self.person_features:
            alpha = 0.7  # Weight for historical features
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

    def create_track(self, bbox, features, color_hist, frame_time):
        """Create a new track"""
        track_id = self.next_id
        self.next_id += 1
        
        self.active_tracks[track_id] = {
            'box': bbox,
            'features': features,
            'last_seen': frame_time,
            'disappeared': 0
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
        """Merge tracks that likely belong to the same person"""
        merged_tracks = set()
        
        for track_id1 in list(self.active_tracks.keys()):
            if track_id1 in merged_tracks:
                continue
                
            for track_id2 in list(self.active_tracks.keys()):
                if track_id1 == track_id2 or track_id2 in merged_tracks:
                    continue
                    
                # Calculate temporal overlap
                time1_start = self.track_timestamps[track_id1]['first_appearance']
                time1_end = self.track_timestamps[track_id1]['last_appearance']
                time2_start = self.track_timestamps[track_id2]['first_appearance']
                time2_end = self.track_timestamps[track_id2]['last_appearance']
                
                # Skip if tracks are temporally distant
                if time1_end < time2_start - 2 or time2_end < time1_start - 2:
                    continue
                
                # Calculate feature similarity
                feature_sim = 1 - cosine(self.person_features[track_id1].flatten(),
                                       self.person_features[track_id2].flatten())
                
                # Calculate positional proximity
                last_pos1 = self.track_positions[track_id1][-1]
                last_pos2 = self.track_positions[track_id2][-1]
                pos_dist = np.sqrt(
                    ((last_pos1[0] + last_pos1[2])/2 - (last_pos2[0] + last_pos2[2])/2)**2 +
                    ((last_pos1[1] + last_pos1[3])/2 - (last_pos2[1] + last_pos2[3])/2)**2
                )
                
                # Merge if similarity is high and tracks are close
                if feature_sim > self.merge_threshold and pos_dist < 150:
                    # Merge track2 into track1
                    self.merge_tracks(track_id1, track_id2)
                    merged_tracks.add(track_id2)
        
        return len(merged_tracks)
        
    def merge_tracks(self, track_id1, track_id2):
        """Merge track2 into track1"""
        # Update timestamps to cover entire span
        self.track_timestamps[track_id1]['first_appearance'] = min(
            self.track_timestamps[track_id1]['first_appearance'],
            self.track_timestamps[track_id2]['first_appearance']
        )
        self.track_timestamps[track_id1]['last_appearance'] = max(
            self.track_timestamps[track_id1]['last_appearance'],
            self.track_timestamps[track_id2]['last_appearance']
        )
        
        # Combine feature histories
        self.feature_history[track_id1].extend(self.feature_history[track_id2])
        
        # Combine position histories
        combined_positions = self.track_positions[track_id1] + self.track_positions[track_id2]
        # Sort positions by timestamp if available
        self.track_positions[track_id1] = combined_positions
        
        # Transfer door interaction flags
        if track_id2 in self.door_entries:
            self.door_entries.add(track_id1)
        if track_id2 in self.door_exits:
            self.door_exits.add(track_id1)
        
        # Remove track2
        if track_id2 in self.active_tracks:
            del self.active_tracks[track_id2]
        if track_id2 in self.person_features:
            del self.person_features[track_id2]
        if track_id2 in self.door_entries:
            self.door_entries.remove(track_id2)
        if track_id2 in self.door_exits:
            self.door_exits.remove(track_id2)

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
        """Filter out invalid detections"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Basic size checks
        if width < 20 or height < 40:
            return False
            
        # Check aspect ratio (typical human aspect ratio)
        aspect_ratio = height / width
        if aspect_ratio < 1.2 or aspect_ratio > 4.0:
            return False
            
        # Camera-specific checks
        if self.camera_id == 1:
            # For café (Camera 1), filter detections at edges
            if x1 < 10 or x2 > self.frame_width - 10:
                return False
        else:
            # For food shop (Camera 2), different constraints
            if y1 < 5:  # Filter detections that start at very top of frame
                return False
        
        return True

    def process_frame(self, frame, frame_time, frame_count):
        """Process a single frame"""
        # Run consolidation periodically
        if frame_count % self.consolidation_frequency == 0:
            merged = self.consolidate_tracks()
            if merged > 0:
                logger.info(f"Merged {merged} tracks at frame {frame_count}")
        
        # Detect persons
        results = self.detector(frame, classes=[0])  # class 0 is person
        
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                conf = float(box.conf[0])
                if conf >= self.detection_threshold:
                    bbox = [int(x) for x in box.xyxy[0]]
                    # Apply additional filtering
                    if self.filter_detection(bbox, conf):
                        detections.append((bbox, conf))
        
        # Process detections
        matched_tracks = set()
        
        for bbox, _ in detections:
            x1, y1, x2, y2 = bbox
            
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
                # Create new track
                track_id = self.create_track(bbox, features, color_hist, frame_time)
                matched_tracks.add(track_id)
                
            # Check if in door region for entry/exit detection
            if self.is_in_door_region(bbox):
                # Add to door interactions set based on position in sequence
                if len(self.track_positions[track_id]) <= 3:  # Near start of track
                    self.door_entries.add(track_id)
                elif len(self.track_positions[track_id]) >= 5:  # Near end of track
                    self.door_exits.add(track_id)
        
        # Update unmatched tracks
        self.update_unmatched_tracks(matched_tracks, frame_time)
        
        # Draw tracking visualization
        return self.draw_tracks(frame)

    def draw_tracks(self, frame):
        """Draw bounding boxes and IDs for visualization"""
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
        """Process the entire video"""
        frame_count = 0
        stride = 1  # Process every frame (adjust for performance if needed)
        
        logger.info(f"Processing video: {self.video_name}")
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process only every nth frame for efficiency
            if frame_count % stride == 0:
                frame_time = frame_count / self.fps
                processed_frame = self.process_frame(frame, frame_time, frame_count)
                
                if visualize:
                    cv2.imshow(f"Camera {self.camera_id}", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        self.cap.release()
        if visualize:
            cv2.destroyAllWindows()
            
        process_time = time.time() - start_time
        logger.info(f"Completed processing {self.video_name} in {process_time:.2f} seconds")
        
        # Perform final track consolidation
        self.consolidate_tracks()
        
        # Detect final door interactions
        for track_id in self.active_tracks:
            is_entering, is_exiting = self.detect_door_interaction(track_id)
            if is_entering:
                self.door_entries.add(track_id)
            if is_exiting:
                self.door_exits.add(track_id)
                
        return self.get_valid_tracks()

    def get_valid_tracks(self):
        """Get valid tracks that meet quality criteria"""
        valid_tracks = {}
        
        all_tracks = set(self.track_timestamps.keys())
        
        for track_id in all_tracks:
            # Skip tracks that are too short
            duration = (self.track_timestamps[track_id]['last_appearance'] - 
                        self.track_timestamps[track_id]['first_appearance'])
                        
            # Skip tracks with too few detections
            if track_id not in self.feature_history or len(self.feature_history[track_id]) < self.min_detections:
                continue
                
            if duration >= self.min_track_duration:
                valid_tracks[track_id] = {
                    'id': track_id,
                    'features': self.person_features.get(track_id),
                    'color_histogram': self.color_histograms.get(track_id),
                    'first_appearance': self.track_timestamps[track_id]['first_appearance'],
                    'last_appearance': self.track_timestamps[track_id]['last_appearance'],
                    'duration': duration,
                    'is_entry': track_id in self.door_entries,
                    'is_exit': track_id in self.door_exits
                }
        
        logger.info(f"Identified {len(valid_tracks)} valid tracks out of {len(all_tracks)} total tracks")
        return valid_tracks
        
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union between two boxes"""
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

def process_video_directory(input_dir, output_dir="tracking_results"):
    """Process all videos in a directory and generate cross-camera analysis"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find video files
    video_files = sorted(list(Path(input_dir).glob("Camera_*_*.mp4")))
    if not video_files:
        logger.error(f"No video files found in {input_dir}")
        return None
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Create global tracker for cross-camera matching
    global_tracker = GlobalTracker()
    
    # Process each video
    all_tracks = {}
    for video_path in video_files:
        logger.info(f"Processing {video_path}")
        
        try:
            # Initialize and run tracker
            tracker = PersonTracker(str(video_path), output_dir)
            valid_tracks = tracker.process_video(visualize=False)
            
            # Store tracks
            video_name = video_path.stem
            all_tracks[video_name] = {
                'camera_id': tracker.camera_id,
                'date': tracker.date,
                'tracks': valid_tracks
            }
            
            logger.info(f"Found {len(valid_tracks)} valid tracks in {video_name}")
            
            # Register valid tracks with global tracker
            for track_id, track_info in valid_tracks.items():
                if track_info['features'] is not None:
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
            logger.error(f"Error processing {video_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Analyze camera transitions
    transition_analysis = global_tracker.analyze_camera_transitions()
    
    # Create summary in CSV format
    csv_data = []
    
    # Group by date
    dates = set(info['date'] for info in all_tracks.values())
    
    for date in dates:
        csv_data.append({
            'Date': date,
            'Camera1_Unique_Individuals': transition_analysis['unique_camera1'],
            'Camera2_Unique_Individuals': transition_analysis['unique_camera2'],
            'Transitions_Camera1_to_Camera2': transition_analysis['camera1_to_camera2']
        })
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'tracking_summary.csv')
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    # Save JSON with detailed results
    json_path = os.path.join(output_dir, 'tracking_results.json')
    results = {
        'unique_camera1': transition_analysis['unique_camera1'],
        'unique_camera2': transition_analysis['unique_camera2'],
        'transitions': {
            'camera1_to_camera2': transition_analysis['camera1_to_camera2'],
            'transition_details': transition_analysis['valid_transitions']
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {csv_path} and {json_path}")
    
    # Print summary
    print("\n===== TRACKING SUMMARY =====")
    print(f"Camera 1 Unique Individuals: {transition_analysis['unique_camera1']}")
    print(f"Camera 2 Unique Individuals: {transition_analysis['unique_camera2']}")
    print(f"Camera 1 to Camera 2 Transitions: {transition_analysis['camera1_to_camera2']}")
    print("============================\n")
    
    return results

def main():
    # Working directory with videos
    working_dir = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
                             'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')
    
    # Process the videos
    results = process_video_directory(working_dir)
    
    # Print the final numbers - with fixed dictionary access
    if results:
        print(f"\nFinal Results: {results['unique_camera1']} individuals in Camera 1, " +
              f"{results['unique_camera2']} individuals in Camera 2, " +
              f"{results['transitions']['camera1_to_camera2']} transitions from Camera 1 to Camera 2")

if __name__ == "__main__":
    main()
