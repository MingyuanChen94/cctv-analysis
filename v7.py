import os
import cv2
import numpy as np
from ultralytics import YOLO
import torchreid
import torch
from collections import defaultdict
import os
import time
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import scipy.spatial.distance as distance
from pathlib import Path
import shutil
import json
import pandas as pd

# Set the working directory
working_directory = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
                                 'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')
ouptut_dir = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
                                 'Documents', 'VISIONARY', 'Durham Experiment', 'test_data', 'tracking_results')

class TrackingState:
    ACTIVE = 'active'          # Fully visible
    OCCLUDED = 'occluded'      # Temporarily hidden
    TENTATIVE = 'tentative'    # New track
    LOST = 'lost'              # Missing too long

class GlobalTracker:
    def __init__(self):
        self.global_identities = {}  # Map camera-specific IDs to global IDs
        self.appearance_sequence = {}  # Track sequence of camera appearances
        self.feature_database = {}  # Store features for cross-camera matching
        self.similarity_threshold = 0.6
        self.min_transition_time = 30  # Minimum seconds between cameras
        self.max_transition_time = 600  # Maximum seconds between cameras
        self.feature_history = defaultdict(list)
        self.max_features_per_identity = 5

        # Track validation parameters
        self.min_track_duration = 2.0    # Minimum 2 seconds for valid track
        self.feature_consistency_threshold = 0.7

    def register_camera_detection(self, camera_id, person_id, features, timestamp):
        """Enhanced registration with temporal consistency"""
        # Only register high-quality tracks
        if not self._is_track_valid(camera_id, person_id, features):
            return
            
        global_id = self._match_or_create_global_id(camera_id, person_id, features, timestamp)
        
        if global_id is not None:
            if global_id not in self.appearance_sequence:
                self.appearance_sequence[global_id] = []
            
            camera_key = f"Camera_{camera_id}"
            
            # Only append if this is a new appearance in this camera
            if not self.appearance_sequence[global_id] or \
               self.appearance_sequence[global_id][-1]['camera'] != camera_key:
                self.appearance_sequence[global_id].append({
                    'camera': camera_key,
                    'timestamp': timestamp
                })

    def _is_track_valid(self, camera_id, person_id, features):
        """Validate track quality"""
        camera_key = f"{camera_id}_{person_id}"
        
        # Check if we've seen this track before
        if camera_key in self.global_identities:
            global_id = self.global_identities[camera_key]
            
            # Check feature consistency with history
            if global_id in self.feature_history:
                similarities = [
                    1 - distance.cosine(features.flatten(), hist_feat.flatten())
                    for hist_feat in self.feature_history[global_id]
                ]
                if similarities and np.mean(similarities) < 0.7:
                    return False
        
        return True

    def _match_or_create_global_id(self, camera_id, person_id, features, timestamp):
        """Modified matching with relaxed constraints"""
        camera_key = f"{camera_id}_{person_id}"
        
        if camera_key in self.global_identities:
            return self.global_identities[camera_key]
        
        best_match = None
        best_score = 0
        
        for global_id, stored_features in self.feature_database.items():
            last_appearance = self.appearance_sequence.get(global_id, [])[-1] if self.appearance_sequence.get(global_id) else None
            
            if last_appearance:
                last_camera = int(last_appearance['camera'].split('_')[1])
                time_diff = timestamp - last_appearance['timestamp']
                
                # Skip if same camera or invalid transition time
                if last_camera == camera_id or time_diff < self.min_transition_time:
                    continue
                
                # Calculate similarity with temporal weighting
                base_similarity = 1 - distance.cosine(features.flatten(), stored_features.flatten())
                
                # Add time-based bonus for realistic transition times
                time_bonus = 0.2 if self.min_transition_time <= time_diff <= 120 else 0
                adjusted_similarity = base_similarity + time_bonus
                
                if adjusted_similarity > self.similarity_threshold and adjusted_similarity > best_score:
                    best_match = global_id
                    best_score = adjusted_similarity
        
        if best_match is None:
            best_match = len(self.global_identities)
            self.feature_database[best_match] = features
            
        self.global_identities[camera_key] = best_match
        return best_match
    
    def analyze_camera_transitions(self):
        """Analyze transitions between cameras"""
        cam1_to_cam2 = 0
        cam2_to_cam1 = 0
        
        for global_id, appearances in self.appearance_sequence.items():
            # Sort appearances by timestamp
            sorted_appearances = sorted(appearances, key=lambda x: x['timestamp'])
            
            # Check for sequential appearances
            for i in range(len(sorted_appearances) - 1):
                current = sorted_appearances[i]
                next_app = sorted_appearances[i + 1]
                
                if current['camera'] == 'Camera_1' and next_app['camera'] == 'Camera_2':
                    cam1_to_cam2 += 1
                elif current['camera'] == 'Camera_2' and next_app['camera'] == 'Camera_1':
                    cam2_to_cam1 += 1
        
        return {
            'camera1_to_camera2': cam1_to_cam2,
            'camera2_to_camera1': cam2_to_cam1,
            'total_unique_individuals': len(self.global_identities)
        }

# Modify the main function to use GlobalTracker
def process_video_directory(input_dir, output_base_dir=None):
    """Process videos with separate parameters for individual counting and transitions"""
    global_tracker = GlobalTracker()
    results = {}
    per_camera_stats = defaultdict(int)
    
    if output_base_dir is None:
        output_base_dir = os.path.join(os.path.dirname(input_dir), "tracking_results")
    
    # Ensure absolute paths
    input_dir = os.path.abspath(input_dir)
    output_base_dir = os.path.abspath(output_base_dir)
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Separate parameters for individual counting and cross-camera matching
    individual_params = {
        'similarity_threshold': 0.5,    # More lenient for same-camera tracking
        'min_track_duration': 0.5,      # Shorter minimum duration (0.5 seconds)
        'max_track_gap': 2.0,           # Allow longer gaps
        'feature_consistency': 0.6,      # More lenient feature consistency
        'min_detections': 2             # Require only 2 detections for counting
    }
    
    transition_params = {
        'similarity_threshold': 0.65,    # Stricter for cross-camera matching
        'min_track_duration': 1.0,      # Longer duration required
        'feature_consistency': 0.7,      # Stricter feature consistency
        'min_detections': 3             # More detections required
    }
    
    # Process videos
    video_files = sorted(list(Path(input_dir).glob('*.[ma][pv][4i]')))
    print(f"\nFound {len(video_files)} videos")
    
    for video_path in video_files:
        print(f"\nProcessing: {video_path}")
        camera_id = video_path.stem.split('_')[1]
        date = video_path.stem.split('_')[-1]
        
        try:
            # Initialize tracker
            tracker = PersonTracker(str(video_path), output_base_dir)
            tracker.similarity_threshold = individual_params['similarity_threshold']
            tracker.min_track_duration = individual_params['min_track_duration']
            tracker.max_track_gap = individual_params['max_track_gap']
            
            # Process video
            tracker.process_video()
            
            # Identify stable tracks for individual counting
            individual_tracks = {}
            transition_tracks = {}
            
            for track_id, track_info in tracker.active_tracks.items():
                track_duration = (track_info['last_seen'] - 
                                tracker.person_timestamps[track_id]['first_appearance'])
                features = np.array([f for f in tracker.appearance_history[track_id]])
                
                # Check for individual counting criteria
                if (track_duration >= individual_params['min_track_duration'] and 
                    len(features) >= individual_params['min_detections']):
                    
                    if len(features) >= 2:
                        consistency = np.mean([
                            1 - distance.cosine(features[i].flatten(), features[i+1].flatten())
                            for i in range(len(features)-1)
                        ])
                        
                        if consistency >= individual_params['feature_consistency']:
                            individual_tracks[track_id] = track_info
                            
                            # Check if track also meets transition criteria
                            if (track_duration >= transition_params['min_track_duration'] and 
                                len(features) >= transition_params['min_detections'] and 
                                consistency >= transition_params['feature_consistency']):
                                transition_tracks[track_id] = track_info
            
            # Update results with individual tracks
            video_results = {
                'video_name': video_path.stem,
                'date': date,
                'camera_id': camera_id,
                'person_details': {
                    track_id: {
                        'first_appearance': tracker.person_timestamps[track_id]['first_appearance'],
                        'last_appearance': track_info['last_seen'],
                        'duration': track_info['last_seen'] - 
                                  tracker.person_timestamps[track_id]['first_appearance']
                    }
                    for track_id, track_info in individual_tracks.items()
                }
            }
            
            results[video_path.stem] = video_results
            per_camera_stats[f"Camera_{camera_id}"] = len(individual_tracks)
            
            # Register only high-quality tracks for cross-camera matching
            for track_id, track_info in transition_tracks.items():
                global_tracker.register_camera_detection(
                    int(camera_id),
                    track_id,
                    track_info['features'],
                    tracker.person_timestamps[track_id]['first_appearance']
                )
            
            print(f"Found {len(individual_tracks)} individuals")
            print(f"Of which {len(transition_tracks)} are suitable for cross-camera matching")
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    # Rest of the code remains the same (CSV creation, etc.)
    transition_analysis = global_tracker.analyze_camera_transitions()
    
    # Create CSV data
    csv_data = []
    for video_name, video_results in results.items():
        date = video_results['date']
        camera_id = video_results['camera_id']
        
        date_entry = next((entry for entry in csv_data if entry['Date'] == date), None)
        if date_entry is None:
            date_entry = {
                'Date': date,
                'Camera1_Unique_Individuals': 0,
                'Camera2_Unique_Individuals': 0,
                'Transitions_Camera1_to_Camera2': transition_analysis['camera1_to_camera2'],
                'Transitions_Camera2_to_Camera1': transition_analysis['camera2_to_camera1']
            }
            csv_data.append(date_entry)
        
        date_entry[f'Camera{camera_id}_Unique_Individuals'] = len(video_results['person_details'])
    
    # Save CSV
    csv_path = os.path.join(output_base_dir, 'daily_tracking_summary.csv')
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to: {csv_path}")
    print("\nCSV Contents:")
    print(df.to_string())
    
    return {
        'per_camera_statistics': dict(per_camera_stats),
        'cross_camera_analysis': transition_analysis,
        'total_unique_global': len(global_tracker.global_identities)
    }

def find_csv_file(working_directory):
    """Find the daily tracking summary CSV file"""
    # Check the default location
    default_path = os.path.join(working_directory, "tracking_results", "daily_tracking_summary.csv")
    
    if os.path.exists(default_path):
        print(f"\nFound CSV file at: {default_path}")
        return default_path
        
    # If not found, search the working directory recursively
    print("\nSearching for CSV file...")
    for root, dirs, files in os.walk(working_directory):
        for file in files:
            if file == "daily_tracking_summary.csv":
                path = os.path.join(root, file)
                print(f"Found CSV file at: {path}")
                return path
                
    print("\nCSV file not found! Checked locations:")
    print(f"1. Default path: {default_path}")
    print("2. All subdirectories of:", working_directory)
    return None

def main():
    # Process videos
    summary = process_video_directory(working_directory)
    
    # Try to find the CSV file
    csv_path = find_csv_file(working_directory)
    
    if csv_path and os.path.exists(csv_path):
        print("\nCSV file contents:")
        try:
            df = pd.read_csv(csv_path)
            print(df.to_string())
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")

class PersonTracker:
    def __init__(self, video_path, output_base_dir="tracking_results"):
        # Get video name for organizing output
        self.video_name = Path(video_path).stem

        # Create video-specific output directory
        self.output_dir = os.path.join(output_base_dir, self.video_name)
        self.images_dir = os.path.join(self.output_dir, "person_images")
        self.features_dir = os.path.join(self.output_dir, "person_features")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

        # Initialize models
        self.detector = YOLO("yolo11x.pt")
        self.reid_model = torchreid.models.build_model(
            name='osnet_ain_x1_0',
            num_classes=1000,
            pretrained=True
        )
        self.reid_model = self.reid_model.cuda() if torch.cuda.is_available() else self.reid_model
        self.reid_model.eval()

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Initialize tracking variables
        self.active_tracks = {}
        self.person_features = {}
        self.person_timestamps = {}
        self.next_id = 0
        self.appearance_history = defaultdict(list)

        # Add new attributes for reentry handling
        self.lost_tracks = {}
        self.max_lost_age = self.fps * 30  # Max time to remember lost tracks (30 seconds)
        self.max_history_length = 10  # Number of recent features to keep

        # Modified tracking parameters
        self.similarity_threshold = 0.65
        self.min_detection_confidence = 0.5
        self.feature_weight = 0.6
        self.position_weight = 0.25
        self.motion_weight = 0.15
        self.reentry_threshold = 0.75

        # Track consolidation parameters
        self.min_track_length = self.fps * 3  # Minimum 3 seconds
        self.max_track_gap = self.fps * 2     # Maximum 2 second gap for merging
        self.consolidation_threshold = 0.75    # Threshold for merging tracks
        
        # Time-based track filtering
        self.min_track_duration = 1.5  # Minimum 1.5 seconds for valid track
        self.stable_track_threshold = self.fps * 5  # 5 seconds for stable track
        
        # Track quality assessment
        self.track_quality_threshold = 0.7
        self.min_detections_for_track = 5
        self.max_disappeared = self.fps * 2

        # Spatial-temporal parameters for individual counting
        self.min_spatial_distance = 50  # Minimum pixels between different individuals
        self.min_temporal_gap = 0.5     # Minimum seconds between appearances
        self.track_memory = 30          # Remember tracks for this many frames
        
        # Track classification
        self.track_categories = {
            'entering': [],     # Tracks near door, moving inward
            'leaving': [],      # Tracks near door, moving outward
            'inside': [],       # Tracks away from door
            'temporary': []     # Short-lived or uncertain tracks
        }
        
        # Door region definition remains the same
        self.camera_id = int(Path(video_path).stem.split('_')[1])
        self.door_coords = {
            1: [(1030, 0), (1700, 560)],
            2: [(400, 0), (800, 470)]
        }
        self.door_region_buffer = 50  # pixels buffer around door region

        # Adjust parameters based on camera
        if self.camera_id == 1:
            # More strict parameters for complex environment in Camera 1
            self.min_detection_confidence = 0.6
            self.consolidation_threshold = 0.8
        else:
            # More relaxed parameters for Camera 2's cleaner environment
            self.min_detection_confidence = 0.5
            self.consolidation_threshold = 0.75

    def classify_track_location(self, box):
        """Classify track based on location and movement"""
        center = self.calculate_box_center(box)
        door = self.door_coords[self.camera_id]
        
        # Check if in door region
        in_door = (door[0][0] <= center[0] <= door[1][0] and 
                  door[0][1] <= center[1] <= door[1][1])
        
        return 'door' if in_door else 'inside'
    
    def is_new_person(self, current_box, current_time):
        """Determine if this detection likely represents a new person"""
        # Check spatial separation from all active tracks
        for track_id, track in self.active_tracks.items():
            if track['disappeared'] < self.track_memory:
                dist = np.linalg.norm(
                    np.array(self.calculate_box_center(current_box)) - 
                    np.array(self.calculate_box_center(track['box']))
                )
                if dist < self.min_spatial_distance:
                    return False
                    
        # Check temporal separation from recent tracks in similar location
        for track_id, timestamps in self.person_timestamps.items():
            if current_time - timestamps['last_appearance'] < self.min_temporal_gap:
                if track_id in self.active_tracks:
                    track_box = self.active_tracks[track_id]['box']
                    dist = np.linalg.norm(
                        np.array(self.calculate_box_center(current_box)) - 
                        np.array(self.calculate_box_center(track_box))
                    )
                    if dist < self.min_spatial_distance:
                        return False
                        
        return True
    
    def consolidate_tracks(self):
        """Merge tracks that likely belong to the same person"""
        merged_tracks = set()
        
        for track_id1 in list(self.active_tracks.keys()):
            if track_id1 in merged_tracks:
                continue
                
            track1 = self.active_tracks[track_id1]
            track1_features = self.person_features[track_id1]
            
            for track_id2 in list(self.active_tracks.keys()):
                if track_id2 in merged_tracks or track_id1 == track_id2:
                    continue
                    
                track2 = self.active_tracks[track_id2]
                track2_features = self.person_features[track_id2]
                
                # Check temporal separation
                time_gap = abs(track1['last_seen'] - track2['last_seen'])
                if time_gap > self.max_track_gap:
                    continue
                
                # Calculate feature similarity
                similarity = 1 - distance.cosine(track1_features.flatten(), 
                                              track2_features.flatten())
                
                if similarity > self.consolidation_threshold:
                    # Merge tracks
                    self._merge_tracks(track_id1, track_id2)
                    merged_tracks.add(track_id2)

    def _merge_tracks(self, track_id1, track_id2):
        """Merge track2 into track1"""
        track1 = self.active_tracks[track_id1]
        track2 = self.active_tracks[track_id2]
        
        # Update timestamps
        self.person_timestamps[track_id1]['first_appearance'] = min(
            self.person_timestamps[track_id1]['first_appearance'],
            self.person_timestamps[track_id2]['first_appearance']
        )
        self.person_timestamps[track_id1]['last_appearance'] = max(
            self.person_timestamps[track_id1]['last_appearance'],
            self.person_timestamps[track_id2]['last_appearance']
        )
        
        # Combine feature history
        self.appearance_history[track_id1].extend(self.appearance_history[track_id2])
        
        # Remove track2
        del self.active_tracks[track_id2]
        del self.person_timestamps[track_id2]
        del self.person_features[track_id2]

    def filter_short_tracks(self):
        """Remove tracks that are too short or unstable"""
        for track_id in list(self.active_tracks.keys()):
            track_duration = (self.person_timestamps[track_id]['last_appearance'] - 
                            self.person_timestamps[track_id]['first_appearance'])
            
            if track_duration < self.min_track_duration:
                self._remove_track(track_id)

    def _remove_track(self, track_id):
        """Remove a track and all its associated data"""
        if track_id in self.active_tracks:
            del self.active_tracks[track_id]
        if track_id in self.person_timestamps:
            del self.person_timestamps[track_id]
        if track_id in self.person_features:
            del self.person_features[track_id]
        if track_id in self.appearance_history:
            del self.appearance_history[track_id]

    def is_in_door_region(self, box):
        """Check if detection is in door region"""
        door = self.door_coords[self.camera_id]
        box_center = self.calculate_box_center(box)
        
        # Add buffer to door region
        door_x_min = max(0, door[0][0] - self.door_region_buffer)
        door_x_max = min(self.frame_width, door[1][0] + self.door_region_buffer)
        door_y_min = max(0, door[0][1] - self.door_region_buffer)
        door_y_max = min(self.frame_height, door[1][1] + self.door_region_buffer)
        
        return (door_x_min <= box_center[0] <= door_x_max and 
                door_y_min <= box_center[1] <= door_y_max)

    def assess_track_quality(self, track_id):
        """Assess the quality of a track based on multiple factors"""
        track = self.active_tracks[track_id]
        
        if track['state'] != TrackingState.ACTIVE:
            return 0.0
            
        # Calculate detection consistency
        detection_count = len(self.appearance_history[track_id])
        if detection_count < self.min_detections_for_track:
            return 0.0
            
        # Calculate feature consistency
        feature_distances = []
        features = np.array([f for f in self.appearance_history[track_id]])
        for i in range(len(features)-1):
            dist = distance.cosine(features[i].flatten(), features[i+1].flatten())
            feature_distances.append(dist)
        
        feature_consistency = 1 - np.mean(feature_distances) if feature_distances else 0
        
        # Factor in track duration
        duration = track['last_seen'] - self.person_timestamps[track_id]['first_appearance']
        duration_score = min(1.0, duration / (self.fps * 5))  # Normalize to 5 seconds
        
        # Combine scores
        quality_score = (0.5 * feature_consistency + 
                        0.3 * duration_score + 
                        0.2 * (detection_count / self.min_detections_for_track))
                        
        return quality_score

    def extract_features(self, person_crop):
        """Extract ReID features from person crop"""
        try:
            # Preprocess image for ReID
            img = cv2.resize(person_crop, (128, 256))
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1).unsqueeze(0)
            if torch.cuda.is_available():
                img = img.cuda()

            # Extract features
            with torch.no_grad():
                features = self.reid_model(img)
            return features.cpu().numpy()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def calculate_box_center(self, box):
        """Calculate center point of a bounding box"""
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def calculate_velocity(self, current_box, previous_box):
        """Calculate velocity vector between two boxes"""
        current_center = self.calculate_box_center(current_box)
        previous_center = self.calculate_box_center(previous_box)
        return [current_center[0] - previous_center[0],
                current_center[1] - previous_center[1]]

    def predict_next_position(self, box, velocity):
        """Predict next position based on current position and velocity"""
        center = self.calculate_box_center(box)
        predicted_center = [center[0] + velocity[0], center[1] + velocity[1]]
        width = box[2] - box[0]
        height = box[3] - box[1]
        return [predicted_center[0] - width/2, predicted_center[1] - height/2,
                predicted_center[0] + width/2, predicted_center[1] + height/2]

    def calculate_motion_similarity(self, current_boxes, tracked_boxes, tracked_velocities):
        """Calculate motion-based similarity"""
        n_detections = len(current_boxes)
        n_tracks = len(tracked_boxes)
        motion_sim = np.zeros((n_detections, n_tracks))

        for i, current_box in enumerate(current_boxes):
            current_center = self.calculate_box_center(current_box)
            for j, (tracked_box, velocity) in enumerate(zip(tracked_boxes, tracked_velocities)):
                # Predict where the tracked box should be
                predicted_box = self.predict_next_position(
                    tracked_box, velocity)
                predicted_center = self.calculate_box_center(predicted_box)

                # Calculate distance between prediction and actual position
                distance = np.sqrt(
                    (current_center[0] - predicted_center[0])**2 +
                    (current_center[1] - predicted_center[1])**2
                )
                # Convert distance to similarity (closer = more similar)
                # 100 is a scaling factor
                motion_sim[i, j] = np.exp(-distance / 100.0)

        return motion_sim

    def detect_occlusion(self, box1, box2):
        """
        Detect if box1 is occluded by box2.
        Returns: 
            - is_occluded (bool): True if box1 is occluded by box2
            - occlusion_score (float): Degree of occlusion (0 to 1)
        """
        # Calculate IoU
        iou = self.calculate_iou(box1, box2)

        # Calculate centers and areas
        center1 = self.calculate_box_center(box1)
        center2 = self.calculate_box_center(box2)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate vertical position (y-coordinate)
        y1 = box1[3]  # bottom of box1
        y2 = box2[3]  # bottom of box2

        # Factors that suggest box1 is behind box2:
        # 1. Significant overlap
        overlap_factor = 1.0 if iou > 0.3 else 0.0

        # 2. Box2 is closer to camera (generally larger and lower in frame)
        size_factor = 1.0 if area2 > area1 else 0.0
        position_factor = 1.0 if y2 > y1 else 0.0

        # 3. Box1 is partially contained within box2
        contained_horizontally = (
            (box1[0] > box2[0] and box1[0] < box2[2]) or
            (box1[2] > box2[0] and box1[2] < box2[2])
        )
        contained_vertically = (
            (box1[1] > box2[1] and box1[1] < box2[3]) or
            (box1[3] > box2[1] and box1[3] < box2[3])
        )
        containment_factor = 1.0 if (
            contained_horizontally and contained_vertically) else 0.0

        # Calculate occlusion score (weighted combination of factors)
        occlusion_score = (
            0.4 * overlap_factor +
            0.2 * size_factor +
            0.2 * position_factor +
            0.2 * containment_factor
        )

        # Determine if occluded based on score threshold
        is_occluded = occlusion_score > 0.5

        return is_occluded, occlusion_score

    def calculate_similarity_matrix(self, current_features, current_boxes, tracked_features, tracked_boxes):
        """Calculate similarity matrix combining appearance, position, and motion"""
        n_detections = len(current_features)
        n_tracks = len(tracked_features)

        if n_detections == 0 or n_tracks == 0:
            return np.array([])

        # Calculate appearance similarity
        appearance_sim = 1 - distance.cdist(
            np.array([f.flatten() for f in current_features]),
            np.array([f.flatten() for f in tracked_features]),
            metric='cosine'
        )

        # Calculate position similarity using IoU
        position_sim = np.zeros((n_detections, n_tracks))
        for i, box1 in enumerate(current_boxes):
            for j, box2 in enumerate(tracked_boxes):
                position_sim[i, j] = self.calculate_iou(box1, box2)

        # Calculate velocities for tracked objects
        tracked_velocities = []
        for track_id in list(self.active_tracks.keys())[:n_tracks]:
            if 'previous_box' in self.active_tracks[track_id]:
                velocity = self.calculate_velocity(
                    self.active_tracks[track_id]['box'],
                    self.active_tracks[track_id]['previous_box']
                )
            else:
                velocity = [0, 0]  # No velocity for new tracks
            tracked_velocities.append(velocity)

        # Calculate motion similarity
        motion_sim = self.calculate_motion_similarity(
            current_boxes, tracked_boxes, tracked_velocities)

        # Combine all similarities
        similarity_matrix = (
            self.feature_weight * appearance_sim +
            self.position_weight * position_sim +
            self.motion_weight * motion_sim
        )

        return similarity_matrix

    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def update_feature_history(self, track_id, features):
        """Maintain rolling window of recent features"""
        self.appearance_history[track_id].append(features)
        if len(self.appearance_history[track_id]) > self.max_history_length:
            self.appearance_history[track_id].pop(0)

        # Update feature representation using exponential moving average
        if track_id in self.person_features:
            alpha = 0.7  # Weight for historical features
            current_features = self.person_features[track_id]
            updated_features = alpha * \
                current_features + (1 - alpha) * features
            self.person_features[track_id] = updated_features
        else:
            self.person_features[track_id] = features

    def recover_lost_tracklet(self, features, current_box, frame_time):
        """Attempt to recover lost tracks"""
        best_match_id = None
        best_match_score = 0

        # Check recently lost tracks
        lost_tracks_to_remove = []
        for lost_id, lost_info in self.lost_tracks.items():
            # Skip if lost track is too old
            if frame_time - lost_info['last_seen'] > self.max_lost_age:
                lost_tracks_to_remove.append(lost_id)
                continue

            # Calculate appearance similarity
            lost_features = lost_info['features']
            appearance_sim = 1 - \
                distance.cosine(features.flatten(), lost_features.flatten())

            # Calculate position similarity based on predicted movement
            predicted_box = self.predict_next_position(
                lost_info['box'],
                lost_info['velocity']
            )
            position_sim = self.calculate_iou(current_box, predicted_box)

            # Combine similarities
            match_score = (
                self.feature_weight * appearance_sim +
                self.position_weight * position_sim
            )

            # Check temporal consistency
            if match_score > 0.6 and match_score > best_match_score:
                best_match_score = match_score
                best_match_id = lost_id

        # Clean up old lost tracks
        for lost_id in lost_tracks_to_remove:
            del self.lost_tracks[lost_id]

        return best_match_id if best_match_score > 0.6 else None
    
    def update_existing_track(self, track_id, box, features, frame_time, frame):
        """Update an existing track with new detection"""
        self.active_tracks[track_id].update({
            'previous_box': self.active_tracks[track_id]['box'],
            'box': box,
            'features': features,
            'last_seen': frame_time,
            'disappeared': 0,
            'state': TrackingState.ACTIVE
        })

        # Update velocity
        if 'previous_box' in self.active_tracks[track_id]:
            self.active_tracks[track_id]['velocity'] = self.calculate_velocity(
                box,
                self.active_tracks[track_id]['previous_box']
            )

        # Update feature history and timestamps
        self.update_feature_history(track_id, features)
        self.person_timestamps[track_id]['last_appearance'] = frame_time
        
        # Save person image
        self.save_person_image(track_id, 
            frame[box[1]:box[3], box[0]:box[2]])
        
    def reactivate_track(self, track_id, box, features, frame_time, frame):
        """Reactivate a previously lost track"""
        # Remove from lost tracks
        lost_info = self.lost_tracks.pop(track_id)
        
        # Reactivate in active tracks
        self.active_tracks[track_id] = {
            'box': box,
            'features': features,
            'last_seen': frame_time,
            'disappeared': 0,
            'state': TrackingState.ACTIVE,
            'velocity': lost_info.get('velocity', [0, 0]),
            'previous_box': lost_info.get('box', box)
        }

        # Update timestamps and save new image
        self.person_timestamps[track_id]['last_appearance'] = frame_time
        self.update_feature_history(track_id, features)
        self.save_person_image(track_id, 
            frame[box[1]:box[3], box[0]:box[2]])
        

    def create_new_track(self, box, features, frame_time, frame):
        """Create a new track for unmatched detection"""
        new_id = self.next_id
        self.next_id += 1

        self.active_tracks[new_id] = {
            'state': TrackingState.TENTATIVE,
            'box': box,
            'features': features,
            'last_seen': frame_time,
            'disappeared': 0,
            'velocity': [0, 0]
        }

        self.person_features[new_id] = features
        self.appearance_history[new_id] = [features]
        self.person_timestamps[new_id] = {
            'first_appearance': frame_time,
            'last_appearance': frame_time
        }

        self.save_person_image(new_id,
            frame[box[1]:box[3], box[0]:box[2]])
        

    def update_lost_tracks(self, matched_track_ids, frame_time):
        """Update status of lost tracks and remove expired ones"""
        # Move unmatched active tracks to lost tracks
        for track_id in list(self.active_tracks.keys()):
            if track_id not in matched_track_ids:
                track_info = self.active_tracks[track_id]
                track_info['disappeared'] += 1

                if track_info['disappeared'] > self.max_disappeared:
                    # Move to lost tracks
                    self.lost_tracks[track_id] = {
                        'features': track_info['features'],
                        'box': track_info['box'],
                        'velocity': track_info.get('velocity', [0, 0]),
                        'last_seen': track_info['last_seen']
                    }
                    del self.active_tracks[track_id]

        # Remove expired lost tracks
        for track_id in list(self.lost_tracks.keys()):
            if frame_time - self.lost_tracks[track_id]['last_seen'] > self.max_lost_age:
                del self.lost_tracks[track_id]

    def save_tracking_results(self):
        """Save results with corrected counting"""
        results = {
            'video_name': self.video_name,
            'date': self.video_name.split('_')[-1],
            'camera_id': self.camera_id,
            'person_details': {}
        }
        
        # Count unique individuals based on spatial-temporal separation
        unique_individuals = set()
        for track_id, track in self.active_tracks.items():
            if track['disappeared'] <= self.track_memory:
                location = self.classify_track_location(track['box'])
                if location != 'temporary':
                    unique_individuals.add(track_id)
        
        # Add all valid tracks to results
        for track_id in unique_individuals:
            results['person_details'][track_id] = {
                'first_appearance': self.person_timestamps[track_id]['first_appearance'],
                'last_appearance': self.active_tracks[track_id]['last_seen'],
                'duration': (self.active_tracks[track_id]['last_seen'] - 
                           self.person_timestamps[track_id]['first_appearance'])
            }
        
        return results

    def update_tracks(self, frame, detections, frame_time):
        """Modified track updating with improved individual counting"""
        current_boxes = []
        current_features = []
        
        # Process detections
        for box, conf in detections:
            if conf < self.min_detection_confidence:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
                
            features = self.extract_features(person_crop)
            if features is not None:
                current_boxes.append([x1, y1, x2, y2])
                current_features.append(features)
        
        # Update existing tracks
        matched_detections = set()
        for track_id, track in list(self.active_tracks.items()):
            best_match = None
            best_score = float('-inf')
            
            # Find best matching detection
            for i, (box, features) in enumerate(zip(current_boxes, current_features)):
                if i in matched_detections:
                    continue
                    
                # Calculate spatial similarity
                dist = np.linalg.norm(
                    np.array(self.calculate_box_center(box)) - 
                    np.array(self.calculate_box_center(track['box']))
                )
                spatial_score = np.exp(-dist / self.min_spatial_distance)
                
                # Calculate feature similarity
                feature_sim = 1 - distance.cosine(
                    features.flatten(),
                    track['features'].flatten()
                )
                
                # Combined score
                score = 0.7 * spatial_score + 0.3 * feature_sim
                
                if score > best_score:
                    best_score = score
                    best_match = i
            
            # Update track or mark as disappeared
            if best_match is not None and best_score > 0.3:
                matched_detections.add(best_match)
                self.update_existing_track(
                    track_id,
                    current_boxes[best_match],
                    current_features[best_match],
                    frame_time,
                    frame
                )
            else:
                track['disappeared'] += 1
        
        # Create new tracks for unmatched detections
        for i, (box, features) in enumerate(zip(current_boxes, current_features)):
            if i in matched_detections:
                continue
                
            # Check if this is likely a new person
            if self.is_new_person(box, frame_time):
                self.create_new_track(box, features, frame_time, frame)
        
        # Clean up disappeared tracks
        for track_id in list(self.active_tracks.keys()):
            if self.active_tracks[track_id]['disappeared'] > self.track_memory:
                del self.active_tracks[track_id]

    def save_person_image(self, person_id, frame):
        """Save person image in video-specific directory"""
        person_dir = os.path.join(self.images_dir, f"person_{person_id}")
        os.makedirs(person_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(person_dir, f"{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        return image_path

    def save_person_features(self, person_id, features, frame_time):
        """Save person features in video-specific directory"""
        # Ensure the features directory exists
        os.makedirs(self.features_dir, exist_ok=True)

        # Save features array
        feature_path = os.path.join(
            self.features_dir, f"person_{person_id}_features.npz")
        np.savez(feature_path,
                 features=features,
                 timestamp=frame_time,
                 video_name=self.video_name)
        return feature_path

    def update_person_features(self, person_id, features, frame_time):
        """Update and save person features"""
        # Update feature history
        self.appearance_history[person_id].append(features)
        if len(self.appearance_history[person_id]) > self.max_history_length:
            self.appearance_history[person_id].pop(0)

        # Update person features with moving average
        if person_id in self.person_features:
            alpha = 0.7  # Weight for historical features
            self.person_features[person_id] = (alpha * self.person_features[person_id] +
                                               (1 - alpha) * features)
        else:
            self.person_features[person_id] = features

        # Save updated features
        self.save_person_features(
            person_id, self.person_features[person_id], frame_time)


    def process_video(self):
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_time = frame_count / self.fps
            frame_count += 1

            # Detect persons using YOLO
            results = self.detector(frame, classes=[0])  # class 0 is person

            # Process detections
            detections = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    detections.append((box.xyxy[0], box.conf[0]))

            # Update tracking
            self.update_tracks(frame, detections, frame_time)

            # Visualize results
            for track_id, track_info in self.active_tracks.items():
                box = track_info['box']
                cv2.rectangle(frame, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}",
                            (int(box[0]), int(box[1])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display frame
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        return self.generate_report()

    def generate_report(self):
        """Generate tracking report"""
        report = {
            'total_unique_persons': self.next_id,
            'person_details': {}
        }

        for person_id in self.person_timestamps.keys():
            report['person_details'][person_id] = {
                'first_appearance': self.person_timestamps[person_id]['first_appearance'],
                'last_appearance': self.person_timestamps[person_id]['last_appearance'],
                'duration': self.person_timestamps[person_id]['last_appearance'] -
                self.person_timestamps[person_id]['first_appearance'],
                'image_path': os.path.join(self.output_dir, f"person_{person_id}")
            }

        return report

def main():
    # Process all videos in the working directory
    summary = process_video_directory(working_directory, ouptut_dir)

    # Verify CSV location
    csv_path = os.path.join(ouptut_dir, 'daily_tracking_summary.csv')
    if os.path.exists(csv_path):
        print(f"\nConfirmed: CSV file exists at {csv_path}")
        df = pd.read_csv(csv_path)
        print("\nFinal CSV Contents:")
        print(df.to_string())
    else:
        print(f"\nWarning: CSV file not found at {csv_path}")

if __name__ == '__main__':
    main()
