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
# working_directory = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
#                                  'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')
working_directory = os.path.join('D:\\', 'OneDrive - University of Exeter',
                                 'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')
ouptut_dir = os.path.join(working_directory, 'tracking_results')

class TrackingState:
    ACTIVE = 'active'          # Fully visible
    OCCLUDED = 'occluded'      # Temporarily hidden
    TENTATIVE = 'tentative'    # New track
    LOST = 'lost'              # Missing too long

class GlobalTracker:
    def __init__(self):
        self.global_identities = {}
        self.appearance_sequence = {}
        self.feature_database = {}
        self.color_database = {}
        
        # Matching parameters
        self.similarity_threshold = 0.6
        self.min_transition_time = 30    # 30 seconds
        self.max_transition_time = 900   # 15 minutes
        self.expected_transition = 150    # Expected transition time
        
        # Track history
        self.track_exits = defaultdict(list)  # Track exit points
        self.track_entries = defaultdict(list)  # Track entry points
        self.feature_history = defaultdict(list)  # Track feature history
        
    def register_camera_detection(self, camera_id, person_id, features, timestamp, 
                                color_features=None, is_entry=False, is_exit=False):
        """Register detection with entry/exit information"""
        camera_key = f"{camera_id}_{person_id}"
        
        # Store color features if provided
        if color_features is not None:
            if camera_key not in self.color_database:
                self.color_database[camera_key] = []
            self.color_database[camera_key].append(color_features)
        
        # Store entry/exit information
        if is_entry:
            self.track_entries[camera_key].append({
                'timestamp': timestamp,
                'features': features
            })
        if is_exit:
            self.track_exits[camera_key].append({
                'timestamp': timestamp,
                'features': features
            })
        
        global_id = self._match_or_create_global_id(camera_id, person_id, features, timestamp)
        
        if global_id not in self.appearance_sequence:
            self.appearance_sequence[global_id] = []
        
        camera_key = f"Camera_{camera_id}"
        # Only append if this is a new appearance
        if not self.appearance_sequence[global_id] or \
           self.appearance_sequence[global_id][-1]['camera'] != camera_key:
            self.appearance_sequence[global_id].append({
                'camera': camera_key,
                'timestamp': timestamp,
                'is_entry': is_entry,
                'is_exit': is_exit
            })
    
    def _match_or_create_global_id(self, camera_id, person_id, features, timestamp):
        """Enhanced matching with entry/exit awareness"""
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
                
                # Skip invalid transitions
                if last_camera == camera_id or \
                   time_diff < self.min_transition_time or \
                   time_diff > self.max_transition_time:
                    continue
                
                # Calculate feature similarity
                reid_sim = 1 - distance.cosine(features.flatten(), stored_features.flatten())
                
                # Calculate color similarity if available
                color_sim = 0
                if camera_key in self.color_database and global_id in self.color_database:
                    color_sim = 1 - distance.cosine(
                        self.color_database[camera_key][-1].flatten(),
                        self.color_database[global_id][-1].flatten()
                    )
                
                # Calculate temporal score
                time_score = 1.0 - abs(time_diff - self.expected_transition) / self.expected_transition
                time_score = max(0, time_score)
                
                # Add entry/exit bonus
                transition_bonus = 0
                if last_camera == 1 and camera_id == 2:
                    # Check if person exited Camera 1 and entered Camera 2
                    if (self.track_exits.get(f"1_{global_id}") and 
                        self.track_entries.get(camera_key)):
                        transition_bonus = 0.2
                
                # Combined similarity with transition awareness
                similarity = (0.4 * reid_sim + 
                            0.2 * color_sim + 
                            0.2 * time_score +
                            0.2 * transition_bonus)
                
                if similarity > self.similarity_threshold and similarity > best_score:
                    best_match = global_id
                    best_score = similarity
        
        if best_match is None:
            best_match = len(self.global_identities)
            self.feature_database[best_match] = features
            self.feature_history[best_match] = [features]
        else:
            # Update feature history
            self.feature_history[best_match].append(features)
            # Update stored features with moving average
            alpha = 0.7
            self.feature_database[best_match] = (
                alpha * self.feature_database[best_match] +
                (1 - alpha) * features
            )
        
        self.global_identities[camera_key] = best_match
        return best_match
    
    def analyze_camera_transitions(self):
        """Analyze transitions with entry/exit validation"""
        cam1_to_cam2 = 0
        cam2_to_cam1 = 0
        valid_transitions = []
        
        for global_id, appearances in self.appearance_sequence.items():
            # Sort appearances by timestamp
            sorted_appearances = sorted(appearances, key=lambda x: x['timestamp'])
            
            # Check for sequential appearances
            for i in range(len(sorted_appearances) - 1):
                current = sorted_appearances[i]
                next_app = sorted_appearances[i + 1]
                
                if (current['camera'] == 'Camera_1' and 
                    next_app['camera'] == 'Camera_2'):
                    # Validate the transition
                    time_diff = next_app['timestamp'] - current['timestamp']
                    if (self.min_transition_time <= time_diff <= self.max_transition_time and
                        current.get('is_exit', False) and
                        next_app.get('is_entry', False)):
                        cam1_to_cam2 += 1
                        valid_transitions.append({
                            'global_id': global_id,
                            'from_camera': 1,
                            'to_camera': 2,
                            'exit_time': current['timestamp'],
                            'entry_time': next_app['timestamp'],
                            'transition_time': time_diff
                        })
                        
                elif (current['camera'] == 'Camera_2' and 
                      next_app['camera'] == 'Camera_1'):
                    time_diff = next_app['timestamp'] - current['timestamp']
                    if self.min_transition_time <= time_diff <= self.max_transition_time:
                        cam2_to_cam1 += 1
        
        return {
            'camera1_to_camera2': cam1_to_cam2,
            'camera2_to_camera1': cam2_to_cam1,
            'valid_transitions': valid_transitions
        }

def process_video_directory(input_dir, output_base_dir=None):
    global_tracker = GlobalTracker()
    results = {}
    per_camera_stats = defaultdict(int)
    
    if output_base_dir is None:
        output_base_dir = os.path.join(os.path.dirname(input_dir), "tracking_results")
    
    # Ensure absolute paths
    input_dir = os.path.abspath(input_dir)
    output_base_dir = os.path.abspath(output_base_dir)
    
    os.makedirs(output_base_dir, exist_ok=True)
    
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
            tracker.process_video()
            
            # Identify stable tracks for individual counting
            individual_tracks = {}
            transition_tracks = {}
            
            for track_id, track_info in tracker.active_tracks.items():
                track_duration = (track_info['last_seen'] - 
                                tracker.person_timestamps[track_id]['first_appearance'])
                features = np.array([f for f in tracker.appearance_history[track_id]])
                
                # Check for individual counting criteria
                if (track_duration >= 1.5 and  # Minimum 1.5 seconds
                    len(features) >= 2):       # Minimum 2 detections
                    individual_tracks[track_id] = track_info
                    
                    # Check if track also meets transition criteria
                    if (track_duration >= 3.0 and  # Minimum 3 seconds
                        len(features) >= 3):       # Minimum 3 detections
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
    
    # Analyze camera transitions
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
        self.similarity_threshold = 0.45     # Much more lenient for initial matching
        self.min_detection_confidence = 0.5 # Lower detection threshold
        self.feature_weight = 0.6          # Equal weight to features and position
        self.position_weight = 0.4
        self.motion_weight = 0.0           # Disable motion weighting temporarily
        
        # Track maintenance parameters
        self.max_disappeared = self.fps * 1800 # Longer disappeared time
        
        # New parameters for track splitting and merging
        self.max_track_gap = self.fps * 1800  # Longer gap allowed
        self.merge_threshold = 0.55         # Lower threshold for merging
        self.reentry_threshold = 0.75

        # Track consolidation parameters
        self.min_track_length = self.fps * 3  # Minimum 3 seconds
        self.max_track_gap = self.fps * 2     # Maximum 2 second gap for merging
        self.consolidation_threshold = 0.75    # Threshold for merging tracks
        
        # Time-based track filtering
        self.min_track_duration = 1.5  # Minimum 1.5 seconds for valid track
        self.stable_track_threshold = self.fps * 5  # 5 seconds for stable track
        
        # Add track stability requirements
        self.min_consecutive_detections = 3  # New parameter
        self.max_position_jump = 100        # Pixels, new parameter

         # Door region parameters
        self.camera_id = int(Path(video_path).stem.split('_')[1])
        self.door_coords = {
            1: [(1030, 0), (1700, 560)],  # Camera 1 door
            2: [(400, 0), (800, 470)]      # Camera 2 door
        }
        self.door_buffer = 50  # pixels around door region
        
        # Track validation parameters
        self.min_track_duration = 1.5      # Minimum 1.5 seconds
        self.min_detections = 5            # Minimum detections required
        self.track_quality_threshold = 0.6  # Quality threshold
        
        # Track analysis
        self.tracks_through_door = set()   # Tracks that entered/exited through door
        self.track_trajectories = defaultdict(list)  # Store movement history
        self.color_features = defaultdict(list)      # Store color histograms

        # Adjust parameters based on camera
        if self.camera_id == 1:
            # More strict parameters for complex environment in Camera 1
            self.min_detection_confidence = 0.6
            self.consolidation_threshold = 0.8
        else:
            # More relaxed parameters for Camera 2's cleaner environment
            self.min_detection_confidence = 0.5
            self.consolidation_threshold = 0.75
    
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
        """Enhanced door region check"""
        box_center = self.calculate_box_center(box)
        door = self.door_coords[self.camera_id]
        
        # Add buffer to door region
        x_min = max(0, door[0][0] - self.door_buffer)
        x_max = min(self.frame_width, door[1][0] + self.door_buffer)
        y_min = max(0, door[0][1] - self.door_buffer)
        y_max = min(self.frame_height, door[1][1] + self.door_buffer)
        
        return (x_min <= box_center[0] <= x_max and 
                y_min <= box_center[1] <= y_max)
    
    def extract_color_features(self, frame, box):
        """Extract color histogram features"""
        x1, y1, x2, y2 = map(int, box)
        person_crop = frame[y1:y2, x1:x2]
        
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

    def analyze_trajectory(self, track_id):
        """Analyze track trajectory and door interaction"""
        if len(self.track_trajectories[track_id]) < self.min_detections:
            return False
            
        positions = np.array(self.track_trajectories[track_id])
        
        # Check if track starts or ends near door
        starts_at_door = self.is_in_door_region(positions[0])
        ends_at_door = self.is_in_door_region(positions[-1])
        
        # Calculate movement direction
        if len(positions) >= 2:
            start_center = self.calculate_box_center(positions[0])
            end_center = self.calculate_box_center(positions[-1])
            
            # Vector from start to end
            movement = np.array(end_center) - np.array(start_center)
            
            # Check if movement is consistent with entry/exit
            if self.camera_id == 1:
                # For Camera 1, check if movement is from door to interior
                is_valid = starts_at_door or ends_at_door
            else:
                # For Camera 2, check if movement is through main area
                is_valid = abs(movement[0]) > 50  # Significant horizontal movement
            
            return is_valid
        
        return False
    
    def validate_track(self, track_id):
        """Comprehensive track validation"""
        if track_id not in self.active_tracks:
            return False
            
        track = self.active_tracks[track_id]
        
        # Check basic requirements
        duration = track['last_seen'] - self.person_timestamps[track_id]['first_appearance']
        if duration < self.min_track_duration:
            return False
            
        if len(self.appearance_history[track_id]) < self.min_detections:
            return False
            
        # Check feature consistency
        features = np.array([f for f in self.appearance_history[track_id]])
        consistencies = []
        for i in range(len(features)-1):
            similarity = 1 - distance.cosine(features[i].flatten(), features[i+1].flatten())
            consistencies.append(similarity)
        
        if not consistencies or np.mean(consistencies) < self.track_quality_threshold:
            return False
            
        # Check trajectory
        if not self.analyze_trajectory(track_id):
            return False
            
        return True

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
        """Save tracking results with corrected structure"""
        video_date = self.video_name.split('_')[-1]  # Extract date from video name
        
        results = {
            'video_name': self.video_name,
            'date': video_date,
            'video_metadata': {
                'width': self.frame_width,
                'height': self.frame_height,
                'fps': self.fps
            },
            'total_persons': len(self.person_timestamps),
            'person_details': {}  # Ensure this key exists
        }
        
        # Process each person's data
        for person_id, timestamps in self.person_timestamps.items():
            # Calculate track duration
            duration = timestamps['last_appearance'] - timestamps['first_appearance']
            
            # Only include tracks that meet minimum duration
            if duration >= self.min_track_duration:
                results['person_details'][person_id] = {
                    'first_appearance': timestamps['first_appearance'],
                    'last_appearance': timestamps['last_appearance'],
                    'duration': duration,
                    'appearances': len(self.appearance_history.get(person_id, [])),
                }
        
        # Save results to JSON
        results_path = os.path.join(self.output_dir, "tracking_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

    def update_tracks(self, frame, detections, frame_time):
        """Enhanced track updating with door awareness"""
        current_boxes = []
        current_features = []
        current_colors = []
        
        # Process detections
        for box, conf in detections:
            if conf < self.min_detection_confidence:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
                
            features = self.extract_features(person_crop)
            color_feat = self.extract_color_features(frame, [x1, y1, x2, y2])
            
            if features is not None:
                current_boxes.append([x1, y1, x2, y2])
                current_features.append(features)
                current_colors.append(color_feat)
        
        # Match detections to tracks
        matched_tracks = set()
        matched_detections = set()
        
        for det_idx, (det_features, det_box, det_colors) in enumerate(
            zip(current_features, current_boxes, current_colors)):
            
            best_match = None
            best_score = 0
            
            for track_id, track in self.active_tracks.items():
                if track_id in matched_tracks:
                    continue
                
                # Calculate feature similarity
                reid_sim = 1 - distance.cosine(det_features.flatten(), 
                                             track['features'].flatten())
                
                # Calculate color similarity
                if track_id in self.color_features:
                    color_sim = 1 - distance.cosine(det_colors.flatten(),
                                                  self.color_features[track_id][-1].flatten())
                else:
                    color_sim = 0
                
                # Calculate position similarity
                pos_sim = self.calculate_iou(det_box, track['box'])
                
                # Combined similarity
                similarity = (0.5 * reid_sim + 0.3 * color_sim + 0.2 * pos_sim)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = track_id
            
            if best_match is not None and best_score > self.similarity_threshold:
                self.update_existing_track(best_match, det_box, det_features, frame_time, frame)
                self.track_trajectories[best_match].append(det_box)
                self.color_features[best_match].append(det_colors)
                matched_tracks.add(best_match)
                matched_detections.add(det_idx)
                
                # Check door interaction
                if self.is_in_door_region(det_box):
                    self.tracks_through_door.add(best_match)
        
        # Create new tracks for unmatched detections
        for det_idx in range(len(current_features)):
            if det_idx in matched_detections:
                continue
                
            new_id = self.next_id
            self.next_id += 1
            
            self.active_tracks[new_id] = {
                'box': current_boxes[det_idx],
                'features': current_features[det_idx],
                'last_seen': frame_time,
                'disappeared': 0
            }
            
            self.appearance_history[new_id] = [current_features[det_idx]]
            self.color_features[new_id] = [current_colors[det_idx]]
            self.track_trajectories[new_id] = [current_boxes[det_idx]]
            self.person_timestamps[new_id] = {
                'first_appearance': frame_time,
                'last_appearance': frame_time
            }
            
            if self.is_in_door_region(current_boxes[det_idx]):
                self.tracks_through_door.add(new_id)
        
        # Update unmatched tracks
        for track_id in list(self.active_tracks.keys()):
            if track_id not in matched_tracks:
                track = self.active_tracks[track_id]
                track['disappeared'] += 1
                
                if track['disappeared'] > self.max_disappeared:
                    del self.active_tracks[track_id]
    
    def is_track_stable(self, track_id):
        """Check if a track meets stability criteria"""
        if track_id not in self.appearance_history:
            return False
            
        history = self.appearance_history[track_id]
        if len(history) < self.min_consecutive_detections:
            return False
            
        # Check feature consistency
        features = np.array([f for f in history[-self.min_consecutive_detections:]])
        consistencies = []
        for i in range(len(features)-1):
            similarity = 1 - distance.cosine(features[i].flatten(), features[i+1].flatten())
            consistencies.append(similarity)
        
        avg_consistency = np.mean(consistencies) if consistencies else 0
        return avg_consistency >= self.track_quality_threshold
    
    def merge_similar_tracks(self):
        """More conservative track merging"""
        merged = set()
        
        for track_id1 in list(self.active_tracks.keys()):
            if track_id1 in merged:
                continue
                
            track1 = self.active_tracks[track_id1]
            if not self.is_track_stable(track_id1):
                continue
                
            for track_id2 in list(self.active_tracks.keys()):
                if track_id2 in merged or track_id1 == track_id2:
                    continue
                    
                track2 = self.active_tracks[track_id2]
                if not self.is_track_stable(track_id2):
                    continue
                    
                # Check temporal overlap
                time_gap = abs(track1['last_seen'] - track2['last_seen'])
                if time_gap > self.max_track_gap:
                    continue
                
                # Calculate feature similarity
                similarity = 1 - distance.cosine(
                    track1['features'].flatten(),
                    track2['features'].flatten()
                )
                
                # Check position consistency
                pos1 = self.calculate_box_center(track1['box'])
                pos2 = self.calculate_box_center(track2['box'])
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if (similarity > self.merge_threshold and 
                    distance < self.max_position_jump):
                    self.merge_tracks(track_id1, track_id2)
                    merged.add(track_id2)

    def process_video(self):
        frame_count = 0
        valid_tracks = set()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_time = frame_count / self.fps
            frame_count += 1
            
            # Process frame
            results = self.detector(frame, classes=[0])
            detections = [(box.xyxy[0], box.conf[0]) 
                         for result in results
                         for box in result.boxes.cpu().numpy()]
            
            # Update tracks
            self.update_tracks(frame, detections, frame_time)
            
            # Validate tracks
            for track_id in list(self.active_tracks.keys()):
                if self.validate_track(track_id):
                    valid_tracks.add(track_id)
        
        # Final cleanup - keep only valid tracks that interacted with door
        final_tracks = {}
        for track_id in valid_tracks:
            if track_id in self.tracks_through_door:
                final_tracks[track_id] = self.active_tracks[track_id]
        
        self.active_tracks = final_tracks
        return self.generate_report()
    
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
