"""
Person Tracking System with Multi-Camera Support
This module implements a person tracking system using YOLO detection and ReID features.
"""

import os
import cv2
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import scipy.spatial.distance as distance
from ultralytics import YOLO
import torchreid

# Type aliases
Box = Tuple[float, float, float, float]  # x1, y1, x2, y2
Features = np.ndarray
Timestamp = float

class TrackingState(Enum):
    """Enumeration of possible tracking states"""
    ACTIVE = 'active'
    OCCLUDED = 'occluded'
    TENTATIVE = 'tentative'
    LOST = 'lost'

@dataclass
class TrackConfig:
    """Configuration parameters for tracking"""
    similarity_threshold: float = 0.6
    min_transition_time: int = 30    # seconds
    max_transition_time: int = 900   # seconds
    expected_transition: int = 150   # seconds
    min_track_duration: float = 1.5  # seconds
    min_detections: int = 5
    track_quality_threshold: float = 0.6
    max_history_length: int = 10
    door_buffer: int = 50  # pixels

@dataclass
class CameraConfig:
    """Camera-specific configuration"""
    door_coords: Dict[int, List[Tuple[int, int]]] = None
    
    def __post_init__(self):
        self.door_coords = {
            1: [(1030, 0), (1700, 560)],  # Camera 1 door
            2: [(400, 0), (800, 470)]      # Camera 2 door
        }

class Track:
    """Represents a single tracking instance"""
    def __init__(self, box: Box, features: Features, frame_time: Timestamp):
        self.box = box
        self.features = features
        self.last_seen = frame_time
        self.disappeared = 0
        self.state = TrackingState.TENTATIVE
        self.velocity = [0, 0]
        self.previous_box = None
        self.color_features = []
        self.trajectory = [box]

class GlobalTracker:
    """Manages cross-camera tracking and identity association"""
    
    def __init__(self, config: TrackConfig):
        self.config = config
        self.global_identities = {}
        self.appearance_sequence = {}
        self.feature_database = {}
        self.color_database = {}
        self.track_exits = defaultdict(list)
        self.track_entries = defaultdict(list)
        self.feature_history = defaultdict(list)
    
    def register_camera_detection(self, 
                                camera_id: int, 
                                person_id: int, 
                                features: Features, 
                                timestamp: Timestamp,
                                color_features: Optional[Features] = None,
                                is_entry: bool = False,
                                is_exit: bool = False) -> int:
        """
        Register a new detection and return global ID
        
        Args:
            camera_id: ID of the camera
            person_id: Local ID of the person
            features: ReID features
            timestamp: Detection timestamp
            color_features: Optional color features
            is_entry: Whether this is an entry point
            is_exit: Whether this is an exit point
            
        Returns:
            int: Global ID assigned to this person
        """
        camera_key = f"{camera_id}_{person_id}"
        
        if color_features is not None:
            if camera_key not in self.color_database:
                self.color_database[camera_key] = []
            self.color_database[camera_key].append(color_features)
        
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
        
        global_id = self._match_or_create_global_id(
            camera_id, person_id, features, timestamp)
        
        if global_id not in self.appearance_sequence:
            self.appearance_sequence[global_id] = []
        
        camera_key = f"Camera_{camera_id}"
        if not self.appearance_sequence[global_id] or \
           self.appearance_sequence[global_id][-1]['camera'] != camera_key:
            self.appearance_sequence[global_id].append({
                'camera': camera_key,
                'timestamp': timestamp,
                'is_entry': is_entry,
                'is_exit': is_exit
            })
            
        return global_id

    def _match_or_create_global_id(self, 
                                 camera_id: int, 
                                 person_id: int, 
                                 features: Features, 
                                 timestamp: Timestamp) -> int:
        """Match detection to existing global ID or create new one"""
        camera_key = f"{camera_id}_{person_id}"
        
        if camera_key in self.global_identities:
            return self.global_identities[camera_key]
            
        best_match = None
        best_score = 0
        
        for global_id, stored_features in self.feature_database.items():
            last_appearance = self.appearance_sequence.get(global_id, [])[-1] \
                            if self.appearance_sequence.get(global_id) else None
            
            if not last_appearance:
                continue
                
            last_camera = int(last_appearance['camera'].split('_')[1])
            time_diff = timestamp - last_appearance['timestamp']
            
            # Skip invalid transitions
            if last_camera == camera_id or \
               time_diff < self.config.min_transition_time or \
               time_diff > self.config.max_transition_time:
                continue
            
            # Calculate similarities
            reid_sim = 1 - distance.cosine(features.flatten(), 
                                         stored_features.flatten())
            
            color_sim = self._calculate_color_similarity(
                camera_key, global_id)
            
            time_score = self._calculate_time_score(time_diff)
            
            transition_bonus = self._calculate_transition_bonus(
                last_camera, camera_id, global_id, camera_key)
            
            # Combined similarity score
            similarity = (0.4 * reid_sim + 
                        0.2 * color_sim + 
                        0.2 * time_score +
                        0.2 * transition_bonus)
            
            if similarity > self.config.similarity_threshold and \
               similarity > best_score:
                best_match = global_id
                best_score = similarity
        
        if best_match is None:
            best_match = len(self.global_identities)
            self.feature_database[best_match] = features
            self.feature_history[best_match] = [features]
        else:
            self._update_feature_database(best_match, features)
        
        self.global_identities[camera_key] = best_match
        return best_match

    def _calculate_color_similarity(self, 
                                 camera_key: str, 
                                 global_id: int) -> float:
        """Calculate color feature similarity"""
        if camera_key in self.color_database and \
           global_id in self.color_database:
            return 1 - distance.cosine(
                self.color_database[camera_key][-1].flatten(),
                self.color_database[global_id][-1].flatten()
            )
        return 0

    def _calculate_time_score(self, time_diff: float) -> float:
        """Calculate temporal similarity score"""
        score = 1.0 - abs(time_diff - self.config.expected_transition) / \
                self.config.expected_transition
        return max(0, score)

    def _calculate_transition_bonus(self,
                                 last_camera: int,
                                 current_camera: int,
                                 global_id: int,
                                 camera_key: str) -> float:
        """Calculate transition probability bonus"""
        if last_camera == 1 and current_camera == 2:
            if (self.track_exits.get(f"1_{global_id}") and 
                self.track_entries.get(camera_key)):
                return 0.2
        return 0

    def _update_feature_database(self, 
                              global_id: int, 
                              features: Features):
        """Update feature database with moving average"""
        self.feature_history[global_id].append(features)
        alpha = 0.7
        self.feature_database[global_id] = (
            alpha * self.feature_database[global_id] +
            (1 - alpha) * features
        )

class PersonTracker:
    """Single-camera person tracking system"""
    
    def __init__(self, video_path: str, output_base_dir: str = "tracking_results"):
        self.video_name = Path(video_path).stem
        self.setup_directories(output_base_dir)
        self.initialize_models()
        self.setup_video_capture(video_path)
        self.initialize_tracking_state()
        self.config = TrackConfig()
        self.camera_config = CameraConfig()
        
        # Set camera-specific parameters
        self.camera_id = int(Path(video_path).stem.split('_')[1])
        self._adjust_camera_parameters()

    def setup_directories(self, output_base_dir: str):
        """Setup output directories for results"""
        self.output_dir = os.path.join(output_base_dir, self.video_name)
        self.images_dir = os.path.join(self.output_dir, "person_images")
        self.features_dir = os.path.join(self.output_dir, "person_features")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def initialize_models(self):
        """Initialize YOLO and ReID models"""
        self.detector = YOLO("yolo11x.pt")
        self.reid_model = torchreid.models.build_model(
            name='osnet_ain_x1_0',
            num_classes=1000,
            pretrained=True
        )
        self.reid_model = self.reid_model.cuda() \
                         if torch.cuda.is_available() else self.reid_model
        self.reid_model.eval()

    def setup_video_capture(self, video_path: str):
        """Setup video capture and properties"""
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def initialize_tracking_state(self):
        """Initialize tracking state variables"""
        self.active_tracks = {}
        self.person_features = {}
        self.person_timestamps = {}
        self.next_id = 0
        self.appearance_history = defaultdict(list)
        self.lost_tracks = {}
        self.tracks_through_door = set()
        self.track_trajectories = defaultdict(list)
        self.color_features = defaultdict(list)

    def _adjust_camera_parameters(self):
        """Adjust parameters based on camera ID"""
        if self.camera_id == 1:
            self.config.similarity_threshold = 0.65
            self.config.track_quality_threshold = 0.7
        else:
            self.config.similarity_threshold = 0.6
            self.config.track_quality_threshold = 0.65

    def process_video(self) -> dict:
        """
        Process video and track persons
        
        Returns:
            dict: Tracking report
        """
        frame_count = 0
        valid_tracks = set()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_time = frame_count / self.fps
            frame_count += 1
            
            # Process frame
            detections = self._get_detections(frame)
            self.update_tracks(frame, detections, frame_time)
            
            # Validate tracks
            for track_id in list(self.active_tracks.keys()):
                if self.validate_track(track_id):
                    valid_tracks.add(track_id)
        
        # Keep only valid tracks that interacted with door
        self._filter_final_tracks(valid_tracks)
        return self.generate_report()

    def _get_detections(self, frame: np.ndarray) -> List[Tuple[Box, float]]:
        """Get person detections from frame"""
        results = self.detector(frame, classes=[0])
        return [(box.xyxy[0], box.conf[0]) 
                for result in results
                for box in result.boxes.cpu().numpy()]

    def _filter_final_tracks(self, valid_tracks: set):
        """Filter tracks based on validation criteria"""
        final_tracks = {}
        for track_id in valid_tracks:
            if track_id in self.tracks_through_door:
                final_tracks[track_id] = self.active_tracks[track_id]
        self.active_tracks = final_tracks

    # ... [Rest of the PersonTracker methods remain the same]

def process_video_directory(input_dir: str, 
                          output_base_dir: Optional[str] = None) -> dict:
    """
    Process all videos in directory
    
    Args:
        input_dir: Input directory containing videos
        output_base_dir: Output directory for results
        
    Returns:
        dict: Processing summary
    """
    global_tracker = GlobalTracker(TrackConfig())
    results = {}
    per_camera_stats = defaultdict(int)
    
    if output_base_dir is None:
        output_base_dir = os.path.join(
            os.path.dirname(input_dir), 
            "tracking_results"
        )
    
    # Ensure absolute paths
    input_dir = os.path.abspath(input_dir)
    output_base_dir = os.path.abspath(output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process videos
    video_files = sorted(list(Path(input_dir).glob('*.[ma][pv][4i]')))
    print(f"\nFound {len(video_files)} videos")
    
    for video_path in video_files:
        try:
            tracker = PersonTracker(str(video_path), output_base_dir)
            results[video_path.stem] = process_single_video(
                tracker, global_tracker)
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
            
    # Generate summary CSV
    transition_analysis = global_tracker.analyze_camera_transitions()
    csv_data = generate_csv_data(results, transition_analysis)
    save_csv_report(csv_data, output_base_dir)
    
    return {
        'per_camera_statistics': dict(per_camera_stats),
        'cross_camera_analysis': transition_analysis,
        'total_unique_global': len(global_tracker.global_identities)
    }

def process_single_video(tracker: PersonTracker, 
                        global_tracker: GlobalTracker) -> dict:
    """Process a single video and update global tracking"""
    # Process video with individual tracker
    tracker_results = tracker.process_video()
    
    # Register high-quality tracks for cross-camera matching
    for track_id, track_info in tracker.active_tracks.items():
        if tracker.validate_track(track_id):
            global_tracker.register_camera_detection(
                tracker.camera_id,
                track_id,
                track_info['features'],
                tracker.person_timestamps[track_id]['first_appearance']
            )
    
    return {
        'video_name': tracker.video_name,
        'date': tracker.video_name.split('_')[-1],
        'camera_id': tracker.camera_id,
        'person_details': {
            track_id: {
                'first_appearance': tracker.person_timestamps[track_id]['first_appearance'],
                'last_appearance': track_info['last_seen'],
                'duration': track_info['last_seen'] - 
                           tracker.person_timestamps[track_id]['first_appearance']
            }
            for track_id, track_info in tracker.active_tracks.items()
        }
    }

def generate_csv_data(results: Dict[str, dict],
                     transition_analysis: dict) -> List[dict]:
    """Generate data for CSV report"""
    csv_data = []
    
    for video_name, video_results in results.items():
        date = video_results['date']
        camera_id = video_results['camera_id']
        
        date_entry = next(
            (entry for entry in csv_data if entry['Date'] == date), 
            None
        )
        
        if date_entry is None:
            date_entry = {
                'Date': date,
                'Camera1_Unique_Individuals': 0,
                'Camera2_Unique_Individuals': 0,
                'Transitions_Camera1_to_Camera2': 
                    transition_analysis['camera1_to_camera2'],
                'Transitions_Camera2_to_Camera1': 
                    transition_analysis['camera2_to_camera1']
            }
            csv_data.append(date_entry)
        
        date_entry[f'Camera{camera_id}_Unique_Individuals'] = \
            len(video_results['person_details'])
    
    return csv_data

def save_csv_report(csv_data: List[dict], output_dir: str):
    """Save tracking results to CSV"""
    csv_path = os.path.join(output_dir, 'daily_tracking_summary.csv')
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to: {csv_path}")
    print("\nCSV Contents:")
    print(df.to_string())

def find_csv_file(working_directory: str) -> Optional[str]:
    """
    Find the daily tracking summary CSV file
    
    Args:
        working_directory: Directory to search in
        
    Returns:
        str or None: Path to CSV file if found
    """
    # Check default location
    default_path = os.path.join(
        working_directory, 
        "tracking_results",
        "daily_tracking_summary.csv"
    )
    
    if os.path.exists(default_path):
        print(f"\nFound CSV file at: {default_path}")
        return default_path
    
    # Search recursively
    print("\nSearching for CSV file...")
    for root, _, files in os.walk(working_directory):
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
    """Main entry point"""
    # Define working directory
    working_directory = os.path.join(
        'C:\\Users', 'mc1159', 
        'OneDrive - University of Exeter',
        'Documents', 'VISIONARY', 
        'Durham Experiment', 'test_data'
    )
    output_dir = os.path.join(working_directory, 'tracking_results')
    
    # Process videos
    summary = process_video_directory(working_directory, output_dir)
    
    # Find and display CSV
    csv_path = find_csv_file(working_directory)
    if csv_path and os.path.exists(csv_path):
        print("\nCSV file contents:")
        try:
            df = pd.read_csv(csv_path)
            print(df.to_string())
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")

if __name__ == '__main__':
    main()
