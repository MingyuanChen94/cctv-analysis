import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torchreid.utils import FeatureExtractor
from yolox.tracker.byte_tracker import BYTETracker
from collections import defaultdict
import datetime
from filterpy.kalman import KalmanFilter

class CameraConfig:
    def __init__(self, camera_id, door_coords):
        self.camera_id = camera_id
        self.door_coords = door_coords
        self.fps = 6
        self.resolution = (640, 360)

class VideoSynchronizer:
    """Handles video synchronization between cameras"""
    def __init__(self, fps=6):
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.sync_buffer = {}
        
    def add_frame(self, camera_id, timestamp, frame):
        """Add frame to synchronization buffer"""
        self.sync_buffer[camera_id] = {
            'timestamp': timestamp,
            'frame': frame
        }
        
    def get_synced_frames(self):
        """Get temporally aligned frames"""
        if len(self.sync_buffer) < 2:
            return None
            
        # Get timestamps
        times = [data['timestamp'] for data in self.sync_buffer.values()]
        time_diff = abs(times[0] - times[1])
        
        # Check if frames are within sync threshold
        if time_diff <= self.frame_interval / 2:
            return {
                camera_id: data['frame']
                for camera_id, data in self.sync_buffer.items()
            }
        return None

class HistogramMatcher:
    """Matches histograms between cameras"""
    def __init__(self):
        self.reference_hist = None
        
    def set_reference(self, frame):
        """Set reference histogram from Camera 1"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        self.reference_hist = {
            'l': cv2.calcHist([l], [0], None, [256], [0, 256]),
            'a': cv2.calcHist([a], [0], None, [256], [0, 256]),
            'b': cv2.calcHist([b], [0], None, [256], [0, 256])
        }
        
    def match_histogram(self, frame):
        """Match frame histogram to reference"""
        if self.reference_hist is None:
            return frame
            
        # Convert to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Match each channel
        l_matched = cv2.equalizeHist(l)
        a_matched = cv2.equalizeHist(a)
        b_matched = cv2.equalizeHist(b)
        
        # Merge channels
        matched = cv2.merge([l_matched, a_matched, b_matched])
        return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)    """Helper class to manage track buffers and feature updates"""
    def __init__(self, max_size=300):  # 50 seconds at 6fps
        self.max_size = max_size
        self.positions = []
        self.features = []
        self.timestamps = []
        
    def add(self, position, features, timestamp):
        self.positions.append(position)
        self.features.append(features)
        self.timestamps.append(timestamp)
        
        # Maintain buffer size
        if len(self.positions) > self.max_size:
            self.positions.pop(0)
            self.features.pop(0)
            self.timestamps.pop(0)
            
    def get_recent_features(self, num_frames=5):
        """Get average of recent features"""
        if not self.features:
            return None
        recent = self.features[-num_frames:]
        return np.mean(recent, axis=0)
        
    def get_velocity(self):
        """Calculate current velocity"""
        if len(self.positions) < 2:
            return None
        return np.array(self.positions[-1]) - np.array(self.positions[-2])

class PersonTracker:

def analyze_video_pair(camera1_path, camera2_path):
    """
    Analyze a synchronized pair of videos from both cameras
    Returns preprocessed frames and timing information
    """
    cap1 = cv2.VideoCapture(camera1_path)
    cap2 = cv2.VideoCapture(camera2_path)
    
    # Ensure videos exist and can be opened
    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Cannot open video files")
    
    # Get video properties
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    
    # Synchronize frame rates if different
    target_fps = min(fps1, fps2)
    frame_interval = 1.0 / target_fps
    
    # Initialize frame buffers
    frames1, frames2 = [], []
    timestamps = []
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
            
        timestamp = len(timestamps) * frame_interval
        timestamps.append(timestamp)
        
        # Store frames
        frames1.append(frame1)
        frames2.append(frame2)
    
    cap1.release()
    cap2.release()
    
    return frames1, frames2, timestamps

class PersonTracker:
    def __init__(self, camera_configs):
        """
        Initialize the person tracker with enhanced preprocessing
        """
        # Camera configurations
        self.camera_configs = camera_configs
        
        # Video processing parameters
        self.target_fps = 6
        self.min_frame_size = (640, 360)
        self.max_frame_size = (960, 540)  # For upsampling
        
        # Initialize histogram matcher for each camera
        self.hist_matchers = {
            1: HistogramMatcher(),
            2: HistogramMatcher()
        }
        
        # Detection and tracking parameters
        self.detection_params = {
            'conf_thresh': 0.3,  # Lower threshold for low resolution
            'track_buffer': 60,  # Increased buffer for occlusions
            'min_box_area': 100,  # Minimum detection area
            'track_thresh': 0.3   # Lower tracking threshold
        }
        
        # Initialize models with specific configurations
        self.byte_tracker = self._init_byte_tracker()
        self.feature_extractor = self._init_osnet()
        self.yolox_model = self._init_yolox()
        
        # Enhanced kalman filter configuration
        self.kalman = self._init_kalman()
        
        # Track management
        self.tracks = {1: defaultdict(list), 2: defaultdict(list)}
        self.embeddings = {1: {}, 2: {}}
        self.track_buffers = {1: {}, 2: {}}
        
        # Matching and temporal parameters
        self.matching_params = {
            'min_similarity': 0.75,      # Feature matching threshold
            'max_time_gap': 600,         # 10 minutes maximum transition time
            'min_time_gap': 120,         # 2 minutes minimum transition time
            'min_track_length': 6,       # Minimum 1 second of tracking
            'max_velocity': 50,          # Maximum reasonable velocity
            'direction_weight': 0.3      # Weight for direction consistency
        }
        
    def _preprocess_frames(self, frames1, frames2):
        """
        Preprocess frames from both cameras with strict alignment to workflow
        """
        processed1, processed2 = [], []
        
        # Process initial frames to establish reference histograms
        if frames1 and frames2:
            init_frame1 = frames1[0]
            init_frame2 = frames2[0]
            self.hist_matchers[1].set_reference(init_frame1)
            self.hist_matchers[2].set_reference(init_frame2)
        
        # Process all frames
        for frame1, frame2 in zip(frames1, frames2):
            # 1. Resize frames
            frame1_resized = self._resize_frame(frame1)
            frame2_resized = self._resize_frame(frame2)
            
            # 2. Match histograms
            frame1_matched = self.hist_matchers[1].match_histogram(frame1_resized)
            frame2_matched = self.hist_matchers[2].match_histogram(frame2_resized)
            
            processed1.append(frame1_matched)
            processed2.append(frame2_matched)
            
        return processed1, processed2
    
    def _resize_frame(self, frame):
        """
        Resize frame according to workflow specifications
        """
        height, width = frame.shape[:2]
        
        # Calculate target size
        current_res = width * height
        min_res = self.min_frame_size[0] * self.min_frame_size[1]
        max_res = self.max_frame_size[0] * self.max_frame_size[1]
        
        if current_res < min_res:
            # Upsample for small frames
            scale = np.sqrt(min_res / current_res)
            target_width = int(width * scale)
            target_height = int(height * scale)
            interpolation = cv2.INTER_LINEAR
        elif current_res > max_res:
            # Downsample for large frames
            scale = np.sqrt(max_res / current_res)
            target_width = int(width * scale)
            target_height = int(height * scale)
            interpolation = cv2.INTER_AREA
        else:
            return frame
        
        return cv2.resize(frame, (target_width, target_height), 
                         interpolation=interpolation)
                         
    def _detect_in_door_region(self, frame, camera_id):
        """
        Detect persons specifically in door region
        """
        door_coords = self.camera_configs[camera_id].door_coords
        
        # Create door region mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, 
                     (door_coords[0][0], door_coords[0][1]),
                     (door_coords[1][0], door_coords[1][1]), 
                     255, -1)
        
        # Apply mask to frame
        door_region = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Run detection
        detections = self._detect_persons(door_region)
        
        # Filter detections by door region
        valid_detections = []
        for det in detections:
            if self._is_in_door_region(det[:4], door_coords):
                valid_detections.append(det)
                
        return np.array(valid_detections)
        
        # Initialize tracking storage
        self.tracks = {1: defaultdict(list), 2: defaultdict(list)}
        self.embeddings = {1: {}, 2: {}}
        
    def _init_byte_tracker(self):
        tracker_config = {
            'track_thresh': 0.3,  # Lower threshold for noisy environments
            'track_buffer': 60,   # Increased buffer for occlusions
            'match_thresh': 0.8,
            'frame_rate': 6
        }
        return BYTETracker(tracker_config)
    
    def _init_osnet(self):
        return FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='path_to_pretrained_model',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def _preprocess_frame(self, frame, target_size=(640, 360)):
        # Resize frame
        frame = cv2.resize(frame, target_size)
        
        # Normalize colors using histogram equalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        normalized = cv2.merge((cl,a,b))
        return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)
    
    def _is_in_door_region(self, bbox, door_coords):
        x1, y1, x2, y2 = bbox
        door_x1, door_y1 = door_coords[0]
        door_x2, door_y2 = door_coords[1]
        
        # Check if bbox overlaps with door region
        return (x1 < door_x2 and x2 > door_x1 and 
                y1 < door_y2 and y2 > door_y1)
    
    def _init_kalman(self):
        """
        Initialize Kalman filter for trajectory smoothing
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, dx, dy], Measurement: [x, y]
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement function
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        kf.R *= 10
        
        # Process noise
        kf.Q = np.eye(4) * 0.1
        
        return kf
        
    def _smooth_trajectory(self, track_points):
        """
        Apply Kalman filtering to smooth trajectory
        """
        smoothed = []
        self.kalman.reset()
        
        for point in track_points:
            # Predict
            self.kalman.predict()
            
            # Update
            measurement = np.array([point[0], point[1]])
            self.kalman.update(measurement)
            
            # Get smoothed state
            smoothed.append(self.kalman.x[:2])
            
        return np.array(smoothed)

    def _filter_false_detections(self, tracks, min_time=1.0):
        """
        Filter out tracks that are too short or unlikely to be real people
        """
        filtered_tracks = {}
        for camera_id in tracks:
            filtered_tracks[camera_id] = {}
            for track_id, track_data in tracks[camera_id].items():
                # Check track duration
                track_duration = track_data[-1]['timestamp'] - track_data[0]['timestamp']
                if track_duration < min_time:
                    continue
                    
                # Check track consistency
                positions = np.array([d['bbox'][:2] for d in track_data])
                velocities = np.diff(positions, axis=0)
                mean_velocity = np.mean(np.linalg.norm(velocities, axis=1))
                
                # Filter based on reasonable velocity (pixels per frame)
                if 0.1 <= mean_velocity <= 50:  # Adjust thresholds as needed
                    filtered_tracks[camera_id][track_id] = track_data
                    
        return filtered_tracks

    def _compute_door_transitions(self, tracks, camera_id):
        """
        Determine if tracks represent entry or exit through door
        """
        transitions = {}
        door_coords = self.camera_configs[camera_id].door_coords
        
        for track_id, track_data in tracks.items():
            # Get first and last positions
            first_pos = np.array(track_data[0]['bbox'][:2])
            last_pos = np.array(track_data[-1]['bbox'][:2])
            
            # Check if track starts or ends near door
            starts_at_door = self._is_in_door_region(track_data[0]['bbox'], door_coords)
            ends_at_door = self._is_in_door_region(track_data[-1]['bbox'], door_coords)
            
            # Determine transition type
            if starts_at_door and not ends_at_door:
                transitions[track_id] = 'entry'
            elif ends_at_door and not starts_at_door:
                transitions[track_id] = 'exit'
                
        return transitions

    def process_single_video(self, video_path, camera_id):
        """
        Process a single video file with tracking and feature extraction
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        min_track_length = self.matching_params['min_track_length']
        
        # Initialize tracking buffers
        active_tracks = {}
        track_features = {}
        track_timestamps = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame
            frame = self._preprocess_frame(frame, camera_id, frame_count)
            
            # Run detection and tracking
            detections = self._detect_persons(frame)
            tracks = self.byte_tracker.update(
                detections,
                [frame.shape[0], frame.shape[1]],
                [frame.shape[0], frame.shape[1]]
            )
            
            # Filter short tracks
            if frame_count % min_track_length == 0:
                self.tracks[camera_id] = {
                    k: v for k, v in self.tracks[camera_id].items()
                    if len(v) >= min_track_length
                }
            
            # Process tracks
            current_tracks = {}
            timestamp = frame_count / self.camera_configs[camera_id].fps
            
            for track in tracks:
                track_id = track.track_id
                bbox = track.tlbr
                
                # Extract features for all tracked persons
                features = self._extract_features(frame, bbox)
                
                # Update track information
                if track_id not in active_tracks:
                    active_tracks[track_id] = []
                    track_features[track_id] = []
                    track_timestamps[track_id] = []
                
                active_tracks[track_id].append(bbox)
                track_features[track_id].append(features)
                track_timestamps[track_id].append(timestamp)
                current_tracks[track_id] = True
                
                # Store track information if near door
                if self._is_in_door_region(bbox, self.camera_configs[camera_id].door_coords):
                    self.tracks[camera_id][track_id].append({
                        'timestamp': timestamp,
                        'bbox': bbox
                    })
                    self._update_track_features(track_id, features, camera_id)
                    
            # Clean up lost tracks
            finished_tracks = set(active_tracks.keys()) - set(current_tracks.keys())
            for track_id in finished_tracks:
                if len(active_tracks[track_id]) >= min_track_length:
                    # Smooth trajectory
                    smoothed_trajectory = self._smooth_trajectory(np.array(active_tracks[track_id]))
                    
                    # Store final track data
                    self.tracks[camera_id][track_id].extend([
                        {
                            'timestamp': ts,
                            'bbox': pos.tolist(),
                            'features': feat
                        }
                        for ts, pos, feat in zip(
                            track_timestamps[track_id],
                            smoothed_trajectory,
                            track_features[track_id]
                        )
                    ])
                
                # Clean up tracking buffers
                del active_tracks[track_id]
                del track_features[track_id]
                del track_timestamps[track_id]
            
            frame_count += 1
            
        cap.release()
        
        return frame_count
        
    def _process_tracks(self, tracks, frame, timestamp, camera_id, track_dict, feature_dict):
        """
        Process tracks for a single frame
        """
        for track in tracks:
            track_id = track.track_id
            bbox = track.tlbr
            
            if self._is_in_door_region(bbox, self.camera_configs[camera_id].door_coords):
                # Extract features
                features = self._extract_features(frame, bbox)
                feature_dict[track_id].append(features)
                
                # Store track information
                track_dict[track_id].append({
                    'timestamp': timestamp,
                    'bbox': bbox.tolist()
                })
            
            # Run detection and tracking
            detections = self._detect_persons(frame)
            tracks = self.byte_tracker.update(
                detections,
                [frame.shape[0], frame.shape[1]],
                [frame.shape[0], frame.shape[1]]
            )
            
            # Process current tracks
            current_tracks = {}
            for track in tracks:
                track_id = track.track_id
                bbox = track.tlbr
                
                # Initialize or get track buffer
                if track_id not in self.track_buffers[camera_id]:
                    self.track_buffers[camera_id][track_id] = TrackBuffer()
                buffer = self.track_buffers[camera_id][track_id]
                
                # Extract and update features
                if self._is_in_door_region(bbox, self.camera_configs[camera_id].door_coords):
                    features = self._extract_features(frame, bbox)
                    buffer.add(bbox, features, timestamp)
                    self._update_track_features(track_id, features, camera_id)
                
                current_tracks[track_id] = True
                
                # Store track info
                self.tracks[camera_id][track_id].append({
                    'timestamp': timestamp,
                    'bbox': bbox.tolist()
                })
            
            # Process finished tracks
            finished_tracks = set(active_tracks.keys()) - set(current_tracks.keys())
            for track_id in finished_tracks:
                track_data = self.tracks[camera_id][track_id]
                
                if self._validate_track(track_data):
                    # Smooth trajectory
                    positions = np.array([d['bbox'][:2] for d in track_data])
                    smoothed = self._smooth_trajectory(positions)
                    
                    # Update track positions with smoothed trajectory
                    for i, pos in enumerate(smoothed):
                        track_data[i]['bbox'][:2] = pos.tolist()
                
                # Clean up buffers
                if track_id in self.track_buffers[camera_id]:
                    del self.track_buffers[camera_id][track_id]
            
            # Update active tracks
            active_tracks = current_tracks.copy()
            frame_count += 1
        
        cap.release()
        
        # Final cleanup and validation
        self._filter_false_detections(self.tracks)
        
    def analyze_transitions(self, videos_dir):
        """
        Analyze all videos in directory and compute transitions
        """
        # Sort videos by date
        videos = sorted(Path(videos_dir).glob('Camera_*_*.mp4'))
        results = []
        
        # Group videos by date
        video_groups = defaultdict(dict)
        for video in videos:
            parts = video.stem.split('_')
            camera_id = int(parts[1])
            date = parts[2]
            video_groups[date][camera_id] = video
        
        # Process each date
        for date, cameras in video_groups.items():
            # Reset tracking state for new date
            self.tracks = {1: defaultdict(list), 2: defaultdict(list)}
            self.embeddings = {1: {}, 2: {}}
            self.track_buffers = {1: {}, 2: {}}
            
            # Process videos for each camera
            for camera_id, video_path in cameras.items():
                self.process_video(str(video_path), camera_id)
            
            # Compute transitions
            transitions = self.match_transitions()
            
            # Store results
            results.append({
                'date': date,
                'unique_camera1': len(self.tracks[1]),
                'unique_camera2': len(self.tracks[2]),
                'transitions': len(transitions)
            })
        
        return pd.DataFrame(results)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame
            frame = self._preprocess_frame(frame)
            
            # Run detection and tracking
            detections = self._detect_persons(frame)
            tracks = self.byte_tracker.update(
                detections,
                [frame.shape[0], frame.shape[1]],
                [frame.shape[0], frame.shape[1]]
            )
            
            # Filter short tracks
            if frame_count % min_track_length == 0:
                self.tracks[camera_id] = {
                    k: v for k, v in self.tracks[camera_id].items()
                    if len(v) >= min_track_length
                }
            
            # Process tracks
            current_tracks = {}
            timestamp = frame_count / self.camera_configs[camera_id].fps
            
            for track in tracks:
                track_id = track.track_id
                bbox = track.tlbr
                
                # Extract features for all tracked persons
                features = self._extract_features(frame, bbox)
                
                # Update track information
                if track_id not in active_tracks:
                    active_tracks[track_id] = []
                    track_features[track_id] = []
                    track_timestamps[track_id] = []
                
                active_tracks[track_id].append(bbox)
                track_features[track_id].append(features)
                track_timestamps[track_id].append(timestamp)
                current_tracks[track_id] = True
                
                # Store track information if near door
                if self._is_in_door_region(bbox, self.camera_configs[camera_id].door_coords):
                    self.tracks[camera_id][track_id].append({
                        'timestamp': timestamp,
                        'bbox': bbox
                    })
                    self._update_track_features(track_id, features, camera_id)
                    
            # Clean up lost tracks
            finished_tracks = set(active_tracks.keys()) - set(current_tracks.keys())
            for track_id in finished_tracks:
                if len(active_tracks[track_id]) >= min_track_length:
                    # Smooth trajectory
                    smoothed_trajectory = self._smooth_trajectory(np.array(active_tracks[track_id]))
                    
                    # Store final track data
                    self.tracks[camera_id][track_id].extend([
                        {
                            'timestamp': ts,
                            'bbox': pos.tolist(),
                            'features': feat
                        }
                        for ts, pos, feat in zip(
                            track_timestamps[track_id],
                            smoothed_trajectory,
                            track_features[track_id]
                        )
                    ])
                
                # Clean up tracking buffers
                del active_tracks[track_id]
                del track_features[track_id]
                del track_timestamps[track_id]
            
            frame_count += 1
            
        cap.release()
        
    def match_transitions(self, time_window=(2*60, 10*60)):  # 2-10 minutes window
        """
        Match tracks between cameras based on appearance and temporal constraints
        """
        transitions = []
        
        # Get all tracks with valid exit events from Camera 1
        camera1_exits = self._compute_door_transitions(self.tracks[1], 1)
        camera2_entries = self._compute_door_transitions(self.tracks[2], 2)
        
        # Filter for exits from Camera 1
        exit_tracks = {
            track_id: track_data 
            for track_id, trans_type in camera1_exits.items()
            if trans_type == 'exit' and self._validate_track(self.tracks[1][track_id])
        }
        
        # Filter for entries to Camera 2
        entry_tracks = {
            track_id: track_data
            for track_id, trans_type in camera2_entries.items()
            if trans_type == 'entry' and self._validate_track(self.tracks[2][track_id])
        }
        
        # Match tracks between cameras
        for exit_id, exit_data in exit_tracks.items():
            exit_time = exit_data[-1]['timestamp']
            exit_features = self.embeddings[1][exit_id]
            exit_direction = self._estimate_track_direction(self.tracks[1][exit_id])
            
            best_match = None
            best_score = 0
            
            for entry_id, entry_data in entry_tracks.items():
                entry_time = entry_data[0]['timestamp']
                time_diff = entry_time - exit_time
                
                # Check temporal constraint
                if not (time_window[0] <= time_diff <= time_window[1]):
                    continue
                    
                # Check appearance similarity
                entry_features = self.embeddings[2][entry_id]
                appearance_score = self._compute_similarity(exit_features, entry_features)
                
                # Check direction consistency
                direction_score = 1.0
                entry_direction = self._estimate_track_direction(self.tracks[2][entry_id])
                if exit_direction is not None and entry_direction is not None:
                    direction_sim = np.dot(exit_direction, entry_direction)
                    direction_score = (direction_sim + 1) / 2  # Map from [-1,1] to [0,1]
                
                # Compute final matching score
                score = 0.7 * appearance_score + 0.3 * direction_score
                
                if score > best_score and score > self.matching_params['min_similarity']:
                    best_score = score
                    best_match = (entry_id, score, time_diff)
            
            if best_match:
                transitions.append({
                    'camera1_id': exit_id,
                    'camera2_id': best_match[0],
                    'similarity_score': best_match[1],
                    'transition_time': best_match[2]
                })
        
        return transitions

def process_directory(input_dir, output_file):
    # Camera configurations
    camera_configs = {
        1: CameraConfig(1, [(1030, 0), (1700, 560)]),
        2: CameraConfig(2, [(400, 0), (800, 470)])
    }
    
    # Initialize tracker
    tracker = PersonTracker(camera_configs)
    
    # Process all videos
    results = []
    for video_file in sorted(Path(input_dir).glob('Camera_*_*.mp4')):
        # Parse video filename
        parts = video_file.stem.split('_')
        camera_id = int(parts[1])
        date = parts[2]
        
        # Process video
        tracker.process_video(str(video_file), camera_id)
        
        # Calculate statistics
        unique_camera1 = len(tracker.tracks[1])
        unique_camera2 = len(tracker.tracks[2])
        transitions = tracker.match_transitions()
        
        results.append({
            'date': date,
            'unique_camera1': unique_camera1,
            'unique_camera2': unique_camera2,
            'transitions': len(transitions)
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_dir = "path_to_video_directory"
    output_file = "tracking_results.csv"
    process_directory(input_dir, output_file)
