import os
import cv2
import numpy as np
from ultralytics import YOLO
import torchreid
import torch
from collections import defaultdict
import os
import time
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import scipy.spatial.distance as distance
from pathlib import Path
import shutil
import json
from torchvision import transforms
import filterpy.kalman
from filterpy.kalman import KalmanFilter
from collections import deque
from scipy.optimize import linear_sum_assignment

# Set the working directory
working_directory = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
                                 'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')

# Constants for tracking
KALMAN_AVAILABLE = True  # Since we're importing filterpy

def joint_tracks(tlista, tlistb):
    """Join two track lists"""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_tracks(tlista, tlistb):
    """Remove tracks in b from a"""
    tracks = {}
    for t in tlistb:
        tracks[t.track_id] = t
    
    res = []
    for t in tlista:
        if not tracks.get(t.track_id, None):
            res.append(t)
    return res

def remove_duplicate_tracks(tracks_a, tracks_b):
    """Remove duplicate tracks based on IOU"""
    pdist = iou_distance(tracks_a, tracks_b)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    
    for p, q in zip(*pairs):
        timep = tracks_a[p].frame_id - tracks_a[p].start_frame
        timeq = tracks_b[q].frame_id - tracks_b[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
            
    resa = [t for i, t in enumerate(tracks_a) if i not in dupa]
    resb = [t for i, t in enumerate(tracks_b) if i not in dupb]
    return resa, resb

def iou_distance(atracks, btracks):
    """Compute IOU distance matrix between a and b tracks"""
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
        (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = np.zeros((len(atlbrs), len(btlbrs)))
    
    if _ious.size > 0:
        for i, atlbr in enumerate(atlbrs):
            for j, btlbr in enumerate(btlbrs):
                _ious[i, j] = calculate_iou(atlbr, btlbr)
    
    # Convert IOU to distance
    dist = 1 - _ious
    return dist

def gate_cost_matrix(cost_matrix, tracks, detections, track_indices=None, 
                     detection_indices=None, track_buffer=30):
    """Gate cost matrix based on track age"""
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
        
    gated_cost = np.inf * np.ones_like(cost_matrix)
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = track_buffer
        for col, det_idx in enumerate(detection_indices):
            if cost_matrix[row, col] < gating_distance:
                gated_cost[row, col] = cost_matrix[row, col]
    
    return gated_cost

def calculate_iou(box1, box2):
    """Calculate IOU between two boxes"""
    # Convert boxes to correct format if needed
    if isinstance(box1, np.ndarray):
        box1 = box1.flatten()
    if isinstance(box2, np.ndarray):
        box2 = box2.flatten()
        
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / (union + 1e-6)

class TrackingState:
    ACTIVE = 'active'          # Fully visible
    OCCLUDED = 'occluded'      # Temporarily hidden
    TENTATIVE = 'tentative'    # New track
    LOST = 'lost'              # Missing too long

class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialize a tracker using initial bounding box
        """
        # Handle invalid bounding box
        z = convert_bbox_to_z(bbox)
        if z is None:
            raise ValueError("Invalid bounding box dimensions")
            
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],      # state transition matrix
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]])
        
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],      # measurement function
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.   # measurement noise
        self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01  # process noise
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = z
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_bbox = bbox

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        """Returns the current bounding box estimate as [x,y,w,h]."""
        ret = self.kf.x[:4].reshape(-1)
        return [ret.copy()]

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.last_bbox = bbox

def convert_bbox_to_z(bbox):
    """
    Convert bounding box to KF state vector with error checking
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    # Add checks for zero dimensions
    if w <= 0 or h <= 0:
        return None
        
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    # scale is area
    r = w / float(h + 1e-6)  # add small epsilon to prevent division by zero
    
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x):
    """
    Convert KF state vector to bounding box
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))

class STrack:
    _count = 0

    def __init__(self, tlwh, score, features):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.features = features
        self.is_activated = False
        
        # Initialize Kalman tracker
        self.kalman_tracker = KalmanBoxTracker(tlwh)
        
        self.tracklet_len = 0
        self.state = TrackState.NEW
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        
        # Use _next_id() to assign track ID
        self.track_id = self._next_id()
    
    @staticmethod
    def _next_id():
        STrack._count += 1
        return STrack._count - 1

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        return self._tlwh.copy()

    def set_tlwh(self, tlwh):
        """Safely update the bounding box"""
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
    
    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def update(self, new_track, frame_id):
        """Update track state"""
        self.frame_id = frame_id
        self.time_since_update = 0
        
        # Use set_tlwh instead of direct assignment
        self.set_tlwh(new_track.tlwh)
        self.features = new_track.features
        self.score = new_track.score
        
        if self.state == TrackState.NEW:
            self.state = TrackState.TRACKED
            self.is_activated = True
            
        self.tracklet_len += 1

    def reactivate(self, new_track, frame_id):
        """Reactivate a lost track"""
        # Use set_tlwh instead of direct assignment
        self.set_tlwh(new_track.tlwh)
        self.features = new_track.features
        self.score = new_track.score
        self.tracklet_len = 0
        self.frame_id = frame_id
        self.time_since_update = 0
        self.state = TrackState.TRACKED
        self.is_activated = True

    def mark_lost(self):
        """Mark track as lost"""
        self.state = TrackState.LOST

    def predict(self):
        """Predict next position using Kalman filter"""
        if self.kalman_tracker:
            self.kalman_tracker.predict()
            predicted_box = self.kalman_tracker.get_state()[0]
            self.set_tlwh(predicted_box)

class GlobalTracker:
    def __init__(self):
        self.global_identities = {}
        self.appearance_sequence = {}
        self.feature_database = {}
        self.camera_features = {'1': {}, '2': {}}
        self.last_seen_times = {}
        
        # Adjusted timing windows for transitions between cafe and food shop
        self.min_transition_time = 30   # minimum 30 seconds (quick walk between locations)
        self.max_transition_time = 300  # maximum 5 minutes (allowing for browsing)
        
        # High matching thresholds
        self.cross_camera_threshold = 0.85
        self.consecutive_matches_required = 3
        
        # Track consecutive matches
        self.match_history = {}

        self.reid_model = None  # ReID model for feature extraction
        self.feature_buffer = defaultdict(list)  # Buffer for feature history
        self.max_buffer_size = 10
        self.cross_camera_match_thresh = 0.85

    def register_camera_detection(self, camera_id, person_id, features, timestamp):
        """Register a detection from a specific camera"""
        if isinstance(person_id, STrack):
            person_id = person_id.track_id
            
        global_id = self._match_or_create_global_id(camera_id, person_id, features, timestamp)
        
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

    def _match_or_create_global_id(self, camera_id, person_id, features, timestamp):
        """Enhanced matching with feature history"""
        camera_key = f"{camera_id}_{person_id}"
        
        if camera_key in self.global_identities:
            return self.global_identities[camera_key]
            
        best_match = None
        best_score = 0
        
        if camera_id == '2':  # Only try cross-camera matching for Camera 2
            for global_id, stored_features in self.feature_database.items():
                if global_id in self.camera_features['1']:
                    time_diff = timestamp - self.last_seen_times.get(global_id, 0)
                    
                    if self.min_transition_time <= time_diff <= self.max_transition_time:
                        # Compare with feature history
                        feature_scores = []
                        for hist_features in self.feature_buffer[global_id]:
                            sim = 1 - distance.cosine(
                                features.flatten(), 
                                hist_features.flatten()
                            )
                            feature_scores.append(sim)
                            
                        # Use average of top 3 similarity scores
                        if feature_scores:
                            avg_score = np.mean(sorted(feature_scores)[-3:])
                            if avg_score > self.cross_camera_match_thresh and avg_score > best_score:
                                best_match = global_id
                                best_score = avg_score
        
        if best_match is None:
            best_match = len(self.global_identities)
            
        # Update feature history
        self.feature_buffer[best_match].append(features)
        if len(self.feature_buffer[best_match]) > self.max_buffer_size:
            self.feature_buffer[best_match].pop(0)
            
        self.global_identities[camera_key] = best_match
        self.feature_database[best_match] = features
        self.last_seen_times[best_match] = timestamp
        
        return best_match

    def register_camera_detection(self, camera_id, person_id, features, timestamp):
        """Register a detection from a specific camera"""
        global_id = self._match_or_create_global_id(camera_id, person_id, features, timestamp)
        
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

    def analyze_camera_transitions(self):
        """More conservative transition analysis"""
        cam1_to_cam2 = 0
        cam2_to_cam1 = 0
        
        # Track unique transitions
        valid_transitions = set()
        
        for global_id, appearances in self.appearance_sequence.items():
            if len(appearances) < 2:
                continue
                
            # Sort appearances by timestamp
            sorted_appearances = sorted(appearances, key=lambda x: x['timestamp'])
            
            # Analyze sequential appearances
            for i in range(len(sorted_appearances) - 1):
                current = sorted_appearances[i]
                next_app = sorted_appearances[i + 1]
                
                time_diff = next_app['timestamp'] - current['timestamp']
                
                if (self.min_transition_time <= time_diff <= self.max_transition_time and
                    (current['camera'], next_app['camera'], global_id) not in valid_transitions):
                    
                    if current['camera'] == 'Camera_1' and next_app['camera'] == 'Camera_2':
                        cam1_to_cam2 += 1
                        valid_transitions.add((current['camera'], next_app['camera'], global_id))
                    elif current['camera'] == 'Camera_2' and next_app['camera'] == 'Camera_1':
                        cam2_to_cam1 += 1
                        valid_transitions.add((current['camera'], next_app['camera'], global_id))
        
        return {
            'camera1_to_camera2': cam1_to_cam2,
            'camera2_to_camera1': cam2_to_cam1,
            'total_unique_individuals': len(set(self.global_identities.values()))
        }

# Modify the main function to use GlobalTracker
def process_video_directory(input_dir, output_base_dir="tracking_results"):
    """Process all videos in a directory"""
    global_tracker = GlobalTracker()
    results = {}
    per_camera_stats = defaultdict(int)  # Track per-camera unique individuals

    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(input_dir).glob(f'*{ext}')))

    # Process each video
    for video_path in video_files:
        print(f"\nProcessing video: {video_path}")
        camera_id = video_path.stem.split('_')[1]  # Extract camera number from filename

        try:
            # Initialize tracker for this video
            tracker = PersonTracker(str(video_path), output_base_dir)

            # Process video
            tracker.process_video()

            # Save results
            video_results = tracker.save_tracking_results()
            results[video_path.stem] = video_results

            # Update per-camera statistics
            total_persons = len(tracker.person_timestamps)
            per_camera_stats[f"Camera_{camera_id}"] = total_persons

            # Register detections with global tracker
            for person_id, features in tracker.person_features.items():
                timestamp = tracker.person_timestamps[person_id]['first_appearance']
                global_tracker.register_camera_detection(camera_id, person_id, features, timestamp)

            print(f"Completed processing {video_path}")
            print(f"Found {video_results['total_persons']} unique persons")

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            continue

    # Get cross-camera analysis
    transition_analysis = global_tracker.analyze_camera_transitions()
    
    # Create comprehensive summary
    summary = {
        'per_camera_statistics': dict(per_camera_stats),
        'cross_camera_analysis': transition_analysis,
        'total_unique_global': len(global_tracker.global_identities)
    }

    # Save summary of all videos
    summary_path = os.path.join(output_base_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    # Print comprehensive summary
    print("\nProcessing Summary:")
    print("\nPer Camera Statistics:")
    for camera, count in per_camera_stats.items():
        print(f"{camera}: {count} unique individuals")
        
    print("\nCross-Camera Transitions:")
    print(f"Camera 1 to Camera 2: {transition_analysis['camera1_to_camera2']} individuals")
    print(f"Camera 2 to Camera 1: {transition_analysis['camera2_to_camera1']} individuals")
    print(f"\nTotal Unique Global Individuals: {summary['total_unique_global']}")

    return summary

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
        self.reid_model = self.reid_model.cuda(
        ) if torch.cuda.is_available() else self.reid_model
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
        self.appearance_history = defaultdict(list)  # Initialize appearance history

        # Add new attributes for reentry handling
        self.lost_tracks = {}  # Store tracks that have disappeared
        self.max_lost_age = self.fps * 30  # Max time to remember lost tracks (30 seconds)
        self.appearance_history = defaultdict(list)  # Store feature history
        self.max_history_length = 10  # Number of recent features to keep
        self.reentry_threshold = 0.75  # Minimum similarity for reentry matching

        # Tracking parameters
        self.max_disappeared = self.fps * 2  # Max frames to keep track without detection
        # Extract camera ID from video path
        self.camera_id = Path(video_path).stem.split('_')[1]
        
        # Significantly different parameters for cafe vs food shop
        if self.camera_id == '1':  # Cafe
            self.min_detection_confidence = 0.85
            self.similarity_threshold = 0.9
            self.feature_weight = 0.7
            self.position_weight = 0.2
            self.motion_weight = 0.1
            # Allow for much longer disappearance in cafe (up to 30 minutes)
            self.max_disappeared = self.fps * 60 * 30  # 30 minutes at 6fps
            # Extended lost track memory for cafe
            self.max_lost_age = self.fps * 60 * 45  # 45 minutes
        else:  # Food shop
            self.min_detection_confidence = 0.8
            self.similarity_threshold = 0.85
            self.feature_weight = 0.6
            self.position_weight = 0.25
            self.motion_weight = 0.15
            # Shorter but still significant disappearance allowance for food shop
            self.max_disappeared = self.fps * 60 * 5  # 5 minutes at 6fps
            # Lost track memory for food shop
            self.max_lost_age = self.fps * 60 * 7  # 7 minutes

        # Add additional parameters for handling long-term disappearances
        self.appearance_confidence = {}  # Track confidence in identifications
        self.min_reappearance_confidence = 0.95  # Higher confidence needed for long disappearances

        # ByteTrack-specific parameters
        self.track_thresh = 0.5
        self.high_thresh = 0.6
        self.match_thresh = 0.8
        self.track_buffer = 90
        
        # Track states and buffers
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        
        # Initialize Kalman filter tracking
        self.kalman_trackers = []
        self.track_id_count = 0

    def multi_predict(self, tracks):
        """Predict next positions for multiple tracks"""
        for track in tracks:
            if hasattr(track, 'kalman_tracker'):
                track.kalman_tracker.predict()

    def extract_features(self, person_crop):
        """Enhanced feature extraction with better preprocessing"""
        try:
            # Enhanced preprocessing
            person_crop = cv2.resize(person_crop, (128, 256))
            
            # Apply contrast enhancement
            lab = cv2.cvtColor(person_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Normalize
            img = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert to tensor
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1).unsqueeze(0)
            
            # Normalize with ImageNet stats
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            img = normalize(img/255.0)
            
            if torch.cuda.is_available():
                img = img.cuda()
                
            # Extract features
            with torch.no_grad():
                features = self.reid_model(img)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                
            return features.cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def calculate_box_center(self, box):
        """Calculate center point of a bounding box"""
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    
    def filter_detection(self, box, conf, frame_shape):
        """Much stricter filtering of detections"""
        height, width = frame_shape[:2]
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        
        # Stricter size constraints for both cameras
        if self.camera_id == '1':
            # Minimum size thresholds
            min_width_ratio = 0.05
            min_height_ratio = 0.15
            # Maximum size thresholds
            max_width_ratio = 0.3
            max_height_ratio = 0.8
        else:
            # Slightly different thresholds for Camera 2
            min_width_ratio = 0.04
            min_height_ratio = 0.12
            max_width_ratio = 0.35
            max_height_ratio = 0.85
        
        # Check size constraints
        width_ratio = box_width / width
        height_ratio = box_height / height
        
        if (width_ratio < min_width_ratio or 
            width_ratio > max_width_ratio or 
            height_ratio < min_height_ratio or 
            height_ratio > max_height_ratio):
            return False
        
        # Check aspect ratio (height should be greater than width for standing people)
        aspect_ratio = box_height / box_width
        if aspect_ratio < 1.5 or aspect_ratio > 4.0:  # Typical human proportions
            return False
            
        return True

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
        """Enhanced similarity calculation with better motion modeling"""
        n_detections = len(current_features)
        n_tracks = len(tracked_features)

        if n_detections == 0 or n_tracks == 0:
            return np.array([])

        # Calculate appearance similarity with cosine distance
        appearance_sim = 1 - distance.cdist(
            np.array([f.flatten() for f in current_features]),
            np.array([f.flatten() for f in tracked_features]),
            metric='cosine'
        )

        # Enhanced motion-based similarity
        motion_sim = np.zeros((n_detections, n_tracks))
        for i, current_box in enumerate(current_boxes):
            current_center = self.calculate_box_center(current_box)
            
            for j, (tracked_box, track_id) in enumerate(zip(tracked_boxes, list(self.active_tracks.keys())[:n_tracks])):
                if 'velocity' in self.active_tracks[track_id]:
                    predicted_box = self.predict_next_position(
                        tracked_box,
                        self.active_tracks[track_id]['velocity']
                    )
                    predicted_center = self.calculate_box_center(predicted_box)
                    
                    # Calculate distance between prediction and actual position
                    distance = np.sqrt(
                        (current_center[0] - predicted_center[0])**2 +
                        (current_center[1] - predicted_center[1])**2
                    )
                    # Convert distance to similarity score
                    motion_sim[i, j] = np.exp(-distance / 100.0)

        # Enhanced position similarity using GIoU
        position_sim = np.zeros((n_detections, n_tracks))
        for i, box1 in enumerate(current_boxes):
            for j, box2 in enumerate(tracked_boxes):
                position_sim[i, j] = self.calculate_giou(box1, box2)

        # Combine similarities with dynamic weighting
        similarity_matrix = (
            self.feature_weight * appearance_sim +
            self.position_weight * position_sim +
            self.motion_weight * motion_sim
        )

        return similarity_matrix

    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate IOU between two boxes with error checking"""
        try:
            # Convert boxes to correct format if needed
            if isinstance(box1, np.ndarray):
                box1 = box1.flatten()
            if isinstance(box2, np.ndarray):
                box2 = box2.flatten()
                
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = max(0.1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
            box2_area = max(0.1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
            union = box1_area + box2_area - intersection
            
            return intersection / (union + 1e-6)  # add small epsilon to prevent division by zero
            
        except Exception as e:
            print(f"Error calculating IOU: {str(e)}")
            return 0.0

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
        """Enhanced save_tracking_results with reentry information"""
        results = {
            'video_name': self.video_name,
            'total_persons': self.next_id,
            'video_metadata': {
                'width': self.frame_width,
                'height': self.frame_height,
                'fps': self.fps
            },
            'persons': {}
        }

        for person_id in self.person_timestamps.keys():
            # Calculate reentry statistics
            appearances = []
            current_appearance = None
            
            # Sort all timestamps for this person
            all_times = sorted([
                (t, 'start' if k == 'first_appearance' else 'end')
                for t, k in self.person_timestamps[person_id].items()
            ])
            
            for time, event_type in all_times:
                if event_type == 'start':
                    current_appearance = {'start': time}
                else:
                    if current_appearance:
                        current_appearance['end'] = time
                        appearances.append(current_appearance)
                        current_appearance = None

            results['persons'][person_id] = {
                'first_appearance': self.person_timestamps[person_id]['first_appearance'],
                'last_appearance': self.person_timestamps[person_id]['last_appearance'],
                'total_duration': sum(app['end'] - app['start'] for app in appearances),
                'appearances': appearances,
                'reentry_count': len(appearances) - 1 if appearances else 0,
                'images_dir': os.path.join(self.images_dir, f"person_{person_id}"),
                'features_path': os.path.join(self.features_dir, f"person_{person_id}_features.npz")
            }

        # Save results to JSON
        results_path = os.path.join(self.output_dir, "tracking_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        return results

    def _process_detections(self, detections, frame):
        """Process and split detections into high and low confidence with error handling"""
        high_dets = []
        low_dets = []
        
        for box, conf in detections:
            try:
                if not self.filter_detection(box, conf, frame.shape):
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                # Add checks for valid box dimensions
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                    
                features = self.extract_features(person_crop)
                if features is not None:
                    # Convert to TLWH format with width and height
                    tlwh = [x1, y1, x2-x1, y2-y1]
                    
                    # Add checks for valid dimensions
                    if tlwh[2] <= 0 or tlwh[3] <= 0:
                        continue
                        
                    if conf >= self.high_thresh:
                        try:
                            track = STrack(tlwh, conf, features)
                            high_dets.append(track)
                        except ValueError:
                            continue
                    elif conf >= self.track_thresh:
                        try:
                            track = STrack(tlwh, conf, features)
                            low_dets.append(track)
                        except ValueError:
                            continue
                            
            except Exception as e:
                print(f"Error processing detection: {str(e)}")
                continue
        
        return high_dets, low_dets
    
    def _match_high_confidence(self, tracks, detections):
        """Match tracks with high confidence detections"""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
                
        cost_matrix = self._get_cost_matrix(tracks, detections)
        cost_matrix = gate_cost_matrix(
            cost_matrix, tracks, detections, track_buffer=self.track_buffer)
                
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        matched_indices = list(zip(row_ind, col_ind))
        
        # Get unmatched tracks and detections
        unmatched_tracks = list(set(range(len(tracks))) - set(row_ind))
        unmatched_detections = list(set(range(len(detections))) - set(col_ind))
        
        # Filter matches with low similarity
        matches = []
        for row, col in matched_indices:
            if cost_matrix[row, col] > self.match_thresh:
                unmatched_tracks.append(row)
                unmatched_detections.append(col)
            else:
                matches.append((row, col))
        
        return matches, unmatched_tracks, unmatched_detections
        
    def _get_cost_matrix(self, tracks, detections):
        """Calculate cost matrix between tracks and detections"""
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Appearance similarity
                reid_sim = 1 - distance.cosine(
                    track.features.flatten(),
                    det.features.flatten()
                )
                # Motion similarity
                iou_sim = self.calculate_iou(track.tlbr, det.tlbr)
                # Combined similarity
                cost_matrix[i, j] = -(
                    self.feature_weight * reid_sim +
                    (1 - self.feature_weight) * iou_sim
                )
                
        return cost_matrix

    def update_tracks(self, frame, detections, frame_time):
        """ByteTrack-inspired track updating"""
        self.frame_id += 1
        activated_tracks = []
        refined_tracks = []
        lost_tracks = []
        removed_tracks = []
        
        # Get detections
        high_dets, low_dets = self._process_detections(detections, frame)
        
        # First association with high score detections
        track_pool = joint_tracks(self.tracked_tracks, self.lost_tracks)
        self.multi_predict(track_pool)
        
        # Match with high confidence detections - now using only three return values
        matches, unmatched_tracks, unmatched_detections = self._match_high_confidence(
            track_pool, high_dets)
            
        # Process matches
        for track_idx, det_idx in matches:
            track = track_pool[track_idx]
            det = high_dets[det_idx]
            
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated_tracks.append(track)
            else:
                track.reactivate(det, self.frame_id)
                refined_tracks.append(track)

        # Second association with low score detections
        if len(unmatched_tracks) > 0 and len(low_dets) > 0:
            # Match low confidence detections
            matches_low, unmatched_tracks_low, _ = self._match_high_confidence(
                [track_pool[i] for i in unmatched_tracks], 
                low_dets)
            
            for track_idx, det_idx in matches_low:
                track = track_pool[unmatched_tracks[track_idx]]
                det = low_dets[det_idx]
                
                if track.state == TrackState.TRACKED:
                    track.update(det, self.frame_id)
                    activated_tracks.append(track)
                else:
                    track.reactivate(det, self.frame_id)
                    refined_tracks.append(track)
            
            unmatched_tracks = unmatched_tracks_low

        # Update lost tracks
        for track_idx in unmatched_tracks:
            track = track_pool[track_idx]
            if track.state == TrackState.TRACKED:
                track.mark_lost()
                lost_tracks.append(track)
        
        # Update state
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.TRACKED]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_tracks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refined_tracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_tracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks.extend(removed_tracks)
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks)
        
    def _is_near_predicted_track(self, box):
        """Check if detection is near any predicted track location"""
        for track_id, track_info in self.active_tracks.items():
            if 'velocity' in track_info:
                predicted_box = self.predict_next_position(
                    track_info['box'], track_info['velocity'])
                iou = self.calculate_iou(box, predicted_box)
                if iou > 0.3:  # Threshold for "near"
                    return True
        return False

    def _match_detections(self, boxes, features, scores, frame_time, frame):
        """Enhanced detection matching with Kalman filtering"""
        matched_track_ids = set()
        
        if not boxes:
            return matched_track_ids
            
        # Get active track information
        tracked_boxes = []
        tracked_features = []
        tracked_ids = []
        
        for track_id, track_info in self.active_tracks.items():
            # Predict new location using Kalman filter
            predicted_box = self.predict_next_position(
                track_info['box'], 
                track_info.get('velocity', [0, 0])
            )
            tracked_boxes.append(predicted_box)
            tracked_features.append(track_info['features'])
            tracked_ids.append(track_id)
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(
            features, boxes, tracked_features, tracked_boxes)
        
        # Perform matching
        if similarity_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
            
            for detection_idx, track_idx in zip(row_ind, col_ind):
                if similarity_matrix[detection_idx, track_idx] >= self.similarity_threshold:
                    track_id = tracked_ids[track_idx]
                    matched_track_ids.add(track_id)
                    
                    # Update track with new detection
                    self.update_existing_track(
                        track_id,
                        boxes[detection_idx],
                        features[detection_idx],
                        frame_time,
                        frame
                    )
        
        # Handle unmatched detections
        for i in range(len(boxes)):
            if i not in row_ind:
                # Try to recover lost track first
                recovered_id = self.recover_lost_tracklet(
                    features[i],
                    boxes[i],
                    frame_time
                )
                
                if recovered_id is not None:
                    self.reactivate_track(
                        recovered_id,
                        boxes[i],
                        features[i],
                        frame_time,
                        frame
                    )
                    matched_track_ids.add(recovered_id)
                else:
                    # Create new track only for high confidence detections
                    if scores[i] >= self.min_detection_confidence:
                        self.create_new_track(
                            boxes[i],
                            features[i],
                            frame_time,
                            frame
                        )
        
        return matched_track_ids

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
            
            # Save tracking information - Add this part
            for track in self.tracked_tracks:
                if track.is_activated:
                    # Update person timestamps
                    track_id = track.track_id
                    if track_id not in self.person_timestamps:
                        self.person_timestamps[track_id] = {
                            'first_appearance': frame_time,
                            'last_appearance': frame_time
                        }
                    else:
                        self.person_timestamps[track_id]['last_appearance'] = frame_time
                    
                    # Update person features
                    if track_id not in self.person_features:
                        self.person_features[track_id] = track.features
                    
                    # Save person image
                    person_img = frame[int(track.tlwh[1]):int(track.tlwh[1]+track.tlwh[3]), 
                                    int(track.tlwh[0]):int(track.tlwh[0]+track.tlwh[2])]
                    self.save_person_image(track_id, person_img)

            # Visualize results
            for track in self.tracked_tracks:
                if track.is_activated:
                    box = track.tlbr
                    cv2.rectangle(frame, (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track.track_id}",
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
    summary = process_video_directory(working_directory)

    # Print summary (already handled in process_video_directory)
    print("\nResults saved successfully!")

if __name__ == '__main__':
    main()
