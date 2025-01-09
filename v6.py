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

# Set the working directory
working_directory = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
                                 'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')

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
        # Different thresholds for different cameras
        self.similarity_thresholds = {
            '1': 0.85,  # More strict for complex environment
            '2': 0.75   # Less strict for cleaner environment
        }
        # Store camera-specific feature histories
        self.camera_features = {'1': {}, '2': {}}

    def register_camera_detection(self, camera_id, person_id, features, timestamp):
        """Register a detection from a specific camera"""
        global_id = self._match_or_create_global_id(camera_id, person_id, features)
        
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

    def _match_or_create_global_id(self, camera_id, person_id, features):
        """Enhanced matching with camera-specific handling"""
        camera_key = f"{camera_id}_{person_id}"
        
        if camera_key in self.global_identities:
            return self.global_identities[camera_key]
            
        # Get camera-specific threshold
        threshold = self.similarity_thresholds[camera_id]
        
        best_match = None
        best_score = 0
        
        # Compare against features from both cameras
        for global_id, stored_features in self.feature_database.items():
            # Calculate similarity with temporal weighting
            base_similarity = 1 - distance.cosine(features.flatten(), stored_features.flatten())
            
            # Apply additional checks for cross-camera matching
            if camera_id == '2':  # If current detection is in Camera 2
                if global_id in self.camera_features['1']:  # Check if person was seen in Camera 1
                    time_diff = time.time() - self.camera_features['1'][global_id]['timestamp']
                    # Adjust similarity based on reasonable transition time (e.g., few minutes walk)
                    if 60 <= time_diff <= 300:  # 1-5 minutes transition window
                        base_similarity *= 1.2  # Boost similarity for reasonable transition times
                    else:
                        base_similarity *= 0.8  # Reduce similarity for unlikely transition times
            
            if base_similarity > threshold and base_similarity > best_score:
                best_match = global_id
                best_score = base_similarity
        
        if best_match is None:
            best_match = len(self.global_identities)
            self.feature_database[best_match] = features
            
        self.global_identities[camera_key] = best_match
        
        # Store camera-specific features
        self.camera_features[camera_id][best_match] = {
            'features': features,
            'timestamp': time.time()
        }
        
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
            name='osnet_x1_0',
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
        
        # Camera-specific parameters
        if self.camera_id == '1':
            self.min_detection_confidence = 0.7  # Higher threshold for noisy environment
            self.similarity_threshold = 0.8
            self.feature_weight = 0.6    # More weight on appearance
            self.position_weight = 0.2    # Less weight on position
            self.motion_weight = 0.2      # Less weight on motion
        else:  # Camera 2
            self.min_detection_confidence = 0.5  # Lower threshold for cleaner environment
            self.similarity_threshold = 0.7
            self.feature_weight = 0.5
            self.position_weight = 0.25
            self.motion_weight = 0.25

    def extract_features(self, person_crop):
        """Enhanced feature extraction with preprocessing"""
        try:
            # Enhanced preprocessing for different camera environments
            if self.camera_id == '1':
                # Apply additional preprocessing for noisy environment
                person_crop = cv2.GaussianBlur(person_crop, (3, 3), 0)
                person_crop = cv2.equalizeHist(cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY))
                person_crop = cv2.cvtColor(person_crop, cv2.COLOR_GRAY2BGR)
            
            # Normalize image
            img = cv2.resize(person_crop, (128, 256))
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert to tensor and extract features
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1).unsqueeze(0)
            if torch.cuda.is_available():
                img = img.cuda()
                
            with torch.no_grad():
                features = self.reid_model(img)
                
            # Apply feature normalization
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features.cpu().numpy()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def calculate_box_center(self, box):
        """Calculate center point of a bounding box"""
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    
    def filter_detection(self, box, conf, frame_shape):
        """Filter out likely false detections"""
        height, width = frame_shape[:2]
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        
        # Filter based on camera-specific criteria
        if self.camera_id == '1':
            # More strict filtering for noisy environment
            if (box_width < width * 0.02 or  # Too small
                box_width > width * 0.5 or   # Too large
                box_height < height * 0.1 or # Too short
                box_height > height * 0.9):  # Too tall
                return False
        else:
            # Less strict filtering for Camera 2
            if (box_width < width * 0.01 or
                box_width > width * 0.6 or
                box_height < height * 0.05 or
                box_height > height * 0.95):
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

    def update_tracks(self, frame, detections, frame_time):
        """Update tracks with new detections, handling reentries"""
        current_boxes = []
        current_features = []

        # Process new detections
        for box, conf in detections:
            # Add filter_detection check here
            if not self.filter_detection(box, conf, frame.shape):
                continue
                
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

        # Match with active tracks first
        tracked_boxes = []
        tracked_features = []
        tracked_ids = []

        for track_id, track_info in self.active_tracks.items():
            tracked_boxes.append(track_info['box'])
            tracked_features.append(track_info['features'])
            tracked_ids.append(track_id)

        # Calculate similarity matrix for active tracks
        similarity_matrix = self.calculate_similarity_matrix(
            current_features, current_boxes,
            tracked_features, tracked_boxes
        )

        # Perform matching with active tracks
        matched_indices = []
        if similarity_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
            matched_indices = list(zip(row_ind, col_ind))

        # Process matches and handle unmatched detections
        matched_detections = set()
        matched_track_ids = set()

        for detection_idx, track_idx in matched_indices:
            similarity = similarity_matrix[detection_idx, track_idx]
            if similarity >= self.similarity_threshold:
                track_id = tracked_ids[track_idx]
                matched_track_ids.add(track_id)
                matched_detections.add(detection_idx)

                # Update existing track
                self.update_existing_track(
                    track_id, 
                    current_boxes[detection_idx],
                    current_features[detection_idx],
                    frame_time,
                    frame
                )

        # Try to match remaining detections with lost tracks
        for detection_idx in range(len(current_features)):
            if detection_idx in matched_detections:
                continue

            # Try to recover lost track
            recovered_id = self.recover_lost_tracklet(
                current_features[detection_idx],
                current_boxes[detection_idx],
                frame_time
            )

            if recovered_id is not None:
                # Reactivate recovered track
                self.reactivate_track(
                    recovered_id,
                    current_boxes[detection_idx],
                    current_features[detection_idx],
                    frame_time,
                    frame
                )
                matched_detections.add(detection_idx)
            else:
                # Create new track
                self.create_new_track(
                    current_boxes[detection_idx],
                    current_features[detection_idx],
                    frame_time,
                    frame
                )

        # Update lost tracks
        self.update_lost_tracks(matched_track_ids, frame_time)

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
    summary = process_video_directory(working_directory)

    # Print summary (already handled in process_video_directory)
    print("\nResults saved successfully!")

if __name__ == '__main__':
    main()
