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
from bytetracker import BYTETracker

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
        self.similarity_threshold = 0.75

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
        """Match with existing identity or create new global ID"""
        camera_key = f"{camera_id}_{person_id}"
        
        # Check if we've seen this camera-specific ID before
        if camera_key in self.global_identities:
            return self.global_identities[camera_key]
        
        # Try to match with existing identities
        best_match = None
        best_score = 0
        
        for global_id, stored_features in self.feature_database.items():
            similarity = 1 - distance.cosine(features.flatten(), stored_features.flatten())
            if similarity > self.similarity_threshold and similarity > best_score:
                best_match = global_id
                best_score = similarity
        
        if best_match is None:
            # Create new global identity
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
        self.similarity_threshold = 0.7
        self.max_disappeared = self.fps * 2  # Max frames to keep track without detection
        self.min_detection_confidence = 0.5
        self.feature_weight = 0.55   # Weight for ReID features in matching
        self.position_weight = 0.25  # Weight for absolute position (IoU)
        self.motion_weight = 0.3    # Weight for relative motion prediction

        super().__init__()
        
        # Initialize ByteTrack with correct parameters
        self.byte_tracker = BYTETracker(
            track_thresh=0.45,     # Detection confidence threshold
            track_buffer=25,       # Track buffer size (about 4 seconds at 6fps)
            match_thresh=0.8,      # Matching threshold for track association
            min_box_area=100,      # Minimum box area
            frame_rate=6           # Frame rate
        )
        
        # Enhanced tracking parameters
        self.tracking_params = {
            'similarity_threshold': 0.8,    # Increased from 0.7
            'max_disappeared': self.fps * 4,  # About 4 seconds at 6fps
            'min_detection_confidence': 0.6,  # Increased from 0.5
            'feature_weight': 0.6,    # Increased appearance weight
            'position_weight': 0.2,   # Reduced position weight
            'motion_weight': 0.2,     # Reduced motion weight
            'min_track_length': self.fps * 2,  # Minimum frames to confirm track
            'min_confidence_avg': 0.65,  # Minimum average confidence
            'max_velocity': 50,  # Maximum allowed velocity between frames
            'min_area': 1000,    # Minimum detection area
            'max_area': 50000,   # Maximum detection area
            'aspect_ratio_thresh': 1.6,  # Maximum height/width ratio
            'iou_threshold': 0.3,  # Minimum IoU for track association
        }
        
        # Add new tracking components
        self.confidence_history = defaultdict(list)
        self.feature_history = defaultdict(list)
        self.velocity_history = defaultdict(list)
        self.occlusion_status = defaultdict(bool)
        self.track_quality_scores = defaultdict(float)
        self.track_id_mapping = {}  # Map ByteTrack IDs to our track IDs

    def filter_detections(self, detections, frame):
        """Enhanced detection filtering"""
        filtered_dets = []
        
        for box, conf in detections:
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = height / width if width > 0 else float('inf')
            
            # Apply comprehensive filtering
            if (conf >= self.tracking_params['min_detection_confidence'] and
                self.tracking_params['min_area'] <= area <= self.tracking_params['max_area'] and
                aspect_ratio <= self.tracking_params['aspect_ratio_thresh']):
                
                # Check edge cases
                if (x1 > 0 and y1 > 0 and 
                    x2 < frame.shape[1] and 
                    y2 < frame.shape[0]):
                    
                    # Calculate detection quality score
                    quality_score = self.calculate_detection_quality(
                        frame[y1:y2, x1:x2], 
                        conf, 
                        area,
                        aspect_ratio
                    )
                    
                    if quality_score > 0.5:  # Minimum quality threshold
                        filtered_dets.append((box, conf, quality_score))
        
        return filtered_dets

    def calculate_detection_quality(self, crop, conf, area, aspect_ratio):
        """Calculate comprehensive detection quality score"""
        # Base score from confidence
        quality_score = conf
        
        # Adjust based on position in frame
        relative_area = area / (self.frame_width * self.frame_height)
        if relative_area < 0.01 or relative_area > 0.5:
            quality_score *= 0.8
            
        # Penalize extreme aspect ratios
        if aspect_ratio > 1.2:
            quality_score *= (1.2 / aspect_ratio)
            
        # Add image quality metrics
        try:
            # Calculate image sharpness
            laplacian = cv2.Laplacian(crop, cv2.CV_64F).var()
            sharpness_score = min(laplacian / 500.0, 1.0)
            
            # Calculate contrast
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            contrast = gray.std()
            contrast_score = min(contrast / 127.0, 1.0)
            
            # Combine scores
            quality_score *= (0.4 + 0.3 * sharpness_score + 0.3 * contrast_score)
        except:
            pass
            
        return quality_score

    def update_track_quality(self, track_id, detection_quality, features):
        """Update track quality metrics"""
        # Update confidence history
        self.confidence_history[track_id].append(detection_quality)
        if len(self.confidence_history[track_id]) > 30:  # 5 seconds at 6fps
            self.confidence_history[track_id].pop(0)
            
        # Update feature history
        self.feature_history[track_id].append(features)
        if len(self.feature_history[track_id]) > 10:
            self.feature_history[track_id].pop(0)
            
        # Calculate feature consistency
        if len(self.feature_history[track_id]) > 1:
            feature_dists = []
            for i in range(len(self.feature_history[track_id])-1):
                dist = distance.cosine(
                    self.feature_history[track_id][i].flatten(),
                    self.feature_history[track_id][i+1].flatten()
                )
                feature_dists.append(1 - dist)
            feature_consistency = np.mean(feature_dists)
        else:
            feature_consistency = 1.0
            
        # Update track quality score
        conf_avg = np.mean(self.confidence_history[track_id])
        self.track_quality_scores[track_id] = (
            0.4 * conf_avg +
            0.4 * feature_consistency +
            0.2 * (1.0 if not self.occlusion_status[track_id] else 0.5)
        )

    def update_tracks(self, frame, detections, frame_time):
        """Enhanced track updating with ByteTrack integration"""
        # Filter detections
        filtered_dets = self.filter_detections(detections, frame)
        
        # Convert to ByteTrack format - Update the format to include scores
        byte_dets = np.array([
            [*map(int, box), conf, 0]  # [x1, y1, x2, y2, score, class_id]
            for box, conf, _ in filtered_dets
        ])
        
        if len(byte_dets) > 0:
            # Run ByteTrack update with correct dimension format
            online_targets = self.byte_tracker.update(
                byte_dets,
                [self.frame_height, self.frame_width],
                [self.frame_height, self.frame_width]
            )
            
            # Process ByteTrack results with enhanced matching
            current_tracks = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                
                if self._is_valid_track(tlwh, t.score):
                    x1, y1, w, h = tlwh
                    box = [x1, y1, x1 + w, y1 + h]
                    
                    # Extract and verify features
                    features = self._extract_and_verify_features(frame, box)
                    if features is not None:
                        current_tracks.append({
                            'box': box,
                            'features': features,
                            'byte_id': tid,
                            'score': t.score
                        })
            
            # Update existing tracks
            matched_track_ids = self._update_existing_tracks(current_tracks, frame_time, frame)
            
            # Handle lost tracks
            self._handle_lost_tracks(frame_time)
            
            # Clean up tracking states
            self._cleanup_tracking_states()
        else:
            # Handle frame with no detections
            self._handle_lost_tracks(frame_time)
            self._cleanup_tracking_states()

    def _is_valid_track(self, tlwh, score):
        """Validate track based on geometry and score"""
        w, h = tlwh[2], tlwh[3]
        area = w * h
        aspect_ratio = h / w if w > 0 else float('inf')
        
        return (
            area >= self.tracking_params['min_area'] and
            area <= self.tracking_params['max_area'] and
            aspect_ratio <= self.tracking_params['aspect_ratio_thresh'] and
            score >= self.tracking_params['min_detection_confidence']
        )

    def _extract_and_verify_features(self, frame, box):
        """Extract and verify features with quality checks"""
        x1, y1, x2, y2 = map(int, box)
        if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
            return None
            
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None
            
        features = self.extract_features(person_crop)
        if features is None:
            return None
            
        # Verify feature quality
        feature_norm = np.linalg.norm(features)
        if feature_norm < 0.1 or feature_norm > 10:
            return None
            
        return features

    def _update_existing_tracks(self, current_tracks, frame_time, frame):
        """Update existing tracks with enhanced matching"""
        matched_track_ids = set()
        
        for track in current_tracks:
            byte_id = track['byte_id']
            matched_id = None
            
            # Try to match with existing track ID
            if byte_id in self.track_id_mapping:
                matched_id = self.track_id_mapping[byte_id]
                if self._verify_track_consistency(matched_id, track):
                    self.update_existing_track(
                        matched_id, track['box'], track['features'],
                        frame_time, frame
                    )
                else:
                    matched_id = None
            
            # Try to recover lost track if no match
            if matched_id is None:
                recovered_id = self.recover_lost_tracklet(
                    track['features'], track['box'], frame_time
                )
                if recovered_id is not None:
                    matched_id = recovered_id
                    self.track_id_mapping[byte_id] = recovered_id
                    self.reactivate_track(
                        recovered_id, track['box'], track['features'],
                        frame_time, frame
                    )
            
            # Create new track if no recovery
            if matched_id is None:
                new_id = self.create_new_track(
                    track['box'], track['features'], frame_time, frame
                )
                self.track_id_mapping[byte_id] = new_id
                matched_id = new_id
            
            if matched_id is not None:
                matched_track_ids.add(matched_id)
                self.update_track_quality(
                    matched_id, track['score'], track['features']
                )
        
        return matched_track_ids

    def _verify_track_consistency(self, track_id, new_track):
        """Verify track consistency using historical data"""
        if track_id not in self.active_tracks:
            return False
            
        track_info = self.active_tracks[track_id]
        
        # Check feature consistency
        if len(self.feature_history[track_id]) > 0:
            latest_features = self.feature_history[track_id][-1]
            feature_sim = 1 - distance.cosine(
                latest_features.flatten(),
                new_track['features'].flatten()
            )
            if feature_sim < 0.7:  # Minimum feature similarity threshold
                return False
        
        # Check motion consistency
        if 'velocity' in track_info:
            predicted_box = self.predict_next_position(
                track_info['box'],
                track_info['velocity']
            )
            iou = self.calculate_iou(predicted_box, new_track['box'])
            if iou < self.tracking_params['iou_threshold']:
                return False
        
        return True

    def _handle_lost_tracks(self, frame_time):
        """Enhanced lost track handling"""
        for track_id in list(self.active_tracks.keys()):
            track_info = self.active_tracks[track_id]
            
            # Update disappeared counter
            track_info['disappeared'] += 1
            
            # Check if track should be marked as lost
            if track_info['disappeared'] > self.tracking_params['max_disappeared']:
                # Only keep high quality tracks in lost tracks
                if self.track_quality_scores[track_id] > 0.7:
                    self.lost_tracks[track_id] = {
                        'features': track_info['features'],
                        'box': track_info['box'],
                        'velocity': track_info.get('velocity', [0, 0]),
                        'last_seen': track_info['last_seen'],
                        'quality_score': self.track_quality_scores[track_id],
                        'feature_history': self.feature_history[track_id]
                    }
                del self.active_tracks[track_id]
                
        # Clean up old lost tracks
        for track_id in list(self.lost_tracks.keys()):
            if frame_time - self.lost_tracks[track_id]['last_seen'] > self.tracking_params['max_disappeared']:
                del self.lost_tracks[track_id]

    def _cleanup_tracking_states(self):
        """Clean up tracking states and remove low quality tracks"""
        for track_id in list(self.active_tracks.keys()):
            # Remove tracks with consistently low quality
            if (len(self.confidence_history[track_id]) >= self.tracking_params['min_track_length'] and
                self.track_quality_scores[track_id] < 0.5):
                del self.active_tracks[track_id]
                if track_id in self.track_id_mapping:
                    del self.track_id_mapping[track_id]

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
