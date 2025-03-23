#!/usr/bin/env python3
"""
Cross-Camera People Tracker

Counts unique individuals in Camera 1, Camera 2, and tracks movements between cameras.
Optimized for NVIDIA RTX-4090 GPUs and Apple Silicon M1 Max.
"""

import os
import cv2
import time
import json
import torch
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import scipy.spatial.distance as distance
from ultralytics import YOLO
import torchreid

# Constants
CAMERA1_ID = "1"
CAMERA2_ID = "2"

class DeviceManager:
    """Handles device selection and optimization based on available hardware."""
    
    def __init__(self):
        self.device = self._select_device()
        self.platform = self._detect_platform()
        self.optimized = self._optimize_for_platform()
        
        print(f"Using device: {self.device}")
        print(f"Platform detected: {self.platform}")
        
    def _select_device(self):
        """Select the best available device for computation."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
            
    def _detect_platform(self):
        """Detect the platform (NVIDIA, Apple Silicon, or CPU)."""
        if self.device.type == 'cuda':
            return f"NVIDIA {torch.cuda.get_device_name(0)}"
        elif self.device.type == 'mps':
            return "Apple Silicon"
        else:
            return "CPU"
            
    def _optimize_for_platform(self):
        """Apply platform-specific optimizations."""
        if self.device.type == 'cuda':
            # NVIDIA GPU optimizations
            torch.backends.cudnn.benchmark = True
            # Set appropriate memory fraction based on GPU
            if "RTX 4090" in self.platform:
                # RTX 4090 has 24GB VRAM, we can be generous
                torch.cuda.set_per_process_memory_fraction(0.7)
            else:
                torch.cuda.set_per_process_memory_fraction(0.5)
            return True
        elif self.device.type == 'mps':
            # Apple Silicon optimizations
            # Currently not many specific MPS optimizations available
            return True
        return False
    
    def get_device(self):
        """Return the selected device."""
        return self.device

class TrackingState:
    """Enum-like class for tracking states."""
    ACTIVE = 'active'        # Fully visible
    OCCLUDED = 'occluded'    # Temporarily hidden
    TENTATIVE = 'tentative'  # New track
    LOST = 'lost'            # Missing too long

class PersonTracker:
    """Tracks people within a single video."""
    
    def __init__(self, video_path, output_dir, detector, reid_model, device,
                 camera_id=None, config=None):
        """
        Initialize the person tracker.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save results
            detector: YOLO detection model
            reid_model: Re-identification model
            device: Computation device (cuda, mps, or cpu)
            camera_id: Camera identifier (1 or 2)
            config: Dictionary of configuration parameters
        """
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        # Extract camera ID from the filename - for Camera_1_YYYYMMDD format
        self.camera_id = self.video_name.split('_')[1] if '_' in self.video_name else None
        
        # Set up output directories
        self.output_dir = Path(output_dir) / self.video_name
        self.images_dir = self.output_dir / "person_images"
        self.features_dir = self.output_dir / "person_features"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Store models and device
        self.detector = detector
        self.reid_model = reid_model
        self.device = device
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if detection fails
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Camera-specific parameters
        self.is_camera1 = self.camera_id == CAMERA1_ID
        self.config = config or {}
        
        # Set tracking parameters based on camera
        if self.is_camera1:  # Camera 1 (more complex environment)
            self.min_detection_confidence = self.config.get('cam1_min_confidence', 0.6)
            self.similarity_threshold = self.config.get('cam1_similarity_threshold', 0.65)
            self.max_disappeared = self.fps * self.config.get('cam1_max_disappear_seconds', 3)
            self.feature_weight = self.config.get('cam1_feature_weight', 0.6)
            self.position_weight = self.config.get('cam1_position_weight', 0.2)
            self.motion_weight = self.config.get('cam1_motion_weight', 0.2)
            self.reentry_threshold = self.config.get('cam1_reentry_threshold', 0.7)
        else:  # Camera 2 (cleaner environment)
            self.min_detection_confidence = self.config.get('cam2_min_confidence', 0.5)
            self.similarity_threshold = self.config.get('cam2_similarity_threshold', 0.7)
            self.max_disappeared = self.fps * self.config.get('cam2_max_disappear_seconds', 2)
            self.feature_weight = self.config.get('cam2_feature_weight', 0.55)
            self.position_weight = self.config.get('cam2_position_weight', 0.25)
            self.motion_weight = self.config.get('cam2_motion_weight', 0.2)
            self.reentry_threshold = self.config.get('cam2_reentry_threshold', 0.75)
        
        # Initialize tracking variables
        self.active_tracks = {}
        self.lost_tracks = {}
        self.person_features = {}
        self.person_timestamps = {}
        self.appearance_history = defaultdict(list)
        self.next_id = 0
        
        # Additional tracking parameters
        self.max_lost_age = self.fps * self.config.get('max_lost_seconds', 30)
        self.max_history_length = self.config.get('max_history_length', 10)
        self.update_interval = self.config.get('process_every_nth_frame', 1)
        
        print(f"Initialized tracker for {self.video_name} (Camera {self.camera_id})")
        print(f"Video details: {self.frame_width}x{self.frame_height}, {self.fps} FPS, {self.frame_count} frames")
    
    def extract_features(self, person_crop):
        """Extract ReID features from person crop."""
        try:
            # Preprocess image for ReID
            if person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
                return None  # Skip very small detections
                
            # Make a copy of the crop to ensure contiguous memory
            person_crop = person_crop.copy()
            
            # Resize image
            img = cv2.resize(person_crop, (128, 256))
            
            # Convert BGR to RGB and ensure contiguous array
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Convert to tensor (channels first)
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous().unsqueeze(0)
            
            # Move to device
            img = img.to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.reid_model(img)
                
            return features.cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_box_center(self, box):
        """Calculate center point of a bounding box."""
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    
    def calculate_velocity(self, current_box, previous_box):
        """Calculate velocity vector between two boxes."""
        current_center = self.calculate_box_center(current_box)
        previous_center = self.calculate_box_center(previous_box)
        return [current_center[0] - previous_center[0],
                current_center[1] - previous_center[1]]
    
    def predict_next_position(self, box, velocity):
        """Predict next position based on current position and velocity."""
        center = self.calculate_box_center(box)
        predicted_center = [center[0] + velocity[0], center[1] + velocity[1]]
        width = box[2] - box[0]
        height = box[3] - box[1]
        return [predicted_center[0] - width/2, predicted_center[1] - height/2,
                predicted_center[0] + width/2, predicted_center[1] + height/2]
    
    def calculate_motion_similarity(self, current_boxes, tracked_boxes, tracked_velocities):
        """Calculate motion-based similarity."""
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
                # Scale based on frame size
                distance_scale = min(self.frame_width, self.frame_height) / 10
                motion_sim[i, j] = np.exp(-distance / distance_scale)

        return motion_sim
    
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)
    
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
        containment_factor = 1.0 if (contained_horizontally and contained_vertically) else 0.0

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
        """Calculate similarity matrix combining appearance, position, and motion."""
        n_detections = len(current_features)
        n_tracks = len(tracked_features)

        if n_detections == 0 or n_tracks == 0:
            return np.array([])

        # Ensure all feature arrays are contiguous for cosine distance calculation
        current_features_flat = np.array([f.flatten() for f in current_features], dtype=np.float32)
        tracked_features_flat = np.array([f.flatten() for f in tracked_features], dtype=np.float32)
        
        # Ensure both arrays are contiguous in memory
        if not current_features_flat.flags['C_CONTIGUOUS']:
            current_features_flat = np.ascontiguousarray(current_features_flat)
        if not tracked_features_flat.flags['C_CONTIGUOUS']:
            tracked_features_flat = np.ascontiguousarray(tracked_features_flat)

        # Calculate appearance similarity
        appearance_sim = 1 - distance.cdist(
            current_features_flat,
            tracked_features_flat,
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

        # Combine all similarities with weights
        similarity_matrix = (
            self.feature_weight * appearance_sim +
            self.position_weight * position_sim +
            self.motion_weight * motion_sim
        )

        return similarity_matrix
    
    def update_feature_history(self, track_id, features):
        """Maintain rolling window of recent features."""
        self.appearance_history[track_id].append(features)
        if len(self.appearance_history[track_id]) > self.max_history_length:
            self.appearance_history[track_id].pop(0)

        # Update feature representation using exponential moving average
        if track_id in self.person_features:
            alpha = 0.7  # Weight for historical features
            current_features = self.person_features[track_id]
            updated_features = alpha * current_features + (1 - alpha) * features
            self.person_features[track_id] = updated_features
        else:
            self.person_features[track_id] = features
    
    def recover_lost_tracklet(self, features, current_box, frame_time):
        """Attempt to recover lost tracks."""
        best_match_id = None
        best_match_score = 0

        # Ensure features are flattened and contiguous
        features_flat = features.flatten()
        if not features_flat.flags['C_CONTIGUOUS']:
            features_flat = np.ascontiguousarray(features_flat)

        # Check recently lost tracks
        lost_tracks_to_remove = []
        for lost_id, lost_info in self.lost_tracks.items():
            # Skip if lost track is too old
            if frame_time - lost_info['last_seen'] > self.max_lost_age:
                lost_tracks_to_remove.append(lost_id)
                continue

            # Calculate appearance similarity
            lost_features = lost_info['features']
            lost_features_flat = lost_features.flatten()
            if not lost_features_flat.flags['C_CONTIGUOUS']:
                lost_features_flat = np.ascontiguousarray(lost_features_flat)
                
            appearance_sim = 1 - distance.cosine(features_flat, lost_features_flat)

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

            # Check if this is the best match
            if match_score > self.reentry_threshold and match_score > best_match_score:
                best_match_score = match_score
                best_match_id = lost_id

        # Clean up old lost tracks
        for lost_id in lost_tracks_to_remove:
            del self.lost_tracks[lost_id]

        return best_match_id if best_match_score > self.reentry_threshold else None
    
    def update_existing_track(self, track_id, box, features, frame_time, frame=None):
        """Update an existing track with new detection."""
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
        
        # Save person image if frame is provided
        if frame is not None:
            y1, y2 = max(0, int(box[1])), min(int(box[3]), frame.shape[0])
            x1, x2 = max(0, int(box[0])), min(int(box[2]), frame.shape[1])
            if x2 > x1 and y2 > y1:  # Ensure valid crop
                self.save_person_image(track_id, frame[y1:y2, x1:x2])
    
    def reactivate_track(self, track_id, box, features, frame_time, frame=None):
        """Reactivate a previously lost track."""
        # Get info from lost tracks
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

        # Update timestamps and feature history
        self.person_timestamps[track_id]['last_appearance'] = frame_time
        self.update_feature_history(track_id, features)
        
        # Save person image if frame is provided
        if frame is not None:
            y1, y2 = max(0, int(box[1])), min(int(box[3]), frame.shape[0])
            x1, x2 = max(0, int(box[0])), min(int(box[2]), frame.shape[1])
            if x2 > x1 and y2 > y1:  # Ensure valid crop
                self.save_person_image(track_id, frame[y1:y2, x1:x2])
    
    def create_new_track(self, box, features, frame_time, frame=None):
        """Create a new track for unmatched detection."""
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

        # Save person image if frame is provided
        if frame is not None:
            y1, y2 = max(0, int(box[1])), min(int(box[3]), frame.shape[0])
            x1, x2 = max(0, int(box[0])), min(int(box[2]), frame.shape[1])
            if x2 > x1 and y2 > y1:  # Ensure valid crop
                self.save_person_image(track_id=new_id, frame=frame[y1:y2, x1:x2])
        
        return new_id
    
    def update_lost_tracks(self, matched_track_ids, frame_time):
        """Update status of lost tracks and remove expired ones."""
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
    
    def save_person_image(self, track_id, frame):
        """Save person image in video-specific directory."""
        # Skip if frame is empty or invalid
        if frame is None or frame.size == 0 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            return None
            
        person_dir = self.images_dir / f"person_{track_id}"
        person_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = person_dir / f"{timestamp}.jpg"
        cv2.imwrite(str(image_path), frame)
        return image_path
    
    def save_person_features(self, person_id, features, frame_time):
        """Save person features in video-specific directory."""
        # Ensure the features directory exists
        feature_path = self.features_dir / f"person_{person_id}_features.npz"
        np.savez(str(feature_path),
                 features=features,
                 timestamp=frame_time,
                 video_name=self.video_name)
        return feature_path
    
    def update_tracks(self, frame, detections, frame_time):
        """Update tracks with new detections, handling reentries."""
        current_boxes = []
        current_features = []

        # Process new detections
        for box, conf in detections:
            if conf < self.min_detection_confidence:
                continue

            # Convert box coordinates to integers and ensure they're within frame bounds
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.frame_width, x2), min(self.frame_height, y2)
            
            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
                
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
                matched_track_ids.add(recovered_id)
                matched_detections.add(detection_idx)
            else:
                # Create new track
                new_id = self.create_new_track(
                    current_boxes[detection_idx],
                    current_features[detection_idx],
                    frame_time,
                    frame
                )
                matched_track_ids.add(new_id)

        # Update lost tracks
        self.update_lost_tracks(matched_track_ids, frame_time)
        
        return current_boxes, current_features
    
    def process_video(self, visualize=False, save_frames=False):
        """Process the entire video."""
        frame_count = 0
        processed_count = 0
        
        # Create progress bar
        pbar = tqdm(total=self.frame_count if self.frame_count > 0 else None, 
                   desc=f"Processing {self.video_name}")
        
        # Create visualization output directory if needed
        vis_dir = None
        if save_frames:
            vis_dir = self.output_dir / "visualization"
            vis_dir.mkdir(exist_ok=True)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_time = frame_count / self.fps
            frame_count += 1
            pbar.update(1)
            
            # Skip frames for performance
            if frame_count % self.update_interval != 0:
                continue
                
            processed_count += 1

            # Detect persons using YOLO
            results = self.detector(frame, classes=[0])  # class 0 is person

            # Process detections
            detections = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    detections.append((box.xyxy[0], box.conf[0]))

            # Update tracking
            current_boxes, current_features = self.update_tracks(frame, detections, frame_time)

            # Visualize results
            if visualize or save_frames:
                vis_frame = frame.copy()
                
                # Draw current detections
                for box in current_boxes:
                    cv2.rectangle(vis_frame, 
                                 (int(box[0]), int(box[1])),
                                 (int(box[2]), int(box[3])), 
                                 (0, 255, 0), 2)
                
                # Draw active tracks
                for track_id, track_info in self.active_tracks.items():
                    box = track_info['box']
                    state = track_info['state']
                    
                    # Different colors for different states
                    if state == TrackingState.ACTIVE:
                        color = (0, 255, 0)  # Green
                    elif state == TrackingState.OCCLUDED:
                        color = (0, 165, 255)  # Orange
                    elif state == TrackingState.TENTATIVE:
                        color = (0, 0, 255)  # Red
                    else:
                        color = (128, 128, 128)  # Gray
                        
                    cv2.rectangle(vis_frame, 
                                 (int(box[0]), int(box[1])),
                                 (int(box[2]), int(box[3])), 
                                 color, 2)
                    cv2.putText(vis_frame, f"ID: {track_id}",
                                (int(box[0]), int(box[1])-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add camera info
                cv2.putText(vis_frame, f"Camera {self.camera_id}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2)
                cv2.putText(vis_frame, f"Frame: {frame_count}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 255), 2)
                cv2.putText(vis_frame, f"Tracks: {len(self.active_tracks)}",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 0, 255), 2)
                
                # Display or save
                if visualize:
                    cv2.imshow(f'Camera {self.camera_id}', vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                        
                if save_frames and processed_count % 30 == 0:  # Save every 30th processed frame
                    frame_path = vis_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), vis_frame)

        # Release resources
        self.cap.release()
        if visualize:
            cv2.destroyAllWindows()
        pbar.close()
        
        print(f"Processed {processed_count} frames from {self.video_name}")
        print(f"Found {self.next_id} unique individuals")
        
        return self.save_tracking_results()
    
    def save_tracking_results(self):
        """Save tracking results and return summary."""
        results = {
            'video_name': self.video_name,
            'camera_id': self.camera_id,
            'total_persons': self.next_id,
            'video_metadata': {
                'width': self.frame_width,
                'height': self.frame_height,
                'fps': self.fps,
                'frame_count': self.frame_count
            },
            'persons': {}
        }

        for person_id in self.person_timestamps.keys():
            # Calculate person duration
            first_appearance = self.person_timestamps[person_id]['first_appearance']
            last_appearance = self.person_timestamps[person_id]['last_appearance']
            duration = last_appearance - first_appearance
            
            results['persons'][person_id] = {
                'first_appearance': first_appearance,
                'last_appearance': last_appearance,
                'duration': duration,
                'features_path': str(self.features_dir / f"person_{person_id}_features.npz"),
                'images_dir': str(self.images_dir / f"person_{person_id}")
            }

        # Save results to JSON
        results_path = self.output_dir / "tracking_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        return results

class GlobalTracker:
    """Tracks individuals across multiple cameras."""
    
    def __init__(self, config=None):
        """Initialize the global tracker."""
        self.config = config or {}
        self.global_identities = {}  # Map camera-specific IDs to global IDs
        self.appearance_sequence = {}  # Track sequence of camera appearances
        self.feature_database = {}  # Store features for cross-camera matching
        
        # Cross-camera matching parameters
        self.similarity_threshold = self.config.get('global_similarity_threshold', 0.75)
        self.temporal_constraint = self.config.get('temporal_constraint', True)
        self.max_time_gap = self.config.get('max_time_gap', 600)  # Max seconds between appearances
        self.min_time_gap = self.config.get('min_time_gap', 1)    # Min seconds between appearances
        
        # Enhanced feature storage
        self.camera_appearances = defaultdict(list)  # Track appearances by camera
    
    def register_camera_detection(self, camera_id, person_id, features, timestamp):
        """Register a detection from a specific camera."""
        global_id = self._match_or_create_global_id(camera_id, person_id, features, timestamp)
        
        camera_key = f"Camera_{camera_id}"
        self.camera_appearances[camera_key].append({
            'global_id': global_id,
            'camera_id': person_id,
            'timestamp': timestamp
        })
        
        if global_id not in self.appearance_sequence:
            self.appearance_sequence[global_id] = []
        
        # Only append if this is a new appearance in this camera
        if not self.appearance_sequence[global_id] or \
           self.appearance_sequence[global_id][-1]['camera'] != camera_key:
            self.appearance_sequence[global_id].append({
                'camera': camera_key,
                'timestamp': timestamp,
                'camera_specific_id': person_id
            })
    
    def _match_or_create_global_id(self, camera_id, person_id, features, timestamp):
        """Match with existing identity or create new global ID."""
        camera_key = f"{camera_id}_{person_id}"
        
        # Check if we've seen this camera-specific ID before
        if camera_key in self.global_identities:
            return self.global_identities[camera_key]
        
        # Ensure features are flattened and contiguous
        features_flat = features.flatten()
        if not features_flat.flags['C_CONTIGUOUS']:
            features_flat = np.ascontiguousarray(features_flat)
            
        # Try to match with existing identities
        best_match = None
        best_score = 0
        
        for global_id, data in self.feature_database.items():
            stored_features = data['features']
            last_timestamp = data['last_timestamp']
            last_camera = data['last_camera']
            
            # Skip if temporal constraint is violated
            time_diff = abs(timestamp - last_timestamp)
            if self.temporal_constraint:
                # Skip if time gap is too large or too small
                if time_diff > self.max_time_gap or time_diff < self.min_time_gap:
                    continue
                    
                # Skip if same camera (should be handled by single-camera tracker)
                if camera_id == last_camera:
                    continue
                    
            # Ensure stored features are flattened and contiguous
            stored_features_flat = stored_features.flatten()
            if not stored_features_flat.flags['C_CONTIGUOUS']:
                stored_features_flat = np.ascontiguousarray(stored_features_flat)
            
            # Calculate feature similarity
            similarity = 1 - distance.cosine(features_flat, stored_features_flat)
            
            # Adjust similarity based on time gap (closer in time = higher score)
            if self.temporal_constraint and time_diff <= self.max_time_gap:
                # Apply temporal decay - similarity decreases as time gap increases
                time_factor = 1.0 - (time_diff / self.max_time_gap) * 0.3
                similarity = similarity * time_factor
            
            if similarity > self.similarity_threshold and similarity > best_score:
                best_match = global_id
                best_score = similarity
        
        if best_match is None:
            # Create new global identity
            best_match = len(self.global_identities)
            
        # Update feature database
        self.feature_database[best_match] = {
            'features': features,
            'last_timestamp': timestamp,
            'last_camera': camera_id
        }
            
        self.global_identities[camera_key] = best_match
        return best_match
    
    def analyze_camera_transitions(self):
        """Analyze transitions between cameras."""
        cam1_to_cam2 = 0
        cam2_to_cam1 = 0
        cam1_persons = set()
        cam2_persons = set()
        
        # Track unique individuals per camera
        for global_id, appearances in self.appearance_sequence.items():
            has_cam1 = False
            has_cam2 = False
            
            for app in appearances:
                if app['camera'] == f'Camera_{CAMERA1_ID}':
                    has_cam1 = True
                elif app['camera'] == f'Camera_{CAMERA2_ID}':
                    has_cam2 = True
            
            if has_cam1:
                cam1_persons.add(global_id)
            if has_cam2:
                cam2_persons.add(global_id)
        
        # Count transitions
        for global_id, appearances in self.appearance_sequence.items():
            # Sort appearances by timestamp
            sorted_appearances = sorted(appearances, key=lambda x: x['timestamp'])
            
            # Check for sequential appearances
            for i in range(len(sorted_appearances) - 1):
                current = sorted_appearances[i]
                next_app = sorted_appearances[i + 1]
                
                if current['camera'] == f'Camera_{CAMERA1_ID}' and next_app['camera'] == f'Camera_{CAMERA2_ID}':
                    cam1_to_cam2 += 1
                elif current['camera'] == f'Camera_{CAMERA2_ID}' and next_app['camera'] == f'Camera_{CAMERA1_ID}':
                    cam2_to_cam1 += 1
        
        return {
            'camera1_unique': len(cam1_persons),
            'camera2_unique': len(cam2_persons),
            'camera1_to_camera2': cam1_to_cam2,
            'camera2_to_camera1': cam2_to_cam1,
            'total_unique_individuals': len(self.global_identities) // 2 + 1  # Accounting for overlap
        }

def process_video_pair(video1_path, video2_path, output_dir, config=None, visualize=False, source_dir_name=None):
    """Process a pair of videos from Camera 1 and Camera 2.
    
    Args:
        video1_path: Path to Camera 1 video
        video2_path: Path to Camera 2 video
        output_dir: Base directory to save results
        config: Configuration dictionary
        visualize: Whether to show visualization
        source_dir_name: Name of source directory (used for result organization)
    """
    device_manager = DeviceManager()
    device = device_manager.get_device()
    
    # Set default configuration if not provided
    if config is None:
        config = {
            # Camera 1 parameters (more complex environment)
            'cam1_min_confidence': 0.6,
            'cam1_similarity_threshold': 0.65,
            'cam1_max_disappear_seconds': 3,
            'cam1_feature_weight': 0.6, 
            'cam1_position_weight': 0.2,
            'cam1_motion_weight': 0.2,
            'cam1_reentry_threshold': 0.7,
            
            # Camera 2 parameters (cleaner environment)
            'cam2_min_confidence': 0.5,
            'cam2_similarity_threshold': 0.7,
            'cam2_max_disappear_seconds': 2,
            'cam2_feature_weight': 0.55,
            'cam2_position_weight': 0.25,
            'cam2_motion_weight': 0.2,
            'cam2_reentry_threshold': 0.75,
            
            # General parameters
            'max_lost_seconds': 30,
            'max_history_length': 10,
            'process_every_nth_frame': 1,
            
            # Global tracking parameters
            'global_similarity_threshold': 0.75,
            'temporal_constraint': True,
            'max_time_gap': 600,  # 10 minutes
            'min_time_gap': 1     # 1 second
        }
        
    # Create specific output directory using source directory name if provided
    if source_dir_name:
        output_dir = os.path.join(output_dir, source_dir_name)
    
    print("\n===== Loading models =====")
    # Load models
    try:
        # Load YOLO model
        detector = YOLO("yolov12x.pt")
        detector.to(device)
        
        # Load ReID model
        reid_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        reid_model = reid_model.to(device)
        reid_model.eval()
    except Exception as e:
        print(f"Error loading models: {e}")
        return None
    
    # Create global tracker
    global_tracker = GlobalTracker(config)
    
    print("\n===== Processing Camera 1 =====")
    # Process Camera 1 video
    try:
        tracker1 = PersonTracker(
            video_path=video1_path,
            output_dir=output_dir,
            detector=detector,
            reid_model=reid_model,
            device=device,
            camera_id=CAMERA1_ID,
            config=config
        )
        
        results1 = tracker1.process_video(visualize=visualize)
        
        # Register detections with global tracker
        for person_id, person_data in results1['persons'].items():
            features_path = person_data['features_path']
            features_data = np.load(features_path)
            features = features_data['features']
            timestamp = person_data['first_appearance']
            
            global_tracker.register_camera_detection(
                camera_id=CAMERA1_ID,
                person_id=person_id,
                features=features,
                timestamp=timestamp
            )
            
    except Exception as e:
        print(f"Error processing Camera 1 video: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n===== Processing Camera 2 =====")
    # Process Camera 2 video
    try:
        tracker2 = PersonTracker(
            video_path=video2_path,
            output_dir=output_dir,
            detector=detector,
            reid_model=reid_model,
            device=device,
            camera_id=CAMERA2_ID,
            config=config
        )
        
        results2 = tracker2.process_video(visualize=visualize)
        
        # Register detections with global tracker
        for person_id, person_data in results2['persons'].items():
            features_path = person_data['features_path']
            features_data = np.load(features_path)
            features = features_data['features']
            timestamp = person_data['first_appearance']
            
            global_tracker.register_camera_detection(
                camera_id=CAMERA2_ID,
                person_id=person_id,
                features=features,
                timestamp=timestamp
            )
            
    except Exception as e:
        print(f"Error processing Camera 2 video: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Analyze transitions
    analysis = global_tracker.analyze_camera_transitions()
    
    # Prepare summary
    summary = {
        'camera1_video': Path(video1_path).name,
        'camera2_video': Path(video2_path).name,
        'camera1_unique_persons': analysis['camera1_unique'],
        'camera2_unique_persons': analysis['camera2_unique'],
        'camera1_to_camera2_transitions': analysis['camera1_to_camera2'],
        'camera2_to_camera1_transitions': analysis['camera2_to_camera1'],
        'total_unique_global': analysis['total_unique_individuals']
    }
    
    # Save summary to file with source directory name
    if source_dir_name:
        summary['source_directory'] = source_dir_name
        summary_path = Path(output_dir) / f"{source_dir_name}_cross_camera_summary.json"
    else:
        summary_path = Path(output_dir) / "cross_camera_summary.json"
        
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print summary
    print("\n===== Cross-Camera Analysis =====")
    print(f"Camera 1 ({Path(video1_path).name}): {analysis['camera1_unique']} unique individuals")
    print(f"Camera 2 ({Path(video2_path).name}): {analysis['camera2_unique']} unique individuals")
    print(f"Transitions from Camera 1 to Camera 2: {analysis['camera1_to_camera2']}")
    print(f"Transitions from Camera 2 to Camera 1: {analysis['camera2_to_camera1']}")
    print(f"Total unique global individuals: {analysis['total_unique_individuals']}")
    
    # Return the key results as a tuple (Camera1 count, Camera2 count, Camera1->Camera2 count)
    return (analysis['camera1_unique'], analysis['camera2_unique'], analysis['camera1_to_camera2'])

def process_directory(input_dir, output_dir, config=None, visualize=False):
    """Process all video pairs in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get the input directory name for organizing results
    source_dir_name = input_path.name
    
    # Create a specific output directory for this input directory
    source_output_path = output_path / source_dir_name
    source_output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing videos from directory: {source_dir_name}")
    print(f"Saving results to: {source_output_path}")
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(list(input_path.glob(f'*{ext}')))
    
    # Group videos by camera
    cam1_videos = [v for v in all_videos if v.name.startswith(f"Camera_{CAMERA1_ID}_")]
    cam2_videos = [v for v in all_videos if v.name.startswith(f"Camera_{CAMERA2_ID}_")]
    
    print(f"Found {len(cam1_videos)} Camera 1 videos and {len(cam2_videos)} Camera 2 videos")
    
    # Create pairs based on matching date (YYYYMMDD part)
    video_pairs = []
    for cam1_video in cam1_videos:
        # Extract date part (YYYYMMDD) from Camera_1_YYYYMMDD format
        # First split by underscore to get ['Camera', '1', 'YYYYMMDD']
        parts = cam1_video.stem.split('_')
        if len(parts) >= 3:
            date_part = parts[2]  # Get the date part
            # Find matching Camera 2 video with same date
            matching_cam2 = [v for v in cam2_videos if v.stem.split('_')[2] == date_part]
            
            if matching_cam2:
                video_pairs.append((cam1_video, matching_cam2[0]))
                
    print(f"Found {len(video_pairs)} video pairs to process")
    
    # Process each pair
    results = []
    for i, (video1, video2) in enumerate(video_pairs):
        print(f"\n===== Processing Video Pair {i+1}/{len(video_pairs)} =====")
        print(f"Camera 1: {video1.name}")
        print(f"Camera 2: {video2.name}")
        
        # Create pair-specific output directory under the source directory
        date_part = video1.stem.split('_')[2]  # Get the YYYYMMDD part
        pair_output_dir = source_output_path / f"pair_{date_part}"
        
        try:
            result = process_video_pair(
                video1_path=str(video1),
                video2_path=str(video2),
                output_dir=str(pair_output_dir),
                config=config,
                visualize=visualize,
                source_dir_name=source_dir_name
            )
            
            if result:
                results.append({
                    'pair_name': video1.stem.split('_')[2],  # Use the date part as pair name
                    'camera1_video': video1.name,
                    'camera2_video': video2.name,
                    'camera1_count': result[0],
                    'camera2_count': result[1],
                    'camera1_to_camera2': result[2]
                })
                
        except Exception as e:
            print(f"Error processing video pair: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall results
    all_results_path = source_output_path / f"{source_dir_name}_results.json"
    with open(all_results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Also save a simple summary file with just the key metrics
    summary = {
        'source_directory': source_dir_name,
        'processed_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'video_pairs': len(video_pairs),
        'results': []
    }
    
    for result in results:
        summary['results'].append({
            'pair_name': result['pair_name'],
            'result_tuple': (result['camera1_count'], result['camera2_count'], result['camera1_to_camera2'])
        })
    
    summary_path = source_output_path / f"{source_dir_name}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    return results

def download_models():
    """Download required models if they don't exist."""
    print("Checking for required models...")
    
    # Check for YOLO
    try:
        # This will download the model if it doesn't exist
        YOLO("yolov12x.pt")
        print("YOLO model is available.")
    except Exception as e:
        print(f"Error checking YOLO model: {e}")
    
    # Check for ReID model
    try:
        import torchreid
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        print("ReID model is available.")
    except Exception as e:
        print(f"Error checking ReID model: {e}")
        
    print("Model check complete.")

def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description='Cross-Camera People Tracking')
    
    # Input/output options
    parser.add_argument('--input_dir', type=str, default='./videos',
                        help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    
    # Processing options
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization during processing')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save visualization frames')
    parser.add_argument('--skip_frames', type=int, default=1,
                        help='Process every Nth frame (1 = process all frames)')
    
    # Camera-specific parameters
    parser.add_argument('--cam1_conf', type=float, default=0.6,
                        help='Camera 1 detection confidence threshold')
    parser.add_argument('--cam2_conf', type=float, default=0.5,
                        help='Camera 2 detection confidence threshold')
    parser.add_argument('--cam1_sim', type=float, default=0.65,
                        help='Camera 1 similarity threshold')
    parser.add_argument('--cam2_sim', type=float, default=0.7,
                        help='Camera 2 similarity threshold')
    
    # Global tracking parameters
    parser.add_argument('--global_sim', type=float, default=0.75,
                        help='Global tracker similarity threshold')
    parser.add_argument('--max_time_gap', type=int, default=600,
                        help='Maximum time gap (seconds) for cross-camera matching')
    
    # Single pair processing
    parser.add_argument('--video1', type=str, default=None,
                        help='Path to Camera 1 video (for single pair processing)')
    parser.add_argument('--video2', type=str, default=None,
                        help='Path to Camera 2 video (for single pair processing)')
    
    # Model options
    parser.add_argument('--download_models', action='store_true',
                        help='Download required models')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')
    
    return parser

def main():
    """Main function to run the script."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Check if models should be downloaded
    if args.download_models:
        download_models()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up configuration from arguments
    config = {
        # Camera 1 parameters (more complex environment)
        'cam1_min_confidence': args.cam1_conf,
        'cam1_similarity_threshold': args.cam1_sim,
        'cam1_max_disappear_seconds': 3,
        'cam1_feature_weight': 0.6, 
        'cam1_position_weight': 0.2,
        'cam1_motion_weight': 0.2,
        'cam1_reentry_threshold': 0.7,
        
        # Camera 2 parameters (cleaner environment)
        'cam2_min_confidence': args.cam2_conf,
        'cam2_similarity_threshold': args.cam2_sim,
        'cam2_max_disappear_seconds': 2,
        'cam2_feature_weight': 0.55,
        'cam2_position_weight': 0.25,
        'cam2_motion_weight': 0.2,
        'cam2_reentry_threshold': 0.75,
        
        # General parameters
        'max_lost_seconds': 30,
        'max_history_length': 10,
        'process_every_nth_frame': args.skip_frames,
        
        # Global tracking parameters
        'global_similarity_threshold': args.global_sim,
        'temporal_constraint': True,
        'max_time_gap': args.max_time_gap,
        'min_time_gap': 1
    }
    
    # Print configuration
    print("Running with configuration:")
    print(f"  Camera 1 confidence threshold: {config['cam1_min_confidence']}")
    print(f"  Camera 2 confidence threshold: {config['cam2_min_confidence']}")
    print(f"  Processing every {config['process_every_nth_frame']} frame(s)")
    print(f"  Global similarity threshold: {config['global_similarity_threshold']}")
    
    # Process single pair or directory
    if args.video1 and args.video2:
        # Process single pair
        print(f"Processing video pair: {args.video1} and {args.video2}")
        result = process_video_pair(
            video1_path=args.video1,
            video2_path=args.video2,
            output_dir=args.output_dir,
            config=config,
            visualize=args.visualize
        )
        
        if result:
            print("\n===== Final Results =====")
            print(f"Camera 1 unique individuals: {result[0]}")
            print(f"Camera 2 unique individuals: {result[1]}")
            print(f"Individuals moving from Camera 1 to Camera 2: {result[2]}")
            print(f"Result tuple: {result}")
    else:
        # Process all pairs in directory
        print(f"Processing all video pairs in {args.input_dir}")
        results = process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config=config,
            visualize=args.visualize
        )
        
        if results:
            print("\n===== Summary of All Results =====")
            for i, result in enumerate(results):
                print(f"\nPair {i+1}: {result['pair_name']}")
                print(f"Camera 1 unique individuals: {result['camera1_count']}")
                print(f"Camera 2 unique individuals: {result['camera2_count']}")
                print(f"Individuals moving from Camera 1 to Camera 2: {result['camera1_to_camera2']}")
                print(f"Result tuple: ({result['camera1_count']}, {result['camera2_count']}, {result['camera1_to_camera2']})")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")