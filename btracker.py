import logging
import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
import numpy as np
from ultralytics import YOLO
import torchreid
from datetime import datetime
import threading
from collections import defaultdict
import time
import queue
from pathlib import Path
import csv
import math
from bytetracker import BYTETracker
from typing import Dict, List, Tuple

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PersonTracker")

class PersonTracker:
    def __init__(self):
        self.logger = logging.getLogger("PersonTracker")
        
        # Force CUDA if available
        if not torch.cuda.is_available():
            self.logger.warning("CUDA is not available. Please check your PyTorch installation and GPU setup.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
            # Set default tensor type to CUDA
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # Door and counting zones
        self.doors = {
            'camera1': [(1030, 0), (1700, 560)],
            'camera2': [(400, 0), (800, 470)]
        }
        self.counting_zones = {
            'camera1': [(1030, 200), (1700, 300)],
            'camera2': [(400, 200), (800, 300)]
        }

        # Initialize models
        self.yolo_model = self.initialize_yolo()
        self.reid_model = self.initialize_reid()

        # Initialize other tracking attributes
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
        self.track_features = defaultdict(list)
        self.tracked_individuals = {}
        self.completed_tracks = set()
        self.entry_count = 0
        self.camera1_entries = set()
        self.camera1_to_camera2 = set()

        # Tracking parameters
        self.max_age = 30
        self.min_hits = 3
        self.iou_threshold = 0.3
        self.reid_threshold = 0.7
        self.max_feature_history = 5

        self.logger.info("Person tracker initialized successfully")


    def initialize_yolo(self):
        """Initialize YOLO model with forced GPU usage if available"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory
                model = YOLO("yolov8x.pt")
                model.to(self.device)
                self.logger.info("YOLO model loaded on GPU")
            else:
                model = YOLO("yolov8x.pt")
                self.logger.warning("YOLO model running on CPU")
            return model
        except Exception as e:
            self.logger.error(f"Error initializing YOLO model: {e}")
            raise


    def initialize_reid(self):
        """Initialize ReID model with GPU support"""
        try:
            class OSNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Base convolution layers
                    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
                    self.bn1 = nn.BatchNorm2d(64)
                    self.relu = nn.ReLU(inplace=True)
                    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

                    # Feature layers
                    self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
                    self.bn2 = nn.BatchNorm2d(256)
                    self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
                    self.bn3 = nn.BatchNorm2d(512)

                    # Global pooling
                    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                    self.feat_dim = 512

                def forward(self, x):
                    x = self.conv1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.maxpool(x)

                    x = self.conv2(x)
                    x = self.bn2(x)
                    x = self.relu(x)

                    x = self.conv3(x)
                    x = self.bn3(x)
                    x = self.relu(x)

                    x = self.avgpool(x)
                    return x.view(x.size(0), -1)

            model = OSNet()
            model = model.to(self.device)
            model.eval()
            
            self.logger.info(f"ReID model initialized on {self.device}")
            return model

        except Exception as e:
            self.logger.error(f"Error loading ReID model: {e}")
            raise

    def extract_reid_features(self, frame, bbox, camera_id):
        """Extract ReID features with GPU support"""
        try:
            x1, y1, x2, y2 = bbox
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0 or person_img.shape[0] < 20 or person_img.shape[1] < 20:
                return None

            # Camera-specific color normalization
            if camera_id == 'camera1':
                person_img = cv2.addWeighted(person_img, 1.1, person_img, 0, 10)
            else:
                person_img = cv2.addWeighted(person_img, 0.9, person_img, 0, -10)

            features_list = []
            crops = [
                person_img,
                cv2.flip(person_img, 1),
                person_img[:-20, :],
                person_img[20:, :],
            ]

            for crop in crops:
                # Preprocess
                img = cv2.resize(crop, (128, 256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Normalize
                img = img.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img - mean) / std

                # Convert to tensor and move to device
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                img = img.to(self.device)

                with torch.no_grad():
                    feat = self.reid_model(img)
                    feat = F.normalize(feat, p=2, dim=1)
                    features_list.append(feat.cpu().numpy())

            features = np.mean(features_list, axis=0)
            return features

        except Exception as e:
            self.logger.error(f"Error extracting ReID features: {e}")
            return None

    def is_in_door_area(self, bbox, camera_id):
        """Check if detection is in door area"""
        x1, y1, x2, y2 = bbox
        door_coords = self.doors[camera_id]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        door_x1, door_y1 = door_coords[0]
        door_x2, door_y2 = door_coords[1]

        return (door_x1 <= center_x <= door_x2 and
                door_y1 <= center_y <= door_y2)

    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def extract_appearance_features(self, frame, bbox):
        """Extract clothing and appearance features without using scikit-learn"""
        try:
            x1, y1, x2, y2 = bbox
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                return None

            # Split person into regions
            height = y2 - y1
            head_region = person_img[0:height//4]
            upper_body = person_img[height//4:2*height//3]
            lower_body = person_img[2*height//3:]

            features = {}
            
            # Process each region
            for region_name, region in [('upper', upper_body), ('lower', lower_body)]:
                if region.size == 0:
                    continue
                    
                # Convert to HSV for better color analysis
                hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                
                # Calculate color histograms
                hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
                hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
                
                # Normalize histograms
                hist_h = cv2.normalize(hist_h, hist_h).flatten()
                hist_s = cv2.normalize(hist_s, hist_s).flatten()
                hist_v = cv2.normalize(hist_v, hist_v).flatten()
                
                features[f'{region_name}_color'] = np.concatenate([hist_h, hist_s, hist_v])
                
                # Simple dominant color extraction
                average_color = np.mean(region.reshape(-1, 3), axis=0)
                features[f'{region_name}_dominant_color'] = average_color
                
                # Calculate color variance
                color_var = np.var(region.reshape(-1, 3), axis=0)
                features[f'{region_name}_color_variance'] = color_var
                
                # Simple texture features using gradients
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                gradient_direction = np.arctan2(grad_y, grad_x)
                
                # Histogram of gradient directions
                hist_grad = np.histogram(gradient_direction, bins=9, range=(-np.pi, np.pi))[0]
                hist_grad = hist_grad.astype('float32') / sum(hist_grad)
                features[f'{region_name}_texture'] = hist_grad

            return features

        except Exception as e:
            self.logger.error(f"Error extracting appearance features: {e}")
            return None

    def compare_appearance_features(self, features1, features2):
        """Compare appearance features between two detections"""
        if not features1 or not features2:
            return 0.0

        scores = []
        
        # Compare each region
        for region in ['upper', 'lower']:
            # Compare color histograms
            color_key = f'{region}_color'
            if color_key in features1 and color_key in features2:
                color_sim = cv2.compareHist(
                    features1[color_key].reshape(-1, 1),
                    features2[color_key].reshape(-1, 1),
                    cv2.HISTCMP_CORREL
                )
                scores.append(max(0, color_sim))

            # Compare dominant colors
            dom_key = f'{region}_dominant_color'
            if dom_key in features1 and dom_key in features2:
                color_diff = np.linalg.norm(features1[dom_key] - features2[dom_key])
                color_sim = np.exp(-color_diff / 100)
                scores.append(color_sim)

            # Compare color variance
            var_key = f'{region}_color_variance'
            if var_key in features1 and var_key in features2:
                var_diff = np.linalg.norm(features1[var_key] - features2[var_key])
                var_sim = np.exp(-var_diff / 100)
                scores.append(var_sim)

            # Compare texture features
            tex_key = f'{region}_texture'
            if tex_key in features1 and tex_key in features2:
                texture_sim = np.sum(np.minimum(features1[tex_key], features2[tex_key]))
                scores.append(texture_sim)

        return np.mean(scores) if scores else 0.0
    
    def calculate_spatial_similarity(self, bbox1, last_position, timestamp, camera_id):
        """Calculate spatial similarity between detection and last known position"""
        try:
            # Get center points
            center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
            
            # Calculate time difference
            if isinstance(last_position, tuple):
                # If last_position is already a center point
                center2 = last_position
            else:
                # If last_position is a bbox
                center2 = ((last_position[0] + last_position[2]) / 2, 
                        (last_position[1] + last_position[3]) / 2)

            # Calculate Euclidean distance
            distance = np.sqrt((center1[0] - center2[0])**2 + 
                            (center1[1] - center2[1])**2)

            # Normalize distance based on frame size
            if camera_id == 'camera1':
                max_distance = np.sqrt(1700**2 + 560**2)  # Based on camera1 door area
            else:
                max_distance = np.sqrt(800**2 + 470**2)   # Based on camera2 door area

            # Convert distance to similarity score (1 = same location, 0 = max distance)
            similarity = max(0, 1 - (distance / max_distance))

            return similarity

        except Exception as e:
            self.logger.error(f"Error calculating spatial similarity: {e}")
            return 0.0

    def calculate_reid_similarity(self, features1, features2):
        """Calculate ReID feature similarity"""
        try:
            # Convert to numpy arrays if needed
            if isinstance(features1, torch.Tensor):
                features1 = features1.cpu().numpy()
            if isinstance(features2, torch.Tensor):
                features2 = features2.cpu().numpy()

            # Ensure features are 1D
            features1 = features1.flatten()
            features2 = features2.flatten()

            # Normalize features
            features1 = features1 / np.linalg.norm(features1)
            features2 = features2 / np.linalg.norm(features2)

            # Calculate cosine similarity
            similarity = np.dot(features1, features2)

            return float(similarity)

        except Exception as e:
            self.logger.error(f"Error calculating ReID similarity: {e}")
            return 0.0


    def match_detections(self, frame, detections, timestamp, camera_id):
        """Enhanced matching with appearance features"""
        if not self.tracks:
            return {i: self.next_id + i for i in range(len(detections))}

        matched_track_ids = {}
        unmatched_detections = list(range(len(detections)))

        # Calculate all similarity matrices
        reid_matrix = np.zeros((len(detections), len(self.tracks)))
        appearance_matrix = np.zeros((len(detections), len(self.tracks)))
        spatial_matrix = np.zeros((len(detections), len(self.tracks)))

        # Calculate similarities
        for i, det in enumerate(detections):
            det_bbox = det['bbox']
            det_appearance = self.extract_appearance_features(frame, det_bbox)

            for j, (track_id, track) in enumerate(self.tracks.items()):
                if track_id not in self.tracked_individuals:
                    continue
                    
                person_info = self.tracked_individuals[track_id]

                # ReID similarity
                if hasattr(person_info, 'features') and len(person_info.features) > 0:
                    reid_sim = self.calculate_reid_similarity(det['features'], person_info.features[-1])
                    reid_matrix[i, j] = reid_sim

                # Appearance similarity
                if hasattr(person_info, 'appearance_features'):
                    appearance_matrix[i, j] = self.compare_appearance_features(
                        det_appearance, person_info.appearance_features)

                # Spatial similarity using positions and camera info
                if person_info.last_position is not None:
                    spatial_matrix[i, j] = self.calculate_spatial_similarity(
                        det_bbox, person_info.last_position, timestamp, camera_id)

        # Combined matching with camera-aware weighting
        while unmatched_detections:
            best_match = None
            best_score = self.reid_threshold

            for i in unmatched_detections:
                for j, track_id in enumerate(self.tracks.keys()):
                    if track_id not in self.tracked_individuals:
                        continue
                        
                    person_info = self.tracked_individuals[track_id]
                    
                    # Adjust weights based on camera scenario
                    if person_info.last_camera == camera_id:
                        # Same camera: prioritize spatial and appearance
                        weights = {'reid': 0.3, 'appearance': 0.3, 'spatial': 0.4}
                    else:
                        # Cross-camera: rely more on ReID and appearance
                        weights = {'reid': 0.5, 'appearance': 0.4, 'spatial': 0.1}

                    # Combined score
                    score = (
                        weights['reid'] * reid_matrix[i, j] +
                        weights['appearance'] * appearance_matrix[i, j] +
                        weights['spatial'] * spatial_matrix[i, j]
                    )

                    if score > best_score:
                        best_score = score
                        best_match = (i, j, track_id)

            if best_match:
                i, j, track_id = best_match
                matched_track_ids[i] = track_id
                unmatched_detections.remove(i)
            else:
                break

        # Create new tracks for remaining detections
        for det_idx in unmatched_detections:
            matched_track_ids[det_idx] = self.next_id
            self.next_id += 1

        return matched_track_ids

    def process_frame(self, frame, camera_id, timestamp):
        """Process frame using integrated tracking approach"""
        if frame is None:
            return None

        processed_frame = frame.copy()
        current_detections = []

        # Run YOLO detection
        detections = self.yolo_model(frame)[0]

        # Process YOLO detections
        for det in detections.boxes.data:
            if int(det[5]) == 0:  # person class
                bbox = det[:4].cpu().numpy().astype(int)
                conf = float(det[4].cpu().numpy())

                if conf > 0.5:  # confidence threshold
                    features = self.extract_reid_features(frame, bbox, camera_id)
                    if features is not None:
                        if isinstance(features, torch.Tensor):
                            features = features.cpu().numpy()

                        current_detections.append({
                            'bbox': bbox,
                            'features': features,
                            'conf': conf
                        })

        # Match detections with existing tracks - pass frame and camera_id
        matched_track_ids = self.match_detections(frame, current_detections, timestamp, camera_id)

        # Update visualization and tracking
        active_tracks = set()
        for det_idx, track_id in matched_track_ids.items():
            det = current_detections[det_idx]
            bbox = det['bbox']
            active_tracks.add(track_id)

            # Update track info
            if track_id not in self.tracks:
                self.tracks[track_id] = {}
            
            self.tracks[track_id].update({
                'bbox': bbox,
                'last_seen': timestamp,
                'hits': self.tracks[track_id].get('hits', 0) + 1
            })

            # Update track features and update visualization
            if self.tracks[track_id]['hits'] >= self.min_hits:
                # Draw detection rectangle
                cv2.rectangle(processed_frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]),
                            (0, 255, 0), 2)
                
                # Draw ID text
                cv2.putText(processed_frame, 
                            f"ID: {track_id}",
                            (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            2)

                # Update person info
                if track_id not in self.tracked_individuals:
                    self.tracked_individuals[track_id] = PersonInfo(track_id)
                
                person_info = self.tracked_individuals[track_id]
                person_info.update_position(bbox, timestamp)
                person_info.update_features(det['features'])
                person_info.last_camera = camera_id

                # Process entries if in door area
                if self.is_in_door_area(bbox, camera_id):
                    if not person_info.entry_recorded:
                        if camera_id == 'camera1':
                            if track_id not in self.camera1_entries:
                                self.camera1_entries.add(track_id)
                                person_info.entered_camera1 = True
                                person_info.camera1_entry_time = timestamp
                        elif camera_id == 'camera2' and person_info.entered_camera1:
                            if track_id not in self.camera1_to_camera2:
                                self.camera1_to_camera2.add(track_id)
                        person_info.entry_recorded = True
                        self.entry_count += 1

        # Clean up old tracks
        self.clean_old_tracks(timestamp, active_tracks)

        # Draw door areas
        door_coords = self.doors[camera_id]
        cv2.rectangle(processed_frame,
                    (int(door_coords[0][0]), int(door_coords[0][1])),
                    (int(door_coords[1][0]), int(door_coords[1][1])),
                    (255, 0, 255), 2)

        # Draw entry count
        cv2.putText(processed_frame, 
                    f"Valid Entries: {self.entry_count}",
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255), 
                    2)

        return processed_frame

    def is_valid_transition(self, person_info, current_time):
        """Check if transition time is valid"""
        if not person_info.entered_camera1:
            return False

        transit_time = current_time - person_info.camera1_entry_time
        return 30 <= transit_time <= 600  # 30s to 10min

    def clean_old_tracks(self, current_time, active_tracks):
        """Remove old tracks"""
        for track_id in list(self.tracks.keys()):
            if (track_id not in active_tracks and
                    current_time - self.tracks[track_id]['last_seen'] > self.max_age):
                del self.tracks[track_id]
                if track_id in self.track_features:
                    del self.track_features[track_id]


    def process_videos(self, video_dir, output_dir=None):
        """Process videos grouped by date with proper cleanup"""
        if output_dir is None:
            output_dir = os.path.join(video_dir, 'tracking_results')

        try:
            # Group videos by date
            videos_by_date = defaultdict(list)
            for video_file in Path(video_dir).glob("Camera_*_*.mp4"):
                date = self.extract_date_from_filename(video_file)
                if date:
                    videos_by_date[date].append(video_file)

            # Process each date's videos separately
            for date, video_files in videos_by_date.items():
                self.reset_tracking()
                self.logger.info(f"\nProcessing videos for date: {date}")

                # Sort to ensure Camera_1 processes first
                for video_file in sorted(video_files):
                    camera_id = "camera1" if "Camera_1" in str(video_file) else "camera2"
                    self.logger.info(f"Processing {video_file}")

                    cap = cv2.VideoCapture(str(video_file))
                    if not cap.isOpened():
                        self.logger.error(f"Error opening video file: {video_file}")
                        continue

                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    frame_count = 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    # Create output video writer if needed
                    output_path = None
                    if output_dir:
                        output_path = os.path.join(output_dir, f'processed_{os.path.basename(video_file)}')
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_path, fourcc, fps, 
                                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

                    try:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_count += 1
                            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                            processed_frame = self.process_frame(frame, camera_id, timestamp)

                            if processed_frame is not None:
                                if output_path:
                                    out.write(processed_frame)

                            # Log progress
                            if frame_count % 100 == 0:
                                progress = (frame_count / total_frames) * 100
                                self.logger.info(
                                    f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%) from {camera_id}")

                    except Exception as e:
                        self.logger.error(f"Error processing frame: {e}")
                    finally:
                        cap.release()
                        if output_path:
                            out.release()

                    # Save results for this date if we have processed frames
                    if frame_count > 0 and output_dir:
                        self.save_tracking_data(output_dir, date)

        except Exception as e:
            self.logger.error(f"Error during video processing: {e}")
            raise

        return self.analyze_tracks()

    def reset_tracking(self):
        """Reset tracking states for new date"""
        self.tracks.clear()
        self.track_features.clear()
        self.tracked_individuals.clear()
        self.completed_tracks.clear()
        self.camera1_entries.clear()
        self.camera1_to_camera2.clear()
        self.entry_count = 0
        self.next_id = 0
        self.logger.info("Reset tracking state for new date")

    def extract_date_from_filename(self, filename):
        """Extract date from filename format Camera_X_YYYYMMDD"""
        try:
            date_str = str(filename).split('_')[-1].split('.')[0]
            return date_str
        except Exception as e:
            self.logger.error(f"Error extracting date from filename: {e}")
            return None

    def analyze_tracks(self):
        """Analyze tracking results and generate statistics"""
        results = {
            'total_unique_individuals': len(self.tracked_individuals),
            'total_entries': self.entry_count,
            'camera1_entries': len(self.camera1_entries),
            'camera2_entries': len(set(pid for pid, info in self.tracked_individuals.items()
                                   if hasattr(info, 'camera_times') and 'camera2' in info.camera_times)),
            'camera1_to_camera2_count': len(self.camera1_to_camera2),
            'camera1_to_camera2_ids': list(self.camera1_to_camera2),
            'transitions': [],
            'tracking_quality': {
                'total_tracks': len(self.tracked_individuals),
                'completed_tracks': len(self.completed_tracks),
                'active_tracks': len(self.tracks)
            }
        }

        # Calculate average track length and quality
        track_lengths = []
        track_qualities = []
        for person_id, person_info in self.tracked_individuals.items():
            if hasattr(person_info, 'prev_positions') and person_info.prev_positions:
                track_length = len(person_info.prev_positions)
                track_lengths.append(track_length)
                track_qualities.append(person_info.track_quality)

        if track_lengths:
            results['tracking_quality']['average_track_length'] = sum(
                track_lengths) / len(track_lengths)
            results['tracking_quality']['average_track_quality'] = sum(
                track_qualities) / len(track_qualities)

        return results

    def save_tracking_data(self, output_dir, date):
        """Save tracking data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save entries data
        entries_file = os.path.join(output_dir, f'entries_{date}.csv')
        with open(entries_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Person_ID', 'Camera_ID',
                            'Entry_Time', 'Exit_Time'])

            for person_id, person_info in self.tracked_individuals.items():
                if hasattr(person_info, 'camera_times'):
                    for camera_id, times in person_info.camera_times.items():
                        writer.writerow([
                            date,
                            person_id,
                            camera_id,
                            f"{times.get('first', ''):.2f}" if times.get(
                                'first') else '',
                            f"{times.get('last', ''):.2f}" if times.get(
                                'last') else ''
                        ])

        # Save summary statistics
        summary_file = os.path.join(output_dir, f'tracking_summary_{date}.csv')
        tracking_results = self.analyze_tracks()
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Metric', 'Value'])
            for metric, value in tracking_results.items():
                if isinstance(value, (int, float)):
                    writer.writerow([date, metric, value])

        self.logger.info(f"Saved tracking data to {output_dir}")

    def update_person_info(self, person_id, frame, bbox, camera_id, timestamp, features):
        """Update person information with improved tracking"""
        if person_id not in self.tracked_individuals:
            self.tracked_individuals[person_id] = PersonInfo(person_id)

        person_info = self.tracked_individuals[person_id]
        person_info.update_features(features)
        person_info.update_position(bbox, timestamp)

        # Camera-specific updates
        if camera_id == 'camera1':
            if not person_info.entered_camera1:
                if self.is_in_door_area(bbox, camera_id):
                    person_info.entered_camera1 = True
                    person_info.camera1_entry_time = timestamp
                    self.camera1_entries.add(person_id)
                    self.logger.info(f"New entry in Camera 1: ID {person_id}")

        elif camera_id == 'camera2':
            if person_info.entered_camera1:
                transit_time = timestamp - person_info.camera1_entry_time
                if 30 <= transit_time <= 300:  # 30s to 5min
                    if person_id not in self.camera1_to_camera2:
                        self.camera1_to_camera2.add(person_id)
                        self.logger.info(
                            f"Valid transition to Camera 2: ID {person_id}")

        # Update camera times
        if camera_id not in person_info.camera_times:
            person_info.camera_times[camera_id] = {
                'first': timestamp,
                'last': timestamp
            }
        else:
            person_info.camera_times[camera_id]['last'] = timestamp

class PersonInfo:
    def __init__(self, person_id):
        # Basic identification
        self.person_id = person_id

        # Appearance tracking
        self.features = []  # ReID features history
        self.appearances = []  # Image patches history

        # Position tracking
        self.prev_positions = []  # List of (position, timestamp) tuples
        self.last_position = None
        self.last_seen = None
        self.last_camera = None
        self.last_bbox = None
        self.track_quality = 1.0
        self.consecutive_misses = 0

        # Entry/Exit tracking
        self.entry_recorded = False
        self.exit_recorded = False
        self.entered_camera1 = False
        self.has_exited_camera1 = False
        self.camera1_entry_time = None
        self.camera1_exit_time = None

        # Camera timestamps
        self.camera_times = {}  # {camera_id: {'first': timestamp, 'last': timestamp}}

        # Track status
        self.hits = 0  # Number of times detected
        self.time_since_update = 0

    def update_features(self, new_features):
        """Store multiple features for better matching"""
        feat = np.array(new_features).flatten()
        feat = feat / np.linalg.norm(feat)  # Normalize feature vector
        self.features.append(feat)
        if len(self.features) > 5:  # Keep last 5 features
            self.features.pop(0)

    def update_appearance(self, image):
        """Store appearance image patches"""
        if image.size > 0:  # Only store valid images
            self.appearances.append(image.copy())
            if len(self.appearances) > 5:  # Keep last 5 appearances
                self.appearances.pop(0)

    def update_position(self, bbox, timestamp):
        """Update position with timestamp and track quality"""
        if bbox is None:
            self.consecutive_misses += 1
            self.track_quality *= 0.9
            return False

        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        # Store position history
        self.prev_positions.append((center, timestamp))
        if len(self.prev_positions) > 60:  # 10 seconds at 6fps
            self.prev_positions.pop(0)

        # Update tracking info
        self.last_position = center
        self.last_bbox = bbox
        self.last_seen = timestamp
        self.hits += 1
        self.consecutive_misses = 0

        # Update track quality
        self.track_quality = min(1.0, self.track_quality + 0.1)
        self.time_since_update = 0

        return True

    def get_velocity(self):
        """Calculate current velocity from recent positions"""
        if len(self.prev_positions) < 2:
            return None

        recent_pos = self.prev_positions[-2:]
        time_diff = recent_pos[1][1] - recent_pos[0][1]

        if time_diff > 0:
            dx = recent_pos[1][0][0] - recent_pos[0][0][0]
            dy = recent_pos[1][0][1] - recent_pos[0][0][1]
            return (dx/time_diff, dy/time_diff)

        return None

    def predict_position(self, timestamp):
        """Predict position at given timestamp using velocity"""
        if self.last_position is None or self.last_seen is None:
            return None

        velocity = self.get_velocity()
        if velocity is None:
            return self.last_position

        time_gap = timestamp - self.last_seen
        predicted_x = self.last_position[0] + velocity[0] * time_gap
        predicted_y = self.last_position[1] + velocity[1] * time_gap

        return (predicted_x, predicted_y)

    def get_track_status(self):
        """Get current status of the track"""
        return {
            'id': self.person_id,
            'quality': self.track_quality,
            'hits': self.hits,
            'misses': self.consecutive_misses,
            'last_camera': self.last_camera,
            'time_since_update': self.time_since_update,
            'entered_camera1': self.entered_camera1,
            'has_exited_camera1': self.has_exited_camera1
        }

    def is_track_valid(self):
        """Check if track is still valid based on quality and hits"""
        return (self.track_quality > 0.3 and
                self.hits >= 3 and
                self.consecutive_misses <= 10)
    
if __name__ == "__main__":
    try:
        # Initialize tracker
        tracker = PersonTracker()

        # Define video directory paths
        video_dir = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
                                'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')

        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")

        output_dir = os.path.join(video_dir, 'tracking_results')
        os.makedirs(output_dir, exist_ok=True)

        # Process videos and get results
        results = tracker.process_videos(video_dir, output_dir)

        # Print tracking results
        print("\nTracking Results:")
        print(f"Total unique individuals: {results['total_unique_individuals']}")
        print(f"Total entries in Camera 1: {results['camera1_entries']}")
        print(f"Total entries in Camera 2: {results['camera2_entries']}")
        print(f"People moving from Camera 1 to Camera 2: {results['camera1_to_camera2_count']}")

        if 'average_transit_time' in results:
            print(f"Average transit time between cameras: {results['average_transit_time']:.2f} seconds")

    except FileNotFoundError as e:
        logger.error(f"Error: Video directory or files not found - {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during tracking: {e}")
        sys.exit(1)
