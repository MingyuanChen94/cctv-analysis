import os
import cv2
import numpy as np
import torch
import torchreid
from ultralytics import YOLO
from collections import defaultdict
import logging
from pathlib import Path
import csv
import math

class PersonTracker:
    def __init__(self):
        # Door areas (using coordinates from original code)
        self.doors = {
            'camera1': [(1030, 0), (1700, 560)],
            'camera2': [(400, 0), (800, 470)]
        }
        
        # Initialize models
        self.yolo_model = self.initialize_yolo()
        self.reid_model = self.initialize_reid()
        
        # Tracking state
        self.tracked_individuals = {}
        self.camera_appearances = defaultdict(dict)
        self.cross_camera_matches = set()
        
        # Improved matching parameters
        self.same_camera_reid_threshold = 0.7    # More lenient for same camera
        self.cross_camera_reid_threshold = 0.5   # Much more lenient for cross-camera
        self.max_frame_gap = 5                   # Maximum frames to maintain tracking (at 6fps)
        self.min_track_length = 4                # Minimum frames to consider a valid track
        self.min_entry_frames = 3                # Minimum frames in door area for entry
        
        # Transit time constraints (in seconds)
        self.min_transit_time = 60   # 1 minute minimum
        self.max_transit_time = 300  # 5 minutes maximum
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("PersonTracker")

    def initialize_yolo(self):
        try:
            model = YOLO("yolov8x.pt")  # Using YOLOv8 for better detection
            return model
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            raise

    def initialize_reid(self):
        try:
            model = torchreid.models.build_model(
                name='osnet_ain_x1_0',  # Using improved ReID model
                num_classes=1000,
                loss='softmax',
                pretrained=True
            )
            model.classifier = torch.nn.Identity()
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            return model
        except Exception as e:
            self.logger.error(f"Error loading ReID model: {e}")
            raise

    def extract_features(self, frame, bbox):
        """Extract ReID features with improved preprocessing"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            if x1 >= x2 or y1 >= y2:
                return None
                
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                return None

            # Enhanced preprocessing
            person_img = cv2.resize(person_img, (128, 256))
            person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            
            # Improved normalization
            person_img = person_img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            person_img = (person_img - mean) / std

            # Convert to tensor
            person_img = torch.from_numpy(person_img).permute(2, 0, 1).unsqueeze(0).float()
            if torch.cuda.is_available():
                person_img = person_img.cuda()

            # Extract features
            with torch.no_grad():
                features = self.reid_model(person_img)
                features = features.cpu().numpy()

            # Normalize features
            features = features / np.linalg.norm(features)
            return features

        except Exception as e:
            self.logger.error(f"Error in feature extraction: {e}")
            return None

    def compute_similarity(self, features1, features2):
        """Compute cosine similarity with improved normalization"""
        feat1 = np.array(features1).flatten()
        feat2 = np.array(features2).flatten()
        
        # Ensure normalization
        feat1 = feat1 / (np.linalg.norm(feat1) + 1e-6)
        feat2 = feat2 / (np.linalg.norm(feat2) + 1e-6)
        
        return np.dot(feat1, feat2)

    def match_person(self, features, bbox, camera_id, frame_time):
        """Enhanced matching logic with adaptive thresholds"""
        best_match_id = None
        best_match_score = 0
        
        for person_id, person_info in self.tracked_individuals.items():
            if not person_info.is_active(frame_time, self.max_frame_gap):
                continue

            # Calculate feature similarity
            similarity = self.compute_similarity(features, person_info.get_average_features())
            
            # Determine threshold based on matching context
            if person_info.last_camera == camera_id:
                threshold = self.same_camera_reid_threshold
            else:
                # Check if this could be a valid camera transition
                if camera_id == 'camera2' and person_info.last_camera == 'camera1':
                    transit_time = frame_time - person_info.last_seen
                    if self.min_transit_time <= transit_time <= self.max_transit_time:
                        threshold = self.cross_camera_reid_threshold
                    else:
                        continue
                else:
                    continue

            # Update best match if threshold is met
            if similarity > threshold and similarity > best_match_score:
                best_match_score = similarity
                best_match_id = person_id

        return best_match_id

    def validate_detection(self, bbox, camera_id):
        """Enhanced detection validation"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Basic size checks
        if width < 20 or height < 40:  # Too small
            return False
        if width > height:  # People should be taller than wide
            return False
        if width/height > 0.8:  # Aspect ratio check
            return False

        # Check if detection is in valid door area
        door_coords = self.doors[camera_id]
        door_x1, door_y1 = door_coords[0]
        door_x2, door_y2 = door_coords[1]
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if not (door_x1 <= center_x <= door_x2):
            return False
            
        # For top of door area entry validation
        door_height = door_y2 - door_y1
        if y1 > door_y1 + door_height * 0.2:  # Must start near top
            return False

        return True

    def process_frame(self, frame, camera_id, frame_time):
        """Process frame with improved detection and tracking"""
        # Run YOLO detection
        results = self.yolo_model(frame)
        
        # Process each detection
        for result in results[0].boxes.data:
            if len(result) < 6:
                continue
                
            x1, y1, x2, y2, conf, cls = result[:6]
            if int(cls) != 0 or float(conf) < 0.5:  # Skip non-person or low confidence
                continue

            bbox = [int(x1), int(y1), int(x2), int(y2)]
            
            # Validate detection
            if not self.validate_detection(bbox, camera_id):
                continue

            # Extract features
            features = self.extract_features(frame, bbox)
            if features is None:
                continue

            # Match or create new track
            person_id = self.match_person(features, bbox, camera_id, frame_time)
            
            if person_id is None:
                # Create new track
                person_id = len(self.tracked_individuals)
                self.tracked_individuals[person_id] = PersonInfo(person_id)

            # Update track
            person_info = self.tracked_individuals[person_id]
            person_info.update(bbox, features, camera_id, frame_time)
            
            # Check for camera transition
            if (camera_id == 'camera2' and 
                person_info.has_previous_camera('camera1')):
                transit_time = frame_time - person_info.last_camera_time('camera1')
                if self.min_transit_time <= transit_time <= self.max_transit_time:
                    self.cross_camera_matches.add(person_id)

            # Visualize
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {person_id}", (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def get_statistics(self):
        """Get tracking statistics with improved filtering"""
        # Filter for valid tracks (minimum length requirement)
        valid_tracks = {
            pid: info for pid, info in self.tracked_individuals.items()
            if info.track_length >= self.min_track_length
        }
        
        # Count unique individuals per camera
        camera1_individuals = set()
        camera2_individuals = set()
        
        for pid, info in valid_tracks.items():
            if info.has_appeared_in('camera1'):
                camera1_individuals.add(pid)
            if info.has_appeared_in('camera2'):
                camera2_individuals.add(pid)

        return {
            'camera1_unique': len(camera1_individuals),
            'camera2_unique': len(camera2_individuals),
            'cross_camera': len(self.cross_camera_matches)
        }

class PersonInfo:
    def __init__(self, person_id):
        self.person_id = person_id
        self.appearances = []  # List of (camera_id, time, features)
        self.positions = []    # List of (camera_id, time, bbox)
        self.features_history = []
        self.track_length = 0
        self.last_seen = None
        self.last_camera = None

    def update(self, bbox, features, camera_id, frame_time):
        """Update track information"""
        self.appearances.append((camera_id, frame_time, features))
        self.positions.append((camera_id, frame_time, bbox))
        self.features_history.append(features)
        self.track_length += 1
        self.last_seen = frame_time
        self.last_camera = camera_id
        
        # Keep history manageable
        if len(self.features_history) > 10:
            self.features_history.pop(0)

    def is_active(self, current_time, max_gap):
        """Check if track is still active"""
        return self.last_seen is not None and \
               (current_time - self.last_seen) <= max_gap

    def get_average_features(self):
        """Get average of recent features"""
        if not self.features_history:
            return None
        features = np.array(self.features_history)
        return np.mean(features, axis=0)

    def has_appeared_in(self, camera_id):
        """Check if person has appeared in specific camera"""
        return any(cam == camera_id for cam, _, _ in self.appearances)

    def has_previous_camera(self, camera_id):
        """Check if person has previously appeared in camera"""
        return any(cam == camera_id for cam, _, _ in self.appearances[:-1])

    def last_camera_time(self, camera_id):
        """Get last appearance time in specific camera"""
        times = [t for cam, t, _ in self.appearances if cam == camera_id]
        return max(times) if times else None

def main():
    tracker = PersonTracker()
    
    # video_dir = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
    #                          'Documents', 'VISIONARY', 'Durham Experiment', 'Experiment Data', 'Before')

    video_dir = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
                            'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')    
    # Process videos
    for video_file in sorted(Path(video_dir).glob("Camera_*_*.mp4")):
        camera_id = "camera1" if "Camera_1" in str(video_file) else "camera2"
        print(f"Processing {video_file}")
        
        cap = cv2.VideoCapture(str(video_file))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_time = frame_count / 6.0  # 6fps
            processed_frame = tracker.process_frame(frame, camera_id, frame_time)
            
            cv2.imshow(f"Camera {camera_id}", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
            
        cap.release()
    
    cv2.destroyAllWindows()
    
    # Print results
    stats = tracker.get_statistics()
    print("\nTracking Results:")
    print(f"Camera 1 Unique Individuals: {stats['camera1_unique']}")
    print(f"Camera 2 Unique Individuals: {stats['camera2_unique']}")
    print(f"Camera 1 to Camera 2 Transitions: {stats['cross_camera']}")

if __name__ == "__main__":
    main()
