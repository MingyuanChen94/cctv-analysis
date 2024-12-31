import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchreid
from torchreid.reid import models
from torchreid.reid.utils.feature_extractor import FeatureExtractor
from datetime import datetime
import threading
from collections import defaultdict
import time
import queue
from pathlib import Path
import logging
import csv
import math

class PersonTracker:
    def __init__(self):
        self.doors = {
            'camera1': [(1030, 0), (1700, 560)],
            'camera2': [(400, 0), (800, 470)]
        }

        # Define counting zones (just below door area)
        self.counting_zones = {
            # Adjust coordinates as needed
            'camera1': [(1030, 200), (1700, 300)],
            # Adjust coordinates as needed
            'camera2': [(400, 200), (800, 300)]
        }

        # Initialize models
        self.yolo_model = self.initialize_yolo()
        self.reid_model = self.initialize_reid()

        # Storage for tracked individuals
        self.tracked_individuals = {}  # {id: PersonInfo}
        self.completed_tracks = set()  # Store IDs of completed tracks
        self.current_frame_detections = {}  # Store current frame detections
        # {person_id: {camera_id: (first_time, last_time)}}
        self.camera_appearances = defaultdict(dict)

        # Track counts
        self.entry_count = 0
        self.processed_tracks = set()  # Store IDs of tracks that have been counted

        # Tracking parameters
        self.reid_threshold = 0.92  # Increased threshold for stricter matching
        # Maximum time gap for track continuation (seconds)
        self.max_track_gap = 1.0
        self.min_tracking_frames = 10  # Minimum frames before counting
        self.confidence_threshold = 5   # Required confidence before counting

        # Add new tracking sets
        self.camera1_entries = set()  # Track unique entries in Camera 1
        self.camera1_to_camera2 = set()  # Track people moving between cameras

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("PersonTracker")

    def initialize_yolo(self):
        """Initialize YOLOv11 model"""
        try:
            model = YOLO("yolo11x.pt")
            return model
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            raise

    def initialize_reid(self):
        """Initialize ReID model"""
        try:
            # Initialize the model
            model = torchreid.models.build_model(
                name='osnet_ain_x1_0',
                num_classes=1000,  # Use original number of classes
                loss='softmax',
                pretrained=True
            )

            # Remove classifier layer as we only need features
            model.classifier = torch.nn.Identity()

            # Set to evaluation mode
            model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.cuda()

            return model
        except Exception as e:
            self.logger.error(f"Error loading ReID model: {e}")
            raise

    def is_in_door_area(self, bbox, camera_id):
        """Check if detection is in door area"""
        x1, y1, x2, y2 = bbox
        door_coords = self.doors[camera_id]

        # Check if center of bbox is in door area
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        door_x1, door_y1 = door_coords[0]
        door_x2, door_y2 = door_coords[1]

        return (door_x1 <= center_x <= door_x2 and
                door_y1 <= center_y <= door_y2)

    def extract_reid_features(self, frame, bbox):
        """Extract ReID features from detected person"""
        try:
            x1, y1, x2, y2 = bbox
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                return None

            # Preprocess image for ReID
            person_img = cv2.resize(person_img, (128, 256))
            person_img = cv2.cvtColor(
                person_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Normalize image
            person_img = person_img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            person_img = (person_img - mean) / std

            # Convert to tensor and add batch dimension
            person_img = torch.from_numpy(person_img).permute(
                2, 0, 1).unsqueeze(0).float()

            # Move to GPU if available
            if torch.cuda.is_available():
                person_img = person_img.cuda()

            with torch.no_grad():
                features = self.reid_model(person_img)
                features = features.cpu().numpy()

            if features is None or features.size == 0:
                return None

            # Normalize features
            features = features / np.linalg.norm(features)
            return features

        except Exception as e:
            self.logger.error(f"Error extracting ReID features: {e}")
            return None

    def match_person(self, features, current_time, camera_id):
        """Enhanced matching with multiple feature comparisons"""
        best_match_id = None
        best_match_score = 0.5  # Lower threshold for initial matching

        current_features = np.array(features).flatten()
        current_features = current_features / np.linalg.norm(current_features)

        candidates = []

        # First pass: collect all potential matches
        for person_id, person_info in self.tracked_individuals.items():
            # Skip completed tracks
            if person_id in self.completed_tracks:
                continue

            # Skip if track is too old (increased time gap)
            if (person_info.last_seen is not None and
                    current_time - person_info.last_seen > self.max_track_gap):
                continue

            # Compare with all stored features
            match_scores = []
            for stored_feat in person_info.features:
                score = self.compute_similarity(current_features, stored_feat)
                match_scores.append(score)

            if match_scores:
                # Use both max and average scores for better matching
                max_score = max(match_scores)
                avg_score = sum(match_scores) / len(match_scores)
                combined_score = 0.7 * max_score + 0.3 * avg_score

                if combined_score > best_match_score:
                    candidates.append((person_id, combined_score))

        # Sort candidates by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Second pass: validate top matches
        for candidate_id, score in candidates[:3]:  # Check top 3 matches
            person_info = self.tracked_individuals[candidate_id]

            # Additional validation for camera transitions
            if camera_id == 'camera2' and person_info.has_exited_camera1:
                transit_time = current_time - person_info.camera1_exit_time
                if 30 <= transit_time <= 240:  # Allow 30s to 4min transit time
                    return candidate_id

            # For same camera matching
            elif camera_id == person_info.last_camera:
                time_gap = current_time - person_info.last_seen
                if time_gap < self.max_track_gap:
                    return candidate_id

        return None

    def compute_similarity(self, features1, features2):
        """Compute cosine similarity between feature vectors with improved normalization"""
        # Convert features to numpy arrays if they aren't already
        feat1 = np.array(features1).flatten()
        feat2 = np.array(features2).flatten()

        # Normalize features
        feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feat2 / np.linalg.norm(feat2)

        return np.dot(feat1, feat2)

    def validate_detection(self, bbox, camera_id):
        """Comprehensive validation of initial detection"""
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        door_coords = self.doors[camera_id]

        # Basic size validation
        if width < 20 or height < 40:  # Too small to be a person
            return False
        if width > height:  # Person should be taller than wide
            return False
        if width/height > 0.8:  # Aspect ratio check
            return False

        # Position validation
        door_top = door_coords[0][1]
        door_bottom = door_coords[1][1]
        door_left = door_coords[0][0]
        door_right = door_coords[1][0]
        door_height = door_bottom - door_top

        # Must appear at very top of door
        if bbox[1] > door_top + door_height * 0.15:
            return False

        # Must be fully within door width
        if x_center < door_left or x_center > door_right:
            return False

        # Check if bbox is cut off at the top
        if bbox[1] < 5:  # Too close to frame edge
            return False

        return True

    def validate_movement(self, person_info, current_bbox):
        """Validate consistency of movement pattern"""
        if len(person_info.prev_positions) < 3:
            return True

        current_center = ((current_bbox[0] + current_bbox[2]) / 2,
                          (current_bbox[1] + current_bbox[3]) / 2)

        # Get last known position
        last_center = person_info.prev_positions[-1][0]

        # Calculate displacement
        dx = current_center[0] - last_center[0]
        dy = current_center[1] - last_center[1]

        # Check for unrealistic movements
        max_movement = 100  # Maximum allowed movement between frames
        if abs(dx) > max_movement or abs(dy) > max_movement:
            return False

        return True

    def validate_movement_sequence(self, positions, min_sequence=7):
        """Validate a sequence of movements for consistency"""
        if len(positions) < min_sequence:
            return False

        recent_pos = positions[-min_sequence:]

        # Calculate frame-to-frame movements
        movements = []
        velocities = []
        for i in range(1, len(recent_pos)):
            prev_pos = recent_pos[i-1][0]
            curr_pos = recent_pos[i][0]
            dt = recent_pos[i][1] - recent_pos[i-1][1]

            if dt == 0:  # Avoid division by zero
                return False

            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]

            movements.append((dx, dy))
            velocities.append((dx/dt, dy/dt))

        # Validate vertical movement
        downward_count = sum(1 for _, dy in movements if dy > 0)
        if downward_count < len(movements) * 0.85:  # 85% must be downward
            return False

        # Check for consistent velocity
        avg_vy = np.mean([vy for _, vy in velocities])
        velocity_variations = [abs(vy - avg_vy) for _, vy in velocities]
        if max(velocity_variations) > avg_vy * 0.5:  # Velocity shouldn't vary too much
            return False

        # Check for smooth trajectory
        for i in range(1, len(movements)):
            prev_dx, prev_dy = movements[i-1]
            curr_dx, curr_dy = movements[i]

            # Calculate angle change in trajectory
            angle_change = abs(math.atan2(prev_dy, prev_dx) -
                               math.atan2(curr_dy, curr_dx))
            if angle_change > math.pi/4:  # More than 45 degrees change
                return False

        return True

    def analyze_movement_pattern(self, positions, min_positions=3):
        """Analyze movement pattern from position history"""
        if len(positions) < min_positions:
            return None

        # Analyze the last several positions to determine movement direction
        recent_positions = positions[-min_positions:]

        # Calculate vertical and horizontal movements
        movements = []
        for i in range(1, len(recent_positions)):
            prev_pos = recent_positions[i-1][0]
            curr_pos = recent_positions[i][0]
            dy = curr_pos[1] - prev_pos[1]
            dx = curr_pos[0] - prev_pos[0]
            movements.append((dx, dy))

        # Check if movement is consistently downward
        downward_count = sum(1 for _, dy in movements if dy > 0)
        # 80% of movements must be downward
        if downward_count >= len(movements) * 0.8:
            # Calculate average vertical and horizontal movement
            avg_dy = sum(dy for _, dy in movements) / len(movements)
            avg_dx = sum(dx for dx, _ in movements) / len(movements)

            # Ensure vertical movement is significant compared to horizontal
            if avg_dy > abs(avg_dx) * 1.5:  # Vertical movement should be clearly dominant
                return 'entering'

        return 'other'

    def validate_track_continuity(self, person_info, current_time):
        """Validate track continuity and consistency"""
        if not person_info.prev_positions:
            return False

        # Check temporal continuity
        time_gaps = []
        for i in range(1, len(person_info.prev_positions)):
            gap = person_info.prev_positions[i][1] - \
                person_info.prev_positions[i-1][1]
            time_gaps.append(gap)

        if not time_gaps:  # Need at least two positions for gaps
            return False

        avg_gap = np.mean(time_gaps)
        max_gap = max(time_gaps)

        # Failed if any gap is too large
        if max_gap > avg_gap * 3 or max_gap > 0.5:  # Half second maximum gap
            return False

        # Check spatial continuity
        distances = []
        for i in range(1, len(person_info.prev_positions)):
            prev_pos = person_info.prev_positions[i-1][0]
            curr_pos = person_info.prev_positions[i][0]
            distance = math.sqrt(
                (curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            distances.append(distance)

        avg_distance = np.mean(distances)
        max_distance = max(distances)

        # Failed if any movement is too large
        if max_distance > avg_distance * 3 or max_distance > 100:  # Maximum pixel distance
            return False

        return True

    def analyze_trajectories(self, person_info, bbox, camera_id):
        """Enhanced analysis of person trajectories to detect entry and exit patterns"""
        if len(person_info.prev_positions) < 5:
            return None

        door_coords = self.doors[camera_id]
        door_top = door_coords[0][1]
        door_bottom = door_coords[1][1]
        door_height = door_bottom - door_top

        # Current position
        current_y = (bbox[1] + bbox[3]) / 2

        # Get movement history
        # (y-coord, timestamp)
        positions = [(pos[0][1], pos[1])
                     for pos in person_info.prev_positions[-5:]]

        # Calculate vertical movement
        y_movements = []
        for i in range(1, len(positions)):
            dy = positions[i][0] - positions[i-1][0]
            dt = positions[i][1] - positions[i-1][1]
            if dt > 0:  # Avoid division by zero
                velocity = dy/dt
                y_movements.append(velocity)

        if not y_movements:
            return None

    def update_person_info(self, person_id, frame, bbox, camera_id, timestamp, features):
        """Update person information with improved feature storage"""
        if person_id not in self.tracked_individuals:
            self.tracked_individuals[person_id] = PersonInfo(person_id)

        person_info = self.tracked_individuals[person_id]

        # Update basic tracking data
        person_info.update_features(features.squeeze())
        person_info.update_appearance(frame[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        if not person_info.update_position(bbox, timestamp):
            return

        # Update camera-specific information
        person_info.last_camera = camera_id
        person_info.last_seen = timestamp

        # Track entries and exits
        movement = self.analyze_trajectories(person_info, bbox, camera_id)

        if movement == 'entering':
            if not person_info.entry_recorded:
                if camera_id == 'camera1':
                    self.camera1_entries.add(person_id)
                    person_info.entered_camera1 = True
                    person_info.camera1_entry_time = timestamp
                elif camera_id == 'camera2' and person_info.entered_camera1:
                    self.camera1_to_camera2.add(person_id)
                person_info.entry_recorded = True
                self.entry_count += 1

        elif movement == 'exiting' and camera_id == 'camera1':
            if not person_info.exit_recorded:
                person_info.exit_recorded = True
                person_info.has_exited_camera1 = True
                person_info.camera1_exit_time = timestamp

        # Update camera times
        if camera_id not in person_info.camera_times:
            person_info.camera_times[camera_id] = {
                'first': timestamp, 'last': timestamp}
        else:
            person_info.camera_times[camera_id]['last'] = timestamp

    def get_tracking_stats(self):
        """Get statistics about tracked individuals"""
        stats = {
            'total_entries': self.entry_count,
            'unique_camera1_entries': len(self.camera1_entries),
            'camera1_to_camera2': len(self.camera1_to_camera2),
            'camera1_exits': len([p for p in self.tracked_individuals.values() if p.has_exited_camera1])
        }

        # Analyze timing for camera transitions
        transitions = []
        for person_id in self.camera1_to_camera2:
            person_info = self.tracked_individuals[person_id]
            if ('camera1' in person_info.camera_times and
                    'camera2' in person_info.camera_times):
                camera1_exit = person_info.camera_times['camera1']['last']
                camera2_entry = person_info.camera_times['camera2']['first']
                if camera2_entry > camera1_exit:
                    transit_time = camera2_entry - camera1_exit
                    transitions.append({
                        'person_id': person_id,
                        'camera1_exit': camera1_exit,
                        'camera2_entry': camera2_entry,
                        'transit_time': transit_time
                    })

        stats['transitions'] = transitions
        if transitions:
            stats['avg_transit_time'] = sum(
                t['transit_time'] for t in transitions) / len(transitions)

        return stats

    def process_frame(self, frame, camera_id, timestamp):
        """Process a single frame"""
        # Run YOLO detection
        results = self.yolo_model(frame)

        # Clear current frame detections
        self.current_frame_detections = {}

        for detection in results[0].boxes.data:
            # Convert bbox tensor to integer coordinates
            bbox = [int(coord.item()) for coord in detection[:4]]
            confidence = float(detection[4].item())
            class_id = int(detection[5].item())

            # Only process person detections with high confidence
            if class_id == 0 and confidence > 0.5:  # 0 is person class
                if self.is_in_door_area(bbox, camera_id):
                    # First check initial direction
                    if not self.validate_detection(bbox, camera_id):
                        continue

                    features = self.extract_reid_features(frame, bbox)
                    if features is not None:
                        person_id = self.match_person(
                            features, timestamp, camera_id)

                        if person_id is None:
                            # Create new track for potential new person
                            person_id = len(self.tracked_individuals)
                            self.tracked_individuals[person_id] = PersonInfo(
                                person_id)

                        # Check for duplicate detections in current frame
                        if person_id not in self.current_frame_detections:
                            person_img = frame[bbox[1]                                               :bbox[3], bbox[0]:bbox[2]]
                            self.update_person_info(
                                person_id, person_img, bbox, camera_id, timestamp, features)

                            # Verify movement pattern after updating position
                            person_info = self.tracked_individuals[person_id]
                            if len(person_info.prev_positions) >= 3:
                                movement = self.analyze_movement_pattern(
                                    person_info.prev_positions)
                                if movement != 'entering':
                                    self.completed_tracks.add(person_id)
                                    continue

                            self.current_frame_detections[person_id] = bbox

                            # Only draw boxes for valid tracks
                            if person_id not in self.completed_tracks:
                                cv2.rectangle(frame, (bbox[0], bbox[1]),
                                              (bbox[2], bbox[3]), (0, 255, 0), 2)
                                cv2.putText(frame, f"ID: {person_id}",
                                            (bbox[0], bbox[1]-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw entry count and door area
        cv2.putText(frame, f"Valid Entries: {self.entry_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        door_coords = self.doors[camera_id]
        cv2.rectangle(frame,
                      (int(door_coords[0][0]), int(door_coords[0][1])),
                      (int(door_coords[1][0]), int(door_coords[1][1])),
                      (255, 0, 255), 2)  # Magenta for door area

        return frame

    def extract_date_from_filename(self, filename):
        """Extract date from filename format Camera_X_YYYYMMDD"""
        try:
            # Extract the date part from the filename
            date_str = str(filename).split(
                '_')[-1].split('.')[0]  # Get YYYYMMDD part
            return date_str
        except:
            return None

    def save_tracking_data(self, output_dir, date):
        """Save tracking data to CSV files with date information"""
        os.makedirs(output_dir, exist_ok=True)

        # Save individual entries data
        entries_file = os.path.join(output_dir, f'entries_{date}.csv')
        with open(entries_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Person_ID', 'Camera_ID', 'Entry_Time', 'Exit_Time',
                            'First_Detection_X', 'First_Detection_Y',
                             'Last_Detection_X', 'Last_Detection_Y'])

            for person_id, person_info in self.tracked_individuals.items():
                for camera_id, times in person_info.camera_times.items():
                    if person_info.prev_positions:
                        first_pos = person_info.prev_positions[0][0]
                        last_pos = person_info.prev_positions[-1][0]
                        writer.writerow([
                            date,
                            person_id,
                            camera_id,
                            f"{times['first']:.2f}",
                            f"{times['last']:.2f}",
                            f"{first_pos[0]:.1f}",
                            f"{first_pos[1]:.1f}",
                            f"{last_pos[0]:.1f}",
                            f"{last_pos[1]:.1f}"
                        ])

        # Save camera transitions data
        transitions_file = os.path.join(
            output_dir, f'camera_transitions_{date}.csv')
        with open(transitions_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Person_ID', 'Camera1_Entry', 'Camera1_Exit',
                            'Camera2_Entry', 'Camera2_Exit', 'Transit_Time_Seconds'])

            for person_id in self.camera1_to_camera2:
                person_info = self.tracked_individuals.get(person_id)
                if person_info:
                    camera1_times = person_info.camera_times.get(
                        'camera1', {'first': None, 'last': None})
                    camera2_times = person_info.camera_times.get(
                        'camera2', {'first': None, 'last': None})

                    if camera1_times['last'] is not None and camera2_times['first'] is not None:
                        transit_time = camera2_times['first'] - \
                            camera1_times['last']
                        writer.writerow([
                            date,
                            person_id,
                            f"{camera1_times['first']:.2f}",
                            f"{camera1_times['last']:.2f}",
                            f"{camera2_times['first']:.2f}",
                            f"{camera2_times['last']:.2f}",
                            f"{transit_time:.2f}"
                        ])

        # Save summary statistics
        summary_file = os.path.join(output_dir, f'tracking_summary_{date}.csv')
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Metric', 'Value'])
            writer.writerow([date, 'Total_Camera1_Entries',
                            len(self.camera1_entries)])
            writer.writerow([date, 'Total_Camera2_Entries',
                            len(set(pid for pid, info in self.tracked_individuals.items()
                                if 'camera2' in info.camera_times))])
            writer.writerow(
                [date, 'Camera1_to_Camera2_Transitions', len(self.camera1_to_camera2)])

    def process_videos(self, video_dir, output_dir=None):
        """Process videos grouped by date"""
        if output_dir is None:
            output_dir = os.path.join(video_dir, 'tracking_results')

        # Group videos by date
        videos_by_date = defaultdict(list)
        for video_file in Path(video_dir).glob("Camera_*_*.mp4"):
            date = self.extract_date_from_filename(video_file)
            if date:
                videos_by_date[date].append(video_file)

        # Process each date's videos separately
        for date, video_files in videos_by_date.items():
            # Reset tracking for each date
            self.reset_tracking()

            print(f"\nProcessing videos for date: {date}")

            # Process each camera's video for this date
            # Sort to ensure Camera_1 processes first
            for video_file in sorted(video_files):
                camera_id = "camera1" if "Camera_1" in str(
                    video_file) else "camera2"
                print(f"Processing {video_file}")

                cap = cv2.VideoCapture(str(video_file))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    processed_frame = self.process_frame(
                        frame, camera_id, timestamp)

                    # Show real-time feedback
                    cv2.imshow(f"Camera {camera_id}", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()

            # Save results for this date
            if output_dir:
                self.save_tracking_data(output_dir, date)

        cv2.destroyAllWindows()

    def validate_intercamera_timing(self, first_camera_time, second_camera_time):
        """Validate the timing between camera appearances"""
        time_diff = second_camera_time - first_camera_time

        # Expected walking time is around 2 minutes
        MIN_TRANSIT_TIME = 60  # 1 minute minimum
        MAX_TRANSIT_TIME = 600  # 10 minutes maximum

        return MIN_TRANSIT_TIME <= time_diff <= MAX_TRANSIT_TIME

    def track_between_cameras(self):
        """Track people moving from Camera 1 to Camera 2"""
        camera1_tracks = {}  # {person_id: last_appearance_time}
        camera2_tracks = {}  # {person_id: first_appearance_time}
        matches = []  # [(person_id, camera1_time, camera2_time)]

        # Collect all valid tracks from both cameras
        for person_id, camera_info in self.camera_appearances.items():
            if 'camera1' in camera_info:
                # Use last appearance
                camera1_tracks[person_id] = camera_info['camera1'][1]
            if 'camera2' in camera_info:
                # Use first appearance
                camera2_tracks[person_id] = camera_info['camera2'][0]

        # Match tracks between cameras
        for person_id, camera1_time in camera1_tracks.items():
            if person_id in camera2_tracks:
                camera2_time = camera2_tracks[person_id]

                # Check if person appeared in Camera 2 after Camera 1
                if camera2_time > camera1_time:
                    # Validate the timing between appearances
                    if self.validate_intercamera_timing(camera1_time, camera2_time):
                        matches.append((person_id, camera1_time, camera2_time))

        return matches

    def analyze_tracks(self):
        """Analyze tracking results"""
        camera_matches = self.track_between_cameras()

        results = {
            'total_unique_individuals': len(self.tracked_individuals) - len(self.completed_tracks),
            'total_entries': self.entry_count,
            'camera1_entries': len(self.camera1_entries),
            'camera2_entries': len(set(pid for pid, info in self.tracked_individuals.items()
                                       if 'camera2' in info.camera_times)),
            'camera1_to_camera2_count': len(self.camera1_to_camera2),
            'camera1_to_camera2_ids': list(self.camera1_to_camera2),
            'transitions': [
                {
                    'person_id': pid,
                    'camera1_exit': self.tracked_individuals[pid].camera1_exit_time,
                    'camera2_entry': self.tracked_individuals[pid].camera_times.get('camera2', {}).get('first')
                }
                for pid in self.camera1_to_camera2
                if pid in self.tracked_individuals
            ]
        }

        # Calculate average transit time for valid transitions
        valid_transitions = [t for t in results['transitions']
                             if t['camera1_exit'] is not None and t['camera2_entry'] is not None]
        if valid_transitions:
            transit_times = [(t['camera2_entry'] - t['camera1_exit'])
                             for t in valid_transitions]
            results['average_transit_time'] = sum(
                transit_times) / len(transit_times)

        return results

    def reset_tracking(self):
        """Reset tracking states for new date"""
        self.tracked_individuals.clear()
        self.completed_tracks.clear()
        self.current_frame_detections.clear()
        self.camera1_entries.clear()
        self.camera1_to_camera2.clear()
        self.entry_count = 0

class PersonInfo:
    def __init__(self, person_id):
        self.person_id = person_id
        self.appearances = []
        self.features = []  # Store multiple features
        self.prev_positions = []
        self.last_position = None
        self.last_seen = None
        self.last_camera = None

        # Entry/Exit tracking
        self.entry_recorded = False
        self.exit_recorded = False
        self.entered_camera1 = False
        self.has_exited_camera1 = False
        self.camera1_entry_time = None
        self.camera1_exit_time = None
        self.camera_times = {}

    def update_appearance(self, image):
        """Store appearance image"""
        if image.size > 0:  # Only store valid images
            # Store a copy to prevent reference issues
            self.appearances.append(image.copy())
            if len(self.appearances) > 10:  # Keep last 10 appearances
                self.appearances.pop(0)

    def update_features(self, new_features):
        """Store multiple features for better matching"""
        feat = np.array(new_features).flatten()
        feat = feat / np.linalg.norm(feat)  # Normalize feature vector
        self.features.append(feat)
        if len(self.features) > 10:  # Keep more feature history
            self.features.pop(0)

    def get_average_features(self):
        """Get average of recent features"""
        if not self.features:
            return None
        # Stack features and compute mean
        stacked_features = np.vstack(self.features)
        return np.mean(stacked_features, axis=0)

    def update_position(self, bbox, timestamp):
        """Update position with timestamp"""
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.prev_positions.append((center, timestamp))
        if len(self.prev_positions) > 30:  # Keep last 30 positions
            self.prev_positions.pop(0)
        self.last_position = center
        self.last_seen = timestamp
        return True
    
tracker = PersonTracker()
video_dir = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
                         'Documents', 'VISIONARY', 'Durham Experiment', 'Experiment Data', 'Before')

# video_dir = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
#                          'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')

try:
    tracker.process_videos(video_dir)
    results = tracker.analyze_tracks()

    print("\nTracking Results:")
    print(f"Total unique individuals: {results['total_unique_individuals']}")
    print(
        f"People moving from Camera 1 to Camera 2: {results['camera1_to_camera2_count']}")

except Exception as e:
    logging.error(f"Error during tracking: {e}")
    raise