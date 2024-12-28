from ultralytics import YOLO
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import urllib.request
import os
from collections import deque

class PersonTracker:
    def __init__(self):
        self.model = YOLO('yolov8x.pt')
        
        # Download and load face cascade file
        cascade_file = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_file):
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, cascade_file)
        
        self.face_cascade = cv2.CascadeClassifier(cascade_file)
        
        self.unique_persons = []
        self.position_history = []
        self.track_length = 30
        
        # Tracking parameters
        self.similarity_threshold = 0.7
        self.position_threshold = 150
        
        # Door zone parameters
        self.door_zone = None  # Will be set by user
        self.door_direction = None  # Will be set by user
        self.entry_buffer = 50  # Buffer area around door to detect entries
        
        # Track people near door
        self.door_area_tracks = {}  # Track people near the door
        self.track_timeout = 30  # Frames to keep tracking a person
        
    def set_door_area(self, x1, y1, x2, y2, direction='right'):
        """
        Set the door area and entry direction
        direction: 'left', 'right', 'up', 'down' - direction of movement that indicates entry
        """
        self.door_zone = (x1, y1, x2, y2)
        self.door_direction = direction
        
    def is_in_door_zone(self, bbox):
        """Check if a detection overlaps with the door zone"""
        if self.door_zone is None:
            return False
            
        dx1, dy1, dx2, dy2 = self.door_zone
        bx1, by1, bx2, by2 = bbox
        
        # Check for overlap
        return not (bx2 < dx1 or bx1 > dx2 or by2 < dy1 or by1 > dy2)
    
    def check_entry_direction(self, current_pos, track_history):
        """
        Check if movement direction indicates entry through door
        """
        if len(track_history) < 2:
            return False
            
        # Get movement vector
        start_pos = np.array(track_history[0])
        current_pos = np.array(current_pos)
        movement = current_pos - start_pos
        
        # Check movement direction based on door configuration
        if self.door_direction == 'right':
            return movement[0] > self.entry_buffer
        elif self.door_direction == 'left':
            return movement[0] < -self.entry_buffer
        elif self.door_direction == 'down':
            return movement[1] > self.entry_buffer
        elif self.door_direction == 'up':
            return movement[1] < -self.entry_buffer
            
        return False
    
    def extract_features(self, frame, bbox):
        features = {}
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        # Store position for temporal tracking
        features['position'] = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Ensure coordinates are within frame boundaries
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
        
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return None
            
        try:
            # Extract face features
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                face_x, face_y, face_w, face_h = faces[0]
                face_roi = gray[face_y:face_y+face_h, face_x:face_x+face_w]
                face_roi = cv2.resize(face_roi, (64, 64))
                features['face'] = face_roi.flatten()
            else:
                features['face'] = None
                
            # Extract body features
            h_roi = person_roi.shape[0]
            upper_body = person_roi[:h_roi//2, :]
            lower_body = person_roi[h_roi//2:, :]
            
            if upper_body.size > 0 and lower_body.size > 0:
                # Convert to HSV for better color matching
                upper_hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
                lower_hsv = cv2.cvtColor(lower_body, cv2.COLOR_BGR2HSV)
                
                # Calculate color histograms
                features['upper_body_color'] = cv2.calcHist([upper_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                features['lower_body_color'] = cv2.calcHist([lower_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
                
                # Normalize histograms
                cv2.normalize(features['upper_body_color'], features['upper_body_color'], 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(features['lower_body_color'], features['lower_body_color'], 0, 1, cv2.NORM_MINMAX)
                
                # Add build features
                features['build'] = {
                    'aspect_ratio': h/w,
                    'area': w*h,
                    'height': h
                }
                
                return features
                
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            return None
        
        return None

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        # Set up door area on first frame
        ret, first_frame = cap.read()
        if not ret:
            return 0
            
        # Let user select door area
        cv2.namedWindow('Select Door Area')
        door_zone = cv2.selectROI('Select Door Area', first_frame, False)
        cv2.destroyWindow('Select Door Area')
        
        # Set door area with some padding
        x, y, w, h = door_zone
        self.set_door_area(x-self.entry_buffer, y-self.entry_buffer, 
                          x+w+self.entry_buffer, y+h+self.entry_buffer)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 3 != 0:  # Process every 3rd frame
                continue
                
            # Run YOLOv8 inference
            results = self.model(frame, classes=[0])  # Only detect persons
            
            # Update door area tracks
            current_tracks = {}
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf.item()
                    if confidence > 0.5:
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # Extract features
                        person_features = self.extract_features(frame, bbox)
                        if not person_features:
                            continue
                            
                        current_pos = person_features['position']
                        
                        # Check if person is in door zone
                        if self.is_in_door_zone(bbox):
                            found_match = False
                            
                            # Check existing door tracks
                            for track_id, track_data in self.door_area_tracks.items():
                                if self.compare_features(person_features, track_data['features']) > self.similarity_threshold:
                                    # Update existing track
                                    track_data['positions'].append(current_pos)
                                    track_data['timeout'] = self.track_timeout
                                    track_data['features'] = person_features
                                    current_tracks[track_id] = track_data
                                    found_match = True
                                    break
                            
                            if not found_match:
                                # Start new track
                                track_id = len(self.door_area_tracks)
                                current_tracks[track_id] = {
                                    'positions': deque([current_pos], maxlen=self.track_length),
                                    'features': person_features,
                                    'timeout': self.track_timeout,
                                    'counted': False
                                }
                        
                        # Check if any tracks indicate entry
                        for track_id, track_data in current_tracks.items():
                            if not track_data['counted'] and len(track_data['positions']) >= 2:
                                if self.check_entry_direction(current_pos, list(track_data['positions'])):
                                    # Add to unique persons
                                    self.unique_persons.append(track_data['features'])
                                    track_data['counted'] = True
                        
                        # Draw detection box
                        color = (0, 255, 0) if self.is_in_door_zone(bbox) else (0, 255, 255)
                        cv2.rectangle(frame, 
                                    (int(bbox[0]), int(bbox[1])), 
                                    (int(bbox[2]), int(bbox[3])), 
                                    color, 2)
            
            # Update door area tracks
            self.door_area_tracks = {k: v for k, v in current_tracks.items() if v['timeout'] > 0}
            for track in self.door_area_tracks.values():
                track['timeout'] -= 1
            
            # Draw door zone
            if self.door_zone:
                x1, y1, x2, y2 = self.door_zone
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            # Display count
            cv2.putText(frame, f"Unique persons: {len(self.unique_persons)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            display_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('Video', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return len(self.unique_persons)

# Usage
tracker = PersonTracker()
video_path = camera_1_files_sorted[0]
unique_count = tracker.process_video(video_path)
print(f"Number of unique individuals detected: {unique_count}")
