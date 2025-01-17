import os
import cv2
import numpy as np
from ultralytics import YOLO
import torchreid
import torch
from collections import defaultdict
from datetime import datetime
from scipy.spatial.distance import cdist
import math
from pathlib import Path
import json

class GlobalTracker:
    def __init__(self):
        self.global_identities = {}
        self.appearance_sequence = {}
        self.feature_database = {}
        # Lowered similarity threshold for cross-camera matching
        self.same_camera_threshold = 0.6  # More lenient same-camera matching
        self.cross_camera_threshold = 0.4  # Much more lenient for cross-camera
        self.max_transit_time = 360  # 6 minutes maximum transit time
        self.min_transit_time = 30   # 30 seconds minimum transit time

    def register_camera_detection(self, camera_id, person_id, features, timestamp):
        global_id = self._match_or_create_global_id(camera_id, person_id, features, timestamp)
        
        if global_id not in self.appearance_sequence:
            self.appearance_sequence[global_id] = []
        
        camera_key = f"{camera_id}"
        
        # Only append if this is a new appearance in this camera
        if not self.appearance_sequence[global_id] or \
           self.appearance_sequence[global_id][-1]['camera'] != camera_key:
            self.appearance_sequence[global_id].append({
                'camera': camera_key,
                'timestamp': timestamp
            })

    def _match_or_create_global_id(self, camera_id, person_id, features, timestamp):
        camera_key = f"{camera_id}_{person_id}"
        
        if camera_key in self.global_identities:
            return self.global_identities[camera_key]
        
        best_match = None
        best_score = 0
        
        for global_id, stored_features in self.feature_database.items():
            last_appearance = self.appearance_sequence[global_id][-1]
            last_camera = last_appearance['camera']
            time_diff = timestamp - last_appearance['timestamp']
            
            # Different thresholds for same-camera vs cross-camera matching
            if camera_id == last_camera:
                threshold = self.same_camera_threshold
                # Skip if too much time has passed in same camera
                if time_diff > 30:  # 30 seconds for same camera
                    continue
            else:
                threshold = self.cross_camera_threshold
                # Validate transit time for cross-camera matches
                if not (self.min_transit_time <= time_diff <= self.max_transit_time):
                    continue
                
                # Only allow Camera_1 to Camera_2 transitions
                if last_camera == "Camera_2" and camera_id == "1":
                    continue
            
            similarity = 1 - cdist(features.reshape(1, -1), 
                                 stored_features.reshape(1, -1), 
                                 metric='cosine')[0][0]
            
            if similarity > threshold and similarity > best_score:
                best_match = global_id
                best_score = similarity
        
        if best_match is None:
            best_match = len(self.global_identities)
            self.feature_database[best_match] = features
            
        self.global_identities[camera_key] = best_match
        return best_match

    def analyze_camera_transitions(self):
        cam1_to_cam2 = 0
        valid_transitions = []
        
        for global_id, appearances in self.appearance_sequence.items():
            sorted_appearances = sorted(appearances, key=lambda x: x['timestamp'])
            
            # Track transitions
            for i in range(len(sorted_appearances) - 1):
                current = sorted_appearances[i]
                next_app = sorted_appearances[i + 1]
                
                if (current['camera'] == '1' and 
                    next_app['camera'] == '2'):
                    time_diff = next_app['timestamp'] - current['timestamp']
                    if self.min_transit_time <= time_diff <= self.max_transit_time:
                        cam1_to_cam2 += 1
                        valid_transitions.append((global_id, time_diff))
        
        return {
            'camera1_to_camera2': cam1_to_cam2,
            'valid_transitions': valid_transitions,
            'unique_camera1': len(set(id for id, appearances in self.appearance_sequence.items()
                               if any(app['camera'] == '1' for app in appearances))),
            'unique_camera2': len(set(id for id, appearances in self.appearance_sequence.items()
                               if any(app['camera'] == '2' for app in appearances)))
        }

class PersonTracker:
    def __init__(self, video_path, output_base_dir="tracking_results"):
        self.video_name = Path(video_path).stem
        self.output_dir = os.path.join(output_base_dir, self.video_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize models
        self.detector = YOLO("yolov8x.pt")  # Using YOLOv8x for better detection
        self.reid_model = torchreid.models.build_model(
            name='osnet_ain_x1_0',  # Using improved ReID model
            num_classes=1000,
            pretrained=True
        )
        self.reid_model = self.reid_model.cuda() if torch.cuda.is_available() else self.reid_model
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

        # Enhanced tracking parameters
        self.min_detection_confidence = 0.3  # Lower confidence threshold
        self.min_track_length = 2  # Reduced minimum track length
        self.max_disappeared = self.fps * 5  # Increased disappearance allowance
        
        # Door areas (adjusted based on your description)
        self.doors = {
            '1': [(1030, 0), (1700, 560)],
            '2': [(400, 0), (800, 470)]
        }

    def extract_features(self, person_crop):
        """Enhanced feature extraction with improved preprocessing"""
        try:
            # Handle poor quality images
            if person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
                return None
                
            # Apply contrast enhancement
            lab = cv2.cvtColor(person_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Improved preprocessing
            img = cv2.resize(enhanced, (128, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Enhanced normalization with padding
            img = np.pad(img, ((2,2), (2,2), (0,0)), mode='edge')
            img = cv2.resize(img, (128, 256))
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std

            # Convert to tensor
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            if torch.cuda.is_available():
                img = img.cuda()

            # Extract features
            with torch.no_grad():
                features = self.reid_model(img)
            return features.cpu().numpy()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def validate_detection(self, bbox, camera_id):
        # Ensure camera_id is a string
        camera_id = str(camera_id)
        """Enhanced detection validation"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Basic size checks
        if width < 15 or height < 30:  # More lenient size thresholds
            return False
        if width > height * 1.2:  # More lenient aspect ratio
            return False
        if width/height > 1.0:  # More lenient width/height ratio
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
        if y1 > door_y1 + door_height * 0.4:  # More lenient top area check
            return False

        return True

    def process_video(self):
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_time = frame_count / self.fps
            frame_count += 1

            # Detect persons using YOLO
            results = self.detector(frame)
            
            # Process detections
            detections = []
            for result in results[0].boxes.data:
                if len(result) >= 6:  # Ensure we have class and confidence
                    bbox = result[:4].cpu().numpy()
                    conf = float(result[4])
                    cls = int(result[5])
                    
                    if cls == 0 and conf > self.min_detection_confidence:  # person class
                        detections.append((bbox, conf))

            # Update tracking
            self._update_tracks(frame, detections, frame_time)

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

    def _update_tracks(self, frame, detections, frame_time):
        """Update tracks with improved matching and validation"""
        # Process each detection
        for bbox, conf in detections:
            if not self.validate_detection(bbox, self.video_name.split('_')[1]):
                continue

            # Extract features
            person_crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            if person_crop.size == 0:
                continue
                
            features = self.extract_features(person_crop)
            if features is None:
                continue

            # Update timestamps and features
            best_match_id = None
            best_match_score = 0.4  # Much lower threshold for matching

            # Match with existing tracks
            for track_id, track_info in self.active_tracks.items():
                if track_info['disappeared'] > self.max_disappeared:
                    continue

                similarity = 1 - cdist(features.reshape(1, -1), 
                                     track_info['features'].reshape(1, -1), 
                                     metric='cosine')[0][0]
                
                if similarity > best_match_score:
                    best_match_id = track_id
                    best_match_score = similarity

            if best_match_id is not None:
                # Update existing track
                self.active_tracks[best_match_id].update({
                    'box': bbox,
                    'features': features,
                    'last_seen': frame_time,
                    'disappeared': 0
                })
                self.person_timestamps[best_match_id]['last_appearance'] = frame_time
            else:
                # Create new track
                new_id = self.next_id
                self.next_id += 1
                
                self.active_tracks[new_id] = {
                    'box': bbox,
                    'features': features,
                    'last_seen': frame_time,
                    'disappeared': 0
                }
                
                self.person_features[new_id] = features
                self.person_timestamps[new_id] = {
                    'first_appearance': frame_time,
                    'last_appearance': frame_time
                }

        # Update disappeared counts
        for track_id in list(self.active_tracks.keys()):
            if track_id not in [match_id for match_id in 
                              [best_match_id for best_match_id in 
                               [None] if best_match_id is not None]]:
                self.active_tracks[track_id]['disappeared'] += 1

    def save_tracking_results(self):
        """Save tracking results with enhanced filtering"""
        results = {
            'video_name': self.video_name,
            'total_persons': len([id for id, info in self.active_tracks.items()
                                if info['disappeared'] <= self.max_disappeared]),
            'video_metadata': {
                'width': self.frame_width,
                'height': self.frame_height,
                'fps': self.fps
            },
            'persons': {}
        }

        # Only include tracks that meet minimum length requirement
        valid_tracks = {
            id: info for id, info in self.active_tracks.items()
            if info['disappeared'] <= self.max_disappeared
        }

        for person_id in valid_tracks.keys():
            results['persons'][person_id] = {
                'first_appearance': self.person_timestamps[person_id]['first_appearance'],
                'last_appearance': self.person_timestamps[person_id]['last_appearance'],
                'duration': (self.person_timestamps[person_id]['last_appearance'] -
                           self.person_timestamps[person_id]['first_appearance'])
            }

        return results

def process_video_directory(input_dir, output_base_dir="tracking_results"):
    global_tracker = GlobalTracker()
    results = {}

    video_files = sorted(Path(input_dir).glob("Camera_*_*.mp4"))

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

            # Register detections with global tracker
            for person_id, features in tracker.person_features.items():
                timestamp = tracker.person_timestamps[person_id]['first_appearance']
                global_tracker.register_camera_detection(camera_id, person_id, features, timestamp)

            print(f"Completed processing {video_path}")
            print(f"Found {video_results['total_persons']} potential persons")

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            continue

    # Get cross-camera analysis
    transition_analysis = global_tracker.analyze_camera_transitions()
    
    # Create comprehensive summary
    summary = {
        'per_camera_statistics': {
            'Camera 1': transition_analysis['unique_camera1'],
            'Camera 2': transition_analysis['unique_camera2']
        },
        'cross_camera_analysis': {
            'camera1_to_camera2': transition_analysis['camera1_to_camera2'],
            'transitions': transition_analysis['valid_transitions']
        }
    }

    # Save summary
    summary_path = os.path.join(output_base_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    # Print comprehensive summary
    print("\nProcessing Summary:")
    print("\nPer Camera Statistics:")
    print(f"Camera 1: {summary['per_camera_statistics']['Camera 1']} unique individuals")
    print(f"Camera 2: {summary['per_camera_statistics']['Camera 2']} unique individuals")
    print("\nCross-Camera Transitions:")
    print(f"Camera 1 to Camera 2: {summary['cross_camera_analysis']['camera1_to_camera2']} individuals")
    
    if transition_analysis['valid_transitions']:
        transit_times = [t[1] for t in transition_analysis['valid_transitions']]
        avg_transit = sum(transit_times) / len(transit_times)
        print(f"Average transit time: {avg_transit:.1f} seconds")

    return summary

def main():
    video_dir = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
                            'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')

    try:
        summary = process_video_directory(video_dir)
        print("\nResults saved successfully!")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == '__main__':
    main()
