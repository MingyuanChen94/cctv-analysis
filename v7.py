import os
import cv2
import numpy as np
from ultralytics import YOLO
import torchreid
import torch
from collections import defaultdict
import time
from datetime import datetime
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import scipy.spatial.distance as distance
from pathlib import Path
import shutil
import json
from bytetracker import BYTETracker

class TrackingState:
    ACTIVE = 'active'
    OCCLUDED = 'occluded'
    TENTATIVE = 'tentative'
    LOST = 'lost'

class GlobalTracker:
    def __init__(self):
        self.global_identities = {}
        self.appearance_sequence = {}
        self.feature_database = {}
        self.similarity_threshold = 0.75

    def register_camera_detection(self, camera_id, person_id, features, timestamp):
        global_id = self._match_or_create_global_id(camera_id, person_id, features)
        
        if global_id not in self.appearance_sequence:
            self.appearance_sequence[global_id] = []
        
        camera_key = f"Camera_{camera_id}"
        
        if not self.appearance_sequence[global_id] or \
           self.appearance_sequence[global_id][-1]['camera'] != camera_key:
            self.appearance_sequence[global_id].append({
                'camera': camera_key,
                'timestamp': timestamp
            })

    def _match_or_create_global_id(self, camera_id, person_id, features):
        camera_key = f"{camera_id}_{person_id}"
        
        if camera_key in self.global_identities:
            return self.global_identities[camera_key]
        
        best_match = None
        best_score = 0
        
        for global_id, stored_features in self.feature_database.items():
            similarity = 1 - distance.cosine(features.flatten(), stored_features.flatten())
            if similarity > self.similarity_threshold and similarity > best_score:
                best_match = global_id
                best_score = similarity
        
        if best_match is None:
            best_match = len(self.global_identities)
            self.feature_database[best_match] = features
            
        self.global_identities[camera_key] = best_match
        return best_match

    def analyze_camera_transitions(self):
        cam1_to_cam2 = 0
        cam2_to_cam1 = 0
        
        for global_id, appearances in self.appearance_sequence.items():
            sorted_appearances = sorted(appearances, key=lambda x: x['timestamp'])
            
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

class PersonTracker:
    def __init__(self, video_path, output_base_dir="tracking_results"):
        self.video_name = Path(video_path).stem
        self.output_dir = os.path.join(output_base_dir, self.video_name)
        self.images_dir = os.path.join(self.output_dir, "person_images")
        self.features_dir = os.path.join(self.output_dir, "person_features")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

        # Initialize models
        self.detector = YOLO("yolov8x.pt")
        self.reid_model = torchreid.models.build_model(
            name='osnet_x1_0',
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

        # Initialize ByteTracker
        self.byte_tracker = BYTETracker(
            track_thresh=0.45,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=self.fps
        )

        # Enhanced tracking parameters
        self.min_detection_confidence = 0.6
        self.feature_weight = 0.6
        self.position_weight = 0.2
        self.motion_weight = 0.2
        self.similarity_threshold = 0.8
        self.feature_update_momentum = 0.8
        self.min_track_length = 5
        self.max_feature_distance = 0.3
        self.max_disappeared = self.fps * 2
        self.max_lost_age = self.fps * 30

        # Initialize tracking containers
        self.active_tracks = {}
        self.lost_tracks = {}
        self.person_features = {}
        self.person_timestamps = {}
        self.appearance_history = defaultdict(list)
        self.next_id = 0

    def extract_features(self, person_crop):
        try:
            img = cv2.resize(person_crop, (128, 256))
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1).unsqueeze(0)
            if torch.cuda.is_available():
                img = img.cuda()

            with torch.no_grad():
                features = self.reid_model(img)
            return features.cpu().numpy()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def calculate_box_center(self, box):
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def calculate_velocity(self, current_box, previous_box):
        current_center = self.calculate_box_center(current_box)
        previous_center = self.calculate_box_center(previous_box)
        return [current_center[0] - previous_center[0],
                current_center[1] - previous_center[1]]

    def predict_next_position(self, box, velocity):
        center = self.calculate_box_center(box)
        predicted_center = [center[0] + velocity[0], center[1] + velocity[1]]
        width = box[2] - box[0]
        height = box[3] - box[1]
        return [predicted_center[0] - width/2, predicted_center[1] - height/2,
                predicted_center[0] + width/2, predicted_center[1] + height/2]

    def calculate_iou(self, box1, box2):
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
        self.appearance_history[track_id].append(features)
        if len(self.appearance_history[track_id]) > self.min_track_length:
            self.appearance_history[track_id].pop(0)

        if track_id in self.person_features:
            current_features = self.person_features[track_id]
            momentum = self.feature_update_momentum
            
            dist = distance.cosine(features.flatten(), current_features.flatten())
            if dist > self.max_feature_distance:
                momentum = 0.95
                
            updated_features = momentum * current_features + (1 - momentum) * features
            self.person_features[track_id] = updated_features
        else:
            self.person_features[track_id] = features

    def process_detections(self, frame, detections):
        try:
            byte_dets = []
            processed_detections = []
            
            for box, conf in detections:
                if conf < self.min_detection_confidence:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                byte_dets.append([x1, y1, x2 - x1, y2 - y1, conf])
                processed_detections.append({
                    'box': [x1, y1, x2, y2],
                    'conf': conf
                })
            
            if not byte_dets:
                return [], []
                
            byte_dets = np.array(byte_dets)
            online_targets = self.byte_tracker.update(
                byte_dets,
                [self.frame_height, self.frame_width],
                [self.frame_height, self.frame_width]
            )
            
            return online_targets, processed_detections
            
        except Exception as e:
            print(f"Error in process_detections: {str(e)}")
            return [], []

    def update_tracks(self, frame, detections, frame_time):
        try:
            online_targets, processed_detections = self.process_detections(frame, detections)
            
            if not online_targets:
                return
            
            tracked_ids = set()
            for target in online_targets:
                track_id = target.track_id
                tracked_ids.add(track_id)
                
                tlwh = target.tlwh
                box = [int(tlwh[0]), int(tlwh[1]), 
                      int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])]
                
                try:
                    person_crop = frame[box[1]:box[3], box[0]:box[2]]
                    if person_crop.size == 0:
                        continue
                        
                    features = self.extract_features(person_crop)
                    if features is None:
                        continue
                        
                    if track_id in self.active_tracks:
                        self._update_existing_track(track_id, box, features, frame_time, frame)
                    else:
                        self._create_new_track(box, features, frame_time, frame)
                        
                except Exception as e:
                    print(f"Error processing target {track_id}: {str(e)}")
                    continue
            
            self._update_lost_tracks(tracked_ids, frame_time)
            
        except Exception as e:
            print(f"Error in update_tracks: {str(e)}")

    def _update_existing_track(self, track_id, box, features, frame_time, frame):
        try:
            if track_id not in self.active_tracks:
                return

            stored_features = self.active_tracks[track_id]['features']
            feature_dist = distance.cosine(features.flatten(), stored_features.flatten())
            
            if feature_dist > self.max_feature_distance:
                self._create_new_track(box, features, frame_time, frame)
                return
            
            self.active_tracks[track_id].update({
                'previous_box': self.active_tracks[track_id].get('box', box),
                'box': box,
                'features': features,
                'last_seen': frame_time,
                'disappeared': 0,
                'state': TrackingState.ACTIVE
            })
            
            self.update_feature_history(track_id, features)
            self.person_timestamps[track_id]['last_appearance'] = frame_time
            self.save_person_image(track_id, frame[box[1]:box[3], box[0]:box[2]])
            
        except Exception as e:
            print(f"Error in _update_existing_track: {str(e)}")

    def _create_new_track(self, box, features, frame_time, frame):
        try:
            track_id = self.next_id
            self.next_id += 1

            self.active_tracks[track_id] = {
                'state': TrackingState.TENTATIVE,
                'box': box,
                'features': features,
                'last_seen': frame_time,
                'disappeared': 0,
                'velocity': [0, 0]
            }

            self.person_features[track_id] = features
            self.appearance_history[track_id] = [features]
            self.person_timestamps[track_id] = {
                'first_appearance': frame_time,
                'last_appearance': frame_time
            }

            self.save_person_image(track_id, frame[box[1]:box[3], box[0]:box[2]])
            
        except Exception as e:
            print(f"Error in _create_new_track: {str(e)}")

    def _update_lost_tracks(self, tracked_ids, frame_time):
        try:
            for track_id in list(self.active_tracks.keys()):
                if track_id not in tracked_ids:
                    track_info = self.active_tracks[track_id]
                    track_info['disappeared'] += 1
                    
                    if track_info['disappeared'] > self.max_disappeared:
                        self.lost_tracks[track_id] = {
                            'features': track_info['features'],
                            'box': track_info['box'],
                            'last_seen': track_info['last_seen']
                        }
                        del self.active_tracks[track_id]
            
            for track_id in list(self.lost_tracks.keys()):
                if frame_time - self.lost_tracks[track_id]['last_seen'] > self.max_lost_age:
                    del self.lost_tracks[track_id]
                    
        except Exception as e:
            print(f"Error in _update_lost_tracks: {str(e)}")

    def save_person_image(self, person_id, frame):
        person_dir = os.path.join(self.images_dir, f"person_{person_id}")
        os.makedirs(person_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(person_dir, f"{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        return image_path

    def save_person_features(self, person_id, features, frame_time):
        os.makedirs(self.features_dir, exist_ok=True)
        feature_path = os.path.join(self.features_dir, f"person_{person_id}_features.npz")
        np.savez(feature_path,
                 features=features,
                 timestamp=frame_time,
                 video_name=self.video_name)
        return feature_path

    def process_video(self):
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                frame_time = frame_count / self.fps
                frame_count += 1
                
                # Run YOLO detection
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
                    
        except Exception as e:
            print(f"Error in process_video: {str(e)}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            
        return self.generate_report()

    def save_tracking_results(self):
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
            appearances = []
            current_appearance = None
            
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

        results_path = os.path.join(self.output_dir, "tracking_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        return results

    def generate_report(self):
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


def process_video_directory(input_dir, output_base_dir="tracking_results"):
    global_tracker = GlobalTracker()
    results = {}
    per_camera_stats = defaultdict(int)

    os.makedirs(output_base_dir, exist_ok=True)

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(input_dir).glob(f'*{ext}')))

    for video_path in video_files:
        print(f"\nProcessing video: {video_path}")
        camera_id = video_path.stem.split('_')[1]

        try:
            tracker = PersonTracker(str(video_path), output_base_dir)
            tracker.process_video()
            video_results = tracker.save_tracking_results()
            results[video_path.stem] = video_results

            total_persons = len(tracker.person_timestamps)
            per_camera_stats[f"Camera_{camera_id}"] = total_persons

            for person_id, features in tracker.person_features.items():
                timestamp = tracker.person_timestamps[person_id]['first_appearance']
                global_tracker.register_camera_detection(camera_id, person_id, features, timestamp)

            print(f"Completed processing {video_path}")
            print(f"Found {video_results['total_persons']} unique persons")

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            continue

    transition_analysis = global_tracker.analyze_camera_transitions()
    
    summary = {
        'per_camera_statistics': dict(per_camera_stats),
        'cross_camera_analysis': transition_analysis,
        'total_unique_global': len(global_tracker.global_identities)
    }

    summary_path = os.path.join(output_base_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print("\nProcessing Summary:")
    print("\nPer Camera Statistics:")
    for camera, count in per_camera_stats.items():
        print(f"{camera}: {count} unique individuals")
        
    print("\nCross-Camera Transitions:")
    print(f"Camera 1 to Camera 2: {transition_analysis['camera1_to_camera2']} individuals")
    print(f"Camera 2 to Camera 1: {transition_analysis['camera2_to_camera1']} individuals")
    print(f"\nTotal Unique Global Individuals: {summary['total_unique_global']}")

    return summary


if __name__ == '__main__':
    # Set the working directory - modify this path as needed
    # working_directory = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
    #                                'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')
    working_directory = os.path.join('Users', 'chenm', 'Library', 'CloudStorage', 'UniversityofExeter',
                                   'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')
    
    # Process all videos in the directory
    summary = process_video_directory(working_directory)
    print("\nResults saved successfully!")