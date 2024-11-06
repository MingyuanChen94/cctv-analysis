import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from tqdm import tqdm

from cctv_analysis.utils.config import Config
from cctv_analysis.utils.data_types import PersonDetection, DetectionDatabase
from cctv_analysis.person_tracker import PersonTracker
from cctv_analysis.utils.logger import setup_logger

class CameraProcessor:
    """Handles video processing and person detection"""
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("CameraProcessor")
        self.tracker = PersonTracker(config)
        self.detection_db = DetectionDatabase()
        
        # Ensure output directories exist
        self.output_dir = Path("output")
        self.tracks_dir = self.output_dir / "tracks"
        self.analysis_dir = self.output_dir / "analysis"
        
        for dir_path in [self.tracks_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process_video(self, video_path: str, camera_id: int):
        """Process video with batch processing"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        self.logger.info(f"Processing video from camera {camera_id}: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_batch = []
        frame_ids = []
        
        try:
            for frame_idx in tqdm(range(total_frames), 
                                desc=f"Processing camera {camera_id}"):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)
                frames_batch.append(processed_frame)
                frame_ids.append(frame_idx)
                
                # Process batch when ready
                if (len(frames_batch) == self.config.processing.batch_size or 
                    frame_idx == total_frames - 1):
                    batch_detections = self._process_batch(
                        frames_batch, frame_ids, fps, camera_id
                    )
                    for detection in batch_detections:
                        self.detection_db.add_detection(camera_id, detection)
                    frames_batch = []
                    frame_ids = []
            
            # Save tracking results
            self._save_tracking_results(camera_id)
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            raise
        finally:
            cap.release()
        
        return self.detection_db.get_camera_detections(camera_id)

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for detection"""
        img = cv2.resize(
            frame,
            (self.config.detector.input_size[1],
             self.config.detector.input_size[0])
        )
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = torch.from_numpy(img).float()
        img /= 255.0  # normalize to [0, 1]
        return img

    def _process_batch(self, frames: List[torch.Tensor],
                      frame_ids: List[int],
                      fps: float,
                      camera_id: int) -> List[PersonDetection]:
        """Process a batch of frames"""
        device = torch.device(self.config.processing.device)
        batch = torch.stack(frames).to(device)
        
        # Run detection and tracking
        detections = self.tracker.process_batch(
            batch, frame_ids, fps
        )
        
        return detections

    def _save_tracking_results(self, camera_id: int):
        """Save tracking results to output directory"""
        output_file = self.tracks_dir / f"camera_{camera_id}_tracks.npz"
        
        # Get all detections for this camera
        detections = self.detection_db.get_camera_detections(camera_id)
        
        # Convert to numpy arrays for efficient storage
        track_ids = np.array([d.track_id for d in detections])
        frame_ids = np.array([d.frame_id for d in detections])
        bboxes = np.array([d.bbox for d in detections])
        timestamps = np.array([d.timestamp.timestamp() for d in detections])
        confidences = np.array([d.confidence for d in detections])
        reid_features = np.array([d.reid_features for d in detections])
        
        # Save to compressed numpy format
        np.savez_compressed(
            output_file,
            track_ids=track_ids,
            frame_ids=frame_ids,
            bboxes=bboxes,
            timestamps=timestamps,
            confidences=confidences,
            reid_features=reid_features
        )
        
        self.logger.info(f"Saved tracking results to {output_file}")
