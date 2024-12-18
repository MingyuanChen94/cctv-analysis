import torch
from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

class PersonDetector:
    """Person detection module using YOLOv8."""
    
    def __init__(self, model_path: str = "models/detector/yolov8x6.pt", conf_thresh: float = 0.5):
        """
        Initialize the detector with YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 model weights
            conf_thresh: Confidence threshold for detections
        """
        self.conf_thresh = conf_thresh
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect persons in a frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            
        Returns:
            List of tuples containing bounding boxes (x1, y1, x2, y2) and confidence scores
        """
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for r in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = r
            
            # Filter for person class (typically class 0 in COCO)
            if cls == 0 and conf >= self.conf_thresh:
                detections.append(((int(x1), int(y1), int(x2), int(y2)), float(conf)))
                
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Tuple[Tuple[int, int, int, int], float]]]:
        """
        Batch detection on multiple frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection results for each frame
        """
        results = self.model(frames, verbose=False)
        batch_detections = []
        
        for result in results:
            frame_detections = []
            for r in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = r
                if cls == 0 and conf >= self.conf_thresh:
                    frame_detections.append(((int(x1), int(y1), int(x2), int(y2)), float(conf)))
            batch_detections.append(frame_detections)
            
        return batch_detections
