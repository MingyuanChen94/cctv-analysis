# src/cctv_analysis/detector.py

import torch
from ultralytics import YOLO
import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_path=None, device="cuda"):
        """
        Initialize YOLOv8x6 detector
        Args:
            model_path: Path to model weights (if None, downloads from ultralytics)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.device = "cpu"
            print("GPU not available, using CPU")

        try:
            # Initialize YOLOv8x6 model
            if model_path:
                self.model = YOLO(model_path)
            else:
                self.model = YOLO('yolov8x6.pt')
            
            # Move model to specified device
            self.model.to(self.device)
            
            print("Successfully loaded YOLOv8x6 model")
            
        except Exception as e:
            raise RuntimeError(f"Error initializing YOLOv8x6 model: {e}")

    def detect(self, img, conf_thresh=0.3):
        """
        Detect persons in image
        Args:
            img: OpenCV image in BGR format
            conf_thresh: Confidence threshold
        Returns:
            List of detections, each in (x1, y1, x2, y2, confidence) format
        """
        try:
            # Check for valid image
            if img is None or img.size == 0:
                print("Invalid image input")
                return []

            # Run inference
            results = self.model(img, conf=conf_thresh, classes=0)  # class 0 is person
            
            # Extract person detections
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.conf.item() >= conf_thresh:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf.item()
                        
                        # Filter out unrealistic detections
                        w = x2 - x1
                        h = y2 - y1
                        if w < 20 or h < 40 or w/h > 2 or h/w > 4:
                            continue
                            
                        detections.append([x1, y1, x2, y2, conf])

            return np.array(detections)

        except Exception as e:
            print(f"Error in detection: {e}")
            return []

    def draw_detections(self, img, detections, show_conf=True, color=(0, 255, 0)):
        """
        Draw detection boxes on image
        Args:
            img: OpenCV image
            detections: List of detections from detect()
            show_conf: Whether to show confidence scores
            color: BGR color tuple for boxes
        Returns:
            Image with drawn detections
        """
        img_draw = img.copy()
        for det in detections:
            x1, y1, x2, y2, conf = map(float, det)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence score
            if show_conf:
                label = f"person {conf:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    img_draw,
                    (x1, y1 - label_h - baseline),
                    (x1 + label_w, y1),
                    color,
                    -1
                )
                cv2.putText(
                    img_draw,
                    label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )
        
        return img_draw
