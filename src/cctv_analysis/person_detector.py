import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


class PersonDetector:
    """
    Person detection class using YOLOv5 model.
    Supports multiple detection backends with easy switching between them.
    """

    SUPPORTED_BACKENDS = ["yolov5", "ssd", "hog"]

    def __init__(self, config: dict):
        """
        Initialize the person detector.

        Args:
            config (dict): Configuration dictionary containing:
                - backend: Detection backend ('yolov5', 'ssd', 'hog')
                - model_path: Path to model weights (for deep learning backends)
                - confidence_threshold: Detection confidence threshold
                - device: Device to run inference on ('cpu' or 'cuda')
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.backend = config.get("backend", "yolov5")
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize the detection backend."""
        try:
            if self.backend == "yolov5":
                self.model = torch.hub.load(
                    "ultralytics/yolov5", "yolov5s", pretrained=True
                )
                self.model.to(self.device)
                self.model.eval()

            elif self.backend == "ssd":
                self.model = cv2.dnn.readNet(
                    str(Path(self.config["model_path"]) / "ssd_weights.pb"),
                    str(Path(self.config["model_path"]) / "ssd_config.pbtxt"),
                )
                if self.device == "cuda":
                    self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

            elif self.backend == "hog":
                self.model = cv2.HOGDescriptor()
                self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        except Exception as e:
            self.logger.error(f"Error initializing {self.backend} detector: {str(e)}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, any]]:
        """
        Detect persons in the given frame.

        Args:
            frame (np.ndarray): Input frame

        Returns:
            List[Dict]: List of detections, each containing:
                - bbox: Tuple[int, int, int, int] (x, y, width, height)
                - confidence: float
                - features: np.ndarray (extracted features for re-identification)
        """
        if frame is None:
            return []

        try:
            if self.backend == "yolov5":
                return self._detect_yolov5(frame)
            elif self.backend == "ssd":
                return self._detect_ssd(frame)
            elif self.backend == "hog":
                return self._detect_hog(frame)

        except Exception as e:
            self.logger.error(f"Error during detection: {str(e)}")
            return []

    def _detect_yolov5(self, frame: np.ndarray) -> List[Dict[str, any]]:
        """YOLOv5 detection implementation."""
        results = self.model(frame)
        detections = []

        # Process YOLOv5 results
        for *xyxy, conf, cls in results.xyxy[0]:
            if int(cls) == 0 and conf >= self.confidence_threshold:  # class 0 is person
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1

                detection = {
                    "bbox": (x1, y1, w, h),
                    "confidence": float(conf),
                    "features": self._extract_features(frame[y1:y2, x1:x2]),
                }
                detections.append(detection)

        return detections

    def _detect_ssd(self, frame: np.ndarray) -> List[Dict[str, any]]:
        """SSD detection implementation."""
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [127.5, 127.5, 127.5])
        self.model.setInput(blob)
        detections = self.model.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >= self.confidence_threshold:
                class_id = int(detections[0, 0, i, 1])
                if class_id == 15:  # SSD person class
                    box = detections[0, 0, i, 3:7] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                    )
                    x1, y1, x2, y2 = map(int, box)
                    w, h = x2 - x1, y2 - y1

                    detection = {
                        "bbox": (x1, y1, w, h),
                        "confidence": float(confidence),
                        "features": self._extract_features(frame[y1:y2, x1:x2]),
                    }
                    results.append(detection)

        return results

    def _detect_hog(self, frame: np.ndarray) -> List[Dict[str, any]]:
        """HOG detection implementation."""
        boxes, weights = self.model.detectMultiScale(
            frame, winStride=(8, 8), padding=(4, 4), scale=1.05
        )

        results = []
        for (x, y, w, h), confidence in zip(boxes, weights):
            if confidence >= self.confidence_threshold:
                detection = {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "confidence": float(confidence),
                    "features": self._extract_features(frame[y : y + h, x : x + w]),
                }
                results.append(detection)

        return results

    def _extract_features(self, person_img: np.ndarray) -> np.ndarray:
        """
        Extract features from detected person image for re-identification.

        Args:
            person_img (np.ndarray): Cropped person image

        Returns:
            np.ndarray: Extracted features
        """
        try:
            # Resize to standard size
            resized = cv2.resize(person_img, (128, 256))

            # Convert to float and normalize
            normalized = resized.astype(np.float32) / 255.0

            # Simple feature extraction (can be replaced with more sophisticated methods)
            features = cv2.calcHist(
                [normalized], [0, 1, 2], None, [8, 8, 8], [0, 1, 0, 1, 0, 1]
            )
            features = features.flatten()
            features = features / features.sum()  # L1 normalization

            return features

        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.array([])
