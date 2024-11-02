import datetime
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

from .demographic_analyzer import DemographicAnalyzer
from .person_detector import PersonDetector
from .person_tracker import PersonTracker
from .utils import setup_logging


class CameraProcessor:
    """Main class for processing CCTV footage and tracking individuals across cameras."""

    def __init__(self, config: dict):
        """
        Initialize the camera processor with configuration.

        Args:
            config (dict): Configuration dictionary containing processing parameters
        """
        self.config = config
        self.detector = PersonDetector(config["detector"])
        self.tracker = PersonTracker(config["tracker"])
        self.demographic_analyzer = DemographicAnalyzer(config["demographics"])
        self.logger = setup_logging(__name__, config["logging"])

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[Dict]:
        """
        Process a single frame from the video feed.

        Args:
            frame (np.ndarray): Input frame
            timestamp (float): Frame timestamp

        Returns:
            List[Dict]: List of detected persons with their features and demographics
        """
        try:
            # Detect persons in frame
            detections = self.detector.detect(frame)

            # Track detected persons
            tracked_persons = self.tracker.update(detections)

            # Analyze demographics for each tracked person
            results = []
            for person in tracked_persons:
                person_roi = self._extract_roi(frame, person["bbox"])
                if person_roi is None:
                    continue

                demographics = self.demographic_analyzer.analyze(person_roi)

                results.append(
                    {
                        "timestamp": timestamp,
                        "track_id": person["track_id"],
                        "bbox": person["bbox"],
                        "features": person["features"],
                        **demographics,
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return []

    def process_video(self, video_path: str, camera_id: int) -> List[Dict]:
        """
        Process entire video file.

        Args:
            video_path (str): Path to video file
            camera_id (int): Camera identifier

        Returns:
            List[Dict]: All detections from the video
        """
        self.logger.info(f"Processing video from camera {camera_id}: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_detections = self.process_frame(frame, timestamp)

            for detection in frame_detections:
                detection["camera_id"] = camera_id
                detections.append(detection)

        cap.release()
        return detections

    def analyze_traffic_flow(
        self, camera1_detections: List[Dict], camera2_detections: List[Dict]
    ) -> pd.DataFrame:
        """
        Analyze traffic flow between two cameras.

        Args:
            camera1_detections (List[Dict]): Detections from first camera
            camera2_detections (List[Dict]): Detections from second camera

        Returns:
            pd.DataFrame: Analysis results
        """
        matches = []

        for det1 in camera1_detections:
            for det2 in camera2_detections:
                if det2["timestamp"] > det1[
                    "timestamp"
                ] and self.tracker.match_features(det1["features"], det2["features"]):
                    matches.append(
                        {
                            "camera1_time": det1["timestamp"],
                            "camera2_time": det2["timestamp"],
                            "time_difference": det2["timestamp"] - det1["timestamp"],
                            "track_id": det1["track_id"],
                            "age": det1["age"],
                            "gender": det1["gender"],
                        }
                    )

        return pd.DataFrame(matches)

    def _extract_roi(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Extract region of interest from frame using bounding box."""
        try:
            x, y, w, h = bbox
            return frame[y : y + h, x : x + w]
        except Exception as e:
            self.logger.error(f"Error extracting ROI: {str(e)}")
            return None


def process_surveillance_footage(
    camera1_path: str, camera2_path: str, config_path: str = "config/config.yaml"
) -> pd.DataFrame:
    """
    Main function to process surveillance footage from two cameras.

    Args:
        camera1_path (str): Path to camera 1 footage
        camera2_path (str): Path to camera 2 footage
        config_path (str): Path to configuration file

    Returns:
        pd.DataFrame: Analysis results
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize processor
    processor = CameraProcessor(config)

    # Process both camera feeds
    camera1_detections = processor.process_video(camera1_path, camera_id=1)
    camera2_detections = processor.process_video(camera2_path, camera_id=2)

    # Analyze traffic flow
    results = processor.analyze_traffic_flow(camera1_detections, camera2_detections)

    return results
