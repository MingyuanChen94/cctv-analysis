import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from .logging_utils import setup_logging

logger = setup_logging(__name__)


class VideoProcessor:
    """Handles video processing operations including reading, writing, and frame manipulation."""

    def __init__(self, output_dir: str = "output/videos"):
        """
        Initialize video processor.

        Args:
            output_dir (str): Directory for output videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def read_video_info(self, video_path: str) -> Dict:
        """
        Get video file information.

        Args:
            video_path (str): Path to video file

        Returns:
            Dict: Video information including:
                - width: Frame width
                - height: Frame height
                - fps: Frames per second
                - frame_count: Total number of frames
                - duration: Video duration in seconds
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }

            # Calculate duration
            info["duration"] = info["frame_count"] / info["fps"]

            cap.release()
            return info

        except Exception as e:
            logger.error(f"Error reading video info: {str(e)}")
            raise

    def frame_generator(
        self, video_path: str
    ) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generate frames from video file with timestamps.

        Args:
            video_path (str): Path to video file

        Yields:
            Tuple[np.ndarray, float]: Frame and its timestamp in milliseconds
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                yield frame, timestamp

        finally:
            cap.release()

    def create_video_writer(
        self, filename: str, frame_size: Tuple[int, int], fps: float = 30.0
    ) -> cv2.VideoWriter:
        """
        Create a video writer object.

        Args:
            filename (str): Output filename
            frame_size (Tuple[int, int]): Frame width and height
            fps (float): Frames per second

        Returns:
            cv2.VideoWriter: Video writer object
        """
        output_path = self.output_dir / filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        return cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    def draw_detections(
        self, frame: np.ndarray, detections: List[Dict], show_demographics: bool = True
    ) -> np.ndarray:
        """
        Draw detection boxes and information on frame.

        Args:
            frame (np.ndarray): Input frame
            detections (List[Dict]): List of detections
            show_demographics (bool): Whether to show demographic information

        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()

        for detection in detections:
            # Get detection info
            bbox = detection["bbox"]
            track_id = detection.get("track_id", None)
            confidence = detection.get("confidence", None)

            # Draw bounding box
            x, y, w, h = map(int, bbox)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw track ID
            if track_id is not None:
                cv2.putText(
                    annotated_frame,
                    f"ID: {track_id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Draw confidence
            if confidence is not None:
                cv2.putText(
                    annotated_frame,
                    f"Conf: {confidence:.2f}",
                    (x, y + h + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Draw demographics
            if show_demographics:
                demo_text = []

                if "age" in detection:
                    demo_text.append(f"Age: {detection['age']}")
                if "gender" in detection:
                    demo_text.append(f"Gender: {detection['gender']}")

                if demo_text:
                    y_offset = h + 30
                    for text in demo_text:
                        cv2.putText(
                            annotated_frame,
                            text,
                            (x, y + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
                        y_offset += 15

        return annotated_frame

    def add_timestamp(
        self, frame: np.ndarray, timestamp: float, position: str = "top-left"
    ) -> np.ndarray:
        """
        Add timestamp to frame.

        Args:
            frame (np.ndarray): Input frame
            timestamp (float): Timestamp in milliseconds
            position (str): Position of timestamp ('top-left', 'top-right',
                          'bottom-left', 'bottom-right')

        Returns:
            np.ndarray: Frame with timestamp
        """
        # Convert timestamp to datetime
        timestamp_dt = datetime.fromtimestamp(timestamp / 1000.0)
        timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Get position coordinates
        height, width = frame.shape[:2]
        positions = {
            "top-left": (10, 30),
            "top-right": (width - 200, 30),
            "bottom-left": (10, height - 10),
            "bottom-right": (width - 200, height - 10),
        }

        pos = positions.get(position, positions["top-left"])

        # Add timestamp to frame
        cv2.putText(
            frame, timestamp_str, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        return frame

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        interval: float = 1.0,
        max_frames: Optional[int] = None,
    ):
        """
        Extract frames from video at specified intervals.

        Args:
            video_path (str): Path to video file
            output_dir (str): Output directory for frames
            interval (float): Interval between frames in seconds
            max_frames (Optional[int]): Maximum number of frames to extract
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * interval)
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_count >= max_frames):
                    break

                if frame_count % frame_interval == 0:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                    frame_filename = f"frame_{frame_count:06d}_{int(timestamp)}.jpg"
                    cv2.imwrite(str(output_dir / frame_filename), frame)

                frame_count += 1

        finally:
            cap.release()

    def create_timelapse(self, input_dir: str, output_file: str, fps: float = 30.0):
        """
        Create timelapse video from extracted frames.

        Args:
            input_dir (str): Directory containing frames
            output_file (str): Output video file path
            fps (float): Frames per second for output video
        """
        input_dir = Path(input_dir)
        frames = sorted(input_dir.glob("*.jpg"))

        if not frames:
            raise ValueError(f"No frames found in {input_dir}")

        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frames[0]))
        height, width = first_frame.shape[:2]

        # Create video writer
        writer = self.create_video_writer(output_file, (width, height), fps)

        try:
            for frame_path in frames:
                frame = cv2.imread(str(frame_path))
                writer.write(frame)

        finally:
            writer.release()


# Example usage
if __name__ == "__main__":
    processor = VideoProcessor()

    # Read video info
    video_path = "path/to/video.mp4"
    info = processor.read_video_info(video_path)
    print("Video Info:", info)

    # Extract frames
    processor.extract_frames(
        video_path=video_path, output_dir="frames", interval=1.0, max_frames=100
    )

    # Create timelapse
    processor.create_timelapse(
        input_dir="frames", output_file="timelapse.mp4", fps=30.0
    )
