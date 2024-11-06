import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
from datetime import datetime

from cctv_analysis.utils.data_types import PersonDetection, PersonMatch
from cctv_analysis.utils.logger import setup_logger

class Visualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.logger = setup_logger("Visualizer")
        self.colors = self._generate_colors(100)  # Generate colors for tracking visualization
        
        # Create output directories
        self.tracks_viz_dir = self.output_dir / "visualizations" / "tracks"
        self.analysis_viz_dir = self.output_dir / "visualizations" / "analysis"
        for dir_path in [self.tracks_viz_dir, self.analysis_viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _generate_colors(self, n: int) -> List[tuple]:
        """Generate n distinct colors for visualization"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.9
            value = 0.9
            # Convert HSV to RGB
            rgb = plt.cm.hsv(hue)[:3]
            # Convert to BGR for OpenCV
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[PersonDetection]) -> np.ndarray:
        """Draw detection boxes and IDs on frame"""
        frame_viz = frame.copy()
        
        for det in detections:
            color = self.colors[det.track_id % len(self.colors)]
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Draw bounding box
            cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            text = f"ID: {det.track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame_viz, (x1, y1 - text_size[1] - 4),
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame_viz, text, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame_viz
    
    def plot_transitions(self, matches: List[PersonMatch], 
                        save_name: str = "transitions.png"):
        """Plot transition times between cameras"""
        transition_times = [match.transition_time for match in matches]
        similarity_scores = [match.similarity_score for match in matches]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot transition time histogram
        ax1.hist(transition_times, bins=30, edgecolor='black')
        ax1.set_title("Distribution of Transition Times Between Cameras")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Count")
        
        # Plot similarity scores
        ax2.hist(similarity_scores, bins=30, edgecolor='black')
        ax2.set_title("Distribution of ReID Similarity Scores")
        ax2.set_xlabel("Similarity Score")
        ax2.set_ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(self.analysis_viz_dir / save_name)
        plt.close()
    
    def plot_activity_timeline(self, detections1: List[PersonDetection],
                             detections2: List[PersonDetection],
                             save_name: str = "activity_timeline.png"):
        """Plot activity timeline for both cameras"""
        # Extract timestamps
        times1 = [d.timestamp for d in detections1]
        times2 = [d.timestamp for d in detections2]
        
        # Create hourly bins
        min_time = min(min(times1), min(times2))
        max_time = max(max(times1), max(times2))
        
        plt.figure(figsize=(15, 6))
        
        # Plot histograms
        plt.hist(times1, bins=50, alpha=0.5, label='Camera 1')
        plt.hist(times2, bins=50, alpha=0.5, label='Camera 2')
        
        plt.title("Activity Timeline")
        plt.xlabel("Time")
        plt.ylabel("Number of Detections")
        plt.legend()
        
        plt.savefig(self.analysis_viz_dir / save_name)
        plt.close()
    
    def save_tracking_video(self, video_path: Path, 
                          detections: List[PersonDetection],
                          output_name: Optional[str] = None):
        """Create visualization video with tracking results"""
        if output_name is None:
            output_name = f"tracking_viz_{datetime.now():%Y%m%d_%H%M%S}.mp4"
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Could not open video: {video_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        output_path = self.tracks_viz_dir / output_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Group detections by frame
        detections_by_frame = {}
        for det in detections:
            if det.frame_id not in detections_by_frame:
                detections_by_frame[det.frame_id] = []
            detections_by_frame[det.frame_id].append(det)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Draw detections for current frame
            frame_detections = detections_by_frame.get(frame_idx, [])
            frame_viz = self.draw_detections(frame, frame_detections)
            
            # Write frame
            out.write(frame_viz)
            frame_idx += 1
        
        cap.release()
        out.release()
