import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import colorsys
from datetime import datetime

class Visualizer:
    """Utility class for visualization of detection and tracking results."""
    
    def __init__(self, num_colors: int = 100):
        """
        Initialize visualizer.
        
        Args:
            num_colors: Number of unique colors to generate for visualization
        """
        self.colors = self._generate_colors(num_colors)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2
        
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate evenly spaced colors in HSV space."""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append(tuple(int(255 * x) for x in rgb))
        return colors
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Tuple[Tuple[int, int, int, int], float]],
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw detection boxes on frame."""
        vis_frame = frame.copy()
        
        for (x1, y1, x2, y2), conf in detections:
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, self.thickness)
            # Draw confidence score
            text = f"{conf:.2f}"
            text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
            cv2.putText(vis_frame, text, (x1, y1 - 5), self.font, 
                       self.font_scale, color, 1)
            
        return vis_frame
    
    def draw_tracks(self, frame: np.ndarray, 
                   tracks: List[Tuple[int, Tuple[int, int, int, int]]],
                   demographics: Optional[Dict[int, Dict]] = None) -> np.ndarray:
        """Draw tracking results with optional demographic information."""
        vis_frame = frame.copy()
        
        for track_id, bbox in tracks:
            color = self.colors[track_id % len(self.colors)]
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, self.thickness)
            
            # Draw track ID
            text = f"ID: {track_id}"
            if demographics and track_id in demographics:
                demo = demographics[track_id]
                text += f" | {demo['gender']} | {demo['age_group']}"
            
            text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
            cv2.putText(vis_frame, text, (x1, y1 - 5), self.font, 
                       self.font_scale, color, 1)
            
        return vis_frame
    
    def draw_multicamera_matches(self, frames: Dict[int, np.ndarray],
                               matches: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]],
                               global_ids: Dict[int, Dict[int, int]]) -> Dict[int, np.ndarray]:
        """
        Draw tracking results across multiple cameras.
        
        Args:
            frames: Dict of camera_id to frames
            matches: Dict of camera_id to tracking results
            global_ids: Dict of camera_id to track_id to global_id mapping
        
        Returns:
            Dict of camera_id to visualized frames
        """
        vis_frames = {}
        
        for camera_id, frame in frames.items():
            vis_frame = frame.copy()
            if camera_id in matches:
                for track_id, bbox in matches[camera_id]:
                    if camera_id in global_ids and track_id in global_ids[camera_id]:
                        global_id = global_ids[camera_id][track_id]
                        color = self.colors[global_id % len(self.colors)]
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, self.thickness)
                        
                        # Draw ID information
                        text = f"Global ID: {global_id} (Local: {track_id})"
                        text_size = cv2.getTextSize(text, self.font, self.font_scale, 1)[0]
                        cv2.putText(vis_frame, text, (x1, y1 - 5), self.font, 
                                  self.font_scale, color, 1)
            
            vis_frames[camera_id] = vis_frame
            
        return vis_frames
    
    def create_summary_visualization(self, 
                                  frames: Dict[int, np.ndarray],
                                  stats: Dict[str, Dict],
                                  timestamp: datetime) -> np.ndarray:
        """Create a summary visualization with statistics."""
        # Calculate layout
        n_cameras = len(frames)
        grid_size = int(np.ceil(np.sqrt(n_cameras)))
        
        # Get frame size
        sample_frame = next(iter(frames.values()))
        frame_h, frame_w = sample_frame.shape[:2]
        
        # Create canvas
        canvas_w = grid_size * frame_w
        canvas_h = grid_size * frame_h
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Draw frames
        for idx, (camera_id, frame) in enumerate(frames.items()):
            row = idx // grid_size
            col = idx % grid_size
            y1 = row * frame_h
            y2 = y1 + frame_h
            x1 = col * frame_w
            x2 = x1 + frame_w
            canvas[y1:y2, x1:x2] = frame
            
            # Draw camera ID
            cv2.putText(canvas, f"Camera {camera_id}", (x1 + 10, y1 + 30),
                       self.font, 1, (255, 255, 255), 2)
        
        # Draw statistics
        stats_x = 10
        stats_y = canvas_h - 150
        
        # Draw timestamp
        cv2.putText(canvas, f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                   (stats_x, stats_y), self.font, 1, (255, 255, 255), 2)
        
        # Draw people count
        if 'total_count' in stats:
            cv2.putText(canvas, f"Total People: {stats['total_count']}",
                       (stats_x, stats_y + 30), self.font, 1, (255, 255, 255), 2)
        
        # Draw demographic statistics if available
        if 'demographics' in stats:
            demo_stats = stats['demographics']
            demo_text = []
            
            if 'gender_distribution' in demo_stats:
                gender_stats = [f"{k}: {v:.1%}" for k, v in 
                              demo_stats['gender_distribution'].items()]
                demo_text.append("Gender: " + ", ".join(gender_stats))
                
            if 'age_distribution' in demo_stats:
                age_stats = [f"{k}: {v:.1%}" for k, v in 
                           demo_stats['age_distribution'].items()]
                demo_text.append("Age: " + ", ".join(age_stats))
            
            for i, text in enumerate(demo_text):
                cv2.putText(canvas, text, (stats_x, stats_y + 60 + i * 30),
                           self.font, 1, (255, 255, 255), 2)
        
        return canvas
