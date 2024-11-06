from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import numpy as np

from cctv_analysis.utils.config import Config
from cctv_analysis.utils.data_types import (
    PersonDetection,
    PersonMatch,
    DetectionDatabase
)
from cctv_analysis.utils.visualization import Visualizer
from cctv_analysis.utils.logger import setup_logger
from cctv_analysis.camera_processor import CameraProcessor

class CCTVAnalysis:
    """Main analysis class for multi-camera tracking"""
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger("CCTVAnalysis")
        self.processor = CameraProcessor(config)
        self.visualizer = Visualizer(config.processing.output_dir)
        self.db = DetectionDatabase()
    
    def process_videos(self, video_paths: Dict[int, Path]):
        """Process videos from all cameras"""
        self.logger.info("Starting video processing...")
        
        for camera_id, video_path in video_paths.items():
            self.logger.info(f"Processing camera {camera_id}: {video_path}")
            detections = self.processor.process_video(video_path, camera_id)
            
            # Save tracking visualization
            output_name = f"camera_{camera_id}_tracking.mp4"
            self.visualizer.save_tracking_video(
                video_path, detections, output_name
            )
    
    def analyze_transitions(self) -> List[PersonMatch]:
        """Analyze transitions between cameras"""
        self.logger.info("Analyzing transitions between cameras...")
        
        matches = []
        cam1_tracks = self.processor.detection_db.get_camera_detections(1)
        cam2_tracks = self.processor.detection_db.get_camera_detections(2)
        
        # Group detections by track ID
        cam1_by_track = self._group_by_track_id(cam1_tracks)
        cam2_by_track = self._group_by_track_id(cam2_tracks)
        
        for track_id1, detections1 in cam1_by_track.items():
            features1 = self._get_track_features(detections1)
            earliest_time1 = min(d.timestamp for d in detections1)
            
            for track_id2, detections2 in cam2_by_track.items():
                features2 = self._get_track_features(detections2)
                earliest_time2 = min(d.timestamp for d in detections2)
                
                # Check time difference
                time_diff = abs((earliest_time2 - earliest_time1).total_seconds())
                if time_diff > self.config.matching.max_time_difference:
                    continue
                
                # Calculate similarity
                similarity = self._compute_similarity(features1, features2)
                
                # Check if match criteria are met
                if similarity > 1 - self.config.matching.max_cosine_distance:
                    match = PersonMatch(
                        track_id_cam1=track_id1,
                        track_id_cam2=track_id2,
                        first_appearance_cam1=earliest_time1,
                        first_appearance_cam2=earliest_time2,
                        similarity_score=similarity,
                        transition_time=time_diff
                    )
                    matches.append(match)
        
        # Create visualizations
        self._create_analysis_visualizations(matches)
        
        return matches
    
    def generate_report(self, matches: List[PersonMatch]):
        """Generate analysis report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_matches': len(matches),
            'average_transition_time': np.mean([m.transition_time for m in matches]),
            'average_similarity': np.mean([m.similarity_score for m in matches]),
            'transitions': [match.to_dict() for match in matches]
        }
        
        # Save report
        report_path = self.config.processing.output_dir / 'analysis' / 'report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Analysis report saved to {report_path}")
        
        return report
    
    def _group_by_track_id(self, detections: List[PersonDetection]) -> Dict:
        """Group detections by track ID"""
        tracks = {}
        for det in detections:
            if det.track_id not in tracks:
                tracks[det.track_id] = []
            tracks[det.track_id].append(det)
        return tracks
    
    def _get_track_features(self, detections: List[PersonDetection]) -> np.ndarray:
        """Get representative features for a track"""
        features = [d.reid_features for d in detections]
        return np.mean(features, axis=0)
    
    def _compute_similarity(self, features1: np.ndarray,
                          features2: np.ndarray) -> float:
        """Compute cosine similarity between feature vectors"""
        return np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2)
        )
    
    def _create_analysis_visualizations(self, matches: List[PersonMatch]):
        """Create visualizations for analysis results"""
        # Plot transition time distributions
        self.visualizer.plot_transitions(matches)
        
        # Plot activity timeline
        self.visualizer.plot_activity_timeline(
            self.processor.detection_db.get_camera_detections(1),
            self.processor.detection_db.get_camera_detections(2)
        )
