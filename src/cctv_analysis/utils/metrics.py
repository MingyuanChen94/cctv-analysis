from typing import Dict, List, Tuple, Set
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta

class TrackingMetrics:
    """Calculate and store tracking-related metrics."""
    
    def __init__(self):
        """Initialize tracking metrics."""
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.track_lengths = defaultdict(int)
        self.track_durations = defaultdict(list)
        self.track_velocities = defaultdict(list)
        self.track_positions = defaultdict(list)
        self.track_timestamps = defaultdict(list)
        
    def update_track(self, track_id: int, bbox: Tuple[int, int, int, int], 
                    timestamp: datetime):
        """Update metrics for a track."""
        self.track_lengths[track_id] += 1
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        self.track_positions[track_id].append((center_x, center_y))
        self.track_timestamps[track_id].append(timestamp)
        
        # Calculate velocity if we have at least two positions
        if len(self.track_positions[track_id]) >= 2:
            pos1 = self.track_positions[track_id][-2]
            pos2 = self.track_positions[track_id][-1]
            time1 = self.track_timestamps[track_id][-2]
            time2 = self.track_timestamps[track_id][-1]
            
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dt = (time2 - time1).total_seconds()
            
            if dt > 0:
                velocity = np.sqrt(dx*dx + dy*dy) / dt
                self.track_velocities[track_id].append(velocity)
        
        # Update duration
        if len(self.track_timestamps[track_id]) >= 2:
            duration = (self.track_timestamps[track_id][-1] - 
                       self.track_timestamps[track_id][0]).total_seconds()
            self.track_durations[track_id] = duration
    
    def get_track_statistics(self, track_id: int) -> Dict:
        """Get statistics for a specific track."""
        stats = {}
        
        if track_id in self.track_lengths:
            stats['length'] = self.track_lengths[track_id]
            stats['duration'] = self.track_durations[track_id]
            
            if self.track_velocities[track_id]:
                stats['avg_velocity'] = np.mean(self.track_velocities[track_id])
                stats['max_velocity'] = np.max(self.track_velocities[track_id])
            
            positions = np.array(self.track_positions[track_id])
            stats['total_distance'] = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
            
        return stats
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for all tracks."""
        summary = {
            'total_tracks': len(self.track_lengths),
            'avg_track_length': np.mean(list(self.track_lengths.values())),
            'avg_track_duration': np.mean(list(self.track_durations.values())),
            'total_unique_people': len(self.track_lengths)
        }
        
        if any(self.track_velocities.values()):
            all_velocities = [v for velocities in self.track_velocities.values() 
                            for v in velocities]
            summary.update({
                'avg_velocity': np.mean(all_velocities),
                'max_velocity': np.max(all_velocities),
                'min_velocity': np.min(all_velocities)
            })
            
        return summary

class MultiCameraMetrics:
    """Calculate metrics for multi-camera tracking."""
    
    def __init__(self):
        """Initialize multi-camera metrics."""
        self.camera_transitions = defaultdict(lambda: defaultdict(int))
        self.global_appearances = defaultdict(set)
        self.transition_times = defaultdict(list)
        
    def update_transition(self, global_id: int, from_camera: int, 
                         to_camera: int, timestamp: datetime):
        """Record a transition between cameras."""
        self.camera_transitions[from_camera][to_camera] += 1
        self.global_appearances[global_id].add(from_camera)
        self.global_appearances[global_id].add(to_camera)
        
        # Store transition time
        key = (from_camera, to_camera)
        self.transition_times[key].append(timestamp)
        
    def get_transition_matrix(self) -> Dict[Tuple[int, int], int]:
        """Get the camera transition count matrix."""
        transitions = {}
        for from_cam in self.camera_transitions:
            for to_cam, count in self.camera_transitions[from_cam].items():
                transitions[(from_cam, to_cam)] = count
        return transitions
    
    def get_popular_paths(self, top_k: int = 5) -> List[Tuple[Tuple[int, int], int]]:
        """Get the most common camera transitions."""
        transitions = self.get_transition_matrix()
        sorted_transitions = sorted(transitions.items(), 
                                 key=lambda x: x[1], reverse=True)
        return sorted_transitions[:top_k]
    
    def get_camera_coverage(self) -> Dict[int, float]:
        """Calculate the percentage of people seen by each camera."""
        total_people = len(self.global_appearances)
        if total_people == 0:
            return {}
            
        camera_counts = defaultdict(int)
        for appearances in self.global_appearances.values():
            for camera in appearances:
                camera_counts[camera] += 1
                
        return {camera: count/total_people 
                for camera, count in camera_counts.items()}
    
    def get_average_transition_times(self) -> Dict[Tuple[int, int], float]:
        """Calculate average time between camera transitions."""
        avg_times = {}
        
        for (from_cam, to_cam), timestamps in self.transition_times.items():
            if len(timestamps) >= 2:
                # Calculate time differences between consecutive appearances
                time_diffs = [(t2 - t1).total_seconds() 
                            for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
                avg_times[(from_cam, to_cam)] = np.mean(time_diffs)
                
        return avg_times

class DemographicMetrics:
    """Calculate demographic-related metrics."""
    
    def __init__(self):
        """Initialize demographic metrics."""
        self.gender_counts = defaultdict(int)
        self.age_counts = defaultdict(int)
        self.gender_age_counts = defaultdict(lambda: defaultdict(int))
        self.temporal_stats = defaultdict(lambda: defaultdict(list))
        
    def update(self, timestamp: datetime, gender: str, age_group: str):
        """Update demographic counts."""
        self.gender_counts[gender] += 1
        self.age_counts[age_group] += 1
        self.gender_age_counts[gender][age_group] += 1
        
        # Record temporal statistics
        hour = timestamp.hour
        self.temporal_stats[hour]['gender'].append(gender)
        self.temporal_stats[hour]['age_group'].append(age_group)
        
    def get_distribution(self) -> Dict:
        """Get overall demographic distribution."""
        total = sum(self.gender_counts.values())
        if total == 0:
            return {}
            
        distribution = {
            'gender': {k: v/total for k, v in self.gender_counts.items()},
            'age': {k: v/total for k, v in self.age_counts.items()},
            'gender_age': {}
        }
        
        for gender in self.gender_age_counts:
            distribution['gender_age'][gender] = {}
            gender_total = sum(self.gender_age_counts[gender].values())
            if gender_total > 0:
                for age_group, count in self.gender_age_counts[gender].items():
                    distribution['gender_age'][gender][age_group] = count/gender_total
                    
        return distribution
    
    def get_hourly_patterns(self) -> Dict:
        """Analyze temporal patterns in demographics."""
        patterns = {}
        
        for hour in range(24):
            if hour in self.temporal_stats:
                hour_data = self.temporal_stats[hour]
                
                # Calculate gender distribution for this hour
                gender_counts = defaultdict(int)
                for gender in hour_data['gender']:
                    gender_counts[gender] += 1
                total = len(hour_data['gender'])
                
                patterns[hour] = {
                    'total_count': total,
                    'gender_distribution': {k: v/total for k, v in gender_counts.items()},
                    'age_distribution': self._calculate_distribution(hour_data['age_group'])
                }
                
        return patterns
    
    def _calculate_distribution(self, values: List[str]) -> Dict[str, float]:
        """Helper function to calculate distribution of values."""
        if not values:
            return {}
            
        counts = defaultdict(int)
        for v in values:
            counts[v] += 1
        total = len(values)
        
        return {k: v/total for k, v in counts.items()}
