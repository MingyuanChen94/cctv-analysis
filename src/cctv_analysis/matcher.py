import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import datetime
import pandas as pd

@dataclass
class PersonInstance:
    id: int
    timestamp: datetime.datetime
    features: np.ndarray
    demographics: Optional[Dict] = None

class PersonMatcher:
    def __init__(self, similarity_threshold=0.5, max_time_diff=3600):
        """
        Initialize PersonMatcher
        Args:
            similarity_threshold: Threshold for cosine similarity (0-1), lower means more matches
            max_time_diff: Maximum time difference in seconds between matches
        """
        self.similarity_threshold = similarity_threshold
        self.max_time_diff = max_time_diff  # Maximum time difference in seconds
        self.camera1_persons: List[PersonInstance] = []
        self.camera2_persons: List[PersonInstance] = []
        self.matches = []  # List of (cam1_id, cam2_id) tuples
        self.matched_cam2_ids = set()  # Track matched camera 2 IDs
        
    def add_person(self, camera_id: int, person_id: int, timestamp: datetime.datetime, 
                  features: np.ndarray, demographics: Optional[Dict] = None):
        """Add a person detection from either camera"""
        if features is None:
            return
            
        person = PersonInstance(person_id, timestamp, features, demographics)
        if camera_id == 1:
            self.camera1_persons.append(person)
        else:
            self.camera2_persons.append(person)
            # Try to match with camera 1 persons
            self._find_matches(person)
    
    def _find_matches(self, cam2_person: PersonInstance):
        """Find matches for a new camera 2 person among camera 1 persons"""
        if cam2_person.id in self.matched_cam2_ids:
            return
            
        best_match = None
        best_similarity = self.similarity_threshold
        
        for cam1_person in self.camera1_persons:
            # Check if time difference is within limit
            time_diff = (cam2_person.timestamp - cam1_person.timestamp).total_seconds()
            
            # Only consider matches where camera 1 timestamp is earlier and within time limit
            if time_diff < 0 or time_diff > self.max_time_diff:
                continue
            
            similarity = self._compute_similarity(cam1_person.features, cam2_person.features)
            
            # Update best match if similarity is higher
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cam1_person
        
        # Add the best match if found
        if best_match is not None:
            self.matches.append((best_match.id, cam2_person.id))
            self.matched_cam2_ids.add(cam2_person.id)
    
    def _compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between feature vectors
        Returns similarity score between 0 and 1
        """
        if feat1 is None or feat2 is None:
            return 0.0
            
        # Normalize features
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
        
        similarity = np.dot(feat1_norm, feat2_norm)
        return float(np.clip(similarity, 0, 1))
    
    def get_matches(self, sort_by_time=True) -> List[Dict]:
        """
        Get all matches with timestamps and demographics
        Args:
            sort_by_time: Sort matches by time difference if True
        Returns:
            List of match dictionaries with details
        """
        results = []
        for cam1_id, cam2_id in self.matches:
            # Find corresponding PersonInstances
            cam1_person = next(p for p in self.camera1_persons if p.id == cam1_id)
            cam2_person = next(p for p in self.camera2_persons if p.id == cam2_id)
            
            # Calculate similarity score
            similarity = self._compute_similarity(cam1_person.features, cam2_person.features)
            
            # Calculate time difference
            time_diff = (cam2_person.timestamp - cam1_person.timestamp).total_seconds()
            
            results.append({
                "camera1_id": cam1_id,
                "camera2_id": cam2_id,
                "camera1_timestamp": cam1_person.timestamp,
                "camera2_timestamp": cam2_person.timestamp,
                "time_difference": time_diff,
                "similarity_score": similarity,
                "demographics": cam1_person.demographics
            })
        
        if sort_by_time:
            results.sort(key=lambda x: x['time_difference'])
            
        return results
    
    def print_matching_stats(self):
        """Print matching statistics"""
        print(f"\nMatching Statistics:")
        print(f"Number of people detected in Camera 1: {len(self.camera1_persons)}")
        print(f"Number of people detected in Camera 2: {len(self.camera2_persons)}")
        print(f"Number of matches found: {len(self.matches)}")
        
        if self.matches:
            matches = self.get_matches()
            similarities = [m['similarity_score'] for m in matches]
            time_diffs = [m['time_difference'] for m in matches]
            
            print(f"\nSimilarity Scores:")
            print(f"  Mean: {np.mean(similarities):.3f}")
            print(f"  Min:  {np.min(similarities):.3f}")
            print(f"  Max:  {np.max(similarities):.3f}")
            
            print(f"\nTime Differences (seconds):")
            print(f"  Mean: {np.mean(time_diffs):.1f}")
            print(f"  Min:  {np.min(time_diffs):.1f}")
            print(f"  Max:  {np.max(time_diffs):.1f}")
            
    def visualize_matches(self, n_samples=5):
        """Print detailed information for a sample of matches"""
        if not self.matches:
            print("No matches to visualize")
            return
            
        matches = self.get_matches()
        n_samples = min(n_samples, len(matches))
        
        print(f"\nDetailed view of {n_samples} matches:")
        for i, match in enumerate(matches[:n_samples]):
            print(f"\nMatch {i+1}:")
            print(f"  Camera 1 ID: {match['camera1_id']}")
            print(f"  Camera 2 ID: {match['camera2_id']}")
            print(f"  Time Difference: {match['time_difference']:.1f} seconds")
            print(f"  Similarity Score: {match['similarity_score']:.3f}")
            if match['demographics']:
                print(f"  Demographics: {match['demographics']}")
