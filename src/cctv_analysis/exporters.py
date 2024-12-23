import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from .matcher import PersonMatcher

class DetectionExporter:
    def __init__(self, matcher: PersonMatcher, output_dir: str = '../data'):
        """
        Initialize the exporter
        Args:
            matcher: PersonMatcher instance containing detections
            output_dir: Directory to save export files
        """
        self.matcher = matcher
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_basic_detections(self) -> pd.DataFrame:
        """Export basic detection information"""
        # Export Camera 1 detections
        camera1_detections = [{
            'camera_id': 1,
            'person_id': person.id,
            'timestamp': person.timestamp,
            'gender': person.demographics.get('gender') if person.demographics else None,
            'age_group': person.demographics.get('age_group') if person.demographics else None,
        } for person in self.matcher.camera1_persons]
        
        # Export Camera 2 detections
        camera2_detections = [{
            'camera_id': 2,
            'person_id': person.id,
            'timestamp': person.timestamp,
            'gender': person.demographics.get('gender') if person.demographics else None,
            'age_group': person.demographics.get('age_group') if person.demographics else None,
        } for person in self.matcher.camera2_persons]
        
        # Combine all detections
        all_detections = camera1_detections + camera2_detections
        
        # Convert to DataFrame
        df_all = pd.DataFrame(all_detections)
        
        if not df_all.empty:
            # Sort by timestamp
            df_all = df_all.sort_values('timestamp')
            # Add formatted timestamp
            df_all['formatted_time'] = df_all['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to CSV
            output_path = self.output_dir / 'all_detections.csv'
            df_all.to_csv(output_path, index=False)
            print(f"Basic detections saved to {output_path}")
            
        return df_all
    
    def export_detailed_detections(self) -> pd.DataFrame:
        """Export detailed detection information including feature vectors"""
        def create_detailed_detections(persons, camera_id):
            return [{
                'camera_id': camera_id,
                'person_id': person.id,
                'timestamp': person.timestamp,
                'formatted_time': person.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'gender': person.demographics.get('gender') if person.demographics else None,
                'age_group': person.demographics.get('age_group') if person.demographics else None,
                'confidence': person.demographics.get('confidence') if person.demographics else None,
                'feature_vector_mean': np.mean(person.features) if person.features is not None else None,
                'feature_vector_std': np.std(person.features) if person.features is not None else None,
                'feature_vector_norm': np.linalg.norm(person.features) if person.features is not None else None
            } for person in persons]
        
        # Create detailed detections
        camera1_detailed = create_detailed_detections(self.matcher.camera1_persons, camera_id=1)
        camera2_detailed = create_detailed_detections(self.matcher.camera2_persons, camera_id=2)
        all_detailed = camera1_detailed + camera2_detailed
        
        # Convert to DataFrame
        df_detailed = pd.DataFrame(all_detailed)
        
        if not df_detailed.empty:
            # Sort by timestamp
            df_detailed = df_detailed.sort_values('timestamp')
            
            # Save to CSV
            output_path = self.output_dir / 'detailed_detections.csv'
            df_detailed.to_csv(output_path, index=False)
            print(f"Detailed detections saved to {output_path}")
            
        return df_detailed
    
    def export_feature_vectors(self):
        """Export raw feature vectors to JSON"""
        feature_vectors = {
            'camera1': {str(p.id): p.features.tolist() if p.features is not None else None 
                       for p in self.matcher.camera1_persons},
            'camera2': {str(p.id): p.features.tolist() if p.features is not None else None 
                       for p in self.matcher.camera2_persons}
        }
        
        output_path = self.output_dir / 'feature_vectors.json'
        with open(output_path, 'w') as f:
            json.dump(feature_vectors, f)
        print(f"Feature vectors saved to {output_path}")
        
        return feature_vectors
    
    def export_matches(self) -> pd.DataFrame:
        """Export matching results"""
        matches = self.matcher.get_matches()
        df_matches = pd.DataFrame(matches)
        
        if not df_matches.empty:
            output_path = self.output_dir / 'matches.csv'
            df_matches.to_csv(output_path, index=False)
            print(f"Matches saved to {output_path}")
            
        return df_matches
    
    def print_summary_statistics(self, df_detailed: Optional[pd.DataFrame] = None):
        """Print summary statistics of detections"""
        if df_detailed is None:
            df_detailed = self.export_detailed_detections()
            
        print("\nDetection Statistics:")
        print("-" * 50)
        print(f"Total detections: {len(df_detailed)}")
        print(f"Camera 1 detections: len({self.matcher.camera1_persons})")
        print(f"Camera 2 detections: len({self.matcher.camera2_persons})")
        
        if not df_detailed.empty:
            # Time analysis
            time_range = df_detailed['timestamp'].max() - df_detailed['timestamp'].min()
            duration_minutes = time_range.total_seconds() / 60
            detections_per_minute = len(df_detailed) / duration_minutes
            
            print("\nTime Analysis:")
            print("-" * 50)
            print(f"Time range covered: {time_range}")
            print(f"Average detections per minute: {detections_per_minute:.2f}")
            
            # Demographics
            print("\nDemographic Analysis:")
            print("-" * 50)
            print("\nGender distribution by camera:")
            print(pd.crosstab(df_detailed['camera_id'], df_detailed['gender'], margins=True))
            print("\nAge group distribution by camera:")
            print(pd.crosstab(df_detailed['camera_id'], df_detailed['age_group'], margins=True))
    
    def plot_visualizations(self, df_detailed: Optional[pd.DataFrame] = None):
        """Create visualizations of detection patterns"""
        if df_detailed is None:
            df_detailed = self.export_detailed_detections()
            
        if df_detailed.empty:
            print("No data to visualize")
            return
            
        plt.figure(figsize=(15, 10))
        
        # 1. Detections over time
        plt.subplot(2, 2, 1)
        df_detailed['timestamp'].hist(bins=50)
        plt.title('Detections Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Detections')
        
        # 2. Gender distribution
        if 'gender' in df_detailed.columns:
            plt.subplot(2, 2, 2)
            df_detailed['gender'].value_counts().plot(kind='bar')
            plt.title('Gender Distribution')
            plt.xlabel('Gender')
            plt.ylabel('Count')
        
        # 3. Age group distribution
        if 'age_group' in df_detailed.columns:
            plt.subplot(2, 2, 3)
            df_detailed['age_group'].value_counts().plot(kind='bar')
            plt.title('Age Group Distribution')
            plt.xlabel('Age Group')
            plt.ylabel('Count')
        
        # 4. Feature vector norms
        if 'feature_vector_norm' in df_detailed.columns:
            plt.subplot(2, 2, 4)
            df_detailed['feature_vector_norm'].hist(bins=50)
            plt.title('Feature Vector Norms')
            plt.xlabel('Norm')
            plt.ylabel('Count')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'detection_analysis.png'
        plt.savefig(plot_path)
        print(f"Visualizations saved to {plot_path}")
        plt.show()
        
    def export_all(self, create_plots: bool = True):
        """Export all data and create visualizations"""
        # Export all data
        df_detailed = self.export_detailed_detections()
        self.export_basic_detections()
        self.export_feature_vectors()
        self.export_matches()
        
        # Print statistics
        self.print_summary_statistics(df_detailed)
        
        # Create visualizations
        if create_plots:
            self.plot_visualizations(df_detailed)