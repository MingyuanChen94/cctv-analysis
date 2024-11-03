"""
Simplified utilities for CCTV analysis data export.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

def save_individual_records(
    detections: List[Dict],
    matches: List[Dict],
    camera1_file: str,
    camera2_file: str,
    output_path: str
) -> pd.DataFrame:
    """
    Save information about unique individuals to a CSV file.
    
    Args:
        detections (List[Dict]): All detections from both cameras
        matches (List[Dict]): Matches between cameras (appearing in both)
        camera1_file (str): Name of camera 1 video file
        camera2_file (str): Name of camera 2 video file
        output_path (str): Path to save the CSV file
    
    Returns:
        pd.DataFrame: DataFrame containing the saved information
    """
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Helper function to determine age group
    def get_age_group(age: Optional[float]) -> str:
        if age is None:
            return "Unknown"
        elif age <= 18:
            return "0-18"
        elif age <= 35:
            return "19-35"
        elif age <= 50:
            return "36-50"
        else:
            return "51+"
    
    # Process all detections into a dictionary of individuals
    individuals = {}
    for detection in detections:
        track_id = detection['track_id']
        camera_id = detection['camera_id']
        
        if track_id not in individuals:
            individuals[track_id] = {
                'individual_id': track_id,
                'age_group': get_age_group(detection.get('age')),
                'gender': detection.get('gender', "Unknown"),
                'appear_in_camera_1': 0,
                'appear_in_camera_2': 0,
                'appear_camera1_then_camera2': 0,
                'camera_1_video_file': camera1_file,
                'camera_2_video_file': camera2_file
            }
        
        # Update camera appearance flags
        if camera_id == 1:
            individuals[track_id]['appear_in_camera_1'] = 1
        elif camera_id == 2:
            individuals[track_id]['appear_in_camera_2'] = 1
    
    # Process matches to identify camera1 -> camera2 sequences
    for match in matches:
        track_id = match['track_id']
        if track_id in individuals:
            individuals[track_id]['appear_camera1_then_camera2'] = 1
    
    # Convert to DataFrame
    df = pd.DataFrame(list(individuals.values()))
    
    # Ensure all required columns are present
    required_columns = [
        'individual_id',
        'age_group',
        'gender',
        'appear_in_camera_1',
        'appear_in_camera_2',
        'appear_camera1_then_camera2',
        'camera_1_video_file',
        'camera_2_video_file'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns
    df = df[required_columns]
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved individual records to: {output_path}")
    
    return df

def print_summary_statistics(df: pd.DataFrame):
    """
    Print summary statistics of the detection results.
    
    Args:
        df (pd.DataFrame): DataFrame containing individual records
    """
    print("\nSummary Statistics:")
    print("-" * 50)
    print(f"Total unique individuals: {len(df)}")
    print(f"Individuals in Camera 1: {df['appear_in_camera_1'].sum()}")
    print(f"Individuals in Camera 2: {df['appear_in_camera_2'].sum()}")
    print(f"Individuals appearing in both cameras: {df['appear_camera1_then_camera2'].sum()}")
    
    print("\nDemographic Statistics:")
    print("-" * 50)
    print("\nAge Group Distribution:")
    print(df['age_group'].value_counts())
    
    print("\nGender Distribution:")
    print(df['gender'].value_counts())
    
    print("\nCamera Appearance Patterns:")
    print(f"Camera 1 only: {len(df[df['appear_in_camera_1'] == 1 & (df['appear_in_camera_2'] == 0)])}")
    print(f"Camera 2 only: {len(df[df['appear_in_camera_2'] == 1 & (df['appear_in_camera_1'] == 0)])}")
    print(f"Both cameras: {len(df[df['appear_in_camera_1'] == 1 & (df['appear_in_camera_2'] == 1)])}")

# Example usage
if __name__ == "__main__":
    # Example data
    sample_detections = [
        {'track_id': 1, 'camera_id': 1, 'age': 25, 'gender': 'Male'},
        {'track_id': 1, 'camera_id': 2, 'age': 25, 'gender': 'Male'},
        {'track_id': 2, 'camera_id': 1, 'age': 35, 'gender': 'Female'}
    ]
    
    sample_matches = [
        {'track_id': 1, 'camera1_time': 1000, 'camera2_time': 2000}
    ]
    
    # Save to CSV
    df = save_individual_records(
        detections=sample_detections,
        matches=sample_matches,
        camera1_file="camera1.mp4",
        camera2_file="camera2.mp4",
        output_path="output/individuals.csv"
    )
    
    # Print statistics
    print_summary_statistics(df)
