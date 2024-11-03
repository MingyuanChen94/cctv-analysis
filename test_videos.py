"""
Test CCTV analysis without package installation.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent.absolute()
src_dir = str(current_dir / "src")
sys.path.append(src_dir)

# Now we can import our modules
from cctv_analysis.utils import save_individual_records, print_summary_statistics

def generate_sample_data(num_individuals=10):
    """Generate sample detection and matching data."""
    import random
    from datetime import datetime, timedelta
    
    # Sample data parameters
    genders = ['Male', 'Female']
    age_ranges = [(15, 18), (19, 35), (36, 50), (51, 70)]
    
    detections = []
    matches = []
    
    for i in range(num_individuals):
        # Randomly decide camera appearances
        appears_in_camera1 = random.random() < 0.8  # 80% chance
        appears_in_camera2 = random.random() < 0.7  # 70% chance
        
        # Generate demographic info
        age_range = random.choice(age_ranges)
        age = random.randint(age_range[0], age_range[1])
        gender = random.choice(genders)
        
        # Base time for this individual
        base_time = datetime.now() + timedelta(minutes=random.randint(0, 60))
        
        # Add Camera 1 detection if applicable
        if appears_in_camera1:
            detections.append({
                'track_id': i,
                'camera_id': 1,
                'age': age,
                'gender': gender,
                'timestamp': base_time.timestamp()
            })
        
        # Add Camera 2 detection if applicable
        if appears_in_camera2:
            camera2_time = base_time + timedelta(minutes=random.randint(1, 15))
            detections.append({
                'track_id': i,
                'camera_id': 2,
                'age': age,
                'gender': gender,
                'timestamp': camera2_time.timestamp()
            })
            
            # If appears in both cameras, add to matches
            if appears_in_camera1:
                matches.append({
                    'track_id': i,
                    'camera1_time': base_time.timestamp(),
                    'camera2_time': camera2_time.timestamp()
                })
    
    return detections, matches

def main():
    """Run the test."""
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")
        
    print("\nStarting CCTV Analysis Test")
    print("-" * 50)
    
    # Generate sample data
    print("\nGenerating sample data...")
    detections, matches = generate_sample_data(num_individuals=20)
    
    print(f"Generated {len(detections)} detections and {len(matches)} matches")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Save to CSV and get DataFrame
    print("\nSaving data to CSV...")
    df = save_individual_records(
        detections=detections,
        matches=matches,
        camera1_file="data/raw/video1.mp4",
        camera2_file="data/raw/video2.mp4",
        output_path="output/test_results.csv"
    )
    
    # Print statistics
    print("\nAnalysis Results:")
    print_summary_statistics(df)
    
    # Print sample records
    print("\nSample Records from CSV:")
    print("-" * 50)
    print(df.head())

if __name__ == "__main__":
    try:
        main()
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        print("\nCurrent directory:", os.getcwd())
        print("\nDirectory contents:", os.listdir())
        raise