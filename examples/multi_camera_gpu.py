import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import gc

from cctv_analysis import PersonDetector, PersonReID, DemographicAnalyzer, PersonMatcher

def setup_gpu():
    """Setup GPU for optimal performance"""
    if torch.cuda.is_available():
        # Print GPU information
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Enable TF32 on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudnn benchmark
        torch.backends.cudnn.benchmark = True
        
        return True
    else:
        print("No GPU available!")
        return False

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class VideoProcessor:
    def __init__(self, batch_size=4):
        self.batch_size = batch_size
        self.use_gpu = setup_gpu()
        
        # Initialize models
        print("Initializing models...")
        self.detector = PersonDetector(
            model_path='../models/detector/yolox_l.pth',
            model_size='l',
            device='cuda' if self.use_gpu else 'cpu'
        )
        
        self.reid_model = PersonReID(
            model_path='../models/reid/osnet_x1_0.pth',
            device='cuda' if self.use_gpu else 'cpu'
        )
        
        self.demographic_analyzer = DemographicAnalyzer(
            device='cuda' if self.use_gpu else 'cpu'
        )
        
        self.matcher = PersonMatcher(similarity_threshold=0.75)
        
    def process_frame_batch(self, frames, timestamps, camera_id):
        """Process a batch of frames"""
        batch_results = []
        
        # Stack frames for batch processing if possible
        if isinstance(self.detector.model, torch.nn.Module):
            stacked_frames = torch.stack([
                self.detector.preprocess(frame)[0] for frame in frames
            ]).to(self.device)
            
            # Batch detection
            with torch.cuda.amp.autocast():  # Enable AMP for faster processing
                detections_batch = self.detector.model(stacked_frames)
            
            # Process each frame's detections
            for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                detections = self.detector.postprocess(detections_batch[i:i+1])
                if detections[0] is not None:
                    for det in detections[0]:
                        x1, y1, x2, y2 = map(int, det[:4])
                        person_crop = frame[y1:y2, x1:x2]
                        
                        if person_crop.size == 0:
                            continue
                            
                        # Get ReID features
                        features = self.reid_model.extract_features(person_crop)
                        
                        # Get demographics
                        demographics = self.demographic_analyzer.analyze(person_crop)
                        
                        batch_results.append({
                            'frame_idx': i,
                            'bbox': (x1, y1, x2, y2),
                            'features': features,
                            'demographics': demographics[0] if demographics else None
                        })
        
        return batch_results

    def process_video(self, video_path, camera_id, start_time):
        """Process video with batch processing"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_buffer = []
        timestamps_buffer = []
        
        with tqdm(total=total_frames, desc=f"Processing Camera {camera_id}") as pbar:
            while True:
                # Read frames to fill buffer
                while len(frames_buffer) < self.batch_size:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_idx = len(frames_buffer)
                    timestamp = start_time + pd.Timedelta(seconds=frame_idx/fps)
                    
                    frames_buffer.append(frame)
                    timestamps_buffer.append(timestamp)
                
                if not frames_buffer:
                    break
                
                # Process batch
                batch_results = self.process_frame_batch(
                    frames_buffer, timestamps_buffer, camera_id
                )
                
                # Add results to matcher
                for result in batch_results:
                    self.matcher.add_person(
                        camera_id=camera_id,
                        person_id=len(self.matcher.camera1_persons if camera_id == 1 
                                    else self.matcher.camera2_persons),
                        timestamp=timestamps_buffer[result['frame_idx']],
                        features=result['features'],
                        demographics=result['demographics']
                    )
                
                # Update progress
                pbar.update(len(frames_buffer))
                
                # Clear buffers
                frames_buffer.clear()
                timestamps_buffer.clear()
                
                # Clear GPU memory periodically
                if self.use_gpu and pbar.n % (self.batch_size * 10) == 0:
                    clear_gpu_memory()
        
        cap.release()

def main():
    # Initialize processor
    processor = VideoProcessor(batch_size=4)
    
    # Process videos
    video1_path = '../data/videos/camera1.mp4'
    video2_path = '../data/videos/camera2.mp4'
    
    # Assuming videos start at these times (adjust as needed)
    video1_start = pd.Timestamp('2024-01-01 09:00:00')
    video2_start = pd.Timestamp('2024-01-01 09:00:00')
    
    # Process both videos
    processor.process_video(video1_path, camera_id=1, start_time=video1_start)
    processor.process_video(video2_path, camera_id=2, start_time=video2_start)
    
    # Get matching results
    matches = processor.matcher.get_matches()
    
    # Convert to DataFrame for analysis
    df_matches = pd.DataFrame(matches)
    
    # Display summary statistics
    print(f"\nTotal number of individuals appearing in both cameras: {len(df_matches)}")
    if not df_matches.empty:
        print("\nAverage time difference between appearances:")
        print(f"{df_matches['time_difference'].mean():.2f} seconds")
        
        # Demographics analysis
        demographics_df = pd.DataFrame([m['demographics'] for m in matches if m['demographics']])
        if not demographics_df.empty:
            print("\nDemographic breakdown:")
            print("\nGender distribution:")
            print(demographics_df['gender'].value_counts())
            print("\nAge group distribution:")
            print(demographics_df['age_group'].value_counts())
    
    # Save results
    output_path = '../data/analysis_results.csv'
    df_matches.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()
