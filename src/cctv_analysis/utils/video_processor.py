# src/cctv_analysis/utils/video_processor.py

import os
import cv2
import pandas as pd
from datetime import datetime
import glob
from pathlib import Path
import numpy as np
import re

class VideoProcessor:
    def __init__(self, input_dir, output_dir):
        """
        Initialize VideoProcessor
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory for processed videos and metadata
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _parse_filename(self, filename):
        """
        Parse filename to extract camera ID and timestamp
        Args:
            filename: Video filename (format: D4(10)_YYYYMMDDHHMMSS.mp4)
        Returns:
            tuple: (camera_id, timestamp)
        """
        try:
            # Extract camera ID and timestamp using regex
            pattern = r'D(\d+)(?:\((\d+)\))?_(\d{14})'
            match = re.match(pattern, Path(filename).stem)
            
            if not match:
                return None, None
                
            # Get primary and secondary numbers (e.g., 4 and 10 from D4(10))
            primary_num = match.group(1)
            secondary_num = match.group(2)
            timestamp_str = match.group(3)
            
            # Format camera ID
            camera_id = f"D{primary_num}"
            if secondary_num:
                camera_id += f"({secondary_num})"
                
            # Parse timestamp
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
            
            return camera_id, timestamp
            
        except (ValueError, AttributeError) as e:
            print(f"Error parsing filename {filename}: {e}")
            return None, None
            
    def _get_video_metadata(self, video_path):
        """Get video metadata including duration and frame count"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height
        }
        
    def combine_daily_videos(self):
        """Combine videos into daily videos for all cameras"""
        # Get all video files
        video_files = sorted(glob.glob(str(self.input_dir / '*.mp4')))
        
        # Group files by camera and date
        daily_groups = {}
        for video_file in video_files:
            camera_id, timestamp = self._parse_filename(Path(video_file).name)
            if camera_id and timestamp:
                date_key = timestamp.strftime('%Y%m%d')
                group_key = (camera_id, date_key)
                
                if group_key not in daily_groups:
                    daily_groups[group_key] = []
                daily_groups[group_key].append((timestamp, video_file))
        
        # Process each camera's daily videos
        metadata_records = []
        
        for (camera_id, date_key), video_group in daily_groups.items():
            print(f"Processing {camera_id} for date {date_key}...")
            
            # Sort videos by timestamp
            video_group.sort(key=lambda x: x[0])
            
            # Create output filename
            output_filename = f'{camera_id}_{date_key}.mp4'
            output_path = self.output_dir / output_filename
            
            # Get metadata from first video to set output parameters
            first_video = cv2.VideoCapture(video_group[0][1])
            fps = first_video.get(cv2.CAP_PROP_FPS)
            width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            first_video.release()
            
            # Initialize video writer with GPU acceleration if available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("Using CUDA acceleration for video processing")
                fourcc = cv2.VideoWriter_fourcc(*'H264')  # H264 for GPU acceleration
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            start_time = video_group[0][0]
            end_time = None
            total_frames = 0
            source_files = []
            
            # Combine videos
            for timestamp, video_file in video_group:
                source_files.append(Path(video_file).name)
                cap = cv2.VideoCapture(video_file)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Write frame
                    out.write(frame)
                    total_frames += 1
                    
                cap.release()
                end_time = timestamp
            
            out.release()
            
            # Calculate total duration
            duration = total_frames / fps if fps > 0 else 0
            
            # Record metadata
            metadata_records.append({
                'date': date_key,
                'camera_id': camera_id,
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': duration,
                'duration_formatted': str(pd.Timedelta(seconds=duration)),
                'frame_count': total_frames,
                'fps': fps,
                'width': width,
                'height': height,
                'output_file': output_filename,
                'source_files': ';'.join(source_files)
            })
        
        # Save metadata to CSV
        if metadata_records:
            df = pd.DataFrame(metadata_records)
            
            # Save per-camera metadata
            for camera_id in df['camera_id'].unique():
                camera_df = df[df['camera_id'] == camera_id]
                metadata_path = self.output_dir / f'{camera_id}_metadata.csv'
                camera_df.to_csv(metadata_path, index=False)
            
            # Save combined metadata
            combined_metadata_path = self.output_dir / 'combined_metadata.csv'
            df.to_csv(combined_metadata_path, index=False)
            
        return metadata_records
    
    def verify_processed_videos(self):
        """Verify the processed videos and generate a verification report"""
        verification_results = []
        
        # Load metadata
        metadata_path = self.output_dir / 'combined_metadata.csv'
        if not metadata_path.exists():
            print("No metadata file found.")
            return
            
        df = pd.read_csv(metadata_path)
        
        for _, row in df.iterrows():
            output_file = self.output_dir / row['output_file']
            if not output_file.exists():
                status = "Missing"
                actual_duration = 0
            else:
                # Verify video can be opened and check duration
                cap = cv2.VideoCapture(str(output_file))
                if not cap.isOpened():
                    status = "Corrupted"
                    actual_duration = 0
                else:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    actual_duration = frame_count / fps if fps > 0 else 0
                    expected_duration = row['duration_seconds']
                    
                    # Allow for small difference in duration
                    if abs(actual_duration - expected_duration) < 1.0:
                        status = "OK"
                    else:
                        status = "Duration Mismatch"
                        
                cap.release()
            
            verification_results.append({
                'camera_id': row['camera_id'],
                'date': row['date'],
                'file': row['output_file'],
                'expected_duration': row['duration_seconds'],
                'actual_duration': actual_duration,
                'status': status
            })
        
        # Save verification results
        verification_df = pd.DataFrame(verification_results)
        verification_path = self.output_dir / 'verification_report.csv'
        verification_df.to_csv(verification_path, index=False)
        
        # Print summary
        print("\nVerification Summary:")
        print(verification_df['status'].value_counts())
        
        return verification_results
