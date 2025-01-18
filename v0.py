import cv2
import numpy as np
import os
from pathlib import Path
import torch
import pandas as pd
from datetime import datetime
import os
from numba import jit
import cupy as cp
import torch.nn.functional as F
from bytetracker import BYTETracker
from ultralytics import YOLO

# Check GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Modify preprocess_video to use GPU
@jit(nopython=True)  # CPU acceleration for numpy operations
def calculate_hist(channel):
    hist = np.zeros(256)
    for pixel in channel.flatten():
        hist[int(pixel)] += 1
    return hist

def gaussian_blur_tensor(tensor, kernel_size=3, sigma=1.0):
    """
    Apply Gaussian blur to a tensor using PyTorch convolution
    """
    channels = tensor.shape[1]
    
    # Create Gaussian kernel
    kernel_size = kernel_size + (1 - kernel_size % 2)  # Ensure odd size
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2
    kernel = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    
    # Create 2D kernel from 1D kernel
    kernel = kernel.view(1, 1, -1) * kernel.view(1, -1, 1)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size).to(DEVICE)
    
    # Apply separable convolution for efficiency
    padding = kernel_size // 2
    # Apply horizontal convolution
    x = F.conv2d(
        tensor,
        kernel.transpose(2, 3),
        padding=(0, padding),
        groups=channels
    )
    # Apply vertical convolution
    x = F.conv2d(
        x,
        kernel,
        padding=(padding, 0),
        groups=channels
    )
    
    return x

def preprocess_video(video_path, target_fps=6, batch_size=16):  # Reduced batch size
    """
    GPU-accelerated video preprocessing with batch processing
    """
    cap = cv2.VideoCapture(video_path)
    frames_buffer = []
    processed_frames = []
    timestamps = []
    
    frame_interval = max(1, int(cap.get(cv2.CAP_PROP_FPS) / target_fps))
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames_buffer.append(frame)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                
                # Process in batches
                if len(frames_buffer) >= batch_size:
                    if DEVICE.type == 'cuda':
                        with torch.cuda.device(DEVICE):
                            # Convert batch to GPU tensor
                            frames_tensor = torch.from_numpy(
                                np.stack(frames_buffer)
                            ).float().permute(0, 3, 1, 2).to(DEVICE) / 255.0
                            
                            # Resize frames
                            frames_tensor = F.interpolate(
                                frames_tensor, 
                                size=(360, 640),
                                mode='bilinear',
                                align_corners=False
                            )
                            
                            # Apply Gaussian blur
                            frames_tensor = gaussian_blur_tensor(frames_tensor)
                            
                            # Move back to CPU and convert to numpy
                            processed_batch = (frames_tensor.cpu().numpy() * 255).astype(np.uint8)
                            processed_batch = processed_batch.transpose(0, 2, 3, 1)
                            processed_frames.extend(list(processed_batch))
                            
                            # Clear GPU cache
                            torch.cuda.empty_cache()
                    else:
                        # CPU fallback
                        for frame in frames_buffer:
                            frame = cv2.resize(frame, (640, 360))
                            frame = cv2.GaussianBlur(frame, (3, 3), 0)
                            processed_frames.append(frame)
                    
                    frames_buffer = []
            
            frame_count += 1
        
        # Process remaining frames
        if frames_buffer:
            if DEVICE.type == 'cuda':
                with torch.cuda.device(DEVICE):
                    frames_tensor = torch.from_numpy(
                        np.stack(frames_buffer)
                    ).float().permute(0, 3, 1, 2).to(DEVICE) / 255.0
                    
                    frames_tensor = F.interpolate(
                        frames_tensor,
                        size=(360, 640),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    frames_tensor = gaussian_blur_tensor(frames_tensor)
                    
                    processed_batch = (frames_tensor.cpu().numpy() * 255).astype(np.uint8)
                    processed_batch = processed_batch.transpose(0, 2, 3, 1)
                    processed_frames.extend(list(processed_batch))
                    
                    torch.cuda.empty_cache()
            else:
                for frame in frames_buffer:
                    frame = cv2.resize(frame, (640, 360))
                    frame = cv2.GaussianBlur(frame, (3, 3), 0)
                    processed_frames.append(frame)
    
    finally:
        cap.release()
        
    print(f"Video path: {video_path}")
    print(f"Total frames processed: {len(processed_frames)}")
    
    return processed_frames, timestamps

def normalize_colors(frame1, frame2):
    """
    Match histograms between two frames to normalize color appearance
    Args:
        frame1: Reference frame (from Camera 1)
        frame2: Frame to be matched (from Camera 2)
    Returns:
        normalized_frame: Color-normalized version of frame2
    """
    # Convert frames to LAB color space for better color matching
    if DEVICE.type == 'cuda':
        # Move data to GPU
        frame1_gpu = cp.asarray(frame1)
        frame2_gpu = cp.asarray(frame2)
        
        # Color normalization operations on GPU
        lab1 = cv2.cvtColor(cp.asnumpy(frame1_gpu), cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(cp.asnumpy(frame2_gpu), cv2.COLOR_BGR2LAB)
    else:
        lab1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2LAB)
    
    # Split the LAB channels
    l1, a1, b1 = cv2.split(lab1)
    l2, a2, b2 = cv2.split(lab2)
    
    # Calculate histograms for each channel
    def calculate_hist(channel):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        return hist
    
    def match_histograms(src, ref):
        # Calculate cumulative distribution functions (CDF)
        src_hist = calculate_hist(src)
        ref_hist = calculate_hist(ref)
        
        src_cdf = src_hist.cumsum()
        ref_cdf = ref_hist.cumsum()
        
        # Normalize CDFs
        src_cdf_normalized = src_cdf / src_cdf[-1]
        ref_cdf_normalized = ref_cdf / ref_cdf[-1]
        
        # Create lookup table for histogram matching
        lookup_table = np.zeros(256)
        j = 0
        for i in range(256):
            while j < 255 and ref_cdf_normalized[j] < src_cdf_normalized[i]:
                j += 1
            lookup_table[i] = j
        
        # Apply lookup table to source channel
        return cv2.LUT(src, lookup_table.astype('uint8'))
    
    # Match histograms for each channel
    l2_matched = match_histograms(l2, l1)
    a2_matched = match_histograms(a2, a1)
    b2_matched = match_histograms(b2, b1)
    
    # Merge channels and convert back to BGR
    lab2_matched = cv2.merge([l2_matched, a2_matched, b2_matched])
    normalized_frame = cv2.cvtColor(lab2_matched, cv2.COLOR_LAB2BGR)
    
    return normalized_frame

def process_video_pair(video1_path, video2_path):
    """
    Process a pair of videos from Camera 1 and Camera 2, handling FPS differences
    """
    frames1, timestamps1 = preprocess_video(video1_path)
    frames2, timestamps2 = preprocess_video(video2_path)
    
    # Find the common time range
    start_time = max(timestamps1[0], timestamps2[0])
    end_time = min(timestamps1[-1], timestamps2[-1])
    
    # Function to get frames within the common time range
    def get_frames_in_range(frames, timestamps, start_time, end_time):
        valid_frames = []
        valid_timestamps = []
        for frame, timestamp in zip(frames, timestamps):
            if start_time <= timestamp <= end_time:
                valid_frames.append(frame)
                valid_timestamps.append(timestamp)
        return valid_frames, valid_timestamps
    
    # Get synchronized frames
    frames1_sync, timestamps1_sync = get_frames_in_range(frames1, timestamps1, start_time, end_time)
    frames2_sync, timestamps2_sync = get_frames_in_range(frames2, timestamps2, start_time, end_time)
    
    # Ensure we have the same number of frames from both cameras
    min_frames = min(len(frames1_sync), len(frames2_sync))
    frames1_sync = frames1_sync[:min_frames]
    frames2_sync = frames2_sync[:min_frames]
    timestamps1_sync = timestamps1_sync[:min_frames]
    timestamps2_sync = timestamps2_sync[:min_frames]
    
    # Normalize colors for Camera 2 frames based on Camera 1
    normalized_frames2 = []
    for i in range(min_frames):
        normalized_frame = normalize_colors(frames1_sync[i], frames2_sync[i])
        normalized_frames2.append(normalized_frame)
    
    print(f"Synchronized frames: {min_frames}")
    print(f"Time range: {start_time:.2f}s to {end_time:.2f}s")
    
    return frames1_sync, normalized_frames2, timestamps1_sync, timestamps2_sync

class PersonTracker:
    def __init__(self, conf_thresh=0.3, track_buffer=60):
        """
        Initialize person detector and tracker
        """
        # Initialize YOLO model for person detection
        self.detector = YOLO('yolov8n.pt')
        self.detector.to(DEVICE)  # Move model to GPU if available
        
        # Initialize ByteTracker
        self.tracker = BYTETracker(
            track_thresh=conf_thresh,
            track_buffer=track_buffer,
            match_thresh=0.8,
            frame_rate=6
        )
        
        # Store door regions
        self.door_regions = {
            'cam1': [(1030, 0), (1700, 560)],
            'cam2': [(400, 0), (800, 470)]
        }
    
    def is_in_door_region(self, bbox, camera):
        """Check if detection is in door region"""
        # Ensure bbox is on CPU and in numpy format
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
        
        x1, y1, x2, y2 = bbox
        door = self.door_regions[camera]
        
        # Check overlap with door region
        dx1, dy1 = door[0]
        dx2, dy2 = door[1]
        
        overlap_x = max(0, min(x2, dx2) - max(x1, dx1))
        overlap_y = max(0, min(y2, dy2) - max(y1, dy1))
        
        if overlap_x > 0 and overlap_y > 0:
            overlap_area = overlap_x * overlap_y
            bbox_area = (x2 - x1) * (y2 - y1)
            overlap_ratio = overlap_area / bbox_area
            return overlap_ratio > 0.5
        return False
    
    def process_frame(self, frame, camera_id):
        """
        Process a single frame
        """
        # Ensure frame is on CPU for YOLO detection
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        
        # Run YOLO detection
        results = self.detector(frame, classes=[0], conf=0.3)  # class 0 is person
        
        # Extract detections
        detections = []
        if len(results) > 0:
            # Get boxes data and ensure it's on CPU
            boxes = results[0].boxes
            if hasattr(boxes, 'cpu'):
                boxes_data = boxes.data.cpu().numpy()
            else:
                boxes_data = boxes.data.numpy()
                
            for r in boxes_data:
                x1, y1, x2, y2, conf, cls = r
                if self.is_in_door_region([x1, y1, x2, y2], camera_id):
                    # Add the detection with class information (0 for person)
                    detections.append([x1, y1, x2, y2, conf, 0])
        
        # Update tracker
        if len(detections) > 0:
            # Convert detections to numpy array
            det_array = np.array(detections)
            
            # Update tracker
            tracks = self.tracker.update(
                det_array,  # Already includes class information
                [frame.shape[0], frame.shape[1]]  # img_size
            )
        else:
            tracks = []
        
        return tracks, detections
    
def process_video_pair_with_tracking(video1_path, video2_path):
    """Process video pair with person tracking"""
    # Get preprocessed frames
    frames1, frames2_norm, timestamps1, timestamps2 = process_video_pair(
        video1_path, video2_path
    )
    
    # Initialize trackers for both cameras
    tracker1 = PersonTracker()
    tracker2 = PersonTracker()
    
    # Store tracking results
    tracks_cam1 = []
    tracks_cam2 = []
    
    try:
        # Process frames
        for i in range(len(frames1)):
            # Ensure frames are in numpy format
            frame1 = frames1[i].cpu().numpy() if isinstance(frames1[i], torch.Tensor) else frames1[i]
            frame2 = frames2_norm[i].cpu().numpy() if isinstance(frames2_norm[i], torch.Tensor) else frames2_norm[i]
            
            # Process Camera 1
            tracks1, dets1 = tracker1.process_frame(frame1, 'cam1')
            if len(tracks1) > 0:
                tracks_cam1.append({
                    'timestamp': timestamps1[i],
                    'tracks': [
                        {
                            'id': int(t.track_id),
                            'bbox': t.tlbr.tolist() if isinstance(t.tlbr, np.ndarray) else t.tlbr,
                            'score': float(t.score)
                        } for t in tracks1
                    ]
                })
            
            # Process Camera 2
            tracks2, dets2 = tracker2.process_frame(frame2, 'cam2')
            if len(tracks2) > 0:
                tracks_cam2.append({
                    'timestamp': timestamps2[i],
                    'tracks': [
                        {
                            'id': int(t.track_id),
                            'bbox': t.tlbr.tolist() if isinstance(t.tlbr, np.ndarray) else t.tlbr,
                            'score': float(t.score)
                        } for t in tracks2
                    ]
                })
            
            # Print progress
            if i % 100 == 0:
                print(f"Processed {i}/{len(frames1)} frames")
                
            # Clear GPU cache periodically
            if i % 500 == 0:
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"Error during tracking: {e}")
        raise e
    
    finally:
        # Final GPU cleanup
        torch.cuda.empty_cache()
    
    return tracks_cam1, tracks_cam2

# Update main function
def main():
    video_dir = os.path.join('C:\\Users', 'mc1159', 'OneDrive - University of Exeter',
                                 'Documents', 'VISIONARY', 'Durham Experiment', 'test_data')
    output_dir = os.path.join(video_dir, 'results')
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all video pairs
    for video1_path in video_dir.glob("Camera_1_*.mp4"):
        date = video1_path.stem.split('_')[-1]
        video2_path = video_dir / f"Camera_2_{date}.mp4"
        
        if video2_path.exists():
            print(f"Processing videos for date: {date}")
            tracks_cam1, tracks_cam2 = process_video_pair_with_tracking(
                str(video1_path), str(video2_path)
            )
            
            # Save tracking results
            output_file = output_dir / f"tracks_{date}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump({
                    'camera1': tracks_cam1,
                    'camera2': tracks_cam2
                }, f)

if __name__ == "__main__":
    main()
