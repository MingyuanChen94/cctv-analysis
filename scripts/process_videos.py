import os
import cv2
from datetime import datetime
import pandas as pd
from collections import defaultdict
import gc
import psutil
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

class VideoProcessor:
    """
    A class to process and combine CCTV video files.
    
    This class handles the combination of video files from multiple CCTV cameras,
    organizing them by date and camera, and generates a report of the processing.
    """
    
    def __init__(self, memory_limit_gb: float = 14):
        """
        Initialize the VideoProcessor.
        
        Args:
            memory_limit_gb (float): Memory limit in GB (default 14GB)
        """
        self.memory_limit_gb = memory_limit_gb
        self._timeout_ms = 86400000  # 24 hours in milliseconds
        
        # Set OpenCV's FFMPEG buffer sizes
        # buffer_size: 4GB (4 * 1024 * 1024 * 1024)
        # max_packet_queue_size: 1GB (1024 * 1024 * 1024)
        buffer_options = [
            f'buffer_size={4 * 1024 * 1024 * 1024}',
            f'max_packet_queue_size={1024 * 1024 * 1024}',
            'rtbufsize=2147483648',  # 2GB real-time buffer
            'probesize=2147483648',   # 2GB probe size
            'analyzeduration=10000000'  # 10 seconds analyze duration
        ]
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = '|'.join(buffer_options)
        
        # Disable debug messages
        os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
    
    def _parse_timestamp(self, filename: str) -> Optional[datetime]:
        """
        Extract datetime from filename in D04_YYYYMMDDHHMMSS or D10_YYYYMMDDHHMMSS format.
        
        Args:
            filename (str): Name of the video file
            
        Returns:
            Optional[datetime]: Parsed datetime object or None if parsing fails
        """
        try:
            timestamp_str = filename.split('_')[1].split('.')[0]
            return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
        except (IndexError, ValueError) as e:
            print(f"Error parsing filename {filename}: {e}")
            return None

    def _get_video_properties(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Get video properties using OpenCV.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing video properties or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_path}")
                return None
            
            # Set timeouts
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._timeout_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._timeout_ms)
            
            properties = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            
            cap.release()
            return properties
        except Exception as e:
            print(f"Error getting video properties for {video_path}: {e}")
            return None

    def _get_video_duration(self, video_path: str) -> float:
        """
        Get duration of a video file in seconds.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            float: Duration in seconds
        """
        props = self._get_video_properties(video_path)
        if props and props['fps'] > 0:
            return props['frame_count'] / props['fps']
        return 0

    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in GB.
        
        Returns:
            float: Current memory usage in GB
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024 / 1024

    def _process_video_batch(self, 
                           frames_batch: List[Any], 
                           video_writer: cv2.VideoWriter) -> None:
        """
        Process a batch of video frames.
        
        Args:
            frames_batch (List): List of frames to process
            video_writer (cv2.VideoWriter): VideoWriter object to write frames
        """
        for frame in frames_batch:
            video_writer.write(frame)
        frames_batch.clear()

    def _organize_videos(self, 
                        input_folder: str) -> Dict[str, Dict[str, List[Tuple[str, datetime]]]]:
        """
        Organize videos by date and camera.
        
        Args:
            input_folder (str): Path to input folder
            
        Returns:
            Dict: Organized videos by date and camera
        """
        videos_by_date = defaultdict(lambda: defaultdict(list))
        
        for filename in os.listdir(input_folder):
            if not (filename.startswith('D04_') or filename.startswith('D10_')):
                continue
            
            camera = 'Camera1' if filename.startswith('D10_') else 'Camera2'
            timestamp = self._parse_timestamp(filename)
            
            if timestamp:
                date_str = timestamp.strftime('%Y%m%d')
                full_path = os.path.join(input_folder, filename)
                videos_by_date[date_str][camera].append((full_path, timestamp))
        
        return videos_by_date

    def combine_videos(self, 
                      input_folder: str, 
                      output_folder: str) -> None:
        """
        Combine CCTV videos by date and camera, and generate a CSV report.
        
        Args:
            input_folder (str): Path to folder containing source videos
            output_folder (str): Path to folder where combined videos will be saved
        """
        # Create output folder
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Organize videos
        print("Scanning input folder for videos...")
        videos_by_date = self._organize_videos(input_folder)
        
        # Process videos
        report_data = []
        
        for date_str, cameras in videos_by_date.items():
            for camera, video_list in cameras.items():
                report_entry = self._process_camera_videos(
                    date_str, camera, video_list, output_path
                )
                if report_entry:
                    report_data.append(report_entry)
        
        # Generate report
        self._generate_report(report_data, output_path)

    def _process_camera_videos(self, 
                             date_str: str, 
                             camera: str, 
                             video_list: List[Tuple[str, datetime]], 
                             output_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process videos for a specific camera and date.
        
        Args:
            date_str (str): Date string
            camera (str): Camera identifier
            video_list (List): List of video paths and timestamps
            output_path (Path): Output directory path
            
        Returns:
            Optional[Dict]: Report entry for the processed videos
        """
        print(f"\nProcessing {camera} for date {date_str}")
        
        sorted_videos = sorted(video_list, key=lambda x: x[1])
        if not sorted_videos:
            return None
        
        output_filename = f"{camera}_{date_str}.mp4"
        output_file = output_path / output_filename
        
        # Get properties and setup writer
        first_video_props = self._get_video_properties(sorted_videos[0][0])
        if not first_video_props:
            print(f"Error: Could not get properties for first video of {camera} on {date_str}")
            return None
        
        writer = self._setup_video_writer(output_file, first_video_props)
        if not writer:
            return None
        
        # Process videos
        total_duration = 0
        frames_processed = 0
        batch_size = self._calculate_batch_size(first_video_props)
        
        print(f"Processing with batch size: {batch_size} frames")
        
        try:
            frames_processed, total_duration = self._combine_video_files(
                sorted_videos, writer, batch_size
            )
            
            return {
                'Date': datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d'),
                'Camera': camera,
                'Duration_Seconds': round(total_duration, 2),
                'Duration_Minutes': round(total_duration / 60, 2),
                'Output_File': output_filename,
                'Number_of_Source_Videos': len(sorted_videos),
                'Total_Frames': frames_processed
            }
        finally:
            writer.release()
            gc.collect()

    def _setup_video_writer(self, 
                          output_file: Path, 
                          video_props: Dict[str, Any]) -> Optional[cv2.VideoWriter]:
        """
        Set up the video writer.
        
        Args:
            output_file (Path): Path to output file
            video_props (Dict): Video properties
            
        Returns:
            Optional[cv2.VideoWriter]: Configured video writer or None if failed
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_file),
            fourcc,
            video_props['fps'],
            (video_props['width'], video_props['height'])
        )
        
        if not writer.isOpened():
            print(f"Error: Could not create output video {output_file}")
            return None
            
        return writer

    def _calculate_batch_size(self, video_props: Dict[str, Any]) -> int:
        """
        Calculate optimal batch size based on frame size and target memory usage.
        
        Args:
            video_props (Dict): Video properties
            
        Returns:
            int: Calculated batch size
        """
        frame_size = video_props['width'] * video_props['height'] * 3
        target_batch_memory = 500 * 1024 * 1024  # 500MB
        return max(1, min(100, target_batch_memory // frame_size))

    def _combine_video_files(self, 
                           sorted_videos: List[Tuple[str, datetime]], 
                           writer: cv2.VideoWriter, 
                           batch_size: int) -> Tuple[int, float]:
        """
        Combine multiple video files into one.
        
        Args:
            sorted_videos (List): List of video paths and timestamps
            writer (cv2.VideoWriter): Video writer object
            batch_size (int): Number of frames to process in each batch
            
        Returns:
            Tuple[int, float]: Total frames processed and total duration
        """
        frames_processed = 0
        total_duration = 0
        
        for video_path, _ in sorted_videos:
            frames_processed, duration = self._process_single_video(
                video_path, writer, batch_size, frames_processed
            )
            total_duration += duration
            
        return frames_processed, total_duration

    def _process_single_video(self, 
                            video_path: str, 
                            writer: cv2.VideoWriter, 
                            batch_size: int, 
                            frames_processed: int,
                            max_retries: int = 3) -> Tuple[int, float]:
        """
        Process a single video file with retry mechanism.
        
        Args:
            video_path (str): Path to video file
            writer (cv2.VideoWriter): Video writer object
            batch_size (int): Number of frames to process in each batch
            frames_processed (int): Current count of processed frames
            max_retries (int): Maximum number of retries for failed frame reads
            
        Returns:
            Tuple[int, float]: Updated frames count and video duration
        """
        print(f"Processing video: {os.path.basename(video_path)}")
        
        for attempt in range(max_retries):
            try:
                # Open video with explicit FFMPEG backend and increased buffer
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                
                # Set capture properties
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._timeout_ms)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._timeout_ms)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 4096)  # Increased internal buffer
                
                # Additional FFMPEG-specific options
                cap_opts = {
                    'video_size': '1920x1080',  # Adjust based on your video size
                    'buffer_size': str(4 * 1024 * 1024 * 1024),  # 4GB buffer
                    'max_packet_queue_size': str(1024 * 1024 * 1024),  # 1GB queue
                    'rtbufsize': '2147483648',  # 2GB real-time buffer
                    'thread_queue_size': '1024'  # Increased thread queue
                }
                
                for opt, val in cap_opts.items():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                
                if not cap.isOpened():
                    print(f"Attempt {attempt + 1}: Could not open video {video_path}")
                    continue
                
                frames_batch = []
                consecutive_failures = 0
                frame_position = 0
                
                while True:
                    try:
                        ret, frame = cap.read()
                        if not ret:
                            # Verify if we've reached the end or if it's an error
                            if frame_position >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                                break
                            
                            consecutive_failures += 1
                            if consecutive_failures > 5:  # Allow some retries per frame
                                print(f"Too many consecutive failures at frame {frame_position}")
                                break
                                
                            # Try to recover by seeking to the next frame
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position + 1)
                            continue
                            
                        consecutive_failures = 0
                        frame_position += 1
                        
                        frames_batch.append(frame)
                        frames_processed += 1
                        
                        if len(frames_batch) >= batch_size:
                            self._process_video_batch(frames_batch, writer)
                            self._check_memory_usage()
                            
                            if frames_processed % 1000 == 0:
                                print(f"Processed {frames_processed} frames...")
                    
                    except cv2.error as e:
                        print(f"OpenCV error at frame {frame_position}: {e}")
                        consecutive_failures += 1
                        if consecutive_failures > 5:
                            break
                        continue
                
                # Process remaining frames
                if frames_batch:
                    self._process_video_batch(frames_batch, writer)
                
                duration = self._get_video_duration(video_path)
                print(f"Completed processing {os.path.basename(video_path)}")
                return frames_processed, duration
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {video_path}: {e}")
                if attempt < max_retries - 1:
                    print("Retrying...")
                    continue
                else:
                    print("Max retries reached, skipping video")
                    return frames_processed, 0
                    
            finally:
                if 'cap' in locals():
                    cap.release()
                gc.collect()
        
        return frames_processed, 0

    def _check_memory_usage(self) -> None:
        """Check current memory usage and perform garbage collection if needed."""
        memory_usage = self._get_memory_usage()
        print(f"Current memory usage: {memory_usage:.2f} GB")
        
        if memory_usage > self.memory_limit_gb * 0.8:
            gc.collect()
            print("Performed garbage collection")

    def _generate_report(self, 
                        report_data: List[Dict[str, Any]], 
                        output_path: Path) -> None:
        """
        Generate and save the processing report.
        
        Args:
            report_data (List): List of report entries
            output_path (Path): Output directory path
        """
        if report_data:
            df = pd.DataFrame(report_data)
            report_path = output_path / 'video_combination_report.csv'
            df.to_csv(report_path, index=False)
            print(f"\nReport saved to {report_path}")
        else:
            print("\nNo videos were processed")

def main():
    """Main function for command line usage."""
    input_folder = r'C:\Users\mc1159\OneDrive - University of Exeter\Documents\VISIONARY\Durham Experiment\Data'
    output_folder = r'C:\Users\mc1159\OneDrive - University of Exeter\Documents\VISIONARY\Durham Experiment\processed_data'
    
    processor = VideoProcessor(memory_limit_gb=14)
    processor.combine_videos(input_folder, output_folder)

if __name__ == "__main__":
    main()
