# scripts/process_videos.py

import argparse
from pathlib import Path
import sys
import logging
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Dict, Any
import traceback
import psutil
import os

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.cctv_analysis.utils.video_processor import VideoProcessor

# Constants for resource management
TOTAL_CPUS = 16
RAM_PER_CORE_MB = 16384  # 16GB per core
# No need to be conservative with memory since each core has its own 16GB
MAX_WORKERS = TOTAL_CPUS  # Use all cores

def setup_directory_structure(output_dir: Path):
    """Create output directory structure"""
    # Create main directories
    processed_dir = output_dir / 'processed'
    metadata_dir = output_dir / 'metadata'
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'processed_dir': processed_dir,
        'metadata_dir': metadata_dir
    }

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = Path(output_dir) / 'video_processing.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def set_process_resources():
    """Set CPU affinity for worker processes"""
    # Set CPU affinity (each process gets one core)
    process = psutil.Process()
    try:
        # Get the process ID modulo number of CPUs to distribute across cores
        cpu_id = os.getpid() % TOTAL_CPUS
        process.cpu_affinity([cpu_id])
    except Exception as e:
        logging.warning(f"Could not set CPU affinity: {e}")

def process_video_chunk(chunk: List[Path], output_dir: Path, **kwargs) -> List[Dict[str, Any]]:
    """
    Process a chunk of videos in parallel
    
    Args:
        chunk: List of video file paths to process
        output_dir: Directory for processed output
        **kwargs: Additional arguments for video processing
    
    Returns:
        List of metadata records for processed videos
    """
    try:
        # Set CPU affinity for this worker
        set_process_resources()
        
        # Initialize processor for this chunk
        processor = VideoProcessor(str(chunk[0].parent), str(output_dir))
        
        # Process videos
        metadata_records = []
        worker_id = os.getpid() % TOTAL_CPUS
        
        for video_path in chunk:
            try:
                # Log processing start
                logging.debug(f"Worker {worker_id} (Core {worker_id}) starting {video_path.name}")
                start_time = time.time()
                
                # Process the video
                result = processor.process_single_video(video_path)
                if result:
                    metadata_records.append(result)
                    
                # Log processing completion and time
                processing_time = time.time() - start_time
                logging.debug(f"Worker {worker_id} completed {video_path.name} in {processing_time:.2f}s")
                    
            except Exception as e:
                logging.error(f"Error processing {video_path}: {e}")
                continue
                
        return metadata_records
        
    except Exception as e:
        logging.error(f"Chunk processing error: {e}\n{traceback.format_exc()}")
        return []

def chunk_files(files: List[Path], chunk_size: int = None) -> List[List[Path]]:
    """Split files into chunks for parallel processing"""
    total_files = len(files)
    
    # If chunk size not specified, distribute files evenly across cores
    if chunk_size is None:
        chunk_size = max(1, total_files // TOTAL_CPUS)
        if total_files % TOTAL_CPUS:
            chunk_size += 1
    
    return [files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

def main():
    parser = argparse.ArgumentParser(description='Process CCTV videos in parallel')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Input directory containing raw videos')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for processed videos')
    parser.add_argument('--skip-verify', action='store_true',
                      help='Skip verification of processed videos')
    parser.add_argument('--chunk-size', type=int, default=None,
                      help='Number of videos to process per worker (default: auto-calculated)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup output directory and logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        start_time = time.time()
        logging.info(f"Starting parallel video processing with {MAX_WORKERS} workers")
        logging.info(f"Available RAM per core: {RAM_PER_CORE_MB}MB")
        
        # Get list of input videos
        input_dir = Path(args.input_dir)
        video_files = sorted(list(input_dir.glob('*.mp4')))  # Adjust extension pattern as needed
        
        if not video_files:
            logging.error("No video files found in input directory")
            sys.exit(1)
            
        logging.info(f"Found {len(video_files)} videos to process")
        
        # Split files into chunks
        chunks = chunk_files(video_files, args.chunk_size)
        chunk_sizes = [len(chunk) for chunk in chunks]
        logging.info(f"Split videos into {len(chunks)} chunks: {chunk_sizes}")
        
        # Process chunks in parallel
        with Pool(MAX_WORKERS) as pool:
            # Create partial function with fixed output_dir
            process_func = partial(process_video_chunk, output_dir=output_dir)
            
            # Map chunks to worker processes
            chunk_results = pool.map(process_func, chunks)
            
        # Combine results from all chunks
        metadata_records = [
            record 
            for chunk_record in chunk_results 
            for record in chunk_record
        ]
        
        processed_count = len(metadata_records)
        total_count = len(video_files)
        success_rate = (processed_count / total_count) * 100 if total_count > 0 else 0
        
        logging.info(f"Processed {processed_count}/{total_count} videos ({success_rate:.1f}% success rate)")
        
        # Verify processed videos if requested
        if not args.skip_verify:
            logging.info("Verifying processed videos...")
            processor = VideoProcessor(args.input_dir, args.output_dir)
            verification_results = processor.verify_processed_videos()
            
        # Calculate and log performance metrics
        processing_time = time.time() - start_time
        videos_per_second = total_count / processing_time if processing_time > 0 else 0
        seconds_per_video = processing_time / total_count if total_count > 0 else 0
        
        logging.info("Performance Summary:")
        logging.info(f"Total processing time: {processing_time:.2f} seconds")
        logging.info(f"Average processing speed: {videos_per_second:.2f} videos/second")
        logging.info(f"Average time per video: {seconds_per_video:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
