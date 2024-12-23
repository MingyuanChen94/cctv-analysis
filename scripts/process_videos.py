# scripts/process_videos.py

import argparse
from pathlib import Path
import sys
import logging
import time

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.cctv_analysis.utils.video_processor import VideoProcessor

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

def main():
    parser = argparse.ArgumentParser(description='Process CCTV videos')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Input directory containing raw videos')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for processed videos')
    parser.add_argument('--skip-verify', action='store_true',
                      help='Skip verification of processed videos')
    
    args = parser.parse_args()
    
    # Setup output directory and logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    
    try:
        start_time = time.time()
        logging.info("Starting video processing...")
        
        # Initialize processor
        processor = VideoProcessor(args.input_dir, args.output_dir)
        
        # Process videos
        metadata_records = processor.combine_daily_videos()
        logging.info(f"Processed {len(metadata_records)} videos")
        
        # Verify processed videos
        if not args.skip_verify:
            logging.info("Verifying processed videos...")
            verification_results = processor.verify_processed_videos()
            
        # Calculate processing time
        processing_time = time.time() - start_time
        logging.info(f"Processing completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
