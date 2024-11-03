#!/usr/bin/env python
"""
Basic example of CCTV video analysis.
Shows simple person detection and tracking functionality.
"""

import argparse
from pathlib import Path
from datetime import datetime

from cctv_analysis import CameraProcessor
from cctv_analysis.utils import (
    VideoProcessor,
    setup_logging,
    create_data_manager
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Basic CCTV video analysis example")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files (default: output)"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save processed video with annotations"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def main():
    """Run basic video analysis."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logging("basic_analysis", level=log_level)
    logger.info(f"Starting analysis of video: {args.video}")
    
    try:
        # Create output directories
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        video_processor = VideoProcessor(output_dir=output_dir / "videos")
        processor = CameraProcessor()
        data_manager = create_data_manager(base_dir=output_dir / "data")
        
        # Get video info
        video_info = video_processor.read_video_info(args.video)
        logger.info(
            f"Video info: {video_info['width']}x{video_info['height']}, "
            f"{video_info['fps']:.2f} fps, {video_info['duration']:.2f} seconds"
        )
        
        # Process video frames
        detections = []
        for frame, timestamp in video_processor.frame_generator(args.video):
            # Detect and track persons
            frame_detections = processor.process_frame(frame, timestamp)
            detections.extend(frame_detections)
            
            # Save processed frame if requested
            if args.save_video:
                annotated_frame = video_processor.draw_detections(frame, frame_detections)
                video_processor.write_frame(annotated_frame)
        
        # Save results
        output_file = output_dir / "data" / "detections.csv"
        data_manager.save_detections_csv(
            detections=detections,
            output_path=output_file
        )
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"Total frames processed: {len(detections)}")
        print(f"Total persons detected: {len(set(d['track_id'] for d in detections))}")
        print(f"\nResults saved to: {output_file}")
        if args.save_video:
            print(f"Processed video saved to: {output_dir}/videos/")
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
