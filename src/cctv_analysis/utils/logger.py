import logging
from pathlib import Path
from datetime import datetime
import sys

def setup_logger(name: str, log_file: Path = None) -> logging.Logger:
    """Set up logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

class VideoLogger:
    """Logger specifically for video processing with progress tracking"""
    def __init__(self, name: str, total_frames: int, log_dir: Path = None):
        self.name = name
        self.total_frames = total_frames
        self.current_frame = 0
        
        # Set up logging
        if log_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{name}_{timestamp}.log"
        else:
            log_file = None
        
        self.logger = setup_logger(name, log_file)
    
    def update(self, frames_processed: int, message: str = None):
        """Update progress and optionally log a message"""
        self.current_frame += frames_processed
        progress = (self.current_frame / self.total_frames) * 100
        
        status = f"Progress: {progress:.1f}% ({self.current_frame}/{self.total_frames})"
        if message:
            status = f"{message} - {status}"
        
        self.logger.info(status)
    
    def info(self, message: str):
        """Log an info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log an error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log a debug message"""
        self.logger.debug(message)
