import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str, level: str = "INFO", log_file: Optional[str] = None, console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the module.

    Args:
        name (str): Logger name
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (Optional[str]): Path to log file
        console (bool): Whether to log to console

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamp_str() -> str:
    """Get current timestamp as formatted string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_process_logging(process_name: str, base_dir: str = "logs") -> logging.Logger:
    """
    Set up logging for a specific process with timestamped file.

    Args:
        process_name (str): Name of the process
        base_dir (str): Base directory for log files

    Returns:
        logging.Logger: Configured logger
    """
    timestamp = get_timestamp_str()
    log_file = Path(base_dir) / f"{process_name}_{timestamp}.log"

    return setup_logging(
        name=process_name, level="INFO", log_file=str(log_file), console=True
    )


# Example usage
if __name__ == "__main__":
    # Set up basic logging
    logger = setup_logging("test_logger")
    logger.info("Test message")

    # Set up process-specific logging
    process_logger = setup_process_logging("camera_analysis")
    process_logger.info("Started camera analysis")
