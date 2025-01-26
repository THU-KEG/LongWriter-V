import logging
from pathlib import Path
from datetime import datetime
import os

def setup_logger():
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create date-specific directory
    date_dir = log_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    date_dir.mkdir(exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('annotation')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler
    log_file = date_dir / 'annotation.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create global logger instance
logger = setup_logger() 