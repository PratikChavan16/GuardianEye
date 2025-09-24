# utils/logger.py
"""
GuardianEye Logging System
Centralized logging configuration for the entire project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class GuardianEyeLogger:
    """Centralized logger for GuardianEye system."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path(__file__).parent.parent / "backend" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup file handler for all logs
        log_file = self.log_dir / f"guardianeye_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        root_logger.propagate = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific module."""
        return logging.getLogger(name)
    
    def log_inference(self, uid: str, message: str, level: str = "INFO"):
        """Log inference-specific messages."""
        logger = self.get_logger(f"inference.{uid}")
        getattr(logger, level.lower())(message)
    
    def log_tracking(self, uid: str, message: str, level: str = "INFO"):
        """Log tracking-specific messages."""
        logger = self.get_logger(f"tracking.{uid}")
        getattr(logger, level.lower())(message)
    
    def log_optimization(self, message: str, level: str = "INFO"):
        """Log optimization-specific messages."""
        logger = self.get_logger("optimization")
        getattr(logger, level.lower())(message)
    
    def log_api(self, message: str, level: str = "INFO"):
        """Log API-specific messages."""
        logger = self.get_logger("api")
        getattr(logger, level.lower())(message)

# Global logger instance
logger_instance = GuardianEyeLogger()

def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module."""
    return logger_instance.get_logger(name)