# utils/__init__.py
"""
GuardianEye Utilities Package
Common utilities for logging, validation, and error handling.
"""

from .logger import get_logger, logger_instance
from .exceptions import (
    GuardianEyeException,
    InferenceError,
    TrackingError,
    OptimizationError,
    ConfigurationError,
    FileProcessingError,
    ValidationError
)
from .validators import (
    validate_video_file,
    validate_junction_id,
    validate_json_file,
    validate_detections_data,
    validate_stats_data,
    validate_unique_counts_data,
    validate_uid_format
)

__all__ = [
    'get_logger',
    'logger_instance',
    'GuardianEyeException',
    'InferenceError',
    'TrackingError',
    'OptimizationError',
    'ConfigurationError',
    'FileProcessingError',
    'ValidationError',
    'validate_video_file',
    'validate_junction_id',
    'validate_json_file',
    'validate_detections_data',
    'validate_stats_data',
    'validate_unique_counts_data',
    'validate_uid_format'
]