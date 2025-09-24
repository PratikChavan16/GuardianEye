# utils/exceptions.py
"""
GuardianEye Custom Exceptions
Centralized exception handling for the GuardianEye system.
"""

class GuardianEyeException(Exception):
    """Base exception for GuardianEye system."""
    pass

class InferenceError(GuardianEyeException):
    """Exception raised during inference processing."""
    pass

class TrackingError(GuardianEyeException):
    """Exception raised during vehicle tracking."""
    pass

class OptimizationError(GuardianEyeException):
    """Exception raised during traffic optimization."""
    pass

class ConfigurationError(GuardianEyeException):
    """Exception raised for configuration issues."""
    pass

class FileProcessingError(GuardianEyeException):
    """Exception raised during file processing."""
    pass

class ValidationError(GuardianEyeException):
    """Exception raised for data validation failures."""
    pass