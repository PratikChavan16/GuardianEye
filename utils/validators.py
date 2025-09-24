# utils/validators.py
"""
GuardianEye Validation Utilities
Input validation and data validation functions.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from .exceptions import ValidationError

def validate_video_file(file_path: Path) -> bool:
    """Validate that a file is a supported video format."""
    if not file_path.exists():
        raise ValidationError(f"Video file does not exist: {file_path}")
    
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    if file_path.suffix.lower() not in supported_formats:
        raise ValidationError(f"Unsupported video format: {file_path.suffix}")
    
    if file_path.stat().st_size == 0:
        raise ValidationError("Video file is empty")
    
    return True

def validate_junction_id(junction_id: str, valid_junctions: List[str]) -> bool:
    """Validate that a junction ID is valid."""
    if not junction_id:
        raise ValidationError("Junction ID cannot be empty")
    
    if junction_id not in valid_junctions:
        raise ValidationError(f"Invalid junction ID: {junction_id}. Valid IDs: {valid_junctions}")
    
    return True

def validate_json_file(file_path: Path) -> Dict[str, Any]:
    """Validate and load a JSON file."""
    if not file_path.exists():
        raise ValidationError(f"JSON file does not exist: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON format in {file_path}: {e}")

def validate_detections_data(detections: List[Dict[str, Any]]) -> bool:
    """Validate detection data structure."""
    required_fields = ['bbox', 'class', 'confidence']
    
    for i, detection in enumerate(detections):
        for field in required_fields:
            if field not in detection:
                raise ValidationError(f"Detection {i} missing required field: {field}")
        
        # Validate bbox format
        bbox = detection['bbox']
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValidationError(f"Detection {i} has invalid bbox format: {bbox}")
        
        # Validate confidence
        confidence = detection['confidence']
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValidationError(f"Detection {i} has invalid confidence: {confidence}")
    
    return True

def validate_stats_data(stats: Dict[str, Any]) -> bool:
    """Validate stats data structure."""
    required_fields = ['total_frames', 'vehicle_count', 'per_frame_counts']
    
    for field in required_fields:
        if field not in stats:
            raise ValidationError(f"Stats missing required field: {field}")
    
    # Validate counts are non-negative
    if stats['total_frames'] < 0:
        raise ValidationError("Total frames cannot be negative")
    
    if stats['vehicle_count'] < 0:
        raise ValidationError("Vehicle count cannot be negative")
    
    return True

def validate_unique_counts_data(unique_counts: List[Dict[str, Any]]) -> bool:
    """Validate unique counts data structure."""
    required_fields = ['frame', 'unique_count_60s', 'unique_tracks_in_frame']
    
    for i, entry in enumerate(unique_counts):
        for field in required_fields:
            if field not in entry:
                raise ValidationError(f"Unique counts entry {i} missing required field: {field}")
        
        # Validate frame number
        if not isinstance(entry['frame'], int) or entry['frame'] < 0:
            raise ValidationError(f"Entry {i} has invalid frame number: {entry['frame']}")
        
        # Validate counts are non-negative
        if entry['unique_count_60s'] < 0:
            raise ValidationError(f"Entry {i} has negative unique count: {entry['unique_count_60s']}")
    
    return True

def validate_uid_format(uid: str) -> bool:
    """Validate UID format (8 character hexadecimal)."""
    if not uid or len(uid) != 8:
        raise ValidationError(f"UID must be 8 characters long: {uid}")
    
    try:
        int(uid, 16)  # Check if it's valid hexadecimal
    except ValueError:
        raise ValidationError(f"UID must be hexadecimal: {uid}")
    
    return True