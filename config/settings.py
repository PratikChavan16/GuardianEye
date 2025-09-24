# config/settings.py
"""
GuardianEye Configuration Settings
Centralized configuration for the entire GuardianEye traffic optimization system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

class GuardianEyeConfig:
    """Centralized configuration management for GuardianEye."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment and config files."""
        # Directory paths
        self.upload_dir = self.root_dir / "backend" / "uploads"
        self.result_dir = self.root_dir / "backend" / "results"
        self.aggregates_dir = self.root_dir / "backend" / "aggregates"
        self.tracks_dir = self.root_dir / "backend" / "tracks"
        self.logs_dir = self.root_dir / "backend" / "logs"
        self.models_dir = self.root_dir / "models"
        self.config_dir = self.root_dir / "config"
        
        # Ensure directories exist
        for directory in [self.upload_dir, self.result_dir, self.aggregates_dir, 
                         self.tracks_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.mappings_file = self.root_dir / "backend" / "mappings.json"
        self.history_file = self.root_dir / "backend" / "history.json"
        self.junctions_file = self.config_dir / "junctions.json"
        
        # Model paths
        self.custom_model_path = self.models_dir / "best.pt"
        self.yolo_fallback_path = self.root_dir / "yolov8n.pt"
        
        # Inference settings
        self.max_frames = int(os.getenv("GUARDIAN_MAX_FRAMES", "300"))
        self.image_size = int(os.getenv("GUARDIAN_IMAGE_SIZE", "640"))
        self.confidence_threshold = float(os.getenv("GUARDIAN_CONFIDENCE", "0.5"))
        self.iou_threshold = float(os.getenv("GUARDIAN_IOU", "0.5"))
        
        # Vehicle classes (COCO format)
        self.vehicle_classes = [1, 2, 3, 5, 6, 7]  # bicycle, car, motorcycle, bus, train, truck
        
        # Tracking settings
        self.tracking_window_seconds = 60
        self.tracker_max_disappeared = 30
        self.tracker_max_distance = 50
        
        # Server settings
        self.server_host = os.getenv("GUARDIAN_HOST", "0.0.0.0")
        self.server_port = int(os.getenv("GUARDIAN_PORT", "8000"))
        
        # ATCS optimization settings
        self.cycle_time = 120  # seconds
        self.min_green_time = 7  # minimum green time per junction
        self.max_green_time = 60  # maximum green time per junction
        self.yellow_time = 3  # yellow phase duration
        self.all_red_time = 1  # all-red clearance time
        
        # Logging settings
        self.log_level = os.getenv("GUARDIAN_LOG_LEVEL", "INFO")
        self.max_history_entries = 1000
        
        # Initialize data files
        self._initialize_data_files()
    
    def _initialize_data_files(self):
        """Initialize JSON data files if they don't exist."""
        # Initialize mappings file
        if not self.mappings_file.exists():
            self.mappings_file.write_text(json.dumps({"mappings": []}, indent=2))
        
        # Initialize history file
        if not self.history_file.exists():
            self.history_file.write_text(json.dumps({"history": []}, indent=2))
        
        # Initialize junctions file if it doesn't exist
        if not self.junctions_file.exists():
            default_junctions = {
                "junctions": [
                    {"id": "J1", "x": 100, "y": 200, "neighbors": [{"to": "J2", "distance": 500}, {"to": "J3", "distance": 700}]},
                    {"id": "J2", "x": 300, "y": 200, "neighbors": [{"to": "J1", "distance": 500}, {"to": "J4", "distance": 600}]},
                    {"id": "J3", "x": 100, "y": 400, "neighbors": [{"to": "J1", "distance": 700}, {"to": "J5", "distance": 450}]},
                    {"id": "J4", "x": 500, "y": 200, "neighbors": [{"to": "J2", "distance": 600}, {"to": "J6", "distance": 800}]},
                    {"id": "J5", "x": 100, "y": 600, "neighbors": [{"to": "J3", "distance": 450}, {"to": "J6", "distance": 700}]},
                    {"id": "J6", "x": 500, "y": 600, "neighbors": [{"to": "J4", "distance": 800}, {"to": "J5", "distance": 700}]}
                ]
            }
            self.junctions_file.write_text(json.dumps(default_junctions, indent=2))
    
    def get_model_path(self) -> Path:
        """Get the appropriate model path (custom or fallback)."""
        if self.custom_model_path.exists():
            return self.custom_model_path
        return self.yolo_fallback_path
    
    def get_inference_script_path(self) -> Path:
        """Get the inference script path."""
        return self.root_dir / "inference" / "run_inference_and_annotate.py"
    
    def get_tracker_script_path(self) -> Path:
        """Get the tracker script path."""
        return self.root_dir / "tracker" / "tracker_service.py"
    
    @property
    def vehicle_class_names(self) -> Dict[int, str]:
        """Get vehicle class names mapping."""
        return {
            1: "bicycle",
            2: "car", 
            3: "motorcycle",
            5: "bus",
            6: "train",
            7: "truck"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for API responses."""
        return {
            "max_frames": self.max_frames,
            "image_size": self.image_size,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "vehicle_classes": self.vehicle_classes,
            "tracking_window_seconds": self.tracking_window_seconds,
            "cycle_time": self.cycle_time,
            "min_green_time": self.min_green_time,
            "max_green_time": self.max_green_time
        }

# Global configuration instance
config = GuardianEyeConfig()