#!/usr/bin/env python3
"""
Progressive video processor for real-time detection streaming during upload.
Processes video frame-by-frame and emits results via WebSocket as they happen.
"""

import cv2
import json
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from collections import deque

class ProgressiveVideoProcessor:
    def __init__(self, uid: str, video_path: Path, model_path: Path, 
                 confidence: float = 0.35, websocket_callback: Optional[Callable] = None):
        """
        Initialize progressive processor.
        websocket_callback: async function to send updates via WebSocket
        """
        self.uid = uid
        self.video_path = video_path
        self.model_path = model_path
        self.confidence = confidence
        self.websocket_callback = websocket_callback
        
        # Processing state
        self.is_processing = False
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 25.0
        self.detections_log = []
        self.running_stats = {
            "total_vehicles": 0,
            "vehicles_per_minute": 0,
            "class_counts": {},
            "progress_percent": 0
        }
        
        # YOLO model (loaded lazily)
        self.model = None
        
    def _load_model(self):
        """Load YOLO model if not already loaded."""
        if self.model is None:
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
                print(f"Model loaded: {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
    
    def _get_video_info(self):
        """Extract video metadata."""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        cap.release()
        print(f"Video info: {self.total_frames} frames, {self.fps} FPS")
    
    async def _emit_progress(self, frame_data: Dict[str, Any]):
        """Emit progress update via WebSocket if callback provided."""
        if self.websocket_callback:
            try:
                await self.websocket_callback({
                    "type": "frame_progress",
                    "uid": self.uid,
                    "frame_index": self.current_frame,
                    "total_frames": self.total_frames,
                    "progress_percent": (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0,
                    "frame_data": frame_data,
                    "running_stats": self.running_stats,
                    "timestamp": time.time()
                })
            except Exception as e:
                print(f"Error emitting progress: {e}")
    
    def _update_running_stats(self, detections):
        """Update cumulative statistics."""
        # Count vehicles in this frame
        vehicle_classes = ["car", "truck", "bus", "van", "bike", "rickshaw", "motorcycle"]
        frame_vehicles = len([d for d in detections if d["class"].lower() in vehicle_classes])
        
        self.running_stats["total_vehicles"] += frame_vehicles
        
        # Update class counts
        for detection in detections:
            cls = detection["class"]
            self.running_stats["class_counts"][cls] = self.running_stats["class_counts"].get(cls, 0) + 1
        
        # Estimate vehicles per minute (rough calculation)
        elapsed_seconds = self.current_frame / self.fps if self.fps > 0 else 1
        self.running_stats["vehicles_per_minute"] = (self.running_stats["total_vehicles"] / elapsed_seconds) * 60 if elapsed_seconds > 0 else 0
        
        self.running_stats["progress_percent"] = (self.current_frame / self.total_frames * 100) if self.total_frames > 0 else 0
    
    async def process_video_progressively(self):
        """Process video frame by frame with real-time updates."""
        self.is_processing = True
        
        try:
            # Load model and get video info
            self._load_model()
            self._get_video_info()
            
            # Open video for processing
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise ValueError("Cannot open video for processing")
            
            # Prepare output video writer
            out_video_path = Path(str(self.video_path).replace('uploads', 'results').replace('.', '_annotated.'))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = None
            
            # Process frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.current_frame += 1
                
                # Run inference on current frame
                results = self.model(frame, conf=self.confidence, verbose=False)
                
                # Extract detections
                detections = []
                annotated_frame = frame.copy()
                
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            class_name = self.model.names[cls]
                            
                            detections.append({
                                "class": class_name,
                                "confidence": float(conf),
                                "bbox": [float(x1), float(y1), float(x2), float(y2)]
                            })
                            
                            # Draw on annotated frame
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"{class_name} {conf:.2f}", 
                                      (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Initialize video writer if first frame
                if out_writer is None and annotated_frame is not None:
                    height, width = annotated_frame.shape[:2]
                    out_writer = cv2.VideoWriter(str(out_video_path), fourcc, self.fps, (width, height))
                
                # Write annotated frame to output video
                if out_writer:
                    out_writer.write(annotated_frame)
                
                # Store frame detection data
                frame_data = {
                    "frame_index": self.current_frame,
                    "timestamp": self.current_frame / self.fps,
                    "detections": detections,
                    "detection_count": len(detections)
                }
                self.detections_log.append(frame_data)
                
                # Update running statistics
                self._update_running_stats(detections)
                
                # Emit progress update (non-blocking)
                await self._emit_progress(frame_data)
                
                # Brief pause to prevent overwhelming
                await asyncio.sleep(0.01)
            
            # Cleanup
            cap.release()
            if out_writer:
                out_writer.release()
            
            # Save final detection and stats files
            await self._save_final_outputs()
            
            # Emit completion
            if self.websocket_callback:
                await self.websocket_callback({
                    "type": "processing_complete",
                    "uid": self.uid,
                    "total_frames": self.total_frames,
                    "final_stats": self.running_stats,
                    "output_files": {
                        "annotated_video": str(out_video_path),
                        "detections": str(out_video_path).replace("_annotated.", "_detections.json"),
                        "stats": str(out_video_path).replace("_annotated.", "_stats.json")
                    }
                })
        
        except Exception as e:
            # Emit error
            if self.websocket_callback:
                await self.websocket_callback({
                    "type": "processing_error",
                    "uid": self.uid,
                    "error": str(e),
                    "frame_index": self.current_frame
                })
            raise
        
        finally:
            self.is_processing = False
    
    async def _save_final_outputs(self):
        """Save final detection and stats JSON files."""
        base_path = str(self.video_path).replace('uploads', 'results').replace(self.video_path.suffix, '')
        
        # Save detections JSON
        detections_path = base_path + "_detections.json"
        with open(detections_path, 'w') as f:
            json.dump({
                "uid": self.uid,
                "video_path": str(self.video_path),
                "total_frames": self.total_frames,
                "fps": self.fps,
                "detections": self.detections_log
            }, f, indent=2)
        
        # Save stats JSON
        stats_path = base_path + "_stats.json"
        
        # Generate per_frame_counts format for compatibility
        per_frame_counts = []
        for frame_data in self.detections_log:
            counts = {}
            for detection in frame_data["detections"]:
                cls = detection["class"]
                counts[cls] = counts.get(cls, 0) + 1
            
            per_frame_counts.append({
                "frame_idx": frame_data["frame_index"] - 1,  # 0-based
                "ts": frame_data["timestamp"],
                "counts": counts
            })
        
        # Generate flow windows (simplified)
        flow_windows = []
        if self.running_stats["vehicles_per_minute"] > 0:
            flow_windows.append({
                "center_frame": self.total_frames // 2,
                "vehicles_per_minute": self.running_stats["vehicles_per_minute"]
            })
        
        stats_data = {
            "per_frame_counts": per_frame_counts,
            "flow_windows": flow_windows,
            "summary": {
                "total_frames": self.total_frames,
                "total_vehicles_detected": self.running_stats["total_vehicles"]
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"Final outputs saved: {detections_path}, {stats_path}")


# Global registry for active processors
active_processors: Dict[str, ProgressiveVideoProcessor] = {}

def get_processor(uid: str) -> Optional[ProgressiveVideoProcessor]:
    """Get active processor by UID."""
    return active_processors.get(uid)

def start_progressive_processing(uid: str, video_path: Path, model_path: Path, 
                               confidence: float = 0.35, websocket_callback: Optional[Callable] = None):
    """Start progressive processing for a video upload."""
    processor = ProgressiveVideoProcessor(uid, video_path, model_path, confidence, websocket_callback)
    active_processors[uid] = processor
    return processor

def stop_progressive_processing(uid: str):
    """Stop and remove progressive processor."""
    if uid in active_processors:
        processor = active_processors[uid]
        processor.is_processing = False
        del active_processors[uid]