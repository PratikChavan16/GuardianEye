#!/usr/bin/env python3
"""
Simple video streaming service for real-time detection display.
Captures frames from webcam or video file and runs inference.
"""

import cv2
import json
import time
import base64
from pathlib import Path
import threading
from collections import deque

class VideoStreamer:
    def __init__(self, source=0, model_path="models/best.pt", confidence=0.35, 
                 max_fps=10, frame_skip=1, resize_width=640):
        """
        source: 0 for webcam, or path to video file
        model_path: path to YOLO model
        confidence: detection confidence threshold
        max_fps: maximum processing/streaming FPS (default 10)
        frame_skip: process every Nth frame (1=all, 2=every 2nd, etc)
        resize_width: resize frames to this width for inference (maintains aspect ratio)
        """
        self.source = source
        self.model_path = model_path
        self.confidence = confidence
        self.cap = None
        self.model = None
        self.frame_count = 0
        self.processed_count = 0
        self.detections_buffer = []
        self.latest_jpeg = None  # raw JPEG bytes
        self.latest_detection = None
        self.latest_frame = None
        self.running = False
        self.thread = None
        self.max_fps = max_fps
        self.frame_skip = frame_skip
        self.resize_width = resize_width
        self.frame_interval = 1.0 / self.max_fps
        self.last_emit_time = 0.0
        
    def initialize(self):
        """Initialize video capture and YOLO model."""
        try:
            # Initialize video capture
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise Exception(f"Could not open video source: {self.source}")
            
            # Initialize YOLO model
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            
            print(f"Video streamer initialized: {self.source}")
            # Start background loop
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Error initializing video streamer: {e}")
            return False
    
    def get_frame_with_detections(self):
        """Return latest annotated frame & detection meta (does not run inference itself)."""
        return self.latest_frame, self.latest_detection  # type: ignore

    def _loop(self):
        """Background capture + inference loop."""
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.5)
                continue
            start_tick = time.time()
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.2)
                continue
            self.frame_count += 1

            # Frame skipping for performance
            if self.frame_count % self.frame_skip != 0:
                time.sleep(0.01)  # brief pause to avoid busy loop
                continue

            self.processed_count += 1

            # Resize frame for inference if configured
            inference_frame = frame
            if self.resize_width and frame.shape[1] > self.resize_width:
                aspect = frame.shape[0] / frame.shape[1]
                new_height = int(self.resize_width * aspect)
                inference_frame = cv2.resize(frame, (self.resize_width, new_height))

            # Run YOLO inference on resized frame
            results = self.model(inference_frame, conf=self.confidence, verbose=False)
            detections = []
            
            # Scale detections back to original frame size for annotation
            scale_x = frame.shape[1] / inference_frame.shape[1] if inference_frame.shape[1] != frame.shape[1] else 1.0
            scale_y = frame.shape[0] / inference_frame.shape[0] if inference_frame.shape[0] != frame.shape[0] else 1.0
            
            annotated_frame = frame.copy()
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # Scale coordinates back to original frame
                        x1, x2 = x1 * scale_x, x2 * scale_x
                        y1, y2 = y1 * scale_y, y2 * scale_y
                        
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[cls]
                        detections.append({
                            "class": class_name,
                            "confidence": float(conf),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)]
                        })
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"{class_name} {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            frame_data = {
                "frame": self.frame_count,
                "processed_frame": self.processed_count,
                "timestamp": time.time(),
                "detections": detections,
                "total_vehicles": len([d for d in detections if d["class"] in ["Car", "Bus", "Truck", "Van", "Bike", "Rickshaw"]]),
                "frame_skip": self.frame_skip
            }
            self.detections_buffer.append(frame_data)
            if len(self.detections_buffer) > 300:
                self.detections_buffer = self.detections_buffer[-300:]

            # Encode JPEG every loop (could throttle if heavy)
            ok, buffer = cv2.imencode('.jpg', annotated_frame)
            if ok:
                self.latest_jpeg = buffer.tobytes()
                self.latest_detection = frame_data
                self.latest_frame = annotated_frame  # for compatibility

            # Throttle loop to max_fps
            elapsed = time.time() - start_tick
            sleep_for = self.frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    
    def get_frame_as_jpeg_base64(self):
        """Get latest frame as base64-encoded JPEG (non-blocking)."""
        if self.latest_jpeg is None:
            return None, None
        jpg_base64 = base64.b64encode(self.latest_jpeg).decode('utf-8')
        return jpg_base64, self.latest_detection
    
    def get_recent_stats(self, seconds=60):
        """Get detection statistics for recent frames."""
        cutoff_time = time.time() - seconds
        recent_frames = [f for f in self.detections_buffer if f["timestamp"] > cutoff_time]
        
        if not recent_frames:
            return {"total_frames": 0, "total_vehicles": 0, "avg_vehicles_per_frame": 0}
        
        total_vehicles = sum(f["total_vehicles"] for f in recent_frames)
        
        return {
            "total_frames": len(recent_frames),
            "processed_frames": len([f for f in recent_frames if f.get("processed_frame")]),
            "total_vehicles": total_vehicles,
            "avg_vehicles_per_frame": total_vehicles / len(recent_frames),
            "vehicles_per_minute": total_vehicles * (60 / seconds) if len(recent_frames) > 0 else 0,
            "frame_skip_ratio": f"1:{self.frame_skip}" if hasattr(self, 'frame_skip') else "1:1",
            "performance": {
                "capture_fps": len(recent_frames) / seconds if seconds > 0 else 0,
                "inference_fps": len([f for f in recent_frames if f.get("processed_frame")]) / seconds if seconds > 0 else 0
            }
        }
    
    def cleanup(self):
        """Release resources."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

# Standalone testing
if __name__ == "__main__":
    streamer = VideoStreamer(source=0)  # Use webcam
    
    if streamer.initialize():
        print("Press 'q' to quit")
        
        while True:
            frame, detection_data = streamer.get_frame_with_detections()
            if frame is None:
                break
            
            # Display frame
            cv2.imshow('Live Detection', frame)
            
            # Print detection count every 30 frames
            if streamer.frame_count % 30 == 0:
                stats = streamer.get_recent_stats()
                print(f"Frame {streamer.frame_count}: {detection_data['total_vehicles']} vehicles, "
                      f"Recent avg: {stats['avg_vehicles_per_frame']:.1f} vehicles/frame")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        streamer.cleanup()
        cv2.destroyAllWindows()
    else:
        print("Failed to initialize video streamer")