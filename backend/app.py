# backend/app.py
"""
GuardianEye Backend API
Main FastAPI application for video upload, processing, and traffic optimization.
"""

import uuid
import asyncio
import subprocess
import time
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Import our custom modules
from config.settings import config
from utils.logger import get_logger
from utils.exceptions import GuardianEyeException, FileProcessingError, ValidationError
from utils.validators import validate_video_file, validate_junction_id, validate_uid_format
from inference.postprocess_counts import compute_stats

# Initialize logger
logger = get_logger("api")

def read_mappings() -> Dict[str, Any]:
    """Read junction mappings from file."""
    try:
        with open(config.mappings_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading mappings: {e}")
        return {"mappings": []}

def write_mappings(data: Dict[str, Any]) -> None:
    """Write junction mappings to file."""
    try:
        with open(config.mappings_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Mappings updated successfully")
    except Exception as e:
        logger.error(f"Error writing mappings: {e}")
        raise FileProcessingError(f"Failed to update mappings: {e}")

def log_historical_data(demands: Dict[str, float], greens: Dict[str, float], total_demand: float) -> None:
    """Log current demand and optimization data for historical analysis."""
    try:
        with open(config.history_file, 'r') as f:
            history_data = json.load(f)
        
        entry = {
            "timestamp": time.time(),
            "demands": demands,
            "greens": greens,
            "total_demand": total_demand
        }
        history_data["history"].append(entry)
        
        # Keep only last entries to prevent file from growing too large
        if len(history_data["history"]) > config.max_history_entries:
            history_data["history"] = history_data["history"][-config.max_history_entries:]
        
        with open(config.history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Historical data logged: total_demand={total_demand}")
    except Exception as e:
        logger.error(f"Error logging historical data: {e}")

app = FastAPI(
    title="GuardianEye Traffic Optimization API",
    description="Video upload, processing, and traffic signal optimization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """API root endpoint with system information."""
    return {
        "name": "GuardianEye Traffic Optimization API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "upload": "POST /upload - Upload video for processing",
            "status": "GET /status/{uid} - Check processing status",
            "results": "GET /results/{filename} - Download result files",
            "junctions": "GET /junctions - Get junction configuration",
            "optimization": "GET /optimizer/plan - Get current optimization plan",
            "mappings": "GET /list_mappings - List junction mappings"
        },
        "description": "Upload traffic videos for vehicle detection, tracking, and signal optimization."
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0"
    }

@app.get("/junctions")
def get_junctions():
    """Return junction map configuration."""
    try:
        with open(config.junctions_file, 'r') as f:
            data = json.load(f)
        logger.info("Junction configuration retrieved successfully")
        return JSONResponse(data)
    except Exception as e:
        logger.error(f"Error loading junction configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration error: {e}")

@app.get("/favicon.ico")
def favicon():
    """Return empty favicon response."""
    return Response(status_code=204)

def run_inference_process(video_path: Path, out_video: Path, out_json: Path, model_path: Path, 
                         imgsz: int = None, conf: float = None) -> tuple:
    """Invoke inference script with proper model selection and configuration."""
    import sys
    
    # Use configuration values
    imgsz = imgsz or config.image_size
    conf = conf or config.confidence_threshold
    
    # Get appropriate model path
    chosen_model = config.get_model_path()
    if not chosen_model.exists():
        logger.warning(f"Model {chosen_model} not found. Using fallback: {config.yolo_fallback_path}")
        chosen_model = config.yolo_fallback_path
    
    cmd = [
        sys.executable, str(config.get_inference_script_path()),
        "--model", str(chosen_model),
        "--source", str(video_path),
        "--out_video", str(out_video),
        "--out_json", str(out_json),
        "--imgsz", str(imgsz),
        "--conf", str(conf)
    ]
    
    logger.info(f"Running inference with model: {chosen_model}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    if proc.returncode == 0:
        logger.info("Inference completed successfully")
    else:
        logger.error(f"Inference failed with return code {proc.returncode}")
    
    return proc.returncode, proc.stdout, proc.stderr

def postprocess_and_write(detections_json: Path, fps: float, out_stats_json: Path):
    """Compute stats from detection data."""
    compute_stats(str(detections_json), str(out_stats_json), fps=fps, window_seconds=60)

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), model: str = None):
    """Upload a video file for traffic analysis."""
    uid = uuid.uuid4().hex[:8]
    logger.info(f"Upload started for UID: {uid}, filename: {file.filename}")
    
    try:
        # Validate file format
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Setup file paths
        in_path = config.upload_dir / f"{uid}_{file.filename}"
        out_video = config.result_dir / f"{uid}_annotated.mp4"
        out_json = config.result_dir / f"{uid}_detections.json"
        out_stats = config.result_dir / f"{uid}_stats.json"
        
        # Save uploaded file
        with open(in_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        def _job():
            """Background processing job."""
            start = time.time()
            logger.info(f"Starting processing for {uid}")
            
            try:
                # Run inference
                code, out, err = run_inference_process(
                    in_path, out_video, out_json, 
                    config.get_model_path(),
                    imgsz=config.image_size,
                    conf=config.confidence_threshold
                )
                
                duration = time.time() - start
                
                if code != 0:
                    logger.error(f"Inference failed for {uid}: {err}")
                    with open(config.result_dir / f"{uid}_error.log", "w") as f:
                        f.write(f"Exit code: {code}\nStdout:\n{out}\nStderr:\n{err}")
                    return
                
                # Log success
                logger.info(f"Inference completed for {uid} in {duration:.2f}s")
                with open(config.result_dir / f"{uid}_inference.log", "w") as f:
                    f.write(f"Success in {duration:.2f}s\nStdout:\n{out}\nStderr:\n{err}")
                
                # Get video FPS for stats computation
                import cv2
                cap = cv2.VideoCapture(str(in_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                cap.release()
                
                # Compute stats
                postprocess_and_write(out_json, fps, out_stats)
                
                # Run tracker
                try:
                    import sys
                    tracker_result = subprocess.run(
                        [sys.executable, str(config.get_tracker_script_path()), uid],
                        capture_output=True,
                        text=True,
                        cwd=str(config.root_dir)
                    )
                    
                    if tracker_result.returncode != 0:
                        logger.error(f"Tracker failed for {uid}: {tracker_result.stderr}")
                    else:
                        logger.info(f"Tracker completed for {uid}")
                        
                except Exception as e:
                    logger.error(f"Failed to run tracker for {uid}: {e}")
                    
            except Exception as e:
                logger.error(f"Processing error for {uid}: {e}")
                with open(config.result_dir / f"{uid}_error.log", "w") as f:
                    f.write(f"Processing Error: {str(e)}")
        
        background_tasks.add_task(_job)
        
        return JSONResponse({
            "status": "processing",
            "uid": uid,
            "filename": file.filename,
            "message": "Upload successful, processing started",
            "video_path": str(in_path),
            "annotated_video": str(out_video),
            "detections_json": str(out_json),
            "stats_json": str(out_stats)
        })
        
    except Exception as e:
        logger.error(f"Upload error for {uid}: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.post("/upload_progressive")
async def upload_video_progressive(background_tasks: BackgroundTasks, file: UploadFile = File(...), model: str = None):
    """Upload a video file for progressive traffic analysis with real-time updates."""
    uid = uuid.uuid4().hex[:8]
    logger.info(f"Progressive upload started for UID: {uid}, filename: {file.filename}")
    
    try:
        # Validate file format
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Setup file paths
        in_path = config.upload_dir / f"{uid}_{file.filename}"
        
        # Save uploaded file
        with open(in_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        async def _progressive_job():
            """Progressive processing job with real-time updates."""
            try:
                logger.info(f"Starting progressive processing for {uid}")
                
                # Import progressive processor
                from streaming.progressive_processor import start_progressive_processing
                
                # Start progressive processing with WebSocket callback
                processor = start_progressive_processing(
                    uid=uid,
                    video_path=in_path,
                    model_path=config.get_model_path(),
                    confidence=config.confidence_threshold,
                    websocket_callback=lambda data: send_progress_update(uid, data)
                )
                
                # Run progressive processing
                await processor.process_video_progressively()
                
                logger.info(f"Progressive processing completed for {uid}")
                
                # Run tracker if needed
                try:
                    import sys
                    tracker_result = subprocess.run(
                        [sys.executable, str(config.get_tracker_script_path()), uid],
                        capture_output=True,
                        text=True,
                        cwd=str(config.root_dir)
                    )
                    
                    if tracker_result.returncode != 0:
                        logger.error(f"Tracker failed for {uid}: {tracker_result.stderr}")
                    else:
                        logger.info(f"Tracker completed for {uid}")
                        
                except Exception as e:
                    logger.error(f"Failed to run tracker for {uid}: {e}")
                    
            except Exception as e:
                logger.error(f"Progressive processing error for {uid}: {e}")
                # Send error via WebSocket if connected
                await send_progress_update(uid, {
                    "type": "processing_error",
                    "uid": uid,
                    "error": str(e),
                    "timestamp": time.time()
                })
        
        def _progressive_job_wrapper():
            """Wrapper to run progressive job in event loop."""
            try:
                import asyncio
                # Get or create event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Run the async job
                loop.run_until_complete(_progressive_job())
            except Exception as e:
                logger.error(f"Progressive job wrapper error for {uid}: {e}")
        
        # Start progressive processing in background
        background_tasks.add_task(_progressive_job_wrapper)
        
        return JSONResponse({
            "status": "processing",
            "uid": uid,
            "filename": file.filename,
            "message": "Progressive upload successful, processing started",
            "video_path": str(in_path),
            "websocket_endpoint": f"/ws/upload_progress/{uid}"
        })
        
    except Exception as e:
        logger.error(f"Progressive upload error for {uid}: {e}")
        raise HTTPException(status_code=500, detail="Progressive upload failed")

@app.get("/results/{filename}")
def get_result_file(filename: str):
    """Download a result file."""
    try:
        # Basic validation
        if '_' in filename:
            validate_uid_format(filename.split('_')[0])
        
        file_path = config.result_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Requested file not found: {filename}")
            raise HTTPException(status_code=404, detail="File not found")
        
        logger.info(f"Serving file: {filename}")
        return FileResponse(str(file_path))
        
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving file")

@app.get("/status/{uid}")
def get_status(uid: str):
    """Get processing status for a UID."""
    try:
        validate_uid_format(uid)
        
        # Check for various result files
        annotated = config.result_dir / f"{uid}_annotated.mp4"
        stats = config.result_dir / f"{uid}_stats.json"
        detections = config.result_dir / f"{uid}_detections.json"
        error_log = config.result_dir / f"{uid}_error.log"
        
        response = {
            "uid": uid,
            "annotated_exists": annotated.exists(),
            "detections_exists": detections.exists(),
            "stats_exists": stats.exists(),
            "error": error_log.read_text() if error_log.exists() else None,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        # Determine overall status
        if error_log.exists():
            response["status"] = "failed"
        elif annotated.exists() and stats.exists():
            response["status"] = "completed"
        else:
            response["status"] = "processing"
        
        logger.info(f"Status check for {uid}: {response['status']}")
        return JSONResponse(response)
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error checking status for {uid}: {e}")
        raise HTTPException(status_code=500, detail="Status check failed")

@app.get("/optimizer/plan")
def get_optimizer_plan():
    """
    Returns optimized green time allocation for all junctions.
    Uses real demand computation from mapped stats files.
    """
    try:
        # Import ATCS service
        try:
            from atcs.service import get_optimized_plan
        except Exception:
            import sys
            sys.path.append(str(config.root_dir / "atcs"))
            from service import get_optimized_plan  # type: ignore
        
        logger.info("Computing optimization plan")
        plan_data = get_optimized_plan(live_demand_overrides=live_demands)
        
        # Log historical data
        log_historical_data(
            plan_data.get("demands", {}), 
            plan_data.get("greens", {}), 
            plan_data.get("total_demand", 0)
        )
        
        logger.info(f"Optimization plan computed: total_demand={plan_data.get('total_demand', 0)}")
        return JSONResponse(plan_data)
        
    except Exception as e:
        # Fallback to demo demands if service fails
        logger.warning(f"ATCS service error: {e}, using fallback demands")
        
        demands = {"J1": 50, "J2": 30, "J3": 80, "J4": 20, "J5": 10, "J6": 60}
        
        try:
            from atcs.optimizer import allocate_greens
            greens = allocate_greens(demands)
        except Exception as fallback_error:
            logger.error(f"Fallback optimization failed: {fallback_error}")
            # Simple proportional allocation as last resort
            total_demand = sum(demands.values())
            available_time = config.cycle_time - (len(demands) * config.min_green_time)
            greens = {}
            for junction, demand in demands.items():
                proportion = demand / total_demand if total_demand > 0 else 1/len(demands)
                greens[junction] = config.min_green_time + (available_time * proportion)
        
        plan_data = {
            "greens": greens,
            "demands": demands,
            "timestamp": time.time(),
            "total_demand": sum(demands.values()),
            "fallback": True
        }
        
        # Log historical data even for fallback
        log_historical_data(demands, greens, sum(demands.values()))
        
        return JSONResponse(plan_data)

@app.post("/map_stats_to_junction")
async def map_stats(payload: dict):
    """
    Map a processed video result to a specific junction.
    payload = {
      "uid": "abcd1234",
      "stats_path": "backend/results/abcd1234_stats.json",
      "junction": "J1",
      "fps": 25
    }
    """
    try:
        data = read_mappings()
        # Remove any existing mapping for this uid
        data['mappings'] = [m for m in data['mappings'] if m.get('uid') != payload['uid']]
        
        entry = {
            "uid": payload['uid'],
            "stats": payload['stats_path'],
            "junction": payload['junction'],
            "fps": payload.get('fps', 25),
            "last_updated": time.time()
        }
        data['mappings'].append(entry)
        write_mappings(data)
        
        return JSONResponse({"status": "ok", "mapping": entry})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/delete_mapping/{uid}")
async def delete_mapping(uid: str):
    """
    Delete a junction mapping by UID.
    """
    try:
        data = read_mappings()
        original_count = len(data['mappings'])
        
        # Remove mapping with the specified UID
        data['mappings'] = [m for m in data['mappings'] if m.get('uid') != uid]
        
        if len(data['mappings']) == original_count:
            return JSONResponse({"error": "Mapping not found"}, status_code=404)
        
        write_mappings(data)
        return JSONResponse({"status": "ok", "message": f"Mapping for UID {uid} deleted"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/list_results")
def list_results():
    """
    List all processed result files for mapping to junctions.
    """
    try:
        files = [f.name for f in config.result_dir.iterdir() if f.is_file()]
        # Filter for relevant files (annotated videos, stats, detections)
        result_files = [f for f in files if any(suffix in f for suffix in ['_annotated.mp4', '_stats.json', '_detections.json'])]
        logger.info(f"Listed {len(result_files)} result files")
        return JSONResponse({"files": result_files})
    except Exception as e:
        logger.error(f"Error listing results: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing results: {e}")

@app.get("/list_mappings")
def list_mappings():
    """
    Get current junction mappings.
    """
    try:
        data = read_mappings()
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/historical_data")
def get_historical_data(hours: int = 24):
    """
    Get historical demand and optimization data.
    hours: number of hours of history to return (default 24)
    """
    try:
        with open(config.history_file, 'r') as f:
            history_data = json.load(f)
        
        # Filter by time range
        cutoff_time = time.time() - (hours * 3600)
        filtered_history = [
            entry for entry in history_data.get("history", [])
            if entry.get("timestamp", 0) > cutoff_time
        ]
        
        logger.info(f"Retrieved {len(filtered_history)} historical entries for last {hours} hours")
        return JSONResponse({
            "history": filtered_history,
            "count": len(filtered_history),
            "hours_requested": hours
        })
        
    except FileNotFoundError:
        logger.warning("No historical data found")
        return JSONResponse({"history": [], "count": 0, "hours_requested": hours})
    except Exception as e:
        logger.error(f"Error retrieving historical data: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving historical data: {e}")

# Global video streamer instance
video_streamer = None
live_demands = {}  # Store live demand overrides {junction_id: demand_value}

@app.post("/start_stream")
async def start_stream(payload: dict = None):
    """
    Start real-time video streaming with detection.
    payload: {"source": 0, "model": "models/best.pt", "confidence": 0.35}
    """
    global video_streamer
    try:
        # Stop existing stream if running
        if video_streamer:
            video_streamer.cleanup()
        
        # Import streamer class with fallback
        try:
            from streaming.video_streamer import VideoStreamer
        except Exception:
            import sys
            sys.path.append(str(config.root_dir / "streaming"))
            from video_streamer import VideoStreamer  # type: ignore
        
        # Get parameters
        source = payload.get("source", 0) if payload else 0
        model_path = payload.get("model", "models/best.pt") if payload else "models/best.pt"
        confidence = payload.get("confidence", 0.35) if payload else 0.35
        max_fps = payload.get("max_fps", 10) if payload else 10
        frame_skip = payload.get("frame_skip", 1) if payload else 1
        resize_width = payload.get("resize_width", 640) if payload else 640
        
        # Initialize streamer
        video_streamer = VideoStreamer(
            source=source, 
            model_path=model_path, 
            confidence=confidence,
            max_fps=max_fps,
            frame_skip=frame_skip,
            resize_width=resize_width
        )
        
        if video_streamer.initialize():
            return JSONResponse({"status": "started", "source": source, "model": model_path})
        else:
            return JSONResponse({"error": "Failed to initialize video stream"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/stream/mjpeg")
def stream_mjpeg():
    """Stream annotated frames as MJPEG multipart."""
    global video_streamer
    if not video_streamer:
        raise HTTPException(status_code=404, detail="No active stream")

    boundary = "frame"

    def frame_generator():
        while True:
            if not video_streamer or not video_streamer.latest_jpeg:
                time.sleep(0.05)
                continue
            frame_bytes = video_streamer.latest_jpeg
            yield (
                b"--" + boundary.encode() + b"\r\n" +
                b"Content-Type: image/jpeg\r\n" +
                f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode() +
                frame_bytes + b"\r\n"
            )
            time.sleep(0.05)

    return StreamingResponse(frame_generator(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """WebSocket pushing rolling stats & optional lightweight per-frame counts."""
    global video_streamer
    await ws.accept()
    try:
        while True:
            if not video_streamer:
                await ws.send_json({"error": "no_active_stream"})
                await asyncio.sleep(1)
                continue
            stats = video_streamer.get_recent_stats()
            # Add latest frame metadata only (no full image to keep light)
            meta = video_streamer.latest_detection or {}
            payload = {
                "type": "live_stats",
                "timestamp": time.time(),
                "frame": meta.get("frame"),
                "total_vehicles_frame": meta.get("total_vehicles"),
                "vehicles_per_minute": stats.get("vehicles_per_minute"),
                "avg_vehicles_per_frame": stats.get("avg_vehicles_per_frame"),
                "recent_total_frames": stats.get("total_frames"),
                "recent_total_vehicles": stats.get("total_vehicles")
            }
            await ws.send_json(payload)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

@app.post("/update_live_demand")
async def update_live_demand(payload: dict):
    """Update live demand for a junction from real-time stream."""
    global live_demands
    try:
        junction = payload.get("junction")
        demand = payload.get("demand", 0)
        
        if not junction or demand < 0:
            return JSONResponse({"error": "Invalid junction or demand"}, status_code=400)
        
        live_demands[junction] = float(demand)
        logger.info(f"Live demand updated: {junction} = {demand} vehicles/min")
        
        return JSONResponse({"status": "ok", "junction": junction, "demand": demand})
    except Exception as e:
        logger.error(f"Error updating live demand: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# Progressive upload WebSocket connections
progressive_connections: Dict[str, WebSocket] = {}

@app.websocket("/ws/upload_progress/{uid}")
async def websocket_upload_progress(ws: WebSocket, uid: str):
    """WebSocket for progressive upload processing updates."""
    await ws.accept()
    progressive_connections[uid] = ws
    
    try:
        logger.info(f"Progressive upload WebSocket connected for UID: {uid}")
        
        # Send initial connection confirmation
        await ws.send_json({
            "type": "connection_established",
            "uid": uid,
            "timestamp": time.time()
        })
        
        # Keep connection alive until processing completes or disconnects
        while True:
            try:
                # Listen for any client messages (like cancellation requests)
                message = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
                
                if message.get("type") == "cancel_processing":
                    logger.info(f"Processing cancellation requested for UID: {uid}")
                    # Stop progressive processor if exists
                    from streaming.progressive_processor import stop_progressive_processing
                    stop_progressive_processing(uid)
                    break
                    
            except asyncio.TimeoutError:
                # No message received, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error in upload progress WebSocket for {uid}: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"Progressive upload WebSocket disconnected for UID: {uid}")
    except Exception as e:
        logger.error(f"Progressive upload WebSocket error for {uid}: {e}")
    finally:
        # Clean up connection
        if uid in progressive_connections:
            del progressive_connections[uid]
        
        # Stop any active processor
        try:
            from streaming.progressive_processor import stop_progressive_processing
            stop_progressive_processing(uid)
        except:
            pass

async def send_progress_update(uid: str, update_data: Dict[str, Any]):
    """Send progress update to WebSocket client if connected."""
    if uid in progressive_connections:
        try:
            await progressive_connections[uid].send_json(update_data)
        except Exception as e:
            logger.error(f"Error sending progress update for {uid}: {e}")
            # Remove broken connection
            if uid in progressive_connections:
                del progressive_connections[uid]

@app.get("/stream_frame")
def get_stream_frame():
    """
    Get current frame from video stream as base64 JPEG with detection data.
    """
    global video_streamer
    try:
        if not video_streamer:
            return JSONResponse({"error": "No active stream"}, status_code=404)
        
        frame_b64, detection_data = video_streamer.get_frame_as_jpeg_base64()
        
        if frame_b64 is None:
            return JSONResponse({"error": "No frame available"}, status_code=404)
        
        return JSONResponse({
            "frame": frame_b64,
            "detection_data": detection_data,
            "frame_count": video_streamer.frame_count
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/stream_stats")
def get_stream_stats():
    """
    Get real-time detection statistics from video stream.
    """
    global video_streamer
    try:
        if not video_streamer:
            return JSONResponse({"error": "No active stream"}, status_code=404)
        
        stats = video_streamer.get_recent_stats()
        stats["frame_count"] = video_streamer.frame_count
        stats["active"] = True
        
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/stop_stream")
def stop_stream():
    """
    Stop the active video stream.
    """
    global video_streamer
    try:
        if video_streamer:
            video_streamer.cleanup()
            video_streamer = None
            return JSONResponse({"status": "stopped"})
        else:
            return JSONResponse({"error": "No active stream"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
