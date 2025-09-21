# GuardianEye Traffic Optimization System

A comprehensive traffic signal optimization system that uses computer vision and AI to analyze vehicle traffic patterns and optimize signal timing for better traffic flow.

## 🚦 System Overview

GuardianEye is an intelligent traffic management prototype that:
- **Detects vehicles** in traffic videos using YOLOv8 object detection
- **Tracks vehicles** across frames to count unique traffic
- **Computes demand** metrics for traffic optimization
- **Optimizes signal timing** using ATCS (Adaptive Traffic Control System) algorithms
- **Provides real-time monitoring** through a web interface

## 🏗️ Architecture

```
GuardianEye/
├── backend/           # FastAPI web server and API endpoints
├── inference/         # YOLOv8 vehicle detection pipeline
├── tracker/           # Vehicle tracking and unique counting
├── atcs/             # Traffic signal optimization algorithms
├── frontend/         # Web interface for monitoring
├── config/           # Configuration management
├── utils/            # Logging, validation, and utilities
└── models/           # AI model files
```

## 🔧 Features

### Core Functionality
- **Video Upload**: Upload traffic videos for analysis
- **Real-time Processing**: Background video processing with status tracking
- **Vehicle Detection**: YOLOv8-based detection of cars, trucks, buses, motorcycles
- **Vehicle Tracking**: DeepSORT and SimpleIOUTracker for unique vehicle counting
- **Traffic Optimization**: ATCS algorithms for signal timing optimization
- **Junction Mapping**: Map processed videos to specific traffic junctions

### API Endpoints
- `POST /upload` - Upload traffic videos
- `GET /status/{uid}` - Check processing status
- `GET /optimizer/plan` - Get optimized signal timing
- `GET /junctions` - Get junction configuration
- `GET /results/{filename}` - Download result files
- `GET /historical_data` - Get historical optimization data

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Webcam or traffic video files

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GuardianEye
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model** (if not present)
   ```bash
   # The system will automatically download yolov8n.pt on first run
   ```

### Running the System

1. **Start the backend server**
   ```bash
   python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the web interface**
   ```
   http://localhost:8000
   ```

3. **Upload traffic videos**
   - Use the web interface or API to upload MP4/AVI videos
   - Monitor processing status
   - View optimization results

## 📊 Usage

### 1. Video Processing Workflow
1. Upload a traffic video through the web interface
2. System automatically:
   - Detects vehicles using YOLOv8
   - Tracks vehicles across frames
   - Computes unique vehicle counts
   - Generates traffic statistics
3. Map the processed video to a junction
4. View optimization recommendations

### 2. Traffic Optimization
- Access `/optimizer/plan` to get current signal timing recommendations
- System considers:
  - Vehicle demand at each junction
  - Historical traffic patterns
  - ATCS optimization algorithms
  - Configurable cycle times and constraints

### 3. Monitoring and Analysis
- View processing status for uploaded videos
- Access historical optimization data
- Download result files (annotated videos, detection data, statistics)

## ⚙️ Configuration

### Environment Variables
```bash
# Optional configuration
GUARDIAN_MAX_FRAMES=300        # Max frames to process per video
GUARDIAN_IMAGE_SIZE=640        # YOLOv8 inference image size
GUARDIAN_CONFIDENCE=0.5        # Detection confidence threshold
GUARDIAN_LOG_LEVEL=INFO        # Logging level
GUARDIAN_HOST=0.0.0.0         # Server host
GUARDIAN_PORT=8000            # Server port
```

### Configuration Files
- `config/junctions.json` - Junction layout and connectivity
- `config/settings.py` - System configuration management
- `backend/mappings.json` - Video-to-junction mappings (auto-generated)

## 🛠️ Development

### Project Structure
```
├── backend/
│   ├── app.py              # FastAPI application
│   ├── uploads/            # Uploaded video files
│   ├── results/            # Processing results
│   └── aggregates/         # Unique count data
├── inference/
│   ├── run_inference_and_annotate.py  # YOLOv8 detection
│   └── postprocess_counts.py          # Statistics computation
├── tracker/
│   ├── tracker_service.py  # Vehicle tracking service
│   └── unique_counter.py   # Unique counting logic
├── atcs/
│   ├── service.py          # ATCS optimization service
│   └── optimizer.py        # Signal timing algorithms
├── config/
│   └── settings.py         # Configuration management
└── utils/
    ├── logger.py           # Logging system
    ├── exceptions.py       # Custom exceptions
    └── validators.py       # Input validation
```

### Adding New Features
1. Follow the modular architecture
2. Use the centralized configuration system (`config.settings`)
3. Implement proper logging with `utils.logger`
4. Add input validation with `utils.validators`
5. Use custom exceptions from `utils.exceptions`

## 🔍 Troubleshooting

### Common Issues

**1. Model not found errors**
- Ensure `yolov8n.pt` is in the project root
- Check that `models/best.pt` exists for custom models

**2. Video processing fails**
- Verify video format is supported (MP4, AVI, MOV, MKV)
- Check that OpenCV can read the video file
- Monitor logs in `backend/logs/`

**3. Tracking not working**
- DeepSORT may fail - system automatically falls back to SimpleIOUTracker
- Check unique count files in `backend/aggregates/`

**4. Optimization returns 0.0 demands**
- Ensure videos are mapped to junctions
- Check that unique counting completed successfully
- Verify aggregated data exists

### Logs and Debugging
- Application logs: `backend/logs/guardianeye_YYYYMMDD.log`
- Processing logs: `backend/results/{uid}_inference.log`
- Error logs: `backend/results/{uid}_error.log`

## 📈 Performance

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores, GPU support
- **Storage**: ~1GB per hour of processed video

### Processing Times
- **Detection**: ~1-2 minutes per minute of video
- **Tracking**: ~30 seconds per minute of video
- **Optimization**: Near real-time (<1 second)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code structure and style
4. Add proper logging and error handling
5. Test your changes thoroughly
6. Submit a pull request

## 📄 License

This project is a prototype for educational and research purposes.

## 🎯 Future Enhancements

- [ ] Real-time video streaming support
- [ ] Multiple camera integration
- [ ] Advanced ATCS algorithms
- [ ] Machine learning model retraining
- [ ] Dashboard with real-time analytics
- [ ] Integration with existing traffic control systems
- [ ] Mobile application for monitoring

---

**GuardianEye** - Intelligent Traffic Signal Optimization System

Vehicle and number plate detection using YOLOv8.

## Contents
- Dataset in `datasets/guardian` (YOLO format)
- Training artifacts (ignored) under `runs/`
- Inference script: `inference/run_inference_and_annotate.py`
- Models: `models/best.pt`, `models/best.onnx`

## Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install ultralytics opencv-python
```

## Training (example)
```powershell
yolo detect train model=yolov8s.pt data=datasets/guardian/data.yaml imgsz=1280 epochs=40 batch=8 device=0 project=runs/detect name=guardian_v1
```

## Validation
```powershell
yolo val model=runs/detect/guardian_v1/weights/best.pt data=datasets/guardian/data.yaml imgsz=1280
```

## Inference on Video
```powershell
python inference/run_inference_and_annotate.py `
  --model models/best.pt `
  --source data/videos/sample.mp4 `
  --out_video outputs/annotated_sample.mp4 `
  --out_json outputs/sample_detections.json
```

## Export to ONNX
```powershell
yolo export model=models/best.pt format=onnx imgsz=1280
```

## Directory Structure
```
GuardianEye/
  .gitignore
  README.md
  models/
  inference/
  datasets/
  data/videos/
  outputs/ (ignored)
  runs/ (ignored)
```

## Notes
- Large raw videos are ignored. Use Git LFS if you need to version them.
- Only curated weights in `models/` are versioned; training runs are reproducible.
- JSON output contains per-frame detections with timestamps.

## Future Improvements
- Confidence threshold experiments
- Batch processing script
- Performance comparison report

## Backend API
Start the server (development hot-reload):
```powershell
uvicorn backend.app:app --reload
```

Endpoints:
- `GET /` Basic API info and listed endpoints
- `GET /health` Returns `{ "status": "ok" }` and timestamp
- `POST /upload` Multipart form: field `file` (video), optional query `model` (default `models/best.pt`). Returns JSON with `uid` and output paths. Processing runs asynchronously.
- `GET /status/{uid}` Shows whether annotated video / detections / stats are ready or if an error log exists
- `GET /results/{filename}` Streams a produced artifact (annotated video, detections JSON, stats JSON)

Example upload with PowerShell (Invoke-WebRequest):
```powershell
Invoke-WebRequest -Method Post `
  -Uri http://127.0.0.1:8000/upload `
  -InFile .\data\videos\sample.mp4 `
  -ContentType 'application/octet-stream'
```
(Alternatively use a tool like curl or a REST client that supports multipart form: field name `file`).

Check status:
```powershell
Invoke-RestMethod http://127.0.0.1:8000/status/<uid>
```

Download detections JSON once ready:
```powershell
Invoke-WebRequest -OutFile detections.json `
  -Uri http://127.0.0.1:8000/results/<uid>_detections.json
```

Notes:
- CORS is permissive (`*`) for ease of local frontend integration; tighten this in production.
- `favicon.ico` returns 204 to keep logs clean.

## ATCS (Adaptive Traffic Control System)

The ATCS module optimizes traffic light timing based on vehicle demand from video analysis.

### Junction Map Configuration
Junction topology is defined in `config/junctions.json` with 6 intersections (J1-J6):
- Each junction has coordinates for UI plotting
- Neighbor relationships with distances in meters
- Forms a connected graph for traffic flow simulation

### Optimizer Service
Run the ATCS optimizer:
```powershell
python atcs/optimizer.py
```

The optimizer:
1. **Load Map**: Reads junction topology from `config/junctions.json`
2. **Compute Demands**: Converts vehicle counts from stats JSON into demand values per junction
3. **Allocate Greens**: Distributes green time (90s cycle) proportionally to demand with minimum 7s per phase
4. **Compute Offsets**: Calculates phase timing offsets for green wave coordination based on distance/speed

### Integration with Video Processing
1. Upload videos via `/upload` endpoint → generates `*_stats.json`
2. Map each video to a junction ID (manual configuration)
3. Run ATCS optimizer with collected stats to optimize signal timing
4. Apply computed green splits and offsets to traffic control system

### Example Output
```
Green splits (sec): {'J1': 16.6, 'J2': 12.8, 'J3': 22.4, 'J4': 10.8, 'J5': 8.9, 'J6': 18.5}
Offsets (sec): {'J1->J2': 50.0, 'J1->J3': 70.0, 'J2->J4': 60.0, ...}
```

### Future Enhancements
- Real-time demand updates from live video streams ✅
- Machine learning for demand prediction
- Multi-objective optimization (minimize delay + emissions)
- Integration with hardware traffic controllers

## Advanced Features

### Junction Mapping System
Map processed videos to specific junctions for personalized optimization:

**Backend Endpoints:**
- `POST /map_stats_to_junction` - Assign processed video results to junctions
- `GET /list_mappings` - View current video-to-junction assignments  
- `DELETE /delete_mapping/{uid}` - Remove mapping assignments
- `GET /list_results` - List available processed files for mapping

**Frontend Interface:**
- **"Manage Junction Mappings"** button opens mapping interface
- Select processed stats files and assign to junctions J1-J6
- View current mappings with demand values and timestamps
- Delete mappings with confirmation

### Real-time Demand Visualization
The system now displays actual demand values alongside signal timings:

- **Total Demand**: Aggregated vehicles/minute across all junctions
- **Per-junction Demand**: Individual demand values in junction cards
- **Last Updated**: Timestamp of most recent optimization
- **Fallback Indicator**: Shows when using demo data vs real mapped data

### Historical Trend Analysis
Comprehensive historical data tracking and visualization:

**Data Storage:**
- Automatic logging of all optimizer calls to `backend/history.json`
- Stores demands, green times, and timestamps
- Maintains last 1000 entries with automatic rotation

**Charts & Analytics:**
- **Total Demand Over Time**: Line chart showing traffic volume trends
- **Junction Comparison**: Bar chart of average demand per junction
- **Statistical Summary**: Min/max/average values with data point counts
- **Time Range Selection**: 1 hour, 6 hours, 24 hours, 1 week

**Endpoint:**
- `GET /historical_data?hours=24` - Retrieve historical trends

### Live Analysis (Annotated Playback & Instant Stats)
The Live Analysis panel (accessible via the new "Live Analysis" button in the frontend) lets you upload a video and view:

- Immediate processing status updates (upload → inference → stats ready)
- Inline embedded annotated video playback (`*_annotated.mp4`)
- Extracted statistics (total frames, total vehicles, avg vehicles/frame, first peak flow window, max concurrent class counts)
- Lightweight counts-over-time sparkline-style chart (no external libraries)
- Manual control: this workflow DOES NOT auto-map to a junction (kept separate so you can inspect quality before affecting optimization). Use the Mapping panel afterwards if you decide to include it.

**How It Works:**
1. Select a video file and click "Upload & Analyze".
2. Backend `/upload` starts asynchronous processing (detection + stats generation + annotation).
3. Frontend polls `/status/{uid}` every 3 seconds.
4. Once `stats_exists` is true it requests:
   - `/{uid}_stats.json` for metrics
   - `/{uid}_annotated.mp4` for playback (src set on the `<video>` element)
5. Statistics are rendered and the simple chart is drawn from `per_frame_counts`.

**Cancellation:**
- Press "Cancel" during polling to stop frontend polling (backend job continues; you can still access results manually later with the UID).

**Limitations / Notes:**
- Large videos still take time (bounded by model + tracking throughput); the panel simply surfaces progress earlier.
- Annotated video might appear a few seconds after stats (file writing delay); a notice is shown if the video tag errors initially.
- Peak flow shown is the first window present in `flow_windows`; future enhancement could visualize multiple rolling windows.
- Per-class numbers show maximum concurrent counts (not total unique) for quick visual density context.
- Does not modify optimization demand until you explicitly map via the Junction Mappings interface.

**Future Enhancements (candidates):**
- Progressive frame streaming (chunked MP4 or MJPEG) instead of waiting for full annotate
- Toggle to auto-map after completion
- Per-class trend plotting and hover details
- Link directly to mapping action with pre-filled UID

### Live Video Streaming
Real-time video analysis with immediate detection feedback:

**Video Sources:**
- Webcam integration (primary/secondary cameras)
- Video file playback for testing
- Configurable confidence thresholds (0.1-0.9)

**Live Features:**
- **Real-time Detection**: YOLO inference on live video frames  
- **Annotated Video Feed**: Bounding boxes and class labels
- **Live Statistics**: 60-second rolling window of vehicle counts
- **Stream Assignment**: Map live streams to specific junctions
- **Frame-by-frame Analysis**: Current detection counts and totals

**Backend Endpoints:**
- `POST /start_stream` - Initialize video stream with detection
- `GET /stream_frame` - Get current frame as base64 JPEG with detections
- `GET /stream_stats` - Real-time detection statistics  
- `POST /stop_stream` - Stop active video stream

**Frontend Interface:**
- **Live Video Display**: Real-time annotated video feed
- **Detection Statistics**: Live vehicle counts and rates
- **Source Selection**: Choose webcam or video file
- **Junction Assignment**: Map live data to traffic optimization

### Real-Time Streaming (New)
Continuous detection streaming with sub-second latency for immediate traffic monitoring:

**Architecture:**
- Background threaded processing loop in `VideoStreamer`
- MJPEG endpoint (`/stream/mjpeg`) for annotated frame streaming
- WebSocket endpoint (`/ws/live`) for rolling statistics updates
- Live demand injection into traffic optimizer

**Performance Options:**
- **Fast**: 2x frame skip, 320px inference, ~15 FPS processing
- **Balanced**: No skip, 640px inference, ~10 FPS processing  
- **Quality**: No skip, 1280px inference, ~8 FPS processing

**Live Optimization Integration:**
- Select junction for live demand override
- Rolling 60-second vehicle counts converted to vehicles/min
- Periodic optimizer refresh (every 15 seconds) with live demand
- Immediate signal plan updates based on real-time traffic

**Real-Time Endpoints:**
- `POST /start_stream` - Start threaded video processing (enhanced with performance params)
- `GET /stream/mjpeg` - Multipart MJPEG annotated frame stream
- `WS /ws/live` - WebSocket push of rolling stats and frame metadata
- `POST /update_live_demand` - Update junction demand from live stream
- `POST /stop_stream` - Stop processing and clean up resources

**Frontend Real-Time Panel:**
- Source selection (webcam/file) with confidence tuning
- Performance preset selection (fast/balanced/quality)
- Live junction assignment for optimizer integration
- MJPEG video display with automatic refresh
- WebSocket stats overlay with sparkline chart
- Rolling performance metrics (capture FPS, inference FPS, frame skip ratio)

**Usage Example:**
```powershell
# Start real-time stream with performance tuning
curl.exe -X POST "http://127.0.0.1:8000/start_stream" -H "Content-Type: application/json" -d "{\"source\": 0, \"confidence\": 0.35, \"max_fps\": 10, \"frame_skip\": 2, \"resize_width\": 640}"

# View MJPEG stream in browser
# Navigate to: http://127.0.0.1:8000/stream/mjpeg

# Connect WebSocket for live stats
# ws://127.0.0.1:8000/ws/live

# Update live demand for junction
curl.exe -X POST "http://127.0.0.1:8000/update_live_demand" -H "Content-Type: application/json" -d "{\"junction\": \"J1\", \"demand\": 45.2}"

# Stop stream
curl.exe -X POST "http://127.0.0.1:8000/stop_stream"
```

**Performance Considerations:**
- Frame skipping reduces CPU load for real-time applications
- Smaller inference resolution (320px-640px) maintains speed with acceptable accuracy
- WebSocket updates throttled to 0.5-second intervals to avoid flooding
- MJPEG frames automatically throttled by processing loop FPS limit
- Live demand updates occur every 15 seconds to balance responsiveness with stability

**Limitations:**
- Single concurrent stream supported (stop existing before starting new)
- MJPEG has no adaptive bitrate (fixed quality)
- WebSocket reconnection must be handled by client
- Live demand overrides persist until stream stops or manual reset

### Usage Examples

**Complete Workflow:**
1. **Upload & Process Videos**:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload?model=models/best.pt" -F "file=@data/videos/junction1.mp4"
```

2. **Map to Junctions**: 
   - Open frontend → "Manage Junction Mappings"
   - Select processed files → assign to specific junctions

3. **Monitor Optimization**:
   - View real-time signal plans with demand-based timing
   - Check historical trends for traffic patterns
   - Start live streams for immediate feedback

4. **Analyze Performance**:
   - Historical charts show optimization effectiveness
   - Compare demand patterns across different time periods
   - Export data for external analysis

**Live Stream Analysis:**
```powershell
# Start live detection stream
curl.exe -X POST "http://127.0.0.1:8000/start_stream" -H "Content-Type: application/json" -d "{\"source\": 0, \"confidence\": 0.35}"

# Get real-time statistics  
curl.exe "http://127.0.0.1:8000/stream_stats"

# Stop stream
curl.exe -X POST "http://127.0.0.1:8000/stop_stream"
```

## Quickstart Demo (End-to-End)

Follow these steps to experience the full GuardianEye + ATCS pipeline:

### 1. Environment Setup
```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start Backend API
```powershell
uvicorn backend.app:app --reload
```
Backend will listen on `http://127.0.0.1:8000`.

### 3. (Optional) Open Frontend
Simply open `frontend/index.html` in your browser (no build step required).

### 4. Upload a Video
```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload?model=models/best.pt" -F "file=@data/videos/sample.mp4"
```
Response returns a `uid` (save it).

### 5. Poll Processing Status
```powershell
curl.exe http://127.0.0.1:8000/status/<uid>
```
Wait until `stats_exists` becomes `true`.

### 6. Tracker & Aggregates
Tracker runs automatically post-processing (produces `backend/aggregates/<uid>_unique_counts.json`). If needed to re-run:
```powershell
python tracker/tracker_service.py <uid>
```

### 7. Map Stats to Junction
```powershell
curl.exe -X POST http://127.0.0.1:8000/map_stats_to_junction ^
  -H "Content-Type: application/json" ^
  -d "{\"uid\":\"<uid>\",\"stats_path\":\"backend/results/<uid>_stats.json\",\"junction\":\"J1\",\"fps\":25}"
```

### 8. Get Optimizer Plan
```powershell
curl.exe http://127.0.0.1:8000/optimizer/plan
```
Returns real demand-based green splits (uses unique track counts when available).

### 9. View Historical Data
```powershell
curl.exe http://127.0.0.1:8000/historical_data?hours=6
```

### 10. Start Live Stream (Optional)
```powershell
curl.exe -X POST http://127.0.0.1:8000/start_stream -H "Content-Type: application/json" -d "{\"source\":0,\"confidence\":0.35}"
curl.exe http://127.0.0.1:8000/stream_stats
curl.exe -X POST http://127.0.0.1:8000/stop_stream
```

### 11. Clean Up (Optional)
Remove old results/mappings if needed:
```powershell
Remove-Item backend\results\* -Exclude *.gitkeep -Force
```

### Success Criteria
- Optimizer plan shows non-fallback demands
- Aggregates file contains rolling unique counts
- Historical endpoint returns recent entries
- (Optional) Live stream endpoints respond with frame/stats

If any step fails, check logs under `backend/results/<uid>_error.log` or console output.

