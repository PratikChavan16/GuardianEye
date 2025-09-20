# GuardianEye

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
