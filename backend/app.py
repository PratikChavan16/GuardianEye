# backend/app.py
import os, uuid, subprocess, time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from inference.postprocess_counts import compute_stats

ROOT = Path.cwd()
UPLOAD_DIR = ROOT / "backend" / "uploads"
RESULT_DIR = ROOT / "backend" / "results"
INFERENCE_SCRIPT = ROOT / "inference" / "run_inference_and_annotate.py"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GuardianEye V1 API")

def run_inference_process(video_path, out_video, out_json, model_path, imgsz=1280, conf=0.35):
    cmd = [
        "python", str(INFERENCE_SCRIPT),
        "--model", str(model_path),
        "--source", str(video_path),
        "--out_video", str(out_video),
        "--out_json", str(out_json),
        "--imgsz", str(imgsz),
        "--conf", str(conf)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def postprocess_and_write(detections_json, fps, out_stats_json):
    compute_stats(str(detections_json), str(out_stats_json), fps=fps, window_seconds=60)

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), model: str = "models/best.pt"):
    uid = uuid.uuid4().hex[:8]
    in_path = UPLOAD_DIR / f"{uid}_{file.filename}"
    with open(in_path, "wb") as f:
        content = await file.read()
        f.write(content)

    out_video = RESULT_DIR / f"{uid}_annotated.mp4"
    out_json = RESULT_DIR / f"{uid}_detections.json"
    out_stats = RESULT_DIR / f"{uid}_stats.json"

    def _job():
        start = time.time()
        code, out, err = run_inference_process(in_path, out_video, out_json, model)
        duration = time.time() - start
        if code != 0:
            with open(RESULT_DIR / f"{uid}_error.log", "w") as L:
                L.write(err or out)
            return
        import cv2
        cap = cv2.VideoCapture(str(in_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        postprocess_and_write(out_json, fps, out_stats)

    background_tasks.add_task(_job)

    return JSONResponse({
        "status": "processing",
        "uid": uid,
        "video_path": str(in_path),
        "annotated_video": str(out_video),
        "detections_json": str(out_json),
        "stats_json": str(out_stats)
    })

@app.get("/results/{filename}")
def get_result_file(filename: str):
    p = RESULT_DIR / filename
    if not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p))

@app.get("/status/{uid}")
def status(uid: str):
    annotated = RESULT_DIR / f"{uid}_annotated.mp4"
    stats = RESULT_DIR / f"{uid}_stats.json"
    detections = RESULT_DIR / f"{uid}_detections.json"
    error_log = RESULT_DIR / f"{uid}_error.log"
    res = {
        "annotated_exists": annotated.exists(),
        "detections_exists": detections.exists(),
        "stats_exists": stats.exists(),
        "error": error_log.read_text() if error_log.exists() else None
    }
    return JSONResponse(res)
