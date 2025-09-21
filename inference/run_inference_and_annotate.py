#!/usr/bin/env python3
"""
Run YOLOv8 detection on a video, save annotated video and per-frame JSON detections.
Usage:
  python inference/run_inference_and_annotate.py \
    --model runs/detect/guardian_v1/weights/best.pt \
    --source data/videos/test1.mp4 \
    --out_video outputs/annotated_test1.mp4 \
    --out_json outputs/test1_detections.json
"""

import argparse, json, os, time
from ultralytics import YOLO
import cv2
import numpy as np

def xyxy_to_xywh(xyxy):
    x1,y1,x2,y2 = xyxy
    return [float(x1), float(y1), float(x2-x1), float(y2-y1)]

def draw_boxes(frame, detections, class_names):
    for det in detections:
        cls = det['class']
        conf = det['conf']
        x,y,w,h = det['bbox']
        x1,y1,x2,y2 = int(x), int(y), int(x+w), int(y+h)
        color = (0,255,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = f"{class_names[int(cls)]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-18), (x1+tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return frame

def main(args):
    model = YOLO(args.model)  # load weights (PyTorch)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit("Cannot open video: " + args.source)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))

    out_json = []
    frame_idx = 0
    start_time = time.time()

    # get class names from model (Ultralytics model.names)
    names = model.names if hasattr(model, 'names') else {}

    max_frames_env = os.getenv("GUARDIAN_MAX_FRAMES")
    max_frames = None
    if max_frames_env:
        try:
            max_frames = int(max_frames_env)
            print(f"Limiting to max {max_frames} frames via GUARDIAN_MAX_FRAMES")
        except ValueError:
            print("Invalid GUARDIAN_MAX_FRAMES value, ignoring")

    print("Starting inference on", args.source, "imgsz=", args.imgsz)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts = time.time()
        # run inference (ultralytics returns a Results object)
        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        # results[0].boxes has xyxy, conf, cls
        detections = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy[0].numpy()
                conf = float(box.conf[0].cpu()) if hasattr(box.conf, 'cpu') else float(box.conf[0])
                cls = int(box.cls[0].cpu()) if hasattr(box.cls, 'cpu') else int(box.cls[0])
                xywh = xyxy_to_xywh(xyxy)
                detections.append({"bbox": xywh, "class": cls, "conf": conf})
        # draw boxes
        vis = frame.copy()
        vis = draw_boxes(vis, detections, names)
        writer.write(vis)

        # append to json
        out_json.append({
            "frame_idx": frame_idx,
            "ts": ts,
            "camera": args.camera_id,
            "detections": detections
        })

        frame_idx += 1
        # optionally print progress
        if max_frames is not None and frame_idx >= max_frames:
            print(f"Reached max frame limit {max_frames}; stopping early.")
            break
        if frame_idx % 100 == 0 and frame_idx > 0:
            elapsed = time.time() - start_time
            print(f"Processed {frame_idx} frames, {elapsed:.1f}s elapsed")

    cap.release()
    writer.release()
    # save json
    with open(args.out_json, 'w') as f:
        json.dump(out_json, f)
    print("Done. Annotated video:", args.out_video, "JSON:", args.out_json)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out_video", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--camera_id", default="video1")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out_video) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    main(args)