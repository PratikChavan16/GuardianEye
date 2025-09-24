#!/usr/bin/env python3
"""
tracker/tracker_service.py

Consumes per-frame detection JSON (file produced by inference script) OR reads
a Redis stream / detections file and outputs track.update events with stable IDs.

Outputs:
 - backend/tracks/<uid>_tracks.json (append-only)
 - backend/aggregates/<uid>_unique_counts.json (sliding window unique counts)

This implementation uses deep_sort_realtime for tracking.
"""

import json, os, time
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import cv2

# folders
RESULTS_DIR = Path("backend/results")
TRACK_OUT = Path("backend/tracks")
AGG_OUT = Path("backend/aggregates")
TRACK_OUT.mkdir(parents=True, exist_ok=True)
AGG_OUT.mkdir(parents=True, exist_ok=True)

# configure tracker (attempt DeepSort, fallback to None)
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
    tracker = DeepSort(max_age=30,
                       n_init=3,
                       max_iou_distance=0.7,
                       max_cosine_distance=0.2,
                       nn_budget=100)
except Exception as e:
    print(f"DeepSort unavailable ({e}); will use IOU fallback only.")
    tracker = None

# helper: load detections produced by inference script
def load_detections_for_uid(uid):
    # expected file: backend/results/<uid>_detections.json
    p = RESULTS_DIR / f"{uid}_detections.json"
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, 'r') as f:
        return json.load(f)

# Speed estimation helper (optional)
def estimate_speed(track_hist_px, homography=None, fps=25):
    """
    track_hist_px: list of (ts, center_x_px, center_y_px)
    homography: a callable that maps (x_px, y_px) -> (x_m, y_m)
    returns speed in km/h estimated from last two positions
    """
    if len(track_hist_px) < 2 or homography is None:
        return None
    (t1,x1,y1) = track_hist_px[-2]
    (t2,x2,y2) = track_hist_px[-1]
    p1 = homography(x1,y1)
    p2 = homography(x2,y2)
    if p1 is None or p2 is None:
        return None
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
    dist_m = (dx*dx + dy*dy)**0.5
    dt = max(1e-6, t2 - t1)
    speed_mps = dist_m / dt
    speed_kph = speed_mps * 3.6
    return speed_kph

# Sliding-window unique counting
class UniqueCounter:
    def __init__(self, window_seconds=60, fps=25):
        self.window_seconds = window_seconds
        self.buffer = deque()  # elements: (timestamp, set_of_track_ids)
        self.fps = fps

    def add_frame(self, ts, track_ids_set):
        # append
        self.buffer.append((ts, set(track_ids_set)))
        # evict old
        cutoff = ts - self.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def unique_count(self):
        union = set()
        for _, s in self.buffer:
            union |= s
        return len(union)

def simple_homography_factory():
    """
    Placeholder homography: return pixel->meters mapping function.
    Replace with calibrated homography per camera in real use.
    For demo, we will assume 1 pixel == 0.02 meter (i.e., 50 pixels per meter).
    """
    px_per_m = 50.0
    def map_px_to_m(x_px, y_px):
        return (x_px/px_per_m, y_px/px_per_m)
    return map_px_to_m

def _find_video_for_uid(uid: str) -> Path | None:
    """Locate original or annotated video for a UID."""
    candidates = []
    uploads = Path("backend/uploads")
    results = Path("backend/results")
    if uploads.exists():
        candidates.extend([p for p in uploads.glob(f"{uid}_*.*") if p.suffix.lower() in {'.mp4', '.avi', '.mov'}])
    if results.exists():
        annotated = results / f"{uid}_annotated.mp4"
        if annotated.exists():
            candidates.append(annotated)
    return candidates[0] if candidates else None

class SimpleIOUTracker:
    def __init__(self, max_age=30, iou_thr=0.3):
        self.max_age = max_age
        self.iou_thr = iou_thr
        self.next_id = 1
        self.tracks = {}  # id -> {bbox,last_seen}

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / max(1e-6, area_a + area_b - inter)

    def update(self, detections, frame_idx):
        # detections: list of ([x1,y1,x2,y2], cls, conf)
        assigned = {}
        used = set()
        for det in detections:
            bbox = det[0]
            best_iou, best_tid = 0.0, None
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                i = self.iou(tr['bbox'], bbox)
                if i > best_iou:
                    best_iou, best_tid = i, tid
            if best_iou >= self.iou_thr and best_tid is not None:
                self.tracks[best_tid]['bbox'] = bbox
                self.tracks[best_tid]['last_seen'] = frame_idx
                assigned[best_tid] = bbox
                used.add(best_tid)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'bbox': bbox, 'last_seen': frame_idx}
                assigned[tid] = bbox
                used.add(tid)
        # prune old
        remove_ids = [tid for tid, tr in self.tracks.items() if frame_idx - tr['last_seen'] > self.max_age]
        for rid in remove_ids:
            self.tracks.pop(rid, None)
        # return pseudo track objects
        return [{'track_id': tid, 'bbox': assigned[tid]} for tid in assigned]

def run_tracker_for_uid(uid, fps=25):
    """Read detections + video frames for uid, feed frames to DeepSort. Fallback to IOU tracker if video not found."""
    print(f"Starting tracking for uid: {uid}")
    dets = load_detections_for_uid(uid)
    video_path = _find_video_for_uid(uid)
    cap = None
    use_fallback = False
    
    # If DeepSort is not available, force fallback
    if tracker is None:
        print("DeepSort not available, using fallback IOU tracker")
        use_fallback = True
    elif video_path and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open video {video_path}, using fallback IOU tracker")
            use_fallback = True
    else:
        print("No video found; using fallback IOU tracker (no embeddings).")
        use_fallback = True

    fallback_tracker = SimpleIOUTracker() if use_fallback else None
    out_tracks = []
    hist = defaultdict(list)
    uc = UniqueCounter(window_seconds=60, fps=fps)
    homography = simple_homography_factory()
    current_vid_idx = 0

    for frame in dets:
        frame_idx = frame['frame_idx']
        ts = frame['ts']
        # advance video to this frame
        frame_img = None
        if not use_fallback and cap is not None:
            while current_vid_idx <= frame_idx:
                ret, img = cap.read()
                if not ret:
                    print("Video ended prematurely; switching to fallback tracker")
                    use_fallback = True
                    fallback_tracker = SimpleIOUTracker()
                    cap.release()
                    frame_img = None
                    break
                if current_vid_idx == frame_idx:
                    frame_img = img
                current_vid_idx += 1

        raw_detections = []
        for d in frame.get('detections', []):
            x, y, w, h = d['bbox']
            xmin = float(x); ymin = float(y); xmax = float(x + w); ymax = float(y + h)
            cls = int(d.get('class', -1))
            conf = float(d.get('conf', 0.0))
            raw_detections.append([[xmin, ymin, xmax, ymax], cls, conf])

        track_ids_in_frame = set()
        if use_fallback:
            tracks = fallback_tracker.update(raw_detections, frame_idx)
            # unify interface with DeepSort track object expectations
            processed_tracks = []
            for tr in tracks:
                tid = tr['track_id']
                x1,y1,x2,y2 = tr['bbox']
                processed_tracks.append((tid, [x1,y1,x2,y2], None, None))
        else:
            if tracker is None:
                # Should not happen because use_fallback would be True, but guard anyway
                tracks = []
                processed_tracks = []
            else:
                tracks = tracker.update_tracks(raw_detections, frame=frame_img)
                processed_tracks = []
                for tr in tracks:
                    if not tr.is_confirmed():
                        continue
                    tlbr = tr.to_tlbr()
                    processed_tracks.append((tr.track_id, tlbr, getattr(tr,'det_class', None), getattr(tr,'det_conf', None)))

        for tid, tlbr, cls, conf in processed_tracks:
            cx = (tlbr[0] + tlbr[2]) / 2.0
            cy = (tlbr[1] + tlbr[3]) / 2.0
            hist[tid].append((ts, cx, cy))
            speed_kph = estimate_speed(hist[tid], homography=homography, fps=fps)
            track_event = {
                "ts": ts,
                "frame_idx": frame_idx,
                "uid": uid,
                "track_id": f"{uid}_t{tid}",
                "bbox": [float(tlbr[0]), float(tlbr[1]), float(tlbr[2]-tlbr[0]), float(tlbr[3]-tlbr[1])],
                "class": cls,
                "conf": conf,
                "centroid": [cx, cy],
                "speed_kph": speed_kph
            }
            out_tracks.append(track_event)
            track_ids_in_frame.add(track_event['track_id'])

        uc.add_frame(ts, track_ids_in_frame)
        track_out_file = TRACK_OUT / f"{uid}_tracks.jsonl"
        with open(track_out_file, "a") as f:
            for ev in out_tracks[-len(track_ids_in_frame):]:
                f.write(json.dumps(ev) + "\n")
        agg_out_file = AGG_OUT / f"{uid}_unique_counts.json"
        snapshot = {
            "ts": ts,
            "frame_idx": frame_idx,
            "unique_count_60s": uc.unique_count(),
            "unique_tracks_in_frame": len(track_ids_in_frame)
        }
        with open(agg_out_file, "a") as f:
            f.write(json.dumps(snapshot) + "\n")

    if cap is not None:
        cap.release()
    print(f"Tracking completed for uid: {uid}")
    return True

if __name__ == "__main__":
    # simple CLI: python tracker_service.py <uid>
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tracker_service.py <uid>")
        sys.exit(1)
    uid = sys.argv[1]
    run_tracker_for_uid(uid)