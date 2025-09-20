# inference/postprocess_counts.py
import json, os
from collections import defaultdict

CLASS_MAP = None  # optional: mapping index->name

def load_detections(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def per_frame_counts(detections):
    """
    detections: list of frames as produced by run_inference_and_annotate
    returns: list of {frame_idx, ts, counts: {class_name: n}}
    """
    global CLASS_MAP
    summaries = []
    for frame in detections:
        counts = defaultdict(int)
        for d in frame.get('detections', []):
            cls = d.get('class', 0)
            name = CLASS_MAP.get(str(cls), str(cls)) if CLASS_MAP else str(cls)
            counts[name] += 1
        summaries.append({
            "frame_idx": frame['frame_idx'],
            "ts": frame['ts'],
            "counts": dict(counts)
        })
    return summaries

def sliding_window_flow(summaries, fps, window_seconds=60):
    """
    compute flow in vehicles per minute for each sliding window.
    We'll sum counts over window_seconds and scale to per-minute.
    fps: frames per second used during inference reading.
    """
    frames_per_window = max(1, int(window_seconds * fps))
    n = len(summaries)
    flows = []  # per center-timestamp
    seq = []
    for s in summaries:
        total = sum(s['counts'].values())
        seq.append(total)
    if not seq:
        return flows
    cur_sum = sum(seq[:frames_per_window])
    flows.append({
        "center_frame": frames_per_window//2,
        "vehicles_per_minute": cur_sum * (60.0 / window_seconds)
    })
    for i in range(1, n - frames_per_window + 1):
        cur_sum = cur_sum - seq[i-1] + seq[i + frames_per_window - 1]
        flows.append({
            "center_frame": i + frames_per_window//2,
            "vehicles_per_minute": cur_sum * (60.0 / window_seconds)
        })
    return flows

def compute_stats(detections_json_path, out_stats_path, fps=25, window_seconds=60, class_map=None):
    global CLASS_MAP
    if class_map:
        CLASS_MAP = {str(k): v for k, v in class_map.items()}
    det = load_detections(detections_json_path)
    per_frame = per_frame_counts(det)
    flows = sliding_window_flow(per_frame, fps, window_seconds=window_seconds)
    stats = {
        "per_frame_counts": per_frame,
        "flow_windows": flows,
        "summary": {
            "total_frames": len(per_frame),
            "total_vehicles_detected": sum(sum(c.values()) for c in (f['counts'] for f in per_frame))
        }
    }
    os.makedirs(os.path.dirname(out_stats_path) or ".", exist_ok=True)
    with open(out_stats_path, 'w') as f:
        json.dump(stats, f)
    return stats

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python postprocess_counts.py detections.json fps out_stats.json")
        sys.exit(1)
    _, in_json, fps, out_json = sys.argv
    compute_stats(in_json, out_json, fps=float(fps))
    print("Wrote", out_json)
