from ultralytics import YOLO
import json

# Load model and data
m = YOLO('runs/detect/guardian_v1/weights/best.pt')
data = json.load(open('outputs/IMG_2811_detections.json'))

print('Class names:', m.names)

# Count total detections
detections = sum(len(frame['detections']) for frame in data)
print(f'Total detections across all frames: {detections}')

# Count by class
classes = {}
for frame in data:
    for det in frame['detections']:
        cls = det['class']
        classes[cls] = classes.get(cls, 0) + 1

print('Detection counts by class:')
for cls, count in sorted(classes.items()):
    print(f'  {m.names[cls]}: {count}')
    
# Average detections per frame
frames_with_detections = sum(1 for frame in data if len(frame['detections']) > 0)
print(f'Frames with detections: {frames_with_detections}/{len(data)} ({frames_with_detections/len(data)*100:.1f}%)')
print(f'Average detections per frame: {detections/len(data):.1f}')