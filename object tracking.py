import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Fast version

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# Read first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Cannot read first frame.")
    exit()

# Let the user select an object to track
bbox = cv2.selectROI("Select Object", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

(x, y, w, h) = map(int, bbox)
selected_box = [x, y, x + w, y + h]  # Convert to (x1, y1, x2, y2)

# Detect objects in the first frame to identify the selected object's class
initial_result = model.predict(source=first_frame, conf=0.3)[0]
selected_class = None

def compute_iou(boxA, boxB):
    # Compute intersection over union
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Find which detection matches the selected box (best IoU)
best_iou = 0
for box in initial_result.boxes:
    detected_box = box.xyxy[0].cpu().numpy().astype(int)
    iou = compute_iou(detected_box, selected_box)
    if iou > best_iou:
        best_iou = iou
        selected_class = int(box.cls[0])

if selected_class is None:
    print("Couldn't detect any object inside the selected box.")
    exit()

print(f"Tracking class: {model.names[selected_class]}")

# Start tracking loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.3, stream=True)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != selected_class:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            iou = compute_iou([x1, y1, x2, y2], selected_box)

            if iou > 0.1:
                label = f"{model.names[cls_id]} {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Update selected_box for next frame (track it)
                selected_box = [x1, y1, x2, y2]

    cv2.imshow("YOLOv8 Object Tracking (User-selected)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
