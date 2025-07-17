import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')  

cam = cv2.VideoCapture(0)

ret, frame1 = cam.read()

bbox = cv2.selectROI("Select Object", frame1, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

(x, y, w, h) =  bbox
x =  (int) (x)
y = (int) (y)
w = (int) (w)
h = (int) (h)

selected_box = [x, y, x + w, y + h]  

initial_result = model.predict(source=frame1, conf=0.3)[0]
selected_class = None

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

   
    if max(0, yB - yA)*max(0, xB - xA) == 0:
        return 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAB = float(boxAArea + boxBArea - interArea)
    iou = interArea / boxAB
    return iou

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

while True:
    ret, frame = cam.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.3, stream=True)

    for result in results:
        for box in result.boxes:
            
            if int(box.cls[0]) != selected_class:
                continue

            x1, y1, x2, y2 =  box.xyxy[0]
            x1 =  (int) (x1)
            y1 = (int)  (y1) 
            x2 = (int)  (x2)
            y2 = (int)  (y2)
            iou = compute_iou([x1, y1, x2, y2], selected_box)

            if iou > 0.1:
                label = f"{model.names[int(box.cls[0])]} {box.conf[0]:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Update selected_box for next frame (track it)
                selected_box = [x1, y1, x2, y2]

    cv2.imshow("YOLOv8 Object Tracking (User-selected)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
