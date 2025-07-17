# Real-time-object-tracker-using-YOLOv8
This project is a real-time object tracking system built using YOLOv8 from Ultralytics and OpenCV. Unlike standard auto-detection pipelines, it allows the user to manually select an object in the first video frame using a bounding box, and then leverages YOLOv8’s detection capabilities to track that specific object across subsequent frames.

# Deployment Instructions

# 1. Clone the Repository
```bash
git clone  https://github.com/Aliweka2020/Real-time-object-tracker-using-YOLOv8.git
```
Or download the repository from the code button as Zip 

# 1. Install Dependencies   
```bash
pip install torch torchvision torchaudio
pip install opencv-python   
pip install ultralytics   
pip install numpy
```

# Implementation Details
The system is implemented in Python using the Ultralytics YOLOv8 model and OpenCV for real-time video processing and user interaction.

## Key Steps:
1. **Webcam Initialization**  
The system captures live video from the default webcam using ``` cv2.VideoCapture()```.

2. **Manual Object Selection**  
The user is prompted to draw a bounding box on the first frame using OpenCV’s ```cv2.selectROI()```. This selects the target object for tracking.

3. **Object Detection with YOLOv8**  
The selected frame is passed to a YOLOv8 model (e.g., yolov8n.pt) using Ultralytics’ ``` model.predict()``` method. All detected objects are returned with class IDs, bounding boxes, and confidence scores.

4. **Object Matching Using IoU**  
The system calculates the Intersection over Union (IoU) between the user-drawn box and YOLO-detected boxes to find the closest match. The class ID of the matched object is stored.

5. **Tracking Loop**  
  For every new frame:

   * YOLOv8 detects objects.

   * Only detections of the same class as the initially selected object are considered.

   * IoU is used again to determine if the detected box overlaps with the tracked region.

   * Matching boxes are visualized and updated as the new tracked object.

6. **Visualization**  
Bounding boxes and labels (class name + confidence) are drawn on the video stream using OpenCV.
