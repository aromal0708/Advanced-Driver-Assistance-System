import torch
import cv2
import numpy as np
import pyttsx3
import threading

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Load YOLOv5s Model (optimized for speed & accuracy)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.eval()

# Open webcam for real-time video capture
cap = cv2.VideoCapture(0)

# Define Safe Distance and Region of Interest (ROI)
MIN_SAFE_DISTANCE = 100  
MIN_BBOX_SIZE = 5000  # Ignore small (far) vehicles
FRAME_WIDTH = int(cap.get(3))  # Get webcam width
FRAME_HEIGHT = int(cap.get(4))  # Get webcam height
CENTER_X_MIN = FRAME_WIDTH // 3  # Left boundary of center region
CENTER_X_MAX = 2 * (FRAME_WIDTH // 3)  # Right boundary of center region

# Vehicle classes to detect
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# Function to play alert in a separate thread
def play_alert():
    tts_engine.say("Collision Alert!")
    tts_engine.runAndWait()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor and process with YOLOv5
    results = model(frame)

    alert_triggered = False  # Track if an alert was already triggered

    for det in results.xyxy[0]:  
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = results.names[int(cls)]  

        # Ignore low-confidence detections
        if conf < 0.5:
            continue  

        # Calculate bounding box center and area
        bbox_center_x = (x1 + x2) / 2
        bbox_area = (x2 - x1) * (y2 - y1)

        # Only process vehicles in front (center region)
        if label in VEHICLE_CLASSES and bbox_area > MIN_BBOX_SIZE and CENTER_X_MIN <= bbox_center_x <= CENTER_X_MAX:
            color = (0, 255, 0) if bbox_area > MIN_SAFE_DISTANCE else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label}: {int(bbox_area)} px", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Trigger alert only if vehicle is too close
            if bbox_area > MIN_SAFE_DISTANCE and not alert_triggered:
                alert_thread = threading.Thread(target=play_alert)
                alert_thread.start()
                alert_triggered = True

    # Display output frame
    cv2.imshow("Collision Warning System", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
