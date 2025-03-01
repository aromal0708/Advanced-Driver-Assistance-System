import torch
import cv2
import numpy as np
import pyttsx3
import threading

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Load YOLOv5s Model (More accurate than YOLOv5n)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.eval()

# Open webcam for real-time video capture
cap = cv2.VideoCapture(0)

# Minimum safe distance in pixels
MIN_SAFE_DISTANCE = 100  

# Vehicle classes for alerting
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# Function to play collision alert using TTS in a separate thread
def play_alert():
    tts_engine.say("Collision Alert!")
    tts_engine.runAndWait()

# Process video frames in real-time
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
        
        # Estimate object size as rough distance metric
        distance = int((x2 - x1) * (y2 - y1) / 100)

        # Only process vehicles
        if label in VEHICLE_CLASSES:
            color = (0, 255, 0) if distance > MIN_SAFE_DISTANCE else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label}: {distance} px", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Trigger alert only if too close and no alert was played in this frame
            if distance < MIN_SAFE_DISTANCE and not alert_triggered:
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
