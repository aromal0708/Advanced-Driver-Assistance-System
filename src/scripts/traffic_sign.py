import torch
import cv2
import pyttsx3  # Text-to-speech library

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech speed

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='src/models/weights/best.pt', force_reload=True)

# Define function to generate speech alert
def play_alert(label):
    alert_text = f"Warning! {label} ahead!"
    print(f"⚠️ Alert: {alert_text}")
    engine.say(alert_text)
    engine.runAndWait()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 inference
    results = model(frame)

    # Process detections
    for det in results.xyxy[0]:  # Bounding boxes
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]  # Get traffic sign label

        # Draw bounding box & label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Generate real-time alert
        play_alert(label)

    # Show frame
    cv2.imshow('Traffic Sign Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
