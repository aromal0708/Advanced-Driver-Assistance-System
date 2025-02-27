import torch
import cv2
import pyttsx3
import threading  # ✅ Run speech without blocking video

# Set device (Use GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize Text-to-Speech (TTS) engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed

# Load YOLOv5 model with half precision for speed
model = torch.hub.load('ultralytics/yolov5', 'custom', path='src/models/weights/best.pt').to(device)
model.half()  # Enable FP16 precision for faster inference
print(f"Model is running on: {model.model.device}")

# Open webcam with lower resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Traffic Sign Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Sign Detection", 800, 600)

frame_skip = 3  # Process 1 out of every 3 frames
frame_count = 0
alerted_labels = set()  # Keep track of alerts to avoid repeated speech

# Function to generate speech alert in a separate thread
def play_alert(label):
    if label not in alerted_labels:  # Avoid repeating the same alert
        alerted_labels.add(label)  # Mark this label as alerted
        alert_text = f"Warning! {label} ahead!"
        print(f"⚠️ Alert: {alert_text}")

        # Run speech synthesis in a separate thread
        threading.Thread(target=speak, args=(alert_text,)).start()

# Function to handle speech synthesis
def speak(text):
    engine.say(text)
    engine.runAndWait()  # This runs without blocking the main loop

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip some frames to reduce lag

    results = model(frame, size=416)  # Reduce image size for faster inference

    for det in results.xyxy[0]:  # Bounding boxes
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]  

        # Draw bounding box & label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Generate real-time alert in a separate thread
        play_alert(label)  

    # Show frame
    cv2.imshow('Traffic Sign Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
