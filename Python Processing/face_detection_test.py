import cv2
import os

# Create folder to save face images
save_dir = "captured_faces"
os.makedirs(save_dir, exist_ok=True)

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open webcam (or use your ESP32 stream)
# cap = cv2.VideoCapture(0)  # Default webcam
cap = cv2.VideoCapture("http://192.168.194.84:81/stream")  # ESP32 camera stream

if not cap.isOpened():
    print("❌ Cannot open video stream.")
    exit()

print("📷 Webcam started. Press 's' to save face, 'q' to quit.")
image_count = 0

while os.path.exists(os.path.join(save_dir, f"face_{image_count}.jpg")):
    image_count += 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Failed to capture frame.")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangle around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save face when 's' key is pressed
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        filename = os.path.join(save_dir, f"face_{image_count}.jpg")
        cv2.imwrite(filename, face_img)
        print(f"✅ Saved: {filename}")
        image_count += 1

    # Quit when 'q' is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
