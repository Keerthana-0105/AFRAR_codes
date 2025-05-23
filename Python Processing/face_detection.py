import cv2
import face_recognition
import os
import numpy as np
from dotenv import load_dotenv

# Load known faces and their filenames
known_face_encodings = []
known_face_names = []

# Folder where student images are saved
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, "Images")

# Load and encode faces from images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append(filename)

# Open default webcam
load_dotenv()
esp32_stream_url = os.getenv("URL")
cap = cv2.VideoCapture(esp32_stream_url)
#cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

print("📷 Webcam stream started. Press 'q' to quit.")

frame_count = 0  # Counter to skip frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Failed to grab frame.")
        break

    # Resize frame to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                print(f"🟢 Recognized: {name}")
                #mark present in the excel sheet
        face_names.append(name)

    # Draw rectangles and labels
    for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame was 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("Webcam Face Recognition", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()