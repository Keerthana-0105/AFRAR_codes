import cv2
import face_recognition
import os
import numpy as np
import pandas as pd
from datetime import datetime

# === Excel and Image Setup ===
original_excel = "attendance.xlsx"
image_folder = "Images"

# Load base Excel sheet (assumed to have Registration Number, Name columns)
df = pd.read_excel(original_excel, engine='openpyxl')
df['Status'] = "Absent"  # Reset status for this session

# Load known face encodings
known_face_encodings = []
known_reg_numbers = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        reg_no = os.path.splitext(filename)[0]
        try:
            image_path = os.path.join(image_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_reg_numbers.append(reg_no)
        except Exception as e:
            print(f"⚠ Error with {filename}: {e}")

# === Webcam Setup ===
esp32_stream_url = "http://192.168.194.84:81/stream"
cap = cv2.VideoCapture(esp32_stream_url)
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam.")
    exit()

print("📷 Webcam started. Press 'q' to stop.")
recognized_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Could not read frame.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_labels = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        label = "Unknown"
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                reg_no = known_reg_numbers[best_match_index]
                student_row = df[df["Registration No"].astype(str) == reg_no]
                if not student_row.empty:
                    label = f"{reg_no} - {student_row.iloc[0]['Name']}"
                    if reg_no not in recognized_ids:
                        df.loc[df["Registration No"].astype(str) == reg_no, "Status"] = "Present"
                        recognized_ids.add(reg_no)
                        print(f"🟢 Marked Present: {label}")
        face_labels.append(label)

    # Draw bounding boxes
    for (top, right, bottom, left), label in zip(face_locations, face_labels):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Save to new dated Excel file ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
output_filename = f"attendance_{timestamp}.xlsx"
df.to_excel(output_filename, index=False)
print(f"✅ Attendance saved to: {output_filename}")
