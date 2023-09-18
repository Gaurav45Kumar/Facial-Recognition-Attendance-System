import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Load known faces and their encodings
akshay_image = face_recognition.load_image_file(r"C:\faces\akshay.jpeg")
akshay_encoding = face_recognition.face_encodings(akshay_image)[0]

armaan_image = face_recognition.load_image_file(r"C:\faces\armaan.jpeg")
armaan_encoding = face_recognition.face_encodings(armaan_image)[0]

diljit_image = face_recognition.load_image_file(r"C:\faces\diljit.jpeg")
diljit_encoding = face_recognition.face_encodings(diljit_image)[0]

modi_image = face_recognition.load_image_file(r"C:\faces\modi.jpeg")
modi_encoding = face_recognition.face_encodings(modi_image)[0]

rahul_image = face_recognition.load_image_file(r"C:\faces\rahul.jpeg")
rahul_encoding = face_recognition.face_encodings(rahul_image)[0]

sood_image = face_recognition.load_image_file(r"C:\faces\sood.jpeg")
sood_encoding = face_recognition.face_encodings(sood_image)[0]

# Create a list of known face encodings
known_face_encodings = [akshay_encoding, armaan_encoding, diljit_encoding, modi_encoding, rahul_encoding, sood_encoding]
known_face_names = ["akshay", "armaan", "diljit", "modi", "rahul", "sood"]

# Create a list of expected students
students = known_face_names.copy()

# Initialize variables for face detection
face_location = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open a CSV file for attendance
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

# Main loop for video capture and face recognition
while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Match faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Add the text if a person is present
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])

            # Write attendance to CSV
            lnwriter.writerow([name, current_date, now.strftime("%H:%M:%S")])

            # Display recognized face on the video feed
            cv2.rectangle(frame, (face_locations[0][3], face_locations[0][0]), (face_locations[0][1], face_locations[0][2]), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (face_locations[0][3] + 6, face_locations[0][2] - 6), font, 0.5, (255, 255, 255), 1)

    # Display the video feed
    cv2.imshow("Attendance", frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
