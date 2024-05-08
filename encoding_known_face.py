import face_recognition
import cv2
import os

# Load sample images and encode known faces
known_face_encodings = []
known_face_names = []

# Specify the directory path
directory = "../djangoAPP/mock_media/employee_face"

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the filename is a file (not a directory)
    if os.path.isfile(os.path.join(directory, filename)):
        # Process the file here
        filepath = os.path.join(directory, filename)
        base_name, _ = os.path.splitext(filename)
        # Encode known faces
        image = face_recognition.load_image_file(filepath)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(base_name)
        print(filepath)