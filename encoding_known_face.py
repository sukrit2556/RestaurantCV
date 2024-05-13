import face_recognition
import cv2
import os

# Load sample images and encode known faces
known_face_encodings = []
known_face_names = []

# Specify the directory path
directory = "../djangoAPP/mock_media/employee_face/first_face_employee"

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the filename is a file (not a directory)
    if os.path.isfile(os.path.join(directory, filename)):
        # Process the file here
        filepath = os.path.join(directory, filename)
        base_name, _ = os.path.splitext(filename)
        print(filepath, end="")
        # Encode known faces
        image = face_recognition.load_image_file(filepath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_location = face_recognition.face_locations(image_rgb, model='hog')
        face_encoding = face_recognition.face_encodings(image)[0]
        print(face_location, end="")
        
        if face_location:
            print('\033[92m' + '  [pass]' + '\033[0m')
        else:
            print('\033[91m' + '  [fail]' + '\033[0m')
        known_face_encodings.append(face_encoding)
        known_face_names.append(base_name)
        