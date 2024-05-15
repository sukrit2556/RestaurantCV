import face_recognition
import cv2
import os
from database_action import *

# Load sample images and encode known faces
known_face_encodings = []
known_face_names = []
employee_face_id = []

# Specify the directory path
directory = "../djangoAPP/mock_media/employee_face/"

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the filename is a file (not a directory)
    if os.path.isfile(os.path.join(directory, filename)):
        # Process the file here
        filepath = os.path.join(directory, filename)
        ###test_img = cv2.imread(filepath)
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

        _, employee_data = select_db("employee", ["employee_ID","employee_name"], [f"employee_image LIKE '%{filename}%'"])
        known_face_names.append(employee_data[0][1])
        employee_face_id.append(employee_data[0][0])
        print(f"employee name: {employee_data[0][1]} | employee_ID: {employee_data[0][0]}")
        ###cv2.imshow(employee_data[0][1], test_img)
        ###cv2.waitKey(0)
###cv2.destroyAllWindows()