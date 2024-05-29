import face_recognition
import cv2
import os

# Load sample images and encode known faces
known_face_encodings = []
known_face_names = []

# Specify the directory path
directory = "known_people"

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
        print(base_name)

# Load the image for recognition
image = face_recognition.load_image_file("to_test/restaurant_zoomed4.jpg")

# Find all the faces and their encodings in the image
face_locations = face_recognition.face_locations(image, model='hog')
face_encodings = face_recognition.face_encodings(image, face_locations)
print("face_location = ", face_locations)
print("face_encoding = ", face_encodings)
face_names = []

for face_encoding in face_encodings:
    # Compare face encoding with known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # If a match is found, use the known face name
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    face_names.append(name)

# Display results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Draw a box around the face
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with the name below the face
    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

# Display the resulting image

cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
