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
        print(filepath)

# Initialize variables
face_encodings = []
face_names = []

# Open video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read video frame
    ret, frame = video_capture.read()

    # Resize frame for faster processing (optional)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (OpenCV default) to RGB color
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and their encodings in the current frame
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face_locations = []
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray)
    #convert to form of face recognition
    for i, item in enumerate(faces):
        x, y, w, h = faces[i]
        top = y
        right = x + w
        bottom = y + h
        left = x
        face_locations.append((top, right, bottom, left))
    
    
    print(face_locations)

    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    print(face_encodings)
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
    print("face name = ", face_names)
    # Display results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since we scaled them down
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        print("top right bottom left = ", top, right, bottom, left)
        # Draw a box around the face
        print(f"(x1, y1) = {(left, top)} and (x2, y2) = {(right, bottom)}")
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()