import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)


while True:
    
    ret, frame = video_capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    rgb_frame = frame[:, :, ::-1]
    try:
        face_locations = face_recognition.face_locations(rgb_frame)
        print(face_locations)
    except Exception as e:
        print("Error occured: ", e)
        exit()

    cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("breaking")
        break

print("exit outside")
video_capture.release()
cv2.destroyAllWindows()
