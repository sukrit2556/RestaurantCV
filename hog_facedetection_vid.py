import dlib
import cv2

#step1: read the image
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #step2: converts to gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #step3: get HOG face detector and faces
    hogFaceDetector = dlib.get_frontal_face_detector()
    faces = hogFaceDetector(gray, 1)

    #step4: loop through each face and draw a rect around it
    for (i, rect) in enumerate(faces):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        #draw a rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("breaking")
        break
cv2.destroyAllWindows()