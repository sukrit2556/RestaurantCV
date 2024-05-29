import dlib
import cv2
import face_recognition
#step1: read the image
image = cv2.imread("to_test/biden_with_cabinet1.webp")

#step2: converts to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#step3: get HOG face detector and faces
hogFaceDetector = dlib.get_frontal_face_detector()
faces = hogFaceDetector(gray, 1)
print(faces)
face_location = []
#step4: loop through each face and draw a rect around it
for (i, rect) in enumerate(faces):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    #draw a rectangle
    print(rect.top(), rect.right(),rect.bottom(),rect.left())
    face_location.append((rect.top(), rect.right(),rect.bottom(),rect.left()))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
print(face_location)
face_encodings = face_recognition.face_encodings(image, face_location)
print(face_encodings)
    
#step5: display the resulted image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()