import face_recognition
import cv2

image = face_recognition.load_image_file("to_test/old_woman.jpg")
face_locations = face_recognition.face_locations(image)

print(face_locations)
for item in face_locations:
    x1, y1  = item[0], item[1]
    x2, y2  = item[2], item[3]
    image = cv2.rectangle(image, (y1, x1), (y2, x2), (0, 0, 255), 2)
cv2.imshow("output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()