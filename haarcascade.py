import cv2

image_path = "to_test/biden_congress.jpg"
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray)
for i, item in enumerate(faces):
    x, y, w, h = faces[i]
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
print(faces)
cv2.imshow("Faces found", image)
cv2.waitKey(0)
cv2.destroyAllWindows()