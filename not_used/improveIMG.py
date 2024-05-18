def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)
# importing libraries 
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
cap = cv2.VideoCapture('inference/videos/ver_1.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Reading image from folder where it is stored 
    img = frame 

    img = adjust_brightness_contrast(img)
    img = sharpen_image(img)
    # denoising of image saving it into dst image 
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
    print("cukk")

    cv2.imshow('Processed Frame', dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()