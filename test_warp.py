#import libraries 
import cv2 
import numpy as np 
x1, y1 = 0, 0  # top-left corner
x2, y2 = 640, 360  

# load the image using file path 
image = cv2.imread("37.jpg") 
image = image[y1:y2, x1:x2]

# defining source and destination points 
src_points = np.float32([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]]) 
dst_points = np.float32([[100, 100], [image.shape[1] - 100, 100], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]]) 

# calculatint the perspective transformation matrix 
matrix = cv2.getPerspectiveTransform(src_points, dst_points) 

# apply the perspective warp to the image using dimensions 
warped_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0])) 

# display the original and warped images 
cv2.imshow('Original Image', image) 
cv2.imshow('Warped Image', warped_image) 

# wait for 0 to be pressed and then close the windows 
cv2.waitKey(0) 
cv2.destroyAllWindows()
