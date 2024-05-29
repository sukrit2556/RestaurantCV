import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('models/300_28.pt')
height=1080 #for some reason, ANYTHING else works for my HD camera for example 1079..
width=1920
# Open the video file
video_path = "inference/cashdrawer2.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    x1, y1 = 487, 203  # top-left corner
    x2, y2 = 869, 573  

    if success:
    

        #frame = frame[y1:y2, x1:x2]

        # Run YOLOv8 inference on the frame
        results = model(frame, conf = 0.6, save=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        #annotated_frame = cv2.resize(annotated_frame, (1920,1080))
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if len(results[0]) > 0:
            cv2.imwrite("result_frame1.jpg", annotated_frame)


        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()