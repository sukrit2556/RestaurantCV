import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("weights\yolov8l.pt")

# Open the video file
video_path = "inference/videos/1hr_part1.mp4"
cap = cv2.VideoCapture(video_path)
frame_count = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame += 1
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        if frame_count %15 == 0:
            frame = frame[110:429, 41:456]
            results = model.track(frame, persist=True, conf=0.4, classes=[0], tracker="bytetrack.yaml")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()