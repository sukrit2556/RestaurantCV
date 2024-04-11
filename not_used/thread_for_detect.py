import threading
import cv2
from ultralytics import YOLO
import sys
all_frames_ready = threading.Event()
stop_thread = False
overrall_frame_count = 0
end = False
blank_video = []
class videoQueue:
    def __init__(self, int_model_no):
        self.model_no = int_model_no
        self.frame_queue = []
        
    def add_frame(self, frame):
        self.frame_queue.append(frame)
        
    def get_frame(self):
        if len(self.frame_queue) > 0:
            return self.frame_queue.pop(0)
        return None
    
    def is_empty(self):
        if len(self.frame_queue) == 0:
            return True
        else:
            return False
        
class personFound:
    def __init__(self, int_model_no):
        self.model_no = int_model_no
        self.person_found_list = []
        self.frame_num_list = []      

def run_tracker_in_thread(filename, model, file_index):
    name_model_list = ["yolov8n","yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    name_model = name_model_list[file_index-1]
    frame_count = 0
    global blank_video
    
    global overrall_frame_count, end
    video = cv2.VideoCapture(filename)  # Read the video file

    while not stop_thread:
        ret, frame = video.read()  # Read the video frames
        

        # Exit the loop if no more frames in either video
        if not ret:
            if file_index == 5:
                end = True

            break

        frame_count += 1

    # Process the frame skipping
        if frame_count == 1 or (frame_count - 1) % 10 == 0:
            
            if file_index == 5:
                overrall_frame_count = frame_count
                blank_video.append(frame)

        # Track objects in frames if available
            results = model(frame,  classes=(0,1))
            res_plotted = results[0].plot()
            cv2.putText(res_plotted,
                        str(len(results[0])) + " " + "person",
                        (1200,900), 
                        cv2.FONT_HERSHEY_COMPLEX,
                        4,  
                        (0,0,255),
                        4
            )
            cv2.putText(res_plotted,
                        str(name_model),
                        (1200,1000), 
                        cv2.FONT_HERSHEY_COMPLEX,
                        3,  
                        (0,0,255),
                        3
            )
            res_plotted = cv2.resize(res_plotted, (512,288))
            objects[file_index-1].add_frame(res_plotted)
            cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
    # Release video sources
    video.release()

def combine_frame():
    completed_frames = [None, None, None, None, None]
    global stop_thread
    black_image = cv2.imread('inference/images/black.png')
    # Define video resolution and frame rate
    frame_width = 1920
    frame_height = 1080
    fps = 30

    size = (frame_width, frame_height) 
    # Define codec and create VideoWriter object
    out = cv2.VideoWriter('result_video/croud_detect_5FPS.avi', 
						cv2.VideoWriter_fourcc(*'MJPG'), 
						10, size) 

    while not stop_thread and not end:
        # Iterate over each video queue to check for frames
        for index, video_object in enumerate(objects):
            print(index)
            if completed_frames[index] is None:  # Check if frame for this index is not yet filled
                frame = video_object.get_frame()  # Get frame from video queue
                
                if frame is not None:  # If frame is available, update completed_frames
                    completed_frames[index] = frame
                
        if all(frame is not None for frame in completed_frames):
            print(completed_frames)
            black_img = cv2.resize(black_image, (512,288))
            top_row = cv2.hconcat(completed_frames[0:3])
            bottom_row = cv2.hconcat(completed_frames[3:5])
            bottom_row = cv2.hconcat([bottom_row, black_img])
            combined_frame = cv2.vconcat([top_row, bottom_row])
            completed_frames = [None, None, None, None, None]
            cv2.putText(combined_frame,
                        "frame: " + str(overrall_frame_count) ,
                        (1100,400), 
                        cv2.FONT_HERSHEY_COMPLEX,
                        2,  
                        (0,0,255),
                        2
            )
            combined_frame = cv2.resize(combined_frame, (1920,1080))
            out.write(combined_frame)
            # Display the combined frame
            cv2.imshow('Combined Video', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    out.release()

def make_video():
    frame_width = 1920
    frame_height = 1080
    fps = 30

    size = (frame_width, frame_height) 
    # Define codec and create VideoWriter object
    out1 = cv2.VideoWriter('result_video/croud_detect_5FPS_blank.avi', 
						cv2.VideoWriter_fourcc(*'MJPG'), 
						10, size) 
    frame_num = 1
    for video_frame in blank_video:
        frame_num += 10
        video_frame = cv2.resize(video_frame, (1920,1080))
        cv2.putText(video_frame,
            "frame: " + str(frame_num) ,
            (1100,400), 
            cv2.FONT_HERSHEY_COMPLEX,
            2,  
            (0,0,255),
            2
        )
        out1.write(video_frame)
        # Display the combined frame
        cv2.imshow('Combined Video', video_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out1.release()

                
       

# Load the models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8s.pt')
model3 = YOLO('yolov8m.pt')
model4 = YOLO('yolov8l.pt')
model5 = YOLO('yolov8x.pt')

# Define the video files for the trackers
video_file1 = "inference/videos/crowd.mp4"  # Path to video file, 0 for webcam
video_file2 = video_file1  # Path to video file, 0 for webcam, 1 for external camera

object1 = videoQueue(1)
object2 = videoQueue(2)
object3 = videoQueue(3)
object4 = videoQueue(4)
object5 = videoQueue(5)

objects = [object1, object2, object3, object4, object5]
personFoundObj = []

for i in range (0,5):
    personFoundObj.append(personFound(i+1))
print(personFoundObj)
# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, 1), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, 2), daemon=True)
tracker_thread3 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model3, 3), daemon=True)
tracker_thread4 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model4, 4), daemon=True)
tracker_thread5 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model5, 5), daemon=True)
tracker_thread6 = threading.Thread(target=combine_frame, daemon=True)


# Start the tracker threads
tracker_thread1.start()
tracker_thread2.start()
tracker_thread3.start()
tracker_thread4.start()
tracker_thread5.start()
tracker_thread6.start()

# Wait for the tracker threads to finish
tracker_thread1.join()
tracker_thread2.join()
tracker_thread3.join()
tracker_thread4.join()
tracker_thread5.join()

tracker_thread6.join()
print("make video")
make_video()

# Clean up and close windows
cv2.destroyAllWindows()

