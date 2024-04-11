# %%
import threading
import cv2
from ultralytics import YOLO
import csv
import sys
all_frames_ready = threading.Event()
stop_thread = False
overrall_frame_count = 0
end = False
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
    counterOBJ = personFoundObj[file_index-1]
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
        if frame_count == 1 or (frame_count - 1) % 1 == 0:
            
            if file_index == 5:
                overrall_frame_count = frame_count
                counterOBJ.frame_num_list.append(frame_count)

        # Track objects in frames if available
            results = model(frame, classes=(0,1))
            res_plotted = results[0].plot()
            counterOBJ.person_found_list.append(len(results[0]))
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
            # Display the combined frame
            cv2.imshow('Combined Video', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



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
#tracker_thread6 = threading.Thread(target=combine_frame, daemon=True)


# Start the tracker threads
tracker_thread1.start()
tracker_thread2.start()
tracker_thread3.start()
tracker_thread4.start()
tracker_thread5.start()
#tracker_thread6.start()

# Wait for the tracker threads to finish
tracker_thread1.join()
tracker_thread2.join()
tracker_thread3.join()
tracker_thread4.join()
tracker_thread5.join()

#tracker_thread6.join()
# Clean up and close windows
cv2.destroyAllWindows()
# %%
def exportCSV():
    data = [
            ['frame',
             'data_from_thread1',
             'data_from_thread2',
             'data_from_thread3',
             'data_from_thread4',
             'data_from_thread5'
        ]
        ]
    apple = [[personFoundObj[4].frame_num_list[0],
                     personFoundObj[0].person_found_list[0],
                     personFoundObj[1].person_found_list[0],
                     personFoundObj[2].person_found_list[0],
                     personFoundObj[3].person_found_list[0],
                     personFoundObj[4].person_found_list[0]
                     ]]
    apple1 = [personFoundObj[4].frame_num_list[1],
                     personFoundObj[0].person_found_list[1],
                     personFoundObj[1].person_found_list[1],
                     personFoundObj[2].person_found_list[1],
                     personFoundObj[3].person_found_list[1],
                     personFoundObj[4].person_found_list[1]
                     ]
    apple.append(apple1)
    print(apple)
    for i in range (0,len(personFoundObj[4].frame_num_list)):
        print("hello")
        data.append([personFoundObj[4].frame_num_list[i],
                     personFoundObj[0].person_found_list[i],
                     personFoundObj[1].person_found_list[i],
                     personFoundObj[2].person_found_list[i],
                     personFoundObj[3].person_found_list[i],
                     personFoundObj[4].person_found_list[i]
                     ])
        print(data)
    filename = "result_video/detect_restaurant30fps.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)
    print("CSV EXPORTED")
print(len(personFoundObj[0].person_found_list))
print(len(personFoundObj[4].frame_num_list))
print("he")
exportCSV()
# %%
