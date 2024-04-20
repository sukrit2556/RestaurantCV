import cv2
import numpy as np
from ultralytics import YOLO
from function_bundle import *
from thread_bundle import *
import threading
import atexit

frame_count = 0
frame_rate = 0
stop_thread = False
check_available_started = False
list_total_count_cache = []
list_realtime_count_cache = []
availability_cache = []
end_recording = False

#### Initialize the color randomizer for detected box ####
detection_colors, class_list = color_selector()

####################### THREADING PROCESS {BEGIN} #######################

def stop_threads():
    global stop_thread
    stop_thread = True

atexit.register(stop_threads)

thread_obj = Mythread()
# Create and start the thread
thread1 = threading.Thread(target=Mythread.calculate_real_people_total)
thread2 = threading.Thread(target=Mythread.now_frame_rate)
thread3 = threading.Thread(target=Mythread.check_available)
thread4 = threading.Thread(target=Mythread.combine_frame)
####################### THREADING PROCESS {END} #######################


### load a pretrained YOLOv8n model ###
model = YOLO("weights/yolov8l.pt", "v8")

### select the source of captured frame ###
source = config['source']
if source == "video_frame":
    url_path = config['video_frame']['url']
    cap = cv2.VideoCapture(url_path)

elif source == "live_frame":
    url_path = config['live_frame']['url']
    cap = cv2.VideoCapture(url_path)

### Start reading frame ###
if not cap.isOpened():
    print("Cannot open camera")
    exit()

### Amount of frame skipped ####
frame_skipped = config['frame_skip']

## delete incomplete data with no customer out time before the process
i = "0000-00-00 00:00:00"
condition_list = [f"customer_OUT = '{i}'"]
delete_data_db("customer_events", condition_list)

object1 = videoQueue()

while True and not stop_thread:
    # Capture frame-by-frame
    ret, frame = cap.read() # if frame is read correctly ret is True
    
    # Frame counting
    frame_count += 1

    ### Process the frame skipped ###
    if frame_count == 1 or (frame_count) % frame_skipped == 0:
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            stop_threads()
            end_recording = True
            break

        ### Predict on image ###
        detect_params = model(source=[frame], conf=0.4, show=False, save=False, classes=[0], tracker="bytetrack.yaml")

        # Convert tensor array to numpy
        DP = detect_params[0].cpu().numpy()
        #print(DP.boxes)
        #print("len", len(detect_params[0]))
        
        if len(DP) != 0:
            for i in range(len(detect_params[0])):

                boxes = detect_params[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.cpu().numpy()[0]
                conf = box.conf.cpu().numpy()[0]
                bb = box.xyxy.cpu().numpy()[0]

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                font = cv2.FONT_HERSHEY_COMPLEX
                
                # show center of a person on the frame
                horizon_center = int((int(bb[2]) + int(bb[0]))/2)
                vertical_center = int((int(bb[3]) + int(bb[1]))/2)
                
                cv2.circle(frame, (horizon_center, vertical_center), 10, (0,0,255), -1)

                count_table_people(horizon_center, vertical_center)

                # Display class name and confidence (only used in track mode)
                try:
                    id = int(boxes[i].id.cpu().numpy()[0])
                    cv2.putText(
                        frame,
                        class_list[int(clsID)] + " " + str(id) + " " + str(round(conf, 3)) + "%",
                        (int(bb[0]), int(bb[1]) - 10),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )
                except:
                    pass
        
        ### draw table area ###
        draw_table_point(frame, availability_cache)
        draw_from_points(frame, table_crop_points)
        
        # put text on the bottom right bottom
        text_to_put_list = []
        text_to_put_list.append("frame " + str(frame_count) + " | " + str(frame_rate) + " Frame/s")
        text_to_put_list.append(str(len(detect_params[0])) + " " + "person")
        text_to_put_list.append("realtime: " + str(list_realtime_count))
        text_to_put_list.append("total: " + str(list_total_count_cache))
        text_to_put_list.append(str(availability_cache))
        put_text_bottom_right(frame, text_to_put_list)
        print("available: ", availability_cache)
        

        # Display the resulting frame
        cv2.imshow("ObjectDetection", frame)
        #cv2.imwrite("sukrit_restaurant.jpg", frame) 
        list_realtime_count_cache = list_realtime_count.copy()
        reset_people_count()
        
        #object1.add_frame(frame) # uncomment without recording cause memory leak!

        if frame_count == 1:
            thread1.start() #calculate total person
            thread2.start() #check availability
            thread3.start() #check framerate
            #thread4.start() #record video
        # Terminate run when "Q" pressed
        if cv2.waitKey(1) == ord("q"):
            break
        with Mythread.lock1:
            frame_count = frame_count

thread1.join()
thread2.join()
thread3.join()
#thread4.join()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()