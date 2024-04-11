import cv2
import numpy as np
from ultralytics import YOLO
from function_bundle import *
import yaml
import threading

frame_count = 0
frame_rate = 0
stop_thread = False
list_total_count = []
list_realtime_count_cache = []
availability = []

#### Initialize the color randomizer for detected box ####
detection_colors, class_list = color_selector()

####################### THREADING PROCESS {BEGIN} #######################
def calculate_real_people_total():
    global list_total_count
    while not stop_thread:
        list_total_count = list_realtime_count_cache.copy()
        time.sleep(5)

def check_available():
    while not stop_thread:
        table_status = [] # 3 = occupied , 0 = unoccupied
        for table_no, _ in enumerate(table_points): #initialize list
            table_status.append(0)
        # start checking for 3 sampling time at each tables
        for i in range (0, 3):
            list_realtime_count = list_realtime_count_cache.copy()
            for table_no, _ in enumerate(table_points):
                if list_realtime_count[table_no] > 0: #occupied detected that moment
                    table_status[table_no] += 1
            time.sleep(5)

        # determining the meaning of state
        for i in table_status:
            if table_status >= 3:   # occupied
                availability.append("occupied")
            else:                   #unoccupied
                availability.append("Unoccupied")

def now_frame_rate():
    global frame_rate
    second = 1
    while not stop_thread:
        frame_count_before = frame_count
        time.sleep(second)
        frame_count_after = frame_count
        frame_rate = (frame_count_after - frame_count_before)/second

# Create and start the thread
thread1 = threading.Thread(target=calculate_real_people_total)
thread2 = threading.Thread(target=now_frame_rate)
thread3 = threading.Thread(target=check_available)
thread1.start()
thread2.start()
thread3.start()

####################### THREADING PROCESS {END} #######################

### load a pretrained YOLOv8n model ###
model = YOLO("weights/yolov8l.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 1920
frame_hyt = 1080

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

while True:
    # Capture frame-by-frame
    ret, frame = cap.read() # if frame is read correctly ret is True
    
    # Frame counting
    frame_count += 1

    ### Process the frame skipped ###
    if frame_count == 1 or (frame_count) % frame_skipped == 0:
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            stop_thread = True
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
        draw_table_point(frame)
        
        # put text on the bottom right bottom
        text_to_put_list = []
        text_to_put_list.append("frame " + str(frame_count) + " " + str(frame_rate) + " Frame/s")
        text_to_put_list.append(str(len(detect_params[0])) + " " + "person")
        text_to_put_list.append("realtime: " + str(list_realtime_count))
        text_to_put_list.append("total: " + str(list_total_count))
        text_to_put_list.append(str(availability))
        put_text_bottom_right(frame, text_to_put_list)

        
        # Display the resulting frame
        cv2.imshow("ObjectDetection", frame)
        list_realtime_count_cache = list_realtime_count.copy()
        reset_people_count()

        # Terminate run when "Q" pressed
        if cv2.waitKey(1) == ord("q"):
            break
thread1.join()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()