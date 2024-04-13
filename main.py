import cv2
import numpy as np
from ultralytics import YOLO
from function_bundle import *
import threading
import statistics

frame_count = 0
frame_rate = 0
stop_thread = False
check_available_started = False
list_total_count_cache = []
list_realtime_count_cache = []
availability_cache = []

#### Initialize the color randomizer for detected box ####
detection_colors, class_list = color_selector()

####################### THREADING PROCESS {BEGIN} #######################
def calculate_real_people_total():
    global list_total_count_cache, stop_thread
    #initialize sampling list
    table_name = "customer_events"
    fields = ("name", "address", "text1", "text2", "text3")
    values = ("John", "Highway21", "fuck", "dsd", "ssss")
    sampling_from_tables = []
    for _ in range (len(table_points)):
        sampling_from_tables.append([])

    while not stop_thread:
        if check_available_started:
            print("________________-Get in calculate now")
            list_total_count = list_realtime_count_cache.copy()
            
            print("sampling_from_tables", sampling_from_tables)

            # operating
            for i in range (len(table_points)):

                print(i)
            
                # if occupied then keep collecting until it reaches 100 collection
                if availability_cache[i] == "occupied" and len(sampling_from_tables[i]) < 100:
                    realtime_count_that_table = list_total_count[i]
                    sampling_from_tables[i].append(realtime_count_that_table)

                elif availability_cache[i] == "unoccupied": #if become unoccupied then reset
                    sampling_from_tables[i] = []
                    
                
            print("helloooooooooooooooooooooooooooooooooooooooooooooooooooo")
            print(sampling_from_tables)

            # determine the meaning
            for i in range (len(table_points)):
                if len(sampling_from_tables[i]) > 0: # have sampling data
                    list_total_count[i] = statistics.mode(sampling_from_tables[i])
                    #list_total_count[i] = max(sampling_from_tables[i])
                    #list_total_count[i] = round(sum(sampling_from_tables[i])/len(sampling_from_tables[i]))
                    if len(sampling_from_tables[i]) >= 100:
                        # add to database
                        pass
                        
                else: # have no sampling data
                    list_total_count[i] = 0
            list_total_count_cache = list_total_count.copy()

            time.sleep(5)
        else:
            pass




def check_available():
    global list_realtime_count_cache, stop_thread, availability_cache, check_available_started
    table_name = "customer_events"
    while not stop_thread:
        table_status = [] # 3 = occupied , 0 = unoccupied
        availability = []
        try:
            for table_no, _ in enumerate(table_points): #initialize list
                table_status.append(0)
            # start checking for 3 sampling time at each tables
            for i in range (0, 5):
                print("check available ====================================================", i)
                list_realtime_count = list_realtime_count_cache.copy()
                for table_no, _ in enumerate(table_points):
                    if list_realtime_count[table_no] > 0: #occupied detected that moment
                        table_status[table_no] += 1
                time.sleep(4)

            print("table_status", table_status)

            # determining the meaning of state

            for i, item in enumerate(table_status):
                if item >= 5/2:   # occupied
                    availability.append("occupied")
                    
                else:                   #unoccupied
                    availability.append("unoccupied")
            print(availability)


            availability_cache = availability.copy()
            if check_available_started == False:
                check_available_started = True
        except:
            pass

def now_frame_rate():
    global frame_rate
    second = 1
    while not stop_thread:
        frame_count_before = frame_count
        time.sleep(second)
        frame_count_after = frame_count
        frame_rate = int((frame_count_after - frame_count_before)/second)

# Create and start the thread
run_event = threading.Event()
run_event.set()
thread1 = threading.Thread(target=calculate_real_people_total)
thread2 = threading.Thread(target=now_frame_rate)
thread3 = threading.Thread(target=check_available)

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
frame_width = 1920
frame_height = 1080
fps = 30

size = (frame_width, frame_height) 
out = cv2.VideoWriter('555.avi', 
						cv2.VideoWriter_fourcc(*'MJPG'), 
						10, size) 


while True and not stop_thread:
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
        draw_table_point(frame, availability_cache)
        
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
        out.write(frame)

        if frame_count == 1:
            thread1.start()
            thread2.start()
            thread3.start()
        # Terminate run when "Q" pressed
        if cv2.waitKey(1) == ord("q"):
            break

thread1.join()
thread2.join()
thread3.join()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
out.release()